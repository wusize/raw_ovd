# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: support context modelling on cascade rcnn
import numpy as np
import torch
from time import time
from detectron2.config import configurable
from .context_modelling import ContextModelling
from .utils import multi_apply
import torch.nn.functional as F
from detectron2.modeling.proposal_generator.proposal_utils \
    import add_ground_truth_to_proposals
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detic.modeling import clip as CLIP
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from .custom_fast_rcnn import CustomFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class DeticCascadeROIHeads(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        mult_proposal_score: bool = False,
        ws_num_props: int = 512,
        mask_weight: float = 1.0,
        one_class_per_proposal: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.ws_num_props = ws_num_props
        self.mask_weight = mask_weight
        self.one_class_per_proposal = one_class_per_proposal

        self.context_modeling_cfg = self.cfg.CONTEXT_MODELLING
        self.context_modeling = ContextModelling(self.context_modeling_cfg,
                                                 num_words=self.box_predictor[0].num_words,
                                                 word_embed_dim=self.box_predictor[0].word_embed_dim,
                                                 word_dropout=self.box_predictor[0].word_dropout)
        clip_cfg = self.cfg.MODEL.CLIP
        self.clip, _ = CLIP.load(name=clip_cfg.NAME,
                                 use_image_encoder=clip_cfg.USE_IMAGE_ENCODER,
                                 download_root=clip_cfg.MODEL_ROOT)
        self.clip.init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'one_class_per_proposal': cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL,
        })
        cls.cfg = cfg
        return ret

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        return ret

    def _forward_box(self, features, proposals, targets=None,
                     ann_types=None, clip_images=None, image_info=None,
                     resized_image_info=None):
        """
        Add mult proposal scores at testing
        Add ann_type
        """
        if (not self.training) and self.mult_proposal_score:
            # TODO: inspect the objectness scores
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]
        
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, proposals)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions, group_infos = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals, group_infos))
        
        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals, group_infos) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(
                        predictions,
                        [p[p.sample_types == 0] for p in proposals])
                    if self.context_modeling_cfg.ENABLE:
                        stage_losses.update(self.context_modeling.get_loss(group_infos,
                                                                           predictions, clip_images,
                                                                           self.clip, image_info))
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})

            if self.cfg.MODEL.WITH_IMAGE_LABELS:
                image_label_losses = self.image_label_loss(resized_image_info,
                                                           device=list(losses.values())[0].device)
                losses.update(image_label_losses)

            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]
            if self.one_class_per_proposal:
                scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
            predictor, predictions, proposals, _ = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    def forward(self, images, features, proposals, targets=None,
                ann_types=None, clip_images=None, image_info=None,
                resized_image_info=None
                ):
        '''
        enable debug and image labels
        classifier_info is shared across the batch
        '''
        if self.training:
            storage = get_event_storage()
            tik = time()
            proposals = self.label_and_sample_proposals(proposals, targets, ann_types)   # TODO sample topk proposals
            tok = time()
            storage.put_scalar('roi_head_time/label_and_sample', tok - tik)
            losses = self._forward_box(features, proposals, targets,
                                       ann_types, clip_images, image_info, resized_image_info)
            tik = time()
            storage.put_scalar('roi_head_time/forward_box', tik - tok)
            proposals = [p[p.sample_types == 0] for p in proposals]
            # TODO: for seg_mask supervision
            mask_losses = self._forward_mask(features, proposals)
            tok = time()
            storage.put_scalar('roi_head_time/forward_mask', tok - tik)
            losses.update({k: v * self.mask_weight
                           for k, v in mask_losses.items()})
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _create_proposals_from_boxes(self, boxes, proposals):
        """
        Add objectness_logits
        """
        boxes = [Boxes(b.detach()) for b in boxes]
        new_proposals = []
        for p, boxes_per_image in zip(proposals, boxes):
            boxes_per_image.clip(p.image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                p = p[inds]
            p.proposal_boxes = boxes_per_image
            new_proposals.append(p)

        return new_proposals

    def _run_stage(self, features, proposals, stage, get_group_info=True):
        """
        Support context modelling
        """
        # TODO support context modelling
        group_infos = None
        if self.training and get_group_info and self.context_modeling_cfg.ENABLE:
            added_instances, group_infos = multi_apply(self.context_modeling.sample_on_topk,
                                                       [p[p.sample_types == -1] for p in proposals])
            proposals = multi_apply(Instances.cat, [[p[p.sample_types == 0],
                                                     added] for added, p in zip(added_instances, proposals)])
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)

        predictions = self._box_forward(stage, box_features, proposals,
                                        get_group_info)

        return predictions, group_infos

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, ann_types=None):
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image, ann_type in zip(proposals, targets, ann_types):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # det: 0
            proposals_per_image.set('sample_types',
                                    torch.zeros_like(matched_idxs).int())

            if self.context_modeling_cfg.ENABLE:
                added_instances = self.context_modeling.sample_topk_proposals(proposals_per_image, self.mask_on)
            else:
                added_instances = self.context_modeling.sample_topk_proposals(proposals_per_image[:0], self.mask_on)
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # consider ['gt_boxes', 'gt_classes', 'gt_masks']
                # from detectron2.structures.masks import PolygonMasks
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            if ann_type == 'only_caption':
                proposals_per_image = proposals_per_image[:0]    # non-det images

            proposals_per_image = Instances.cat([proposals_per_image, added_instances])
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            det_proposals_per_image = proposals_per_image[proposals_per_image.sample_types == 0]
            if len(det_proposals_per_image) == 0:
                num_fg_samples.append(0)
                num_bg_samples.append(0)
                continue
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, det_proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(det_proposals_per_image), 4))
                )
            proposals_per_image[proposals_per_image.sample_types == 0].gt_classes = gt_classes
            proposals_per_image[proposals_per_image.sample_types == 0].gt_boxes = gt_boxes
            # import pdb; pdb.set_trace()
            # proposals_per_image[proposals_per_image.sample_types == 0] = det_proposals_per_image

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _box_forward(self, stage, box_features, proposals, with_context=True):
        box_predictor = self.box_predictor[stage]
        input_box_features = box_predictor.pre_forward(box_features)
        del box_features
        pseudo_words = box_predictor.pred_words(input_box_features)
        if self.training and with_context:
            storage = get_event_storage()
            sample_types = torch.cat([p.sample_types for p in proposals], dim=0)
            tik = time()
            scores = box_predictor.pred_cls_score(pseudo_words[sample_types == 0], self.clip)
            tok = time()
            storage.put_scalar(f'roi_head_time/pred_cls/stage_{stage}', tok - tik)
            proposal_deltas = box_predictor.bbox_pred(input_box_features[sample_types == 0])
            tik = time()
            storage.put_scalar(f'roi_head_time/pred_box/stage_{stage}', tik - tok)
            del input_box_features
            predictions = dict(kd_pseudo_words=pseudo_words[sample_types == 1],
                               caption_pseudo_words=pseudo_words[sample_types == 2],
                               scores=scores,
                               proposal_deltas=proposal_deltas)
        else:
            scores = box_predictor.pred_cls_score(pseudo_words, self.clip)
            proposal_deltas = box_predictor.bbox_pred(input_box_features)
            del input_box_features
            predictions = dict(scores=scores,
                               proposal_deltas=proposal_deltas)

        return predictions

    def image_label_loss(self, resized_image_info, device):
        proposals = resized_image_info['proposals']
        num_imgs = len(proposals)
        if num_imgs == 0:
            return {f'stage_{k}_image_loss': torch.tensor(0.0, device=device)
                    for k in range(self.num_cascade_stages)}
        proposals = [p[:self.cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS] for p in proposals]
        image_labels = resized_image_info['image_labels']
        features = resized_image_info['features']
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
            predictions, _ = self._run_stage(features, proposals, k, get_group_info=False)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((predictions['scores'], proposals))
        losses = {}
        for stage, (scores, proposals) in enumerate(head_outputs):
            num_prs_per_img = [len(p) for p in proposals]
            preds, targets, loss_weights = multi_apply(self._get_pred_target_per_image,
                                                       scores.split(num_prs_per_img, dim=0),
                                                       proposals, image_labels)
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            loss_weights = torch.cat(loss_weights)

            loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
            loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-12)

            if loss > 100.0:
                loss = loss * 0.0
            losses[f'stage_{stage}_image_loss'] = loss * self.cfg.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT

        return losses

    def _get_pred_target_per_image(self, scores, proposals, image_labels):
        areas = proposals.area()
        idx = areas.argmax().item()
        preds = scores[idx, :-1]    # do not consider bg class
        loss_weights = torch.ones_like(preds)
        targets = torch.zeros_like(preds)
        targets[image_labels] = 1.0
        loss_weights[image_labels] = self.cfg.MODEL.ROI_BOX_HEAD.IMAGE_POS_WEIGHT

        return preds, targets, loss_weights
