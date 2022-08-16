# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures.instances import Instances
import torch
import numpy as np
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, get_norm
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads, select_foreground_proposals
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from torch.nn import functional as F
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator.proposal_utils \
    import add_ground_truth_to_proposals
from detectron2.structures import pairwise_iou
from detic.modeling.roi_heads.context_modelling import ContextModelling
from time import time


@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeads(Res5ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        stage_channel_factor = 2 ** 3
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor

        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.add_image_box = cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX
        self.add_feature_to_prop = cfg.MODEL.ROI_BOX_HEAD.ADD_FEATURE_TO_PROP
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE
        self.box_predictor = DeticFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        self.context_modeling_cfg = cfg.CONTEXT_MODELLING
        self.cfg = cfg

        self.context_modeling = ContextModelling(self.context_modeling_cfg,
                                                 num_words=self.box_predictor.num_words,
                                                 word_embed_dim=self.box_predictor.word_embed_dim,
                                                 word_dropout=self.box_predictor.word_dropout)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None,
                ann_types=None, clip_images=None, image_info=None,
                resized_image_info=None):
        '''
        enable debug and image labels
        '''
        del images
        if self.training:
            proposals, group_infos = self.label_and_sample_proposals(
                proposals, targets, ann_types=ann_types, image_ids=list(image_info.keys()))

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        
        if self.add_feature_to_prop:
            feats_per_image = box_features.mean(dim=[2, 3]).split(
                [len(p) for p in proposals], dim=0)
            for feat, p in zip(feats_per_image, proposals):
                p.feat = feat

        if self.training:
            del features
            storage = get_event_storage()
            tik = time()
            predictions = self._box_forward_train(box_features, proposals)
            losses = self.box_predictor.losses(predictions,
                                               [p[p.sample_types == 0] for p in proposals])
            tok = time()
            # print('detector loss:', tok - tik)
            storage.put_scalar("time/detector_forward", np.float32(tok - tik))

            # TODO contrastive learning
            if self.context_modeling_cfg.ENABLE:
                losses.update(self.context_modeling.get_loss(group_infos,
                                                             predictions, clip_images,
                                                             self.box_predictor.clip, image_info))
                storage.put_scalar("time/contrast_learning", np.float32(time() - tok))

            if self.cfg.MODEL.WITH_IMAGE_LABELS:
                loss = self.image_label_loss(resized_image_info)
                if loss is None:
                    loss = list(losses.values())[0] * 0.0
                losses.update(image_label_loss=loss)

            return proposals, losses
        else:
            predictions = self.box_predictor(
                box_features.mean(dim=[2, 3]))
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, ann_types, image_ids):
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        group_infos = []
        for proposals_per_image, targets_per_image, ann_type, image_id \
                in zip(proposals, targets, ann_types, image_ids):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            added_instances, group_info = self.context_modeling.sample(proposals_per_image, self.mask_on, image_id)
            group_infos.append(group_info)
            # sample type: -1 for topk; 0 for det; 1 for clip-img; 2 for caption

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            proposals_per_image.set('sample_types',
                                    torch.zeros_like(proposals_per_image.gt_classes).int())
            if ann_type == 'only_caption':
                proposals_per_image = proposals_per_image[:0]
            proposals_per_image = Instances.cat([proposals_per_image, added_instances])
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        for k in ['base_scores', 'novel_scores', 'base_ious', 'novel_ious']:
            values = np.concatenate([g['gt_ious_scores'][k] for g in group_infos], axis=0)
            if len(values) > 0:
                val = values.mean()
            else:
                if f"gt_ious_scores/{k}" in storage._latest_scalars:
                    val = storage._latest_scalars[f"gt_ious_scores/{k}"][0]
                else:
                    val = np.zeros(1).sum()
            storage.put_scalar(f"gt_ious_scores/{k}", val)

        return proposals_with_gt, group_infos

    def _box_forward_train(self, box_features, proposals):
        sample_types = torch.cat([p.sample_types for p in proposals], dim=0)
        input_box_features = self.box_predictor.pre_forward(
            box_features.mean(dim=[2, 3]))

        pseudo_words = self.box_predictor.pred_words(input_box_features)
        scores = self.box_predictor.pred_cls_score(pseudo_words[sample_types == 0])
        proposal_deltas = self.box_predictor.bbox_pred(input_box_features[sample_types == 0])
        predictions = dict(kd_pseudo_words=pseudo_words[sample_types == 1],
                           caption_pseudo_words=pseudo_words[sample_types == 2],
                           scores=scores,
                           proposal_deltas=proposal_deltas)

        return predictions

    def image_label_loss(self, resized_image_info):
        proposals = resized_image_info['proposals']
        num_imgs = len(proposals)
        if num_imgs == 0:
            return None
        proposals = [p[:self.cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS] for p in proposals]
        image_labels = resized_image_info['image_labels']
        max_size_proposals = []
        for p in proposals:
            assert len(p) > 0
            areas = p.proposal_boxes.area()
            idx = areas.argmax().item()
            max_size_proposals.append(p[idx:idx+1])
        features = resized_image_info['features']
        proposal_boxes = [x.proposal_boxes for x in max_size_proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        box_features = self.box_predictor.pre_forward(
            box_features.mean(dim=[2, 3]))
        pseudo_words = self.box_predictor.pred_words(box_features)  # Nx1024 -> Nx4x512
        scores = self.box_predictor.pred_cls_score(pseudo_words)[..., :-1]
        targets = torch.zeros_like(scores)
        loss_weights = torch.ones_like(scores)
        for i in range(num_imgs):
            targets[i, image_labels[i]] = 1.0
            loss_weights[i, image_labels[i]] = self.cfg.MODEL.ROI_BOX_HEAD.IMAGE_POS_WEIGHT

        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-12)

        if loss > 100.0:
            loss = loss * 0.0

        return loss * self.cfg.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT


@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeadsExtraNorm(CustomRes5ROIHeads):
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels

    def forward(self, images, features, proposals, targets=None,
                ann_types=None, clip_images=None, image_info=None,
                resized_image_info=None):
        '''
        enable debug and image labels
        '''
        del images
        if self.training:
            proposals, group_infos = self.label_and_sample_proposals(
                proposals, targets, ann_types=ann_types)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        if self.training:
            del features
            storage = get_event_storage()
            tik = time()
            predictions = self._box_forward_train(box_features, proposals)
            losses = self.box_predictor.losses(predictions,
                                               [p[p.sample_types == 0] for p in proposals])
            tok = time()
            # print('detector loss:', tok - tik)
            storage.put_scalar("time/detector_forward", np.float32(tok - tik))

            # TODO contrastive learning
            if self.context_modeling_cfg.ENABLE:
                losses.update(self.context_modeling.get_loss(group_infos,
                                                             predictions, clip_images,
                                                             self.box_predictor.clip, image_info))
                storage.put_scalar("time/contrast_learning", np.float32(time() - tok))

            if self.cfg.MODEL.WITH_IMAGE_LABELS:
                loss = self.image_label_loss(resized_image_info)
                if loss is None:
                    loss = list(losses.values())[0] * 0.0
                losses.update(image_label_loss=loss)

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    [p[p.sample_types == 0] for p in proposals], self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                sample_types = torch.cat([p.sample_types for p in proposals], dim=0)
                box_features = box_features[sample_types == 0]
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            return proposals, losses
        else:
            predictions = self.box_predictor(
                box_features.mean(dim=[2, 3]))
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
