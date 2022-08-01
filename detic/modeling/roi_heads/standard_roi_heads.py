# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import torch
import numpy as np
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator.proposal_utils \
    import add_ground_truth_to_proposals
from detectron2.structures import pairwise_iou
from detic.modeling.roi_heads.ucl_context_modelling \
    import UCLContextModelling as ContextModelling
from time import time


@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE
        self.box_predictor = DeticFastRCNNOutputLayers(
            cfg,  self.box_head.output_shape
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

    def _forward_box(self, features, proposals,
                     clip_images=None, image_info=None,
                     resized_image_info=None, group_infos=None):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        if self.training:
            losses = dict()
            storage = get_event_storage()
            tik = time()
            predictions = self._box_forward_train(box_features, proposals)
            losses.update(self.box_predictor.losses(predictions,
                                                    [p[p.sample_types == 0] for p in proposals]))
            tok = time()
            # print('detector loss:', tok - tik)
            storage.put_scalar("time/detector_forward", np.float32(tok - tik))

            if self.cfg.MODEL.WITH_IMAGE_LABELS:
                image_label_info = self.image_label_info(resized_image_info)
            else:
                image_label_info = None

            # TODO contrastive learning
            if self.context_modeling_cfg.ENABLE:
                losses.update(self.context_modeling.get_loss(group_infos,
                                                             predictions, clip_images,
                                                             self.box_predictor.clip, image_info,
                                                             image_label_info=image_label_info))
                storage.put_scalar("time/contrast_learning", np.float32(time() - tok))

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

            return losses
        else:
            predictions = self.box_predictor(box_features)
            del box_features
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

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
            del targets
            losses = self._forward_box(features, proposals, clip_images, image_info,
                                       resized_image_info, group_infos)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets, ann_types):
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        group_infos = []
        for proposals_per_image, targets_per_image, ann_type in zip(proposals, targets, ann_types):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            added_instances, group_info = self.context_modeling.sample(proposals_per_image, self.mask_on)
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

        return proposals_with_gt, group_infos

    def _box_forward_train(self, box_features, proposals):
        input_box_features = self.box_predictor.pre_forward(box_features)
        del box_features
        pseudo_words = self.box_predictor.pred_words(input_box_features)
        sample_types = torch.cat([p.sample_types for p in proposals], dim=0)
        storage = get_event_storage()
        tik = time()
        scores = self.box_predictor.pred_cls_score(pseudo_words[sample_types == 0])
        storage.put_scalar('time/pred_cls', time() - tik)
        proposal_deltas = self.box_predictor.bbox_pred(input_box_features[sample_types == 0])
        del input_box_features
        predictions = dict(kd_pseudo_words=pseudo_words[sample_types == 1],
                           caption_pseudo_words=pseudo_words[sample_types == 2],
                           scores=scores,
                           proposal_deltas=proposal_deltas)

        return predictions

    def image_label_info(self, resized_image_info):
        proposals = resized_image_info['proposals']
        num_imgs = len(proposals)
        if num_imgs == 0:
            return None
        proposals = [p[:self.cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS] for p in proposals]
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
        image_class_features = self.box_predictor.pred_cls_feature_all(pseudo_words)

        image_labels = resized_image_info['image_labels']

        return dict(image_class_features=image_class_features,
                    image_labels=image_labels)
