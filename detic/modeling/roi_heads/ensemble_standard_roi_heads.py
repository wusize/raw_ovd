# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures import Boxes, Instances, PolygonMasks
from detectron2.layers import ShapeSpec
import torch
import numpy as np
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .ensemble_fast_rcnn import EnsembleFastRCNNOutputLayers
from detectron2.utils.events import get_event_storage
from time import time
from detic.modeling import context
from detectron2.modeling.proposal_generator.proposal_utils \
    import add_ground_truth_to_proposals
from detectron2.structures import pairwise_iou
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals, build_box_head
from detectron2.modeling.poolers import ROIPooler


@ROI_HEADS_REGISTRY.register()
class EnsembleStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        box_head_kd = kwargs.pop("box_head_kd")
        super().__init__(**kwargs)
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE

        self.context_modeling_cfg = cfg.CONTEXT_MODELLING
        assert self.context_modeling_cfg.ENABLE
        self.cfg = cfg
        ContextModelling = getattr(context, f'{self.context_modeling_cfg.VERSION}ContextModelling')
        self.context_modeling = ContextModelling(self.context_modeling_cfg,
                                                 num_words=self.box_predictor.num_words,
                                                 word_embed_dim=self.box_predictor.word_embed_dim,
                                                 word_dropout=self.box_predictor.word_dropout)
        self.box_head_kd = box_head_kd

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_head_kd = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        box_predictor = EnsembleFastRCNNOutputLayers(
            cfg,  box_head.output_shape
        )
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_head_kd": box_head_kd,
            "box_predictor": box_predictor,
        }

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

    # TODO: resolve bugs when there is no annotation
    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
        assert self.mask_pooler is not None
        features = [features[f] for f in self.mask_in_features]
        boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        num_boxes = sum([len(b) for b in boxes])

        # when number of positive proposals is 0, set pseudo mask
        if self.training and num_boxes == 0:
            # make pseudo-boxes and pseudo-masks
            h, w = instances[0].image_size
            pseudo_boxes = [Boxes(torch.tensor([[0.0, 0.0, w-1, h-1],
                                                ]).to(features[0].device))]
            pseudo_instances = [Instances(image_size=(h, w),
                                          proposal_boxes=pseudo_boxes[0],
                                          gt_classes=torch.arange(1).to(features[0].device),
                                          gt_masks=PolygonMasks(
                                              [[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]), ], ]))]
            pseudo_features = self.mask_pooler([f[:1] for f in features], pseudo_boxes)
            pseudo_losses = self.mask_head(pseudo_features, pseudo_instances)
            for k, v in pseudo_losses.items():
                pseudo_losses[k] = v * 0.0
            return pseudo_losses

        features = self.mask_pooler(features, boxes)
        return self.mask_head(features, instances)

    def _forward_box(self, features, proposals,
                     clip_images=None, image_info=None,
                     resized_image_info=None, group_infos=None):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

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

            # TODO contrastive learning
            if self.context_modeling_cfg.ENABLE:
                losses.update(self.context_modeling.get_loss(group_infos,
                                                             predictions, clip_images,
                                                             self.box_predictor.clip, image_info))
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
            box_features_kd = self.box_head_kd(box_features)
            box_features_cls = self.box_head(box_features)
            predictions = self.box_predictor(box_features_cls, box_features_kd)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _box_forward_train(self, box_features, proposals):
        sample_types = torch.cat([p.sample_types for p in proposals], dim=0)

        box_features_kd = self.box_head_kd(box_features[sample_types == 1])
        input_box_features_kd = self.box_predictor.pre_forward(box_features_kd)
        pseudo_words_kd = self.box_predictor.pred_words_kd(input_box_features_kd)   # a linear layer

        box_features_cls = self.box_head(box_features[sample_types == 0])
        input_box_features_cls = self.box_predictor.pre_forward(box_features_cls)
        pseudo_words_cls = self.box_predictor.pred_words(input_box_features_cls)

        storage = get_event_storage()
        tik = time()
        scores, _ = self.box_predictor.pred_cls_score(pseudo_words_cls)
        storage.put_scalar('time/pred_cls', time() - tik)
        proposal_deltas = self.box_predictor.bbox_pred(input_box_features_cls)
        predictions = dict(kd_pseudo_words=pseudo_words_kd,
                           scores=scores,
                           proposal_deltas=proposal_deltas)

        return predictions
