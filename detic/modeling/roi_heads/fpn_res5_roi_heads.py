# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import torch
import torch.nn as nn
import numpy as np
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from torch.nn import functional as F
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator.proposal_utils \
    import add_ground_truth_to_proposals
from detectron2.structures import pairwise_iou
from detectron2.layers import ShapeSpec
from detic.modeling.roi_heads.context_modelling import ContextModelling
from time import time
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet


@ROI_HEADS_REGISTRY.register()
class FPNRes5StandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE

        self.context_modeling_cfg = cfg.CONTEXT_MODELLING
        self.cfg = cfg

        self.custom_res5, out_channels = self._build_res5_block(cfg)
        self.merge_conv = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(3, 3),
                                    padding=1,
                                    stride=(1, 1))
        self.box_predictor = DeticFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        self.context_modeling = ContextModelling(self.context_modeling_cfg,
                                                 num_words=self.box_predictor.num_words,
                                                 word_embed_dim=self.box_predictor.word_embed_dim,
                                                 word_dropout=self.box_predictor.word_dropout)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    @classmethod
    def _build_res5_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _forward_box(self, features, proposals,
                     clip_images=None, image_info=None,
                     resized_image_info=None, group_infos=None):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # box_features = self.box_head(box_features)  # replace box_head with res5
        box_features = self.merge_conv(box_features)
        box_features = self.custom_res5(box_features).mean(dim=[2, 3])
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

            if self.cfg.MODEL.WITH_IMAGE_LABELS:
                loss = self.image_label_loss(resized_image_info)
                if loss is None:
                    loss = list(losses.values())[0] * 0.0
                losses.update(image_label_loss=loss)

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
            max_size_proposals.append(p[idx:idx + 1])
        features = resized_image_info['features']
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in max_size_proposals])
        box_features = self.box_head(box_features)
        box_features = self.box_predictor.pre_forward(box_features)
        pseudo_words = self.box_predictor.pred_words(box_features)  # Nx1024 -> Nx4x512
        scores = self.box_predictor.pred_cls_score(pseudo_words)[..., :-1]  # discard bg
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