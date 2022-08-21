# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures.boxes import Boxes
import torch
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
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.poolers import ROIPooler
from typing import List
from .disentangle_context_modelling import ContextModellingV2 as ContextModelling
from time import time
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from torch.autograd.function import Function


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


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
        def _record_gradient(grad):
            val = grad.norm()
            storage = get_event_storage()
            storage.put_scalar("gradients/detection", val.cpu().numpy())
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if self.training:
            box_features.register_hook(_record_gradient)
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

            # TODO contrastive learning
            if self.context_modeling_cfg.ENABLE:
                losses.update(self.context_modeling.get_loss(group_infos,
                                                             clip_images,
                                                             self.box_predictor.clip, image_info,
                                                             self,
                                                             features))
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
                resized_image_info=None, **kwargs):
        '''
        enable debug and image labels
        '''

        del images
        if self.training:
            proposals, group_infos = self.label_and_sample_proposals(
                proposals, targets, ann_types=ann_types, image_ids=list(image_info.keys()))
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
    def label_and_sample_proposals(self, proposals, targets, ann_types, image_ids):
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        group_infos = []
        for proposals_per_image, targets_per_image, ann_type, image_id in \
                zip(proposals, targets, ann_types, image_ids):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            sampled_instances, group_info = self.context_modeling.sample(proposals_per_image, self.mask_on, image_id)
            group_info['sampled_instances'] = sampled_instances
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
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        # for k in ['base_scores', 'novel_scores', 'base_ious', 'novel_ious', 'bg_scores']:
        #     values = np.concatenate([g['gt_ious_scores'][k] for g in group_infos], axis=0)
        #     if len(values) > 0:
        #         val = values.mean()
        #     else:
        #         if f"gt_ious_scores/{k}" in storage._latest_scalars:
        #             val = storage._latest_scalars[f"gt_ious_scores/{k}"][0]
        #         else:
        #             val = np.zeros(1).sum()
        #     storage.put_scalar(f"gt_ious_scores/{k}", val)

        return proposals_with_gt, group_infos

    def _box_forward_train(self, box_features, proposals):
        sample_types = torch.cat([p.sample_types for p in proposals], dim=0)
        input_box_features = self.box_predictor.pre_forward(box_features)
        del box_features

        pseudo_words = self.box_predictor.pred_words(input_box_features)
        scores = self.box_predictor.pred_cls_score(pseudo_words[sample_types == 0])
        proposal_deltas = self.box_predictor.bbox_pred(input_box_features[sample_types == 0])
        del input_box_features
        predictions = dict(scores=scores,
                           proposal_deltas=proposal_deltas)

        return predictions

    def get_pseudo_words(self, sampled_instances, features, *args, **kwargs):
        def _record_gradient(grad):
            val = grad.norm()
            storage = get_event_storage()
            storage.put_scalar("gradients/contrastive", val.cpu().numpy())
        box_features = self.box_pooler(features, [x.proposal_boxes for x in sampled_instances])
        # TODO: reweight the gradients from contrastive loss
        box_features = _ScaleGradient.apply(box_features, self.context_modeling_cfg.GRAD_WEIGHT)
        box_features.register_hook(_record_gradient)
        box_features = self.box_head(box_features)
        sample_types = torch.cat([p.sample_types for p in sampled_instances], dim=0)
        input_box_features = self.box_predictor.pre_forward(box_features)
        pseudo_words = self.box_predictor.pred_words(input_box_features)
        predictions = dict(kd_pseudo_words=pseudo_words[sample_types == 1],
                           caption_pseudo_words=pseudo_words[sample_types == 2])

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


@ROI_HEADS_REGISTRY.register()
class FPNSumStandardROIHeads(CustomStandardROIHeads):
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

        box_pooler = FPNSumROIPooler(
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
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            FPNSumROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret


class FPNSumROIPooler(ROIPooler):
    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            return torch.zeros(
                (0, x[0].shape[1]) + self.output_size, device=x[0].device, dtype=x[0].dtype
            )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        target_shape = x[1].shape[2:]   # resize to level2  1/8
        resized_x = torch.stack(
            [F.interpolate(x_, size=target_shape,
                           mode="bilinear",
                           align_corners=False) for x_ in x], dim=0)    # stack at dim 0
        resized_x = resized_x.sum(0)

        return self.level_poolers[1](resized_x, pooler_fmt_boxes)     # sample at level1  1/8
