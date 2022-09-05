# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List
import torch
from detectron2.structures import Boxes
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from .custom_rpn import CustomRPN
from detectron2.utils.events import get_event_storage
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.layers import cat
import torch.nn.functional as F
from detectron2.structures.boxes import matched_pairwise_iou
import random
from .utils import centerness_score


@PROPOSAL_GENERATOR_REGISTRY.register()
class IOURPN(CustomRPN):
    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": self._iou_loss(anchors, gt_boxes, pred_objectness_logits),
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses

    def _iou_loss(self, anchors, gt_boxes, pred_objectness_logits):
        # sample
        num_images = len(gt_boxes)
        iou_thr = self.anchor_matcher.thresholds[1]              # 0.3 as default
        num_samples = self.batch_size_per_image * num_images     # 256 as default

        pred_objectness_logits = cat(pred_objectness_logits, dim=1).view(-1)
        gt_boxes = torch.cat(gt_boxes)
        anchors = Boxes.cat(anchors * num_images)
        ious = matched_pairwise_iou(anchors, Boxes(gt_boxes))

        positive_samples = torch.where(ious > iou_thr)[0].tolist()
        if len(positive_samples) == 0:
            positive_samples = ious.topk(5).indices.tolist()
        num_samples = min(len(positive_samples), num_samples)
        positive_samples = random.sample(positive_samples, k=num_samples)

        targets = self._get_geometric_targets(anchors[positive_samples],
                                              gt_boxes[positive_samples])
        preds = pred_objectness_logits[positive_samples]

        objectness_loss = F.binary_cross_entropy_with_logits(
            preds,
            targets,
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images

        return objectness_loss / normalizer

    @staticmethod
    def _get_geometric_targets(anchors, gt_boxes):
        return matched_pairwise_iou(anchors, Boxes(gt_boxes))


@PROPOSAL_GENERATOR_REGISTRY.register()
class CenternessRPN(IOURPN):
    @staticmethod
    def _get_geometric_targets(anchors, gt_boxes):
        return centerness_score(anchors.tensor, gt_boxes)


@PROPOSAL_GENERATOR_REGISTRY.register()
class IOUCenternessRPN(IOURPN):
    def _get_geometric_targets(self, anchors, gt_boxes):
        ious = super(IOUCenternessRPN, self)._get_geometric_targets(anchors, gt_boxes)
        return torch.sqrt(centerness_score(anchors.tensor, gt_boxes) * ious)
