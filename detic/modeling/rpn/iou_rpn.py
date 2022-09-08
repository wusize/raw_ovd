# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from .custom_rpn import CustomRPN
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.structures.boxes import matched_pairwise_iou
import random
from typing import List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch.nn import functional as F
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import cat, ciou_loss, diou_loss, ShapeSpec
from detectron2.structures import Boxes


@PROPOSAL_GENERATOR_REGISTRY.register()
class IOURPN(CustomRPN):
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.CLS_LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT,
                "loss_rpn_iou": cfg.MODEL.RPN.IOU_LOSS_WEIGHT,
            },
        })

        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        ann_types: List[str] = [''],
        return_loss=True
    ):
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training and return_loss:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            for i, ann_type in enumerate(ann_types):
                if ann_type not in ['with_instance']:
                    gt_labels[i][:] = -1
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes,
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
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

        # TODO
        # valid_mask = gt_labels >= 0
        # objectness_loss = F.binary_cross_entropy_with_logits(
        #     cat(pred_objectness_logits, dim=1)[valid_mask],
        #     gt_labels[valid_mask].to(torch.float32),
        #     reduction="sum",
        # )

        normalizer = self.batch_size_per_image * num_images
        losses = {
            # "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_iou": self._iou_loss(anchors, gt_boxes, pred_objectness_logits),
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


def _dense_box_regression_loss(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    fg_anchors = torch.stack([anchors] * len(fg_mask))[fg_mask]
    fg_gt_boxes = torch.stack(gt_boxes)[fg_mask]
    with torch.no_grad():
        fg_ious_with_gt = matched_pairwise_iou(Boxes(fg_anchors), Boxes(fg_gt_boxes))
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction="none",
        ).sum(-1)
        loss_box_reg = (loss_box_reg * fg_ious_with_gt).sum() / (fg_ious_with_gt.mean() + 1e-12)  # re weight
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg
