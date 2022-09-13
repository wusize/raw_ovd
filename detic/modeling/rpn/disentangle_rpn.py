# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional
import torch
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN, _dense_box_regression_loss
from detectron2.layers import cat
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage


@PROPOSAL_GENERATOR_REGISTRY.register()
class DisentangleRPN(RPN):
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

        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            if return_loss:
                assert gt_instances is not None, "RPN requires gt_instances in training!"
                gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
                for i, ann_type in enumerate(ann_types):
                    if ann_type not in ['with_instance']:
                        gt_labels[i][:] = -1
                pred_objectness_logits_detach = [           # detached
                    # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                    score[0].permute(0, 2, 3, 1).flatten(1)
                    for score in pred_objectness_logits
                ]
                pred_objectness_logits = [             # not detached
                    # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                    score[1].permute(0, 2, 3, 1).flatten(1)
                    for score in pred_objectness_logits
                ]
                losses = self.losses(
                    anchors, pred_objectness_logits, pred_objectness_logits_detach,
                    gt_labels, pred_anchor_deltas, gt_boxes,
                )
            else:
                pred_objectness_logits = [
                    # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                    score[0].permute(0, 2, 3, 1).flatten(1)
                    for score in pred_objectness_logits
                ]
                losses = {}
        else:
            pred_objectness_logits = [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in pred_objectness_logits
            ]
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits_pos: List[torch.Tensor],
        pred_objectness_logits_neg: List[torch.Tensor],
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

        neg_mask = gt_labels == 0

        pos_pred_logits = cat(pred_objectness_logits_pos, dim=1)[pos_mask]
        neg_pred_logits = cat(pred_objectness_logits_neg, dim=1)[neg_mask]
        pos_labels = gt_labels[pos_mask].to(torch.float32)
        neg_labels = gt_labels[neg_mask].to(torch.float32)

        objectness_loss = F.binary_cross_entropy_with_logits(
            cat([pos_pred_logits, neg_pred_logits]),
            cat([pos_labels, neg_labels]),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            # The original Faster R-CNN paper uses a slightly different normalizer
            # for loc loss. But it doesn't matter in practice
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses
