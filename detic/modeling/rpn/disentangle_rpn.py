# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional
import torch
from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.layers import cat
import torch.nn.functional as F
from detectron2.config import configurable


@PROPOSAL_GENERATOR_REGISTRY.register()
class DisentangleRPN(RPN):
    @configurable
    def __init__(self, **kwargs):
        pos_loss_weight = kwargs.pop("pos_loss_weight")
        super().__init__(**kwargs)
        self.pos_loss_weight = pos_loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['pos_loss_weight'] = cfg.MODEL.RPN.POS_LOSS_WEIGHT
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
                pred_objectness_logits_detached = [           # detached
                    # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                    score[0].permute(0, 2, 3, 1).flatten(1)
                    for score in pred_objectness_logits
                ]
                pred_objectness_logits_undetached = [             # not detached
                    # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                    score[1].permute(0, 2, 3, 1).flatten(1)
                    for score in pred_objectness_logits
                ]
                losses = self.losses(
                    anchors, pred_objectness_logits_detached,
                    gt_labels, pred_anchor_deltas, gt_boxes,
                )
                losses.update(self.pos_losses(
                    gt_labels, pred_objectness_logits_undetached))
                pred_objectness_logits = pred_objectness_logits_detached
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

    def pos_losses(
        self,
        gt_labels,
        pred_objectness_logits: List[torch.Tensor],
    ):
        gt_labels = torch.stack(gt_labels)
        pos_mask = gt_labels > 0
        if pos_mask.sum() == 0:
            return dict(pos_aux_loss=pred_objectness_logits[0].min() * 0.0)    # return 0 loss

        pos_aux_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[pos_mask],
            gt_labels[pos_mask].to(torch.float32),    # all ones
            reduction="mean",
        )

        return {"pos_aux_loss": self.pos_loss_weight * pos_aux_loss}