# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional
import torch
from detectron2.structures import ImageList, Instances
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.rpn import RPN


@PROPOSAL_GENERATOR_REGISTRY.register()
class CustomRPN(RPN):
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
