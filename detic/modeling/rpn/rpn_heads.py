# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.modeling.proposal_generator import RPN_HEAD_REGISTRY, StandardRPNHead
from typing import List
import torch
from torch import nn
from detectron2.config import configurable


@RPN_HEAD_REGISTRY.register()
class DetachRPNHead(StandardRPNHead):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cur_channels = self.objectness_logits.in_channels
        num_anchors = self.objectness_logits.out_channels
        self.objectness_logits = nn.Sequential(
            self._get_rpn_conv(cur_channels, cur_channels),
            nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1))

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self._forward_objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas

    def _forward_objectness_logits(self, feature):
        return self.objectness_logits(feature.detach())


@RPN_HEAD_REGISTRY.register()
class DisentangleRPNHead(DetachRPNHead):
    def _forward_objectness_logits(self, feature):
        if self.training:
            return self.objectness_logits(feature.detach()), \
                   self.objectness_logits(feature)        # bg, fg
        else:
            return self.objectness_logits(feature)