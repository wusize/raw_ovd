# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from detectron2.modeling.proposal_generator import RPN_HEAD_REGISTRY, StandardRPNHead


@RPN_HEAD_REGISTRY.register()
class CustomRPNHead(StandardRPNHead):
    def forward(self, features: List[torch.Tensor]):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            if self.cfg.MODEL.RPN.DETACH_FOR_OBJECTNESS:
                pred_objectness_logits.append(self.objectness_logits(t.detach()))
            else:
                pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas
