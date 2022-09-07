# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.modeling.proposal_generator import RPN_HEAD_REGISTRY, StandardRPNHead
from typing import List
import torch
from torch import nn
from torch.autograd.function import Function
from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.modeling.anchor_generator import build_anchor_generator


@RPN_HEAD_REGISTRY.register()
class CustomRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        super().__init__()
        cur_channels = in_channels
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 1x1 conv for predicting objectness logits
        if self.cfg.MODEL.RPN.DETACH_FOR_OBJECTNESS:
            self.objectness_logits = nn.Sequential(self._get_rpn_conv(cur_channels, cur_channels),
                                                   nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1))
        else:
            self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)
        self.oln_cfg = self.cfg.MODEL.RPN.OLN
        if self.oln_cfg.ENABLE:
            self.location_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        cls.cfg = cfg
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
        }

    def forward(self, features: List[torch.Tensor]):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        pred_location_logits = []
        for x in features:
            t = self.conv(x)
            if self.cfg.MODEL.RPN.DETACH_FOR_OBJECTNESS:
                pred_objectness_logits.append(self.objectness_logits(t.detach()))
            else:
                pred_objectness_logits.append(self.objectness_logits(t))

            if self.oln_cfg.ENABLE:
                if self.oln_cfg.DETACH:
                    pred_location_logits.append(self.location_logits(t.detach()))
                else:
                    pred_location_logits.append(self.location_logits(t))
            else:
                pred_location_logits.append(None)
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas, pred_location_logits


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@RPN_HEAD_REGISTRY.register()
class GradRPNHead(StandardRPNHead):
    @configurable
    def __init__(self, **kwargs):
        grad_scale = kwargs.pop('grad_scale')
        super().__init__(**kwargs)
        self.grad_scale = grad_scale

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["grad_scale"] = cfg.MODEL.RPN.BCE_GRAD
        return ret

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
        if self.training:
            if self.grad_scale > 0.0:
                feature = _ScaleGradient.apply(feature, self.grad_scale)
            else:
                feature = feature.detach()
        return self.objectness_logits(feature)


@RPN_HEAD_REGISTRY.register()
class AddConvRPNHead(GradRPNHead):
    @configurable
    def __init__(self, **kwargs):
        obj_convs = kwargs.pop('obj_convs')
        super().__init__(**kwargs)
        cur_channels = self.objectness_logits.in_channels
        num_anchors = self.objectness_logits.out_channels
        if obj_convs > 1:
            self.objectness_logits = nn.Sequential()
            for k in range(obj_convs - 1):
                conv = self._get_rpn_conv(cur_channels, cur_channels)
                self.objectness_logits.add_module(f"conv{k}", conv)
            self.objectness_logits.add_module(f"conv_final",
                                              nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(obj_convs=cfg.MODEL.RPN.OBJ_CONVS)
        return ret
