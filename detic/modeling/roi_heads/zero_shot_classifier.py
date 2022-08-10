# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.utils.events import get_event_storage


class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        self.norm_temperature = norm_temperature
        self.use_bias = use_bias < 0
        if self.use_bias:
            if self.cfg.MODEL.ROI_BOX_HEAD.FIX_BIAS:
                self.cls_bias = use_bias
            else:
                self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        zs_weight = torch.tensor(
            np.load(zs_weight_path),
            dtype=torch.float32).permute(1, 0).contiguous()  # D x C

        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1)  # D x (C + 1)
        
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
        
        self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        cls.cfg = cfg
        return {
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        zs_weight = self.zs_weight
        x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias and self.training:
            x = x + self.cls_bias
            if not self.cfg.MODEL.ROI_BOX_HEAD.FIX_BIAS:
                storage = get_event_storage()
                value = self.cls_bias.item()
                storage.put_scalar("time/classifier_bias", np.float32(value))
        return x
