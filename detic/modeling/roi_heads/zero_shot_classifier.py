# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable


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
        if self.cfg.MODEL.ROI_BOX_HEAD.LEARN_BG:
            assert self.cfg.MODEL.ROI_BOX_HEAD.BG_BIAS <= 0.0
            assert not self.cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
            self.bg_embedding = nn.Linear(1, zs_weight_dim)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)

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
        if self.cfg.MODEL.ROI_BOX_HEAD.LEARN_BG:
            assert self.cfg.MODEL.ROI_BOX_HEAD.BG_BIAS <= 0.0
            assert not self.cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
            input_one = x[0].new_ones(1, 1)
            bg_class_embedding = self.bg_embedding(input_one)
            bg_class_embedding = F.normalize(bg_class_embedding, p=2, dim=1)  # 1, 512
            zs_weight[:, -1] = bg_class_embedding[0]   # learnable back_groud

        x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias and self.training:
            x = x + self.cls_bias
        if self.cfg.MODEL.ROI_BOX_HEAD.BG_BIAS > 0.0:
            assert not self.use_bias
            assert not self.cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
            assert not self.cfg.MODEL.ROI_BOX_HEAD.LEARN_BG
            x[..., -1] = x[..., -1] + self.cfg.MODEL.ROI_BOX_HEAD.BG_BIAS
        return x
