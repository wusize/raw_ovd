# Copyright (c) Facebook, Inc. and its affiliates.
from .modeling.meta_arch import custom_rcnn
from .modeling.roi_heads import detic_roi_heads
from .modeling.roi_heads import (res5_roi_heads, standard_roi_heads, multilevel_roi_heads, standard_roi_heads2,
                                 fpn_res5_roi_heads, fpn_res5_cat_roi_heads, offline_standard_roi_heads,
                                 standard_roi_heads_default, standard_roi_heads3)
from .modeling.roi_heads.standard_roi_heads import FPNSumStandardROIHeads
from .modeling.backbone import swintransformer
from .modeling.backbone import timm
from .modeling.rpn.custom_rpn import CustomRPN
from .modeling.fpn.pa_fpn import build_resnet_pafpn_backbone
from .modeling.fpn.c4_fpn import build_resnet_c4fpn_backbone
from .modeling.fpn.fpn2c4 import build_resnet_fpn2c4_backbone
from .modeling.fpn.dc5_fpn import build_resnet_dc5fpn_backbone

from .data.datasets import lvis_v1
from .data.datasets import imagenet
from .data.datasets import cc
from .data.datasets import objects365
from .data.datasets import oid
from .data.datasets import coco_zeroshot
from .data.datasets import coco_custom

try:
    from .modeling.meta_arch import d2_deformable_detr
except:
    pass