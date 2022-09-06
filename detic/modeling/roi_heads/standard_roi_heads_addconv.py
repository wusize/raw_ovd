# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from .standard_roi_heads import CustomStandardROIHeads
import torch.nn as nn


@ROI_HEADS_REGISTRY.register()
class AddConvStandardROIHeads(CustomStandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        add_convs = kwargs.pop('add_convs')
        in_channels = kwargs.pop('in_channels')
        super().__init__(**kwargs)
        if add_convs > 0:
            self.convs = nn.Sequential()
            for i in range(add_convs - 1):
                self.convs.add_module(f'conv_{i}', self._get_add_conv(in_channels,
                                                                      in_channels))
            self.convs.add_module(f'conv_{add_convs - 1}', self._get_add_conv(
                in_channels, in_channels, activation=None))
        else:
            self.convs = nn.Identity()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(add_convs=cfg.MODEL.ROI_HEADS.ADD_CONVS)
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        ret.update(in_channels=in_channels)
        return ret

    def forward(self, images, features, proposals, targets=None,
                ann_types=None, clip_images=None, image_info=None,
                resized_image_info=None):
        all_feature_names = set(self.box_in_features + self.mask_in_features)
        for name in all_feature_names:
            if name in features:
                features[name] = self.convs(features[name])

        return super(AddConvStandardROIHeads, self).forward(
            images, features, proposals, targets,
            ann_types, clip_images, image_info,
            resized_image_info)
