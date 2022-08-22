# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Part of the code is from https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/data/transforms.py 
# Modified by Xingyi Zhou
# The original code is under Apache-2.0 License
import numpy as np
import sys
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    VFlipTransform,
)
from PIL import Image
from detectron2.data.transforms.augmentation_impl import FixedSizeCrop
from detectron2.data.transforms.augmentation import Augmentation
from .custom_transform import EfficientDetResizeCropTransform

__all__ = [
    "EfficientDetResizeCrop",
]

class EfficientDetResizeCrop(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, size, scale, interp=Image.BILINEAR
    ):
        """
        """
        super().__init__()
        self.target_size = (size, size)
        self.scale = scale
        self.interp = interp

    def get_transform(self, img):
        # Select a random scale factor.
        scale_factor = np.random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]
        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.shape[1], img.shape[0]
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * np.random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * np.random.uniform(0, 1))
        return EfficientDetResizeCropTransform(
            scaled_h, scaled_w, offset_y, offset_x, img_scale, self.target_size, self.interp)


class CustomFixedSizeCrop(FixedSizeCrop):
    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        return CustomCropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )


class CustomCropTransform(CropTransform):
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = super(CustomCropTransform, self).apply_coords(coords)
        # TODO: limit coords into the valid range
        x_max = min(self.orig_w - self.x0, self.w)
        y_max = min(self.orig_h - self.y0, self.h)
        coords[:, 0] = np.clip(coords[:, 0], a_min=0, a_max=x_max)
        coords[:, 1] = np.clip(coords[:, 1], a_min=0, a_max=y_max)

        return coords
