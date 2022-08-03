from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
import torch.nn.functional as F
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
import torch
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
__all__ = ["build_resnet_fpn2c4_backbone", "FPN2C4"]


class FPN2C4(FPN):
    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        super(FPN2C4, self).__init__(bottom_up, in_features, out_channels, norm, top_block, fuse_type)
        use_bias = norm == ""
        output_norm = get_norm(norm, 1024)
        self.merge_conv = Conv2d(
            1024 + 256 * len(self._out_features), 1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
        )
        weight_init.c2_xavier_fill(self.merge_conv)

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        c4_feature = bottom_up_features['res4']
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)

        output = torch.cat([c4_feature] +
                           [F.interpolate(res,
                                          size=c4_feature.shape[2:],
                                          align_corners=False,
                                          mode='bilinear') for res in results],
                           dim=1)

        return dict(res4=self.merge_conv(output))

    def output_shape(self):
        return {'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16)}


@BACKBONE_REGISTRY.register()
def build_resnet_fpn2c4_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    assert 'res4' in in_features
    backbone = FPN2C4(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
