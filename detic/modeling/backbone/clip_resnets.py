from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detic.modeling import clip as CLIP
from detectron2.layers.batch_norm import FrozenBatchNorm2d
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


__all__ = [
    "build_clip_resnet_backbone", "build_clip_resnet_fpn_backbone"
]


def freeze(model):
    """
    Make this block not trainable.
    This method sets all parameters to `requires_grad=False`,
    and convert all BatchNorm layers to FrozenBatchNorm

    Returns:
        the block itself
    """
    for p in model.parameters():
        p.requires_grad = False
    model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    return model


class CLIPResNet(Backbone):
    def __init__(self, cfg, out_features=None, freeze_at=0, *args):
        super(CLIPResNet, self).__init__()
        clip_cfg = cfg.MODEL.CLIP_RESNET
        clip_model, _ = CLIP.load(name=clip_cfg.NAME,
                                  use_image_encoder=True,
                                  use_text_encoder=False,
                                  download_root=clip_cfg.MODEL_ROOT)
        clip_model.init_weights(fix_params=False)
        self.resnet = clip_model.visual
        del self.resnet.attnpool
        self.stage_names = ("res2", "res3", "res4", "res5")
        if out_features is None:
            out_features = self.stage_names
        self._out_features = out_features
        self.freeze(freeze_at=freeze_at)

        current_stride = 4
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.resnet.conv3.out_channels}
        for name, stage in zip(self.stage_names, range(4)):
            layer = getattr(self.resnet, f'layer{stage + 1}')
            current_stride *= layer[0].stride
            out_channels = layer[-1].conv3.out_channels
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = out_channels

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            for i in range(3):
                bn = getattr(self.resnet, f'bn{i + 1}')
                conv = getattr(self.resnet, f'conv{i + 1}')
                setattr(self.resnet, f'bn{i + 1}', freeze(bn))
                setattr(self.resnet, f'conv{i + 1}', freeze(conv))
        for idx, stage in enumerate(range(4), start=2):
            if freeze_at >= idx:
                layer = getattr(self.resnet, f'layer{stage + 1}')
                setattr(self.resnet, f'layer{stage + 1}', freeze(layer))
        return self

    def stem(self, x):
        resnet = self.resnet
        for conv, bn in [(resnet.conv1, resnet.bn1), (resnet.conv2, resnet.bn2), (resnet.conv3, resnet.bn3)]:
            x = resnet.relu(bn(conv(x)))
        x = resnet.avgpool(x)
        return x

    def forward(self, x):
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        x = x.type(self.resnet.conv1.weight.dtype)
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, range(4)):
            layer = getattr(self.resnet, f'layer{stage + 1}')
            x = layer(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_clip_resnet_backbone(cfg, *args, **kwargs):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    return CLIPResNet(cfg, out_features=out_features, freeze_at=freeze_at)


@BACKBONE_REGISTRY.register()
def build_clip_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_clip_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
