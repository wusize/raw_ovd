# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn.functional as F
from typing import Dict, List, Optional
import torch
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from torchvision.ops import roi_align
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detic.modeling import clip as CLIP


@META_ARCH_REGISTRY.register()
class CLassAgnosticCLIPRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        # set the clip model
        clip_cfg = self.cfg.MODEL.CLIP
        self.clip, _ = CLIP.load(name=clip_cfg.NAME,
                                 use_image_encoder=clip_cfg.USE_IMAGE_ENCODER,
                                 download_root=clip_cfg.MODEL_ROOT)
        self.clip.init_weights()

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        cls.cfg = cfg
        return ret

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        clip_images = self.clip_preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert detected_instances is None
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        results = self._bbox_clip_image(results, clip_images)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def clip_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        mean = [[[122.7709383]], [[116.7460125]], [[104.09373615]]]
        std = [[[68.5005327]], [[66.6321579]], [[70.32316305]]]
        clip_pixel_mean = torch.tensor(mean).to(self.pixel_mean)
        clip_pixel_std = torch.tensor(std).to(self.pixel_std)
        if self.input_format == 'BGR':
            channel_order = [2, 1, 0]
        else:
            channel_order = [0, 1, 2]

        images = [x["image"][channel_order].to(self.device) for x in batched_inputs]
        images = [(x - clip_pixel_mean) / clip_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        return images

    @torch.no_grad()
    def _bbox_clip_image(self, instances, clip_images):
        # TODO: repeat and mask
        clip_input_size = self.clip.visual.input_resolution
        num_proposals_per_image = [len(inst) for inst in instances]
        input_to_clip = roi_align(
            clip_images.tensor, self._expand_boxes(instances, 1.0),
            (clip_input_size, clip_input_size), 1.0, 2, True)
        clip_image_features_0 = self.clip.encode_image(input_to_clip, normalize=True).float()

        input_to_clip = roi_align(
            clip_images.tensor, self._expand_boxes(instances, 1.5),
            (clip_input_size, clip_input_size), 1.0, 2, True)
        clip_image_features_1 = self.clip.encode_image(input_to_clip, normalize=True).float()

        clip_image_features = F.normalize(clip_image_features_0 + clip_image_features_1,
                                          dim=-1, p=2)

        clip_image_features = clip_image_features.split(num_proposals_per_image, dim=0)

        for i in range(len(instances)):
            instances[i].set('clip_image_features', clip_image_features[i])

        return instances

    @staticmethod
    def _expand_boxes(instances, scalar=1.0):
        boxes_list = []
        for inst in instances:
            h, w = inst.image_size
            boxes = inst.proposal_boxes.tensor
            box_whs = boxes[:, 2:] - boxes[:, :2]
            box_centers = (boxes[:, 2:] + boxes[:, :2]) * 0.5
            x0y0s = (box_centers - box_whs * 0.5 * scalar).clamp(min=0.0)
            x1y1s = (box_centers + box_whs * 0.5 * scalar).clamp(max=torch.tensor([w, h]).to(boxes))
            boxes_list.append(torch.cat([x0y0s, x1y1s], dim=-1))
        return boxes_list
