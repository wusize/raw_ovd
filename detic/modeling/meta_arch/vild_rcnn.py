# Copyright (c) Facebook, Inc. and its affiliates.
import torch.nn.functional as F
from typing import Dict, List, Optional
import torch
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from torchvision.ops import roi_align
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class VILDRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        assert self.proposal_generator is not None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        assert "proposals" in batched_inputs[0]
        import pdb; pdb.set_trace()
        clip_proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances,
                                            clip_proposals=clip_proposals)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
