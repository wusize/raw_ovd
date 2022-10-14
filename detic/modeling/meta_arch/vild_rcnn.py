# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List
import torch
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
        clip_proposals = []
        for x in batched_inputs:
            clip_proposals_per_image = x["proposals"].to(self.device)
            clip_image_features = torch.from_numpy(x["clip_image_features"]).to(self.device)
            clip_proposals_per_image.set("clip_image_features", clip_image_features)
            clip_proposals.append(clip_proposals_per_image)

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
