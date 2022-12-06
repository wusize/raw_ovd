# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple
import torch
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .utils import process_proposals
import json
import os
from .custom_rcnn import CustomRCNN

@META_ARCH_REGISTRY.register()
class RefCustomRCNN(CustomRCNN):

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        reference_features = None
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # import pdb; pdb.set_trace()
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, reference_features=reference_features)
        if self.cfg.MODEL.SAVE_PROPOSALS:
            image_proposals = process_proposals(batched_inputs, images, proposals)
            for img_p in image_proposals:
                file_name = img_p['file_name']
                with open(os.path.join(self.cfg.SAVE_DEBUG_PATH,
                                       file_name.split('.')[0] + '.json'), 'w') as f:
                    json.dump(img_p, f)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]],
                reference_features=None):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        assert not self.training
        return self.inference(batched_inputs, reference_features=reference_features)

