# Copyright (c) Facebook, Inc. and its affiliates.
# import copy
import math
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import torch
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from collections import OrderedDict
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from time import time
from torch.cuda.amp import autocast
from ..utils import load_class_freq
from .utils import process_proposals
import json
import os


@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self,
        dataset_loss_weight = [],
        fp16 = False,
        roi_head_name = '',
        dynamic_classifier = False,
        **kwargs):
        """
        """
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        # self.with_caption = with_caption
        # self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        # self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.cfg.MODEL.SAVE_PROPOSALS:
            os.makedirs(self.cfg.SAVE_DEBUG_PATH, exist_ok=True)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        cls.cfg = cfg
        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals)
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

    def forward_backbone(self, images_tensor):
        if self.fp16:
            with autocast():
                features = self.backbone(images_tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images_tensor)

        return features

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        clip_images = self.clip_preprocess_image(batched_inputs)

        # ann_types = [b.get('ann_type', 'with_instance') for b in batched_inputs]
        image_info = {b['image_id']: dict(captions=b.get('captions', []),
                                          pos_category_ids=b.get('pos_category_ids', [])) for b in batched_inputs}
        image_info = OrderedDict(image_info)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        ann_types = ['with_instance' if len(inst) > 0 else 'only_caption' for inst in gt_instances]
        storage = get_event_storage()
        tik = time()
        features = self.forward_backbone(images.tensor)
        tok = time()
        storage.put_scalar("time/backbone", tok - tik)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, ann_types=ann_types)
        tik = time()
        storage.put_scalar("time/proposal_generator", tik-tok)

        resized_image_info = self.get_resized_image_info(images, batched_inputs, features)

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances, clip_images=clip_images,
                image_info=image_info)
        else:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_types=ann_types, clip_images=clip_images, image_info=image_info,
                resized_image_info=resized_image_info)
        tok = time()
        storage.put_scalar("time/roi_heads", tok - tik)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        
        if self.return_proposal:
            return proposals, losses
        else:
            return losses

    def resize_for_image_label(self, images, batched_inputs, features):
        batch_size = len(batched_inputs)
        image_labels = []
        resized_features = {k: [] for k in features.keys()}
        image_sizes = []
        image_tensors = []
        for i in range(batch_size):
            pos_ids = batched_inputs[i]['pos_category_ids']
            if len(pos_ids) == 0:
                continue
            image_size = images.image_sizes[i]
            new_image_size = (math.ceil(image_size[0] / 2),
                              math.ceil(image_size[1] / 2))
            image_sizes.append(new_image_size)
            image_labels.append(pos_ids)
            batched_shape = images.tensor.shape[2:4]
            batched_shape = (math.ceil(batched_shape[0] / 2),
                             math.ceil(batched_shape[1] / 2))
            new_image = F.interpolate(images.tensor[i:i+1],  size=batched_shape,
                                      mode='bilinear', align_corners=False)
            image_tensors.append(new_image)

        if len(image_tensors) > 0:
            image_tensors = torch.cat(image_tensors)
            resized_images = ImageList(tensor=image_tensors, image_sizes=image_sizes)
            resized_features = self.forward_backbone(image_tensors)
        else:
            resized_images = []

        return dict(images=resized_images,
                    image_labels=image_labels,
                    features=resized_features)

    def get_resized_image_info(self, images, batched_inputs, features):
        if not self.training or not self.cfg.MODEL.WITH_IMAGE_LABELS:
            return dict()
        resized_image_info = \
            self.resize_for_image_label(images, batched_inputs, features)
        resized_images, resized_features = resized_image_info['images'], resized_image_info['features']
        new_proposals = []
        if len(resized_images) > 0:
            with torch.no_grad():
                pred_proposals, _ = self.proposal_generator(
                    resized_images, resized_features, None, ann_types=[], return_loss=False)
            for p in pred_proposals:
                if len(p) > 0:
                    new_proposals.append(p)
                else:
                    h, w = p.image_size
                    image_box = torch.tensor([[0.0, 0.0, w - 1.0, h - 1.0]], device=self.device)
                    img_proposal = Instances(image_size=p.image_size,
                                             proposal_boxes=Boxes(image_box),
                                             objectness_logits=22.3 * torch.ones(1, device=self.device))
                    new_proposals.append(img_proposal)

        resized_image_info.update(proposals=new_proposals)

        return resized_image_info
