# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures.boxes import Boxes
import torch
import numpy as np
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from torch.nn import functional as F
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator.proposal_utils \
    import add_ground_truth_to_proposals
from detectron2.structures import pairwise_iou
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.poolers import ROIPooler
from typing import List
from .disentangle_context_modelling import ContextModellingV2 as ContextModelling
from time import time
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from torch.autograd.function import Function


@ROI_HEADS_REGISTRY.register()
class DefaultStandardROIHeads(StandardROIHeads):
    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        def _record_gradient(grad):
            val = grad.norm()
            storage = get_event_storage()
            storage.put_scalar("gradients/detection", val.cpu().numpy())
        if self.training:
            box_features.register_hook(_record_gradient)
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
