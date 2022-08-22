# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures.boxes import Boxes
import torch
import numpy as np
from .standard_roi_heads import CustomStandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.utils.events import get_event_storage
from time import time


@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeadsV3(CustomStandardROIHeads):

    def _get_box_feature(self, proposals, features, backbone):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if self.box_in_features == ['res4']:
            # if use res4, set POOLER_RESOUTION=14 for res5 and 7 for dilated res5!
            box_features = backbone.res5(box_features).mean(dim=[2, 3])
        else:
            if self.cfg.MODEL.ROI_BOX_HEAD.AVE_POOLING:
                # If pooling, set FC_DIM=2048!
                box_features = box_features.mean(dim=[2, 3])
            else:
                box_features = self.box_head(box_features)

        return box_features

    def _forward_box(self, features, proposals,
                     clip_images=None, image_info=None,
                     resized_image_info=None, group_infos=None, backbone=None):
        features = [features[f] for f in self.box_in_features]       # res4
        box_features = self._get_box_feature(proposals, features, backbone)

        if self.training:
            losses = dict()
            storage = get_event_storage()
            tik = time()
            predictions = self._box_forward_train(box_features, proposals)
            losses.update(self.box_predictor.losses(predictions,
                                                    [p[p.sample_types == 0] for p in proposals]))
            tok = time()
            # print('detector loss:', tok - tik)
            storage.put_scalar("time/detector_forward", np.float32(tok - tik))

            # TODO contrastive learning
            if self.context_modeling_cfg.ENABLE:
                losses.update(self.context_modeling.get_loss(group_infos,
                                                             clip_images,
                                                             self.box_predictor.clip, image_info,
                                                             self,
                                                             features,
                                                             backbone=backbone))
                storage.put_scalar("time/contrast_learning", np.float32(time() - tok))

            if self.cfg.MODEL.WITH_IMAGE_LABELS:
                loss = self.image_label_loss(resized_image_info)
                if loss is None:
                    loss = list(losses.values())[0] * 0.0
                losses.update(image_label_loss=loss)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

            return losses
        else:
            predictions = self.box_predictor(box_features)
            del box_features
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def forward(self, images, features, proposals, targets=None,
                ann_types=None, clip_images=None, image_info=None,
                resized_image_info=None, backbone=None):
        '''
        enable debug and image labels
        '''

        del images
        if self.training:
            proposals, group_infos = self.label_and_sample_proposals(
                proposals, targets, ann_types=ann_types, image_ids=list(image_info.keys()))
            del targets
            losses = self._forward_box(features, proposals, clip_images, image_info,
                                       resized_image_info, group_infos, backbone)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, backbone=backbone)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_pseudo_words(self, sampled_instances, features, backbone):
        box_features = self._get_box_feature(sampled_instances, features, backbone)
        sample_types = torch.cat([p.sample_types for p in sampled_instances], dim=0)
        input_box_features = self.box_predictor.pre_forward(box_features)
        pseudo_words = self.box_predictor.pred_words(input_box_features)
        predictions = dict(kd_pseudo_words=pseudo_words[sample_types == 1],
                           caption_pseudo_words=pseudo_words[sample_types == 2])

        return predictions
