# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures import Boxes
import torch
import numpy as np
from detectron2.utils.events import get_event_storage
from time import time
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from .ensemble_standard_roi_heads import EnsembleStandardROIHeads


@ROI_HEADS_REGISTRY.register()
class RefEnsembleStandardROIHeads(EnsembleStandardROIHeads):

    def forward(self, images, features, proposals, targets=None,
                ann_types=None, clip_images=None, image_info=None,
                resized_image_info=None, reference_features=None):
        '''
        enable debug and image labels
        '''

        del images
        if self.training:
            proposals, group_infos = self.label_and_sample_proposals(
                proposals, targets, ann_types=ann_types)
            del targets
            losses = self._forward_box(features, proposals, clip_images, image_info,
                                       resized_image_info, group_infos)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, reference_features=reference_features)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals,
                     clip_images=None, image_info=None,
                     resized_image_info=None, group_infos=None,
                     reference_features=None):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

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
                                                             predictions, clip_images,
                                                             self.box_predictor.clip, image_info))
                storage.put_scalar("time/contrast_learning", np.float32(time() - tok))


            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)

            return losses
        else:
            box_features_kd = self.box_head_kd(box_features)
            box_features_cls = self.box_head(box_features)
            predictions = self.box_predictor(box_features_cls, box_features_kd)
            pred_instances, _ = self.box_predictor.inference_with_reference(predictions,
                                                                            proposals, reference_features)
            return pred_instances
