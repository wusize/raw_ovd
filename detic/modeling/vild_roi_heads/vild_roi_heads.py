# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.structures import Boxes
import torch
from detectron2.config import configurable
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .vild_fast_rcnn import VILDFastRCNNOutputLayers
import torch.nn.functional as F


@ROI_HEADS_REGISTRY.register()
class VILDROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.image_box_size = cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE
        self.box_predictor = VILDFastRCNNOutputLayers(
            cfg,  self.box_head.output_shape
        )

        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def _forward_box(self, features, proposals, clip_proposals=None):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        if self.training:
            losses = dict()
            predictions = self.box_predictor(box_features)
            losses.update(self.box_predictor.losses(predictions, proposals))

            # vild loss
            kd_box_features = self.box_pooler(features, [x.proposal_boxes for x in clip_proposals])
            kd_box_features = self.box_head(kd_box_features)
            pred_embeddings = self.box_predictor(kd_box_features, False)['pseudo_words']
            pred_embeddings = F.normalize(pred_embeddings, p=2, dim=-1)
            clip_embeddings = torch.cat([p.clip_image_features for p in clip_proposals], dim=0)
            # vild_loss = (pred_embeddings - clip_embeddings).norm(dim=-1, p=1).mean()
            vild_loss = 1.0 - (pred_embeddings * clip_embeddings).sum(-1).mean()

            losses.update(vild_loss=self.cfg.VILD.LOSS_WEIGHT * vild_loss)

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
                clip_proposals=None):
        '''
        enable debug and image labels
        '''

        del images
        if self.training:
            proposals = self.label_and_sample_proposals(
                proposals, targets)
            del targets
            losses = self._forward_box(features, proposals, clip_proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
