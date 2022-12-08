# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
import torch.nn as nn

__all__ = ["EnsembleFastRCNNOutputLayers"]


class EnsembleFastRCNNOutputLayers(DeticFastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        assert self.cfg.MODEL.ROI_BOX_HEAD.ALL_ENCODER
        self.word_pred_kd = nn.Linear(self.word_pred.in_features,
                                      self.word_pred.out_features)

    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        # scores, _ = predictions
        scores = predictions.pop('scores')
        scores_kd = predictions.pop('scores_kd')
        # TODO: ensemble the cosine similarity scores

        factor = self.cfg.MODEL.ROI_BOX_HEAD.ENSEMBLE_FACTOR

        num_inst_per_image = [len(p) for p in proposals]
        # scores_kd[..., -1] = self.cfg.MODEL.ROI_BOX_HEAD.MASK_VALUE  # mask the bg for kd score
        # scores_kd[..., is_base > 0.0] = self.cfg.MODEL.ROI_BOX_HEAD.MASK_VALUE
        if self.cfg.MODEL.ROI_BOX_HEAD.COSINE_SCORE:
            probs = scores.clamp(min=0.0) / self.cls_score.norm_temperature
            probs_kd = scores_kd.clamp(min=0.0) / self.cls_score.norm_temperature
        else:
            if self.use_sigmoid_ce:
                probs = scores.sigmoid()
                probs_kd = scores_kd.sigmoid()
            else:
                probs = F.softmax(scores, dim=-1)
                probs_kd = F.softmax(scores_kd * self.cfg.MODEL.ROI_BOX_HEAD.RESCALE_TEMP, dim=-1)

        if self.cfg.MODEL.ROI_BOX_HEAD.TRANSFER > 0.0:
            transfer_factor = self.cfg.MODEL.ROI_BOX_HEAD.TRANSFER
            probs = (probs ** transfer_factor) * (probs_kd ** (1.0 - transfer_factor))
        else:
            assert (self.is_base > 0.0).sum() < self.num_classes
            is_base = torch.cat([
                (self.is_base.view(-1) > 1e-4).float(),
                self.is_base.new_ones(1)])  # C + 1
            probs_base = (probs ** factor) * (probs_kd ** (1.0 - factor)) * is_base[None]
            probs_novel = (probs ** (1.0 - factor)) * (probs_kd ** factor) * (1.0 - is_base[None])
            probs = probs_base + probs_novel
        return probs.split(num_inst_per_image, dim=0)

    def pred_words_kd(self, x):
        pseudo_words = self.word_pred_kd(x).view(-1, self.num_words, self.word_embed_dim)

        return pseudo_words

    def forward(self, x_cls, x_kd):
        # For inference
        # use clip-text to predict cls feature

        x_cls = self.pre_forward(x_cls)
        pseudo_words = self.pred_words(x_cls)
        scores, class_features = self.pred_cls_score(pseudo_words)

        x_kd = self.pre_forward(x_kd)
        pseudo_words_kd = self.pred_words_kd(x_kd)
        scores_kd, kd_features = self.pred_cls_score(pseudo_words_kd)

        predictions = dict()
        predictions.update(scores=scores)
        predictions.update(class_features=class_features)
        predictions.update(scores_kd=scores_kd)
        predictions.update(kd_features=kd_features)
        predictions.update(proposal_deltas=self.bbox_pred(x_cls))

        return predictions

    def inference_with_reference(self, predictions, proposals, reference_features):
        """
        enable use proposal boxes
        """
        assert len(proposals) == 1
        class_features = F.normalize(predictions['class_features'], dim=-1)
        kd_features = F.normalize(predictions['kd_features'], dim=-1)
        cls_scores = class_features @ reference_features.T
        kd_scores = kd_features @ reference_features.T
        assert self.cfg.MODEL.ROI_BOX_HEAD.TRANSFER > 0.0
        transfer_factor = self.cfg.MODEL.ROI_BOX_HEAD.TRANSFER
        scores = cls_scores * transfer_factor + kd_scores * (1.0 - transfer_factor)
        scores = [scores.repeat(1, 2)]
        boxes = self.predict_boxes(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )
