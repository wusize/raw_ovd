# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from .detic_fast_rcnn import DeticFastRCNNOutputLayers
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
        is_base = torch.cat([
            (self.is_base.view(-1) > 1e-4).float(),
            self.is_base.new_ones(1)])  # C + 1

        num_inst_per_image = [len(p) for p in proposals]
        scores_kd[..., -1] = self.cfg.MODEL.ROI_BOX_HEAD.MASK_VALUE  # mask the bg for kd score

        if self.cfg.MODEL.ROI_BOX_HEAD.COSINE_SCORE:
            probs = scores.clamp(min=0.0) / self.cls_score.norm_temperature
            probs_kd = scores_kd.clamp(min=0.0) / self.cls_score.norm_temperature
        else:
            if self.use_sigmoid_ce:
                probs = scores.sigmoid()
                probs_kd = scores_kd.sigmoid()
            else:
                probs = F.softmax(scores, dim=-1)
                probs_kd = F.softmax(scores_kd, dim=-1)

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
        scores = self.pred_cls_score(pseudo_words)

        x_kd = self.pre_forward(x_kd)
        pseudo_words_kd = self.pred_words_kd(x_kd)
        scores_kd = self.pred_cls_score(pseudo_words_kd)

        predictions = dict()
        predictions.update(scores=scores)
        predictions.update(scores_kd=scores_kd)
        predictions.update(proposal_deltas=self.bbox_pred(x_cls))

        return predictions
