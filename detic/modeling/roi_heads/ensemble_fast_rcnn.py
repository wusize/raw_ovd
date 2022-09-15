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
        if "scores" in predictions:
            scores = predictions.pop('scores')
            if self.cfg.MODEL.ROI_BOX_HEAD.COSINE_SCORE:
                probs = scores.clamp(min=0.0) / self.cls_score.norm_temperature
            else:
                if self.use_sigmoid_ce:
                    probs = scores.sigmoid()
                else:
                    probs = F.softmax(scores, dim=-1)
        else:
            # scores, _ = predictions
            scores_cls = predictions.pop('scores_cls')
            scores_kd = predictions.pop('scores_kd')
            # TODO: ensemble the cosine similarity scores

            factor = self.cfg.MODEL.ROI_BOX_HEAD.ENSEMBLE_FACTOR
            assert factor < 0.5
            is_base = torch.cat([
                (self.freq_weight.view(-1) > 1e-4).float(),
                self.freq_weight.new_ones(1)])  # C + 1

            if self.cfg.MODEL.ROI_BOX_HEAD.COSINE_SCORE:
                probs_cls = scores_cls.clamp(min=0.0) / self.cls_score.norm_temperature
                probs_kd = scores_kd.clamp(min=0.0) / self.cls_score.norm_temperature
            else:
                if self.use_sigmoid_ce:
                    probs_cls = scores_cls.sigmoid()
                    probs_kd = scores_kd.sigmoid()
                else:
                    probs_cls = F.softmax(scores_cls, dim=-1)
                    probs_kd = F.softmax(scores_kd, dim=-1)

            probs_base = (probs_cls ** factor) * (probs_kd ** (1.0 - factor)) * is_base[None]
            probs_novel = (probs_cls ** (1.0 - factor)) * (probs_kd ** factor) * (1.0 - is_base[None])
            probs = probs_base + probs_novel

        num_inst_per_image = [len(p) for p in proposals]
        return probs.split(num_inst_per_image, dim=0)

    def pred_words_kd(self, x):
        pseudo_words = self.word_pred_kd(x).view(-1, self.num_words, self.word_embed_dim)

        return pseudo_words

    def forward(self, x_cls, x_kd):
        # For inference
        # use clip-text to predict cls feature

        x_cls = self.pre_forward(x_cls)
        pseudo_words_cls = self.pred_words(x_cls)

        x_kd = self.pre_forward(x_kd)
        pseudo_words_kd = self.pred_words_kd(x_kd)

        predictions = dict()

        if self.cfg.MODEL.ROI_BOX_HEAD.ENSEMBLE_WORDS:
            pseudo_words = torch.cat([pseudo_words_cls, pseudo_words_kd],
                                     dim=1)
            scores = self.pred_cls_score(pseudo_words)
            predictions.update(scores=scores)
        else:
            scores_cls = self.pred_cls_score(pseudo_words_cls)
            scores_kd = self.pred_cls_score(pseudo_words_kd)
            predictions.update(scores_cls=scores_cls)
            predictions.update(scores_kd=scores_kd)

        predictions.update(proposal_deltas=self.bbox_pred(x_cls))

        return predictions
