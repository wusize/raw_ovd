# Copyright (c) Facebook, Inc. and its affiliates.
import math
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from torch.cuda.amp import autocast
from ..utils import load_class_freq, get_fed_loss_inds
from .zero_shot_classifier import ZeroShotClassifier
from detic.modeling import clip as CLIP
from detectron2.utils.events import get_event_storage

__all__ = ["DeticFastRCNNOutputLayers"]


class DeticFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self, 
        input_shape: ShapeSpec,
        *,
        mult_proposal_score=False,
        cls_score=None,
        # clip settings
        clip_cfg,
        num_words,
        word_embed_dim,
        use_sigmoid_ce = False,
        use_fed_loss = False,
        ignore_zero_cats = False,
        fed_loss_num_cat = 50,
        dynamic_classifier = False,
        add_image_box = False,
        debug = False,
        prior_prob = 0.01,
        cat_freq_path = '',
        fed_loss_freq_weight = 0.5,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape, 
            **kwargs,
        )
        self.mult_proposal_score = mult_proposal_score
        self.use_sigmoid_ce = use_sigmoid_ce
        self.use_fed_loss = use_fed_loss
        self.ignore_zero_cats = ignore_zero_cats
        self.fed_loss_num_cat = fed_loss_num_cat
        self.dynamic_classifier = dynamic_classifier
        self.add_image_box = add_image_box
        self.debug = debug

        if self.use_sigmoid_ce:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)
        
        if self.use_fed_loss or self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

        if self.use_fed_loss and len(self.freq_weight) < self.num_classes:
            # assert self.num_classes == 11493
            print('Extending federated loss weight')
            self.freq_weight = torch.cat(
                [self.freq_weight,
                 self.freq_weight.new_zeros(
                    self.num_classes - len(self.freq_weight))]
            )

        assert (not self.dynamic_classifier) or (not self.use_fed_loss)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        # clip_cfg,
        self.num_words = num_words
        self.word_embed_dim = word_embed_dim

        del self.cls_score
        del self.bbox_pred
        assert cls_score is not None
        self.cls_score = cls_score
        self.word_pred = nn.Linear(input_size, num_words * word_embed_dim)

        self.clip, _ = CLIP.load(name=clip_cfg.NAME,
                                 use_image_encoder=clip_cfg.USE_IMAGE_ENCODER,
                                 download_root=clip_cfg.MODEL_ROOT)
        self.clip.init_weights()

        self.bbox_pred = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, 4)
        )
        weight_init.c2_xavier_fill(self.bbox_pred[0])
        nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
        nn.init.constant_(self.bbox_pred[-1].bias, 0)

        self.word_dropout = self.cfg.MODEL.ROI_BOX_HEAD.RANDOM_DROPOUT

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            # add clif cfg
            'clip_cfg': cfg.MODEL.CLIP,
            'num_words': cfg.MODEL.ROI_BOX_HEAD.NUM_WORDS,
            'word_embed_dim': cfg.MODEL.ROI_BOX_HEAD.WORD_EMBED_DIM,

            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
            'fed_loss_num_cat': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'debug': cfg.DEBUG or cfg.SAVE_DEBUG or cfg.IS_DEBUG,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
                            "loss_cls": cfg.MODEL.ROI_BOX_HEAD.CLS_LOSS_WEIGHT},

        })
        ret['cls_score'] = ZeroShotClassifier(cfg, input_shape)
        cls.cfg = cfg

        return ret

    def losses(self, predictions, proposals, \
        use_advanced_loss=True):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions.pop('scores'), predictions.pop('proposal_deltas')
        assert scores.shape[0] == proposal_deltas.shape[0]
        if scores.shape[0] == 0:
            return {"loss_cls": proposal_deltas.sum() * 0.0,
                    "loss_box_reg": proposal_deltas.sum() * 0.0}
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        num_classes = self.num_classes
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes)
        losses = {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes, 
                num_classes=num_classes)
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1
 
        if self.use_fed_loss and (self.freq_weight is not None): # fedloss
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1 # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()
        if self.ignore_zero_cats and (self.freq_weight is not None):
            w = (self.freq_weight.view(-1) > 1e-4).float()
            weight = weight * w.view(1, C).expand(B, C)
            # import pdb; pdb.set_trace()

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none') # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss
    
    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        change _no_instance handling
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        if self.ignore_zero_cats and (self.freq_weight is not None):
            zero_weight = torch.cat([
                (self.freq_weight.view(-1) > 1e-4).float(),
                self.freq_weight.new_ones(1)]) # C + 1
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, 
                weight=zero_weight, reduction="mean")
        elif self.use_fed_loss and (self.freq_weight is not None): # fedloss
            C = pred_class_logits.shape[1] - 1
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1).float()
            appeared_mask[appeared] = 1. # C + 1
            appeared_mask[C] = 1.
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, 
                weight=appeared_mask, reduction="mean")        
        else:
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, reduction="mean")                  
        return loss

    def box_reg_loss(
            self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, num_classes=-1):
        """
        Allow custom background index
        """
        num_classes = num_classes if num_classes > 0 else self.num_classes
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        return loss_box_reg / max(gt_classes.numel(), 1.0)

    def inference(self, predictions, proposals):
        """
        enable use proposal boxes
        """
        # predictions = (predictions[0], predictions[1])
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if self.mult_proposal_score:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None]) ** 0.5
                      for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        # import pdb; pdb.set_trace()
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes(self, predictions, proposals):
        if not len(proposals):
            return []
        proposal_deltas = predictions.pop('proposal_deltas')
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        # scores, _ = predictions
        scores = predictions.pop('scores')
        num_inst_per_image = [len(p) for p in proposals]
        if self.cfg.MODEL.ROI_BOX_HEAD.COSINE_SCORE:
            probs = scores / self.cls_score.norm_temperature
            w = (self.freq_weight.view(-1) > 1e-4).float()   # base
            b = (1.0 - w) * self.cfg.MODEL.ROI_BOX_HEAD.NOVEL_BIAS  # novel
            t = w + (1.0 - w) * self.cfg.MODEL.ROI_BOX_HEAD.NOVEL_TEMP  # novel
            probs[:, :-1] *= t.view(1, -1)
            probs[:, :-1] += b.view(1, -1)
        else:
            if self.use_sigmoid_ce:
                probs = scores.sigmoid()
            else:
                probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def pre_forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        return x

    def pred_words(self, x):
        pseudo_words = self.word_pred(x).view(-1, self.num_words, self.word_embed_dim)

        return pseudo_words

    def _drop_word(self, word_embeddings):
        p = self.word_dropout
        num_preds, num_words, _ = word_embeddings.shape
        mask = F.dropout(word_embeddings.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0       # TODO add random on this
        mask[mask > 0.0] = 1.0
        if self.training:             # TODO discard this
            is_full = (mask > 0.0).sum(dim=-1) == num_words
            mask[is_full, -1] = 0.0
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask

    @staticmethod
    def _record_gradient(grad):
        val = grad.norm()
        storage = get_event_storage()
        storage.put_scalar("gradients/classification", val.cpu().numpy())

    def pred_cls_score(self, pseudo_words, **kwargs):
        if self.training:
            pseudo_words.register_hook(self._record_gradient)
        clip_model = self.clip
        if pseudo_words.shape[0] == 0:
            return pseudo_words.new_zeros(0, self.num_classes + 1)
        clip_model.eval()
        with autocast():
            if self.word_dropout > 0.0:
                valid_mask = self._drop_word(pseudo_words.half())
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text_tensor(
                pseudo_words.half(), valid_mask)  # add start and stop token
            # assert attn_mask.shape[:2] == pseudo_words.shape[:2]
            cls_features, x, end_token_ids = \
                clip_model.encode_pseudo_text_endk(pseudo_text, end_token_ids,
                                                   text_pe=True,
                                                   stepk=12, normalize=True)
            cls_features = cls_features.float()

        cls_scores = self.cls_score(cls_features)
        return cls_scores

    def forward(self, x):
        x = self.pre_forward(x)
        # use clip-text to predict cls feature

        predictions = dict()
        predictions.update(pseudo_words=self.pred_words(x))
        predictions.update(scores=self.pred_cls_score(predictions['pseudo_words']))
        predictions.update(proposal_deltas=self.bbox_pred(x))

        return predictions
