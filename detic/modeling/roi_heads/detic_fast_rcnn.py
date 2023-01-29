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
        # debug = False,
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
        # self.debug = debug

        if self.use_sigmoid_ce:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)
        
        if self.use_fed_loss:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight, min_count=1)   # for fed_loss
            self.register_buffer('freq_weight', freq_weight)   # only for def loss
        else:
            self.freq_weight = None

        is_base = load_class_freq(cat_freq_path, 1.0, min_count=0)  # to mask the novel classes
        # assert (is_base > 0.0).sum() < self.num_classes
        self.register_buffer('is_base', is_base)

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
        word_pred_layers = self.cfg.MODEL.ROI_BOX_HEAD.WORD_PRED_LAYERS
        if word_pred_layers == 1:
            self.word_pred = nn.Linear(input_size, num_words * word_embed_dim)
        else:
            cur_size = input_size
            word_pred = []
            for _ in range(word_pred_layers - 1):
                word_pred += [nn.Linear(cur_size, num_words * word_embed_dim),
                              nn.ReLU()]
                cur_size = num_words * word_embed_dim
            word_pred.append(nn.Linear(cur_size, num_words * word_embed_dim))
            self.word_pred = nn.Sequential(*word_pred)

        self.clip, self.clip_preprocess = CLIP.load(name=clip_cfg.NAME,
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
        if self.cfg.MODEL.ROI_BOX_HEAD.MASK_FOR_POS:
            cls_features = self.cls_score.zs_weight   # note that the last row and col is bg(0)
            similarity_matrix = cls_features.T @ cls_features
            self.register_buffer('similarity_matrix', similarity_matrix)
        if self.cfg.MODEL.ROI_BOX_HEAD.WORD_BACKGROUND:
            assert self.cfg.MODEL.ROI_BOX_HEAD.LEARN_BG
            self.bg_embedding = nn.Linear(1, 2 * num_words * word_embed_dim)   # use more words than the foreground
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)

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
            # 'debug': cfg.DEBUG or cfg.SAVE_DEBUG or cfg.IS_DEBUG,
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
        if self.ignore_zero_cats and (self.is_base is not None):
            w = (self.is_base.view(-1) > 1e-4).float()
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

        if self.ignore_zero_cats:
            zero_weight = torch.cat([
                (self.is_base.view(-1) > 1e-4).float(),
                self.is_base.new_ones(1)])  # C + 1
            # TODO: use direct mask on the pred_class_logits
            if self.cfg.MODEL.ROI_BOX_HEAD.MASK_FOR_NEG:
                neg_preds = gt_classes == self.num_classes   # bg
                assert neg_preds.sum() > 0
                pred_class_logits[neg_preds][:, zero_weight < 1.0] = \
                    self.cfg.MODEL.ROI_BOX_HEAD.MASK_VALUE
            else:
                pred_class_logits[..., zero_weight < 1.0] = self.cfg.MODEL.ROI_BOX_HEAD.MASK_VALUE

            loss = F.cross_entropy(
                pred_class_logits, gt_classes, reduction="mean")
        elif self.use_fed_loss and (self.freq_weight is not None):  # fedloss
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
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def inference_with_referecne(self, predictions, proposals, ref_image_path):
        """
        enable use proposal boxes
        """
        from PIL import Image
        assert len(proposals) == 1
        image = self.clip_preprocess(Image.open(ref_image_path)
                                     ).unsqueeze(0).to(predictions['pseudo_words'].device)
        image_features = self.clip.encode_image(image, normalize=True)   # 1xE
        class_features = F.normalize(predictions['class_features'], dim=-1)
        scores = [class_features @ image_features.T]
        boxes = self.predict_boxes(predictions, proposals)
        if self.mult_proposal_score:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None]) ** 0.5
                      for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            0.0,
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
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)

        if self.cfg.MODEL.ROI_BOX_HEAD.NOVEL_FACTOR < 1.0:
            novel_factor = self.cfg.MODEL.ROI_BOX_HEAD.NOVEL_FACTOR

            novel_classes = torch.cat([
                (self.is_base.view(-1) > 1e-4).float(),
                self.is_base.new_ones(1)]) < 0.5
            probs[:, novel_classes] = probs[:, novel_classes] ** novel_factor

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
        if self.training and self.cfg.MODEL.ROI_BOX_HEAD.DROP_LAST:
            is_full = (mask > 0.0).sum(dim=-1) == num_words
            mask[is_full, -1] = 0.0
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask

    def pred_cls_score(self, pseudo_words, **kwargs):
        clip_model = self.clip
        if pseudo_words.shape[0] == 0:
            return pseudo_words.new_zeros(0, self.num_classes + 1), None
        clip_model.eval()
        cls_features = self.forward_clip_text(pseudo_words, clip_model)
        if self.cfg.MODEL.ROI_BOX_HEAD.WORD_BACKGROUND:
            ones = pseudo_words.new_ones(1, 1)
            bg_words = self.bg_embedding(ones).view(1, 2 * self.num_words,
                                                    self.word_embed_dim)
            bg_feature = self.forward_clip_text(bg_words, clip_model)
        else:
            bg_feature = None

        cls_scores = self.cls_score(cls_features, bg_feature)
        return cls_scores, cls_features

    def forward_clip_text(self, pseudo_words, clip_model):
        with autocast():
            valid_mask = self._drop_word(pseudo_words.half())
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text_tensor(
                pseudo_words.half(), valid_mask)  # add start and stop token
            # assert attn_mask.shape[:2] == pseudo_words.shape[:2]
            if self.cfg.MODEL.ROI_BOX_HEAD.ALL_ENCODER:
                cls_features = \
                    clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                                  text_pe=True, normalize=True)
            else:
                cls_features, _, _ = \
                    clip_model.encode_pseudo_text_endk(pseudo_text, end_token_ids,
                                                       text_pe=True,
                                                       stepk=12, normalize=True)

        return cls_features.float()

    def forward(self, x):
        x = self.pre_forward(x)
        # use clip-text to predict cls feature

        predictions = dict()
        predictions.update(pseudo_words=self.pred_words(x))
        scores, class_features = self.pred_cls_score(predictions['pseudo_words'])
        predictions.update(scores=scores)
        predictions.update(class_features=class_features)
        predictions.update(proposal_deltas=self.bbox_pred(x))

        return predictions
