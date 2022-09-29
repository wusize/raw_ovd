from models.deformable_detr import DeformableDETR, _get_clones
from models.deformable_transformer import DeformableTransformer
from models.backbone import Joiner
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from models.position_encoding import PositionEmbeddingSine

from .d2_deformable_detr import DeformableDetr, MaskedBackbone
from detic.modeling.context.context_modelling_for_detr \
    import DETRContextModelling as ContextModelling
from detic.modeling.utils import multi_apply, load_class_freq
from detic.modeling.roi_heads.zero_shot_classifier import ZeroShotClassifier

from detectron2.structures import Instances, Boxes, ImageList
from detectron2.modeling import META_ARCH_REGISTRY
from detic.modeling import clip as CLIP


import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import nn
import math


class CustomDeformableTransformer(DeformableTransformer):
    def forward(self, srcs, masks, pos_embeds, **kwargs):
        assert self.two_stage

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

        topk = self.two_stage_num_proposals
        topk_objectness_logits, topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()      # detached
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points

        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references

        useful_infos = dict(memory=memory, spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index, valid_ratios=valid_ratios,
                            mask_flatten=mask_flatten, topk_objectness_logits=topk_objectness_logits.detach())

        return hs, init_reference_out, useful_infos, \
               inter_references_out, enc_outputs_class, enc_outputs_coord_unact


class CustomDeformableDETR(DeformableDETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, cfg=None, num_words=4, word_embed_dim=512, **kwargs):
        super().__init__(**kwargs)
        assert self.two_stage
        self.num_words = num_words
        self.word_embed_dim = word_embed_dim
        # pred 1 word per level
        class_embed = nn.Linear(self.transformer.d_model, word_embed_dim)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.bias.data = torch.ones(word_embed_dim) * bias_value
        num_pred = self.transformer.decoder.num_layers + 1

        if self.with_box_refine:
            self.class_embed = _get_clones(class_embed, num_pred)
        else:
            self.class_embed = nn.ModuleList([class_embed for _ in range(num_pred)])

        self.classifier = ZeroShotClassifier(cfg)
        cat_freq_path = cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
        is_base = load_class_freq(cat_freq_path, 1.0, min_count=0)  # to mask the novel classes
        self.register_buffer('is_base', is_base)
        self.cfg = cfg

    def forward(self, samples, clip_model=None, gts=None, context_sampler=None,
                image_ids=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, useful_infos, inter_references, enc_outputs_class, enc_outputs_coord_unact \
            = self.transformer(srcs, masks, pos)
        # topk = self.transformer.two_stage_num_proposals
        outputs_classes = []
        outputs_coords = []
        assert hs.shape[0] >= self.num_words
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes, dim=-2)
        outputs_coord = torch.stack(outputs_coords, dim=-2)
        assert outputs_class.shape[2] >= self.num_words

        out = {'pred_logits': self.interpret_pseudo_words(outputs_class[..., -self.num_words:, :],
                                                          clip_model=clip_model),   # each stage predict a word
               'pred_boxes': outputs_coord[..., -1, :]}
        # if self.aux_loss:
            # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        assert not self.aux_loss

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if gts is not None and context_sampler is not None:
            out.update(self.context_modelling(useful_infos, context_sampler, gts, init_reference.detach(), image_ids))

        return out

    # TODO: from which level to get reference_points
    def context_modelling(self, useful_infos, context_sampler, gts, reference_points, image_ids):
        transformer = self.transformer
        assert self.training
        topk_objectness_logits = useful_infos['topk_objectness_logits']
        added_instances, group_infos = multi_apply(context_sampler.sample,
                                                   [Instances(image_size=(1, 1),  # normalized
                                                              objectness_logits=torch.cat([
                                                                  inverse_sigmoid(torch.zeros_like(gt[:, 0])),
                                                                  topk_obj
                                                              ], dim=0),
                                                              proposal_boxes=Boxes(box_cxcywh_to_xyxy(  # xyxy format
                                                                  torch.cat([gt, ref], dim=0))))
                                                    for gt, ref, topk_obj in zip(gts, reference_points,
                                                                                 topk_objectness_logits)],
                                                   image_ids)
        added_normed_boxes, added_boxes_valid, added_attn_masks = self.instances_to_tensor(added_instances)
        added_pos_trans_out = transformer.pos_trans_norm(
            transformer.pos_trans(
                transformer.get_proposal_pos_embed(
                    inverse_sigmoid(added_normed_boxes)
                )
            )
        )
        bs, _, c = useful_infos['memory'].shape
        added_query_embed, added_tgt = torch.split(added_pos_trans_out, c, dim=2)

        # decoder
        added_hs, _ = transformer.decoder(added_tgt, added_normed_boxes, useful_infos['memory'],
                                          useful_infos['spatial_shapes'],
                                          useful_infos['level_start_index'],
                                          useful_infos['valid_ratios'],
                                          added_query_embed,
                                          useful_infos['mask_flatten'],
                                          self_attn_mask=added_attn_masks)

        kd_pseudo_words = torch.stack([self.class_embed[lvl](added_hs[lvl])
                                       for lvl in range(added_hs.shape[0]-self.num_words, added_hs.shape[0])], dim=-2)

        kd_pseudo_words = [words[valid > 0.0] for words, valid in zip(kd_pseudo_words, added_boxes_valid)]

        return dict(group_infos=group_infos,
                    kd_pseudo_words=torch.cat(kd_pseudo_words))

    def instances_to_tensor(self, instances):
        max_num = max([len(inst) for inst in instances])
        boxes_list = []
        attn_mask_list = []
        boxes_valid_list = []
        for inst in instances:
            boxes = inst.proposal_boxes.tensor  # normalized
            pos_valid = torch.ones_like(boxes[:, 0])
            add_num = max_num - boxes.shape[0]
            add_boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]]).to(boxes).repeat(add_num, 1)
            neg_valid = torch.zeros_like(add_boxes[:, 0])
            boxes_list.append(torch.cat([boxes, add_boxes], dim=0))

            boxes_valid = torch.cat([pos_valid, neg_valid], dim=0)

            boxes_valid_list.append(boxes_valid)

            attn_mask = boxes_valid[None] * boxes_valid[:, None]
            if self.cfg.MODEL.DETR.NO_SELF_ATTN:
                attn_mask = attn_mask * 0.0
            attn_mask = torch.where(attn_mask > 0.0, 0.0, float('-inf'))
            attn_mask.fill_diagonal_(0.0)
            attn_mask = attn_mask[None].repeat(self.transformer.nhead, 1, 1)
            attn_mask_list.append(attn_mask)

        boxes = torch.stack(boxes_list, dim=0)  # normalized in xyxy format
        attn_masks = torch.cat(attn_mask_list, dim=0)  # bs * nheads
        boxes_valid = torch.stack(boxes_valid_list, dim=0)

        return box_xyxy_to_cxcywh(boxes), boxes_valid, attn_masks

    def interpret_pseudo_words(self, pseudo_words, clip_model):
        bs, num_pred, num_words, _ = pseudo_words.shape
        pseudo_words = pseudo_words.flatten(0, 1)
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
        cls_logits = self.classifier(cls_features.float())[..., :-1]
        if self.cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS:
            zero_weight = (self.is_base.view(-1) > 1e-4).float()
            cls_logits[..., zero_weight < 1.0] = self.cfg.MODEL.ROI_BOX_HEAD.MASK_VALUE

        return cls_logits.view(bs, num_pred, -1)

    def _drop_word(self, word_embeddings):
        p = self.cfg.MODEL.ROI_BOX_HEAD.RANDOM_DROPOUT
        num_preds, num_words, _ = word_embeddings.shape
        mask = F.dropout(word_embeddings.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0       # TODO add random on this
        mask[mask > 0.0] = 1.0
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask


@META_ARCH_REGISTRY.register()
class CustomDeformableDetr(DeformableDetr):
    # FIXME: support mask on
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES

        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DETR.TWO_STAGE
        assert two_stage
        with_box_refine = cfg.MODEL.DETR.WITH_BOX_REFINE

        # Loss parameters:
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        transformer = CustomDeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=two_stage,
            two_stage_num_proposals=num_queries)
        num_words = cfg.MODEL.ROI_BOX_HEAD.NUM_WORDS
        word_embed_dim = cfg.MODEL.ROI_BOX_HEAD.WORD_EMBED_DIM
        self.detr = CustomDeformableDETR(
            backbone=backbone, transformer=transformer,
            cfg=cfg,
            num_words=num_words, word_embed_dim=word_embed_dim,
            num_classes=self.num_classes,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            aux_loss=deep_supervision,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
        )
        self.context_modeling_cfg = cfg.CONTEXT_MODELLING
        self.context_modeling = ContextModelling(self.context_modeling_cfg,
                                                 num_words=num_words,
                                                 word_embed_dim=word_embed_dim,
                                                 word_dropout=cfg.MODEL.ROI_BOX_HEAD.RANDOM_DROPOUT)

        clip_cfg = cfg.MODEL.CLIP
        self.clip, _ = CLIP.load(name=clip_cfg.NAME,
                                 use_image_encoder=clip_cfg.USE_IMAGE_ENCODER,
                                 download_root=clip_cfg.MODEL_ROOT)
        self.clip.init_weights()

    def forward(self, batched_inputs):
        """
        Args:
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        image_ids = [b['image_id'] for b in batched_inputs]
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            losses_custom = dict()
            if self.context_modeling_cfg.ENABLE:
                output = self.detr(images, clip_model=self.clip,
                                   gts=[t['boxes'] for t in targets],
                                   context_sampler=self.context_modeling,
                                   image_ids=image_ids)
                # TODO context modelling
                clip_images = self.clip_preprocess_image(batched_inputs)
                losses_custom.update(self.context_modeling.get_loss(output['group_infos'],
                                                                    output, clip_images,
                                                                    self.clip,
                                                                    [dict(image_id=img_id)
                                                                     for img_id in image_ids]))
            else:
                output = self.detr(images, clip_model=self.clip)

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            loss_dict.update(losses_custom)
            return loss_dict
        else:
            output = self.detr(images, clip_model=self.clip)
            image_sizes = output["pred_boxes"].new_tensor(
                [(t["height"], t["width"]) for t in batched_inputs])
            results = self.post_process(output, image_sizes)
            return results

    def clip_preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        mean = [[[122.7709383]], [[116.7460125]], [[104.09373615]]]
        std = [[[68.5005327]], [[66.6321579]], [[70.32316305]]]
        clip_pixel_mean = torch.tensor(mean).to(self.device)
        clip_pixel_std = torch.tensor(std).to(self.device)
        if self.cfg.INPUT.FORMAT == 'BGR':
            channel_order = [2, 1, 0]
        else:
            channel_order = [0, 1, 2]

        images = [x["image"][channel_order].to(self.device) for x in batched_inputs]
        images = [(x - clip_pixel_mean) / clip_pixel_std for x in images]
        images = ImageList.from_tensors(images)
        return images
