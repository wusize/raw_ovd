from .custom_detrs import CustomDeformableDETR, CustomDeformableTransformer, CustomDeformableDetr
from models.deformable_detr import DeformableDETR, _get_clones
from models.backbone import Joiner
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from models.position_encoding import PositionEmbeddingSine

from .d2_deformable_detr import MaskedBackbone
from detic.modeling.utils import multi_apply
from detectron2.structures import Instances, Boxes
from detectron2.modeling import META_ARCH_REGISTRY


import torch
import torch.nn.functional as F
from torch import nn
import math


class CustomDeformableDETRV2(CustomDeformableDETR):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.two_stage
        num_words = self.num_words
        word_embed_dim = self.word_embed_dim
        class_embed = nn.Linear(self.transformer.d_model, num_words * word_embed_dim)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.bias.data = torch.ones(num_words * word_embed_dim) * bias_value
        num_pred = self.transformer.decoder.num_layers + 1

        if self.with_box_refine:
            self.class_embed = _get_clones(class_embed, num_pred)
        else:
            self.class_embed = nn.ModuleList([class_embed for _ in range(num_pred)])

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
        num_levels, batch_size, num_queries, _ = hs.shape[:2]
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
        outputs_class = torch.stack(outputs_classes).view(num_levels, batch_size, num_queries,
                                                          self.num_words, self.word_embed_dim)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': self.interpret_pseudo_words(outputs_class[-1],
                                                          clip_model=clip_model),   # each stage predict a word
               'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, clip_model)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if gts is not None and context_sampler is not None:
            out.update(self.context_modelling(useful_infos, context_sampler, gts, init_reference.detach(), image_ids))

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, clip_model):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # TODO: pre_process class
        return [{'pred_logits': self.interpret_pseudo_words(a, clip_model=clip_model), 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # TODO: from which level to get reference_points
    def context_modelling(self, useful_infos, context_sampler, gts, reference_points, image_ids):
        transformer = self.transformer
        assert self.training
        topk_objectness_logits = useful_infos['topk_objectness_logits']
        added_instances, group_infos = multi_apply(context_sampler.sample,
                                                   [Instances(image_size=(1, 1),  # normalized
                                                              objectness_logits=torch.cat([
                                                                  inverse_sigmoid(torch.ones_like(gt[:, 0])),
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

        kd_pseudo_words = self.class_embed[added_hs.shape[0]-1](added_hs[added_hs.shape[0]-1]).view(
            bs, -1, self.num_words, self.word_embed_dim)

        kd_pseudo_words = [words[valid > 0.0] for words, valid in zip(kd_pseudo_words, added_boxes_valid)]

        return dict(group_infos=group_infos,
                    kd_pseudo_words=torch.cat(kd_pseudo_words))


@META_ARCH_REGISTRY.register()
class CustomDeformableDetrV2(CustomDeformableDetr):
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
        self.detr = CustomDeformableDETRV2(
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
