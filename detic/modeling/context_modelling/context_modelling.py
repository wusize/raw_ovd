import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .shift_window_sampling import ShiftWindowSampling
from torchvision.ops import roi_align, nms
from .utils import (multi_apply, get_normed_boxes, pseudo_permutations,
                    bbox_xyxy_to_cxcywh, get_att_mask)
from detectron2.structures import Instances, Boxes
from .queues import Queues
from detectron2.utils.events import get_event_storage
import numpy as np
from time import time
from torch.cuda.amp import autocast
from detic.modeling import clip as CLIP


def process_single_image_groups(group_info, device):
    # add region dropout
    spanned_boxes = group_info['spanned_boxes']
    normed_boxes = group_info['normed_boxes']
    box_ids = group_info['box_ids']
    seq_ids = [list(map(box_ids2seq_id, box_ids_)) for box_ids_ in box_ids]
    seq_ids_per_image = []
    start_id = 0
    for seq_ids_ in seq_ids:
        seq_ids_per_image.extend([box_id + start_id for box_id in seq_ids_])
        start_id += (max(seq_ids_) + 1)
    group_info['seq_ids'] = seq_ids_per_image
    group_split = [len(grp) * grp[0].shape[0] for ori in normed_boxes for grp in ori]
    origin_split = [sum([len(grp) * grp[0].shape[0] for grp in ori]) for ori in normed_boxes]
    perms_split = [perm.shape[0] for ori in normed_boxes for grp in ori for perm in grp]

    seq_level_origin_split = [sum([len(grp) for grp in ori]) for ori in normed_boxes]
    seq_level_group_split = [len(grp) for ori in normed_boxes for grp in ori]

    normed_boxes = torch.cat([torch.cat(grp, dim=0)
                              for ori in normed_boxes for grp in ori], dim=0).to(device)
    spanned_boxes = torch.cat([torch.stack(ori, dim=0) for ori in spanned_boxes]).to(device)

    return normed_boxes, spanned_boxes, origin_split, group_split, perms_split, \
           seq_level_origin_split, seq_level_group_split


def box_ids2seq_id(box_ids):
    box_ids_copy = box_ids.copy()
    box_ids_sorted = sorted(box_ids_copy, reverse=True)
    box_ids_str = ''.join([str(box_id) for box_id in box_ids_sorted])

    return int(box_ids_str)


def identity_func(x):
    return x


class SinePositionalEncoding(nn.Module):

    def __init__(self,
                 num_feats=128,
                 num_words=4,
                 word_dims=512,
                 temperature=1.2,
                 scale=2 * math.pi):
        super(SinePositionalEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.scale = scale
        self.pos_proj = nn.Sequential(
            nn.Linear(num_feats * 4, word_dims),
            nn.LayerNorm(word_dims),
            nn.Linear(word_dims, num_words * word_dims))
        self.num_words = num_words
        self.word_dims = word_dims

    def forward(self, x):
        embed = x * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** ((dim_t // 2) - (self.num_feats // 4))
        pos = embed[:, :, None] * dim_t[None, None]
        pos[..., 0::2] = pos[..., 0::2].sin()
        pos[..., 1::2] = pos[..., 1::2].cos()

        assert pos.shape[-1] == self.num_feats

        pos = pos.view(-1, 4 * self.num_feats)

        return self.pos_proj(pos).view(-1, self.num_words, self.word_dims)


class ZeroPositionalEncoding(nn.Module):

    def __init__(self,
                 num_words=4,
                 word_dims=512,):
        super(ZeroPositionalEncoding, self).__init__()
        self.num_words = num_words
        self.word_dims = word_dims

    def forward(self, x):

        return x.new_zeros(x.shape[0], self.num_words, self.word_dims)


class ContextModelling(nn.Module):
    def __init__(self, cfg, num_words, word_embed_dim, word_dropout, sigmoid=True):
        super(ContextModelling, self).__init__()
        self.num_words_per_pred = num_words
        self.word_embed_dim = word_embed_dim
        self.word_dropout = word_dropout
        self.cfg = cfg
        checkboard_cfg = cfg.CHECKBOARD
        if sigmoid:
            self.get_objectness = torch.sigmoid
        else:
            self.get_objectness = identity_func
        self.checkboard_sampling = ShiftWindowSampling(cfg)
        self.checkboard_cfg = checkboard_cfg
        if self.cfg.ENABLE:
            self.ce_temp = cfg.CE_TEMP        # 60.0
            self.bce_temp = cfg.BCE_TEMP
            self.token_temp = cfg.TOKEN_TEMP  # 50.0
            self.bce_bias = nn.Parameter(torch.tensor(0.0))

            self.queues = Queues(queue_cfg=self.cfg.QUEUE)
            if self.cfg.POSITIONAL_ENCODING:
                self.positional_embed = SinePositionalEncoding(num_feats=128,
                                                               num_words=num_words,
                                                               word_dims=word_embed_dim)
            else:
                self.positional_embed = ZeroPositionalEncoding(num_words=num_words,
                                                               word_dims=word_embed_dim)

    @torch.no_grad()
    def _drop_word(self, word_embeddings):
        p = self.word_dropout
        num_preds, num_words, _ = word_embeddings.shape
        mask = F.dropout(word_embeddings.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        # check empty
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0
        mask = mask > 0.0

        return mask

    def _sample_for_in_queue(self, features):
        max_num = self.cfg.QUEUE.MAX_UPDATE
        num_feats = features.shape[0]
        if max_num >= num_feats:
            return features
        else:
            return features[random.choices(range(num_feats),
                                           k=max_num)]

    @torch.no_grad()
    def _bbox_clip_image(self, spanned_boxes, normed_boxes, clip_images,
                         clip_model):
        # TODO: repeat and mask
        device = clip_images.tensor.device
        spanned_boxes = [g.to(device) for g in spanned_boxes]
        normed_boxes = [[g.to(device) for g in img] for img in normed_boxes]
        # num_groups_per_image = [img.shape[0] for img in spanned_boxes]
        clip_input_size = self.cfg.INPUT_RESOLUTION
        input_to_clip = roi_align(
            clip_images.tensor, spanned_boxes, (clip_input_size, clip_input_size), 1.0, 2, True)
        # input_to_clip = input_to_clip.split(num_groups_per_image, dim=0)
        storage = get_event_storage()
        tik = time()
        attn_masks = [get_att_mask(img, num_heads=clip_model.visual.num_heads,
                                   grid_size=clip_input_size // 32) for img in normed_boxes]
        storage.put_scalar("contrast_learning_time/generate_attn_mask",
                           np.float32(time() - tik))
        attn_masks = torch.cat(attn_masks, dim=0)
        clip_img_features, clip_img_tokens = clip_model.encode_image(
            input_to_clip, normalize=True, return_image_tokens=True, attn_masks=attn_masks)

        return clip_img_features.float(), clip_img_tokens.float()

    def kd_clip_contrast(self,
                         pseudo_words,
                         box_ids,
                         normed_boxes, spanned_boxes,
                         clip_images,
                         clip_model, image_info):
        device = pseudo_words.device
        box_ids = box_ids.to(device)
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[0]))
        normed_boxes_tensor = torch.cat([g.to(device) for img in normed_boxes for g in img], dim=0)
        positions = bbox_xyxy_to_cxcywh(normed_boxes_tensor)
        position_embeddings = self.positional_embed(positions)
        pseudo_words = pseudo_words + position_embeddings

        nun_permutations = self.cfg.MAX_PERMUTATIONS
        group_split = [g.shape[0] for img in normed_boxes for g in img]
        image_ids = [img_id
                     for img_n_boxes, img_id in zip(normed_boxes, image_info.keys())
                     for g in img_n_boxes]
        word_masks = [self._drop_word(pseudo_words).split(group_split, dim=0)
                      for _ in range(nun_permutations)]
        word_sequences = pseudo_words.split(group_split, dim=0)
        box_ids_after_group_split = box_ids.split(group_split, dim=0)
        normed_boxes_after_group_split = normed_boxes_tensor.split(group_split, dim=0)
        permutations_per_group = [pseudo_permutations(g.shape[0],
                                                      min(math.factorial(g.shape[0]),
                                                          nun_permutations))
                                  for img in normed_boxes for g in img]
        # TODO: permutation
        word_sequences_permuted = []
        group_ids_permuted = []
        image_ids_permuted = []
        word_masks_permuted = []
        box_ids_permuted = []
        normed_boxes_permuted = []
        image_ids_permuted_box_level = []

        for group_id, (word_seq, perms, img_id) in enumerate(zip(word_sequences, permutations_per_group,
                                                                 image_ids)):
            for idx, perm in enumerate(perms):
                image_ids_permuted.append(img_id)
                group_ids_permuted.append(group_id)
                word_seq_flat = word_seq[perm].flatten(0, 1)
                word_mask_flat = word_masks[idx][group_id].flatten(0, 1)
                word_sequences_permuted.append(word_seq_flat[word_mask_flat])
                word_masks_permuted.append(word_masks[idx][group_id])
                normed_boxes_single_group = normed_boxes_after_group_split[group_id]
                box_ids_single_group = box_ids_after_group_split[group_id]

                box_ids_permuted.append(box_ids_single_group[perm])
                normed_boxes_permuted.append(normed_boxes_single_group[perm])

                image_ids_permuted_box_level.extend([img_id] * len(perm))

        word_masks_permuted_split_by_group = word_masks_permuted
        normed_boxes_permuted_split_by_group = normed_boxes_permuted
        image_ids_permuted_box_level = torch.tensor(image_ids_permuted_box_level).to(device)
        box_ids_permuted = torch.cat(box_ids_permuted)

        group_ids_permuted = torch.tensor(group_ids_permuted).to(device)
        image_ids_permuted = torch.tensor(image_ids_permuted).to(device)
        context_length = max(seq.shape[0] for seq in word_sequences_permuted)
        tok = time()
        clip_model.eval()
        with autocast():
            # TODO: get local image tokens
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text(
                word_sequences_permuted,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features, clip_word_tokens = \
                clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                              text_pe=True, normalize=True,
                                              return_word_tokens=True)
            clip_text_features = clip_text_features.float()
            clip_image_features, clip_image_tokens = self._bbox_clip_image(spanned_boxes, normed_boxes,
                                                                           clip_images,
                                                                           clip_model)  # need to repeat for perms?
        # TODO: repeat (clip_image_features, clip_image_tokens) for each permutation
        num_permutations_per_group = [len(p) for p in permutations_per_group]
        clip_image_features = torch.stack(
            [feat for num, feat in zip(num_permutations_per_group, clip_image_features)
             for _ in range(num)], dim=0)
        clip_image_tokens = torch.stack(
            [feat for num, feat in zip(num_permutations_per_group, clip_image_tokens)
             for _ in range(num)], dim=0)

        tik = time()
        storage.put_scalar("contrast_learning_time/clip_model_forward",
                           np.float32(tik-tok))
        global_clip_image_features = self.queues.get_queue('clip_image_features')
        global_clip_text_features = self.queues.get_queue('clip_text_features')
        num_queries = clip_text_features.shape[0]
        assert clip_image_features.shape[0] == num_queries
        label_mask = group_ids_permuted[None] == group_ids_permuted[:, None]
        label_mask.fill_diagonal_(False)
        # mask same synced_img
        global_text_feature_img_ids = global_clip_text_features[..., -1]
        global_image_feature_img_ids = global_clip_image_features[..., -1]

        # text features as queries
        image_keys = torch.cat([clip_image_features, global_clip_image_features[..., :-1]], dim=0)
        similarity_matrix_0 = self.ce_temp * clip_text_features @ image_keys.T
        similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
        if global_image_feature_img_ids.shape[0] > 0:
            img_id_mask_0 = image_ids_permuted[:, None] == global_image_feature_img_ids[None]
            similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
        # image features as queries
        text_keys = torch.cat([clip_text_features, global_clip_text_features[..., :-1]], dim=0)
        similarity_matrix_1 = self.ce_temp * clip_image_features @ text_keys.T
        similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')
        if global_text_feature_img_ids.shape[0] > 0:
            img_id_mask_1 = image_ids_permuted[:, None] == global_text_feature_img_ids[None]
            similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')

        label = torch.arange(num_queries).to(device)

        loss = 0.5 * F.cross_entropy(similarity_matrix_0, label) \
               + 0.5 * F.cross_entropy(similarity_matrix_1, label)
        losses = dict(contrast_loss=loss * self.cfg.CONTRAST_LOSS_WEIGHT)
        # Enqueue
        queues_update = dict(clip_text_features=self._sample_for_in_queue(
                                 torch.cat([clip_text_features,
                                            image_ids_permuted.view(-1, 1)], dim=-1).detach()),
                             clip_image_features=self._sample_for_in_queue(
                                 torch.cat([clip_image_features,
                                            image_ids_permuted.view(-1, 1)], dim=-1).detach())
                             )

        if True:
            tik = time()
            clip_patch_features = F.normalize(roi_align(
                clip_image_tokens, normed_boxes_permuted_split_by_group, (1, 1),
                float(clip_image_tokens.shape[-1]), 2, True)[..., 0, 0], dim=-1)

            num_words_per_pred = [wm.sum(-1).tolist() for wm in word_masks_permuted_split_by_group]
            clip_word_features = [tk.split(spl) for (tk, spl)
                                  in zip(clip_word_tokens, num_words_per_pred)]
            clip_word_features = F.normalize(torch.stack([feat.mean(0).float()
                                                          for g in clip_word_features
                                                          for feat in g], dim=0), dim=-1)
            tok = time()
            storage.put_scalar("contrast_learning_time/prepare_dense_features",
                               np.float32(tok - tik))

            global_clip_word_features = self.queues.get_queue('clip_word_features')
            global_clip_patch_features = self.queues.get_queue('clip_patch_features')

            global_word_feature_img_ids = global_clip_word_features[..., -1]
            global_patch_feature_img_ids = global_clip_patch_features[..., -1]

            num_queries = clip_patch_features.shape[0]
            assert num_queries == clip_word_features.shape[0]

            # text features as queries
            image_keys = torch.cat([clip_patch_features, global_clip_patch_features[..., :-1]])
            similarity_matrix_0 = self.token_temp * clip_word_features @ image_keys.T
            if global_patch_feature_img_ids.shape[0] > 0:
                img_id_mask_0 = image_ids_permuted_box_level[:, None] == global_patch_feature_img_ids[None]
                similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
            # image features as queries
            text_keys = torch.cat([clip_word_features, global_clip_word_features[..., :-1]])
            similarity_matrix_1 = self.token_temp * clip_patch_features @ text_keys.T
            if global_word_feature_img_ids.shape[0] > 0:
                img_id_mask_1 = image_ids_permuted_box_level[:, None] == global_word_feature_img_ids[None]
                similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')
            labels = torch.arange(num_queries, device=device)
            label_mask = box_ids_permuted[None] == box_ids_permuted[:, None]
            label_mask.fill_diagonal_(False)

            similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
            similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')

            loss = F.cross_entropy(similarity_matrix_0, labels) * 0.5 \
                   + F.cross_entropy(similarity_matrix_1, labels) * 0.5
            losses.update(token_loss=loss * self.cfg.TOKEN_LOSS_WEIGHT)

            queues_update.update(
                clip_word_features=self._sample_for_in_queue(
                    torch.cat([clip_word_features,
                               image_ids_permuted_box_level.view(-1, 1)],
                              dim=-1).detach()),
                clip_patch_features=self._sample_for_in_queue(
                    torch.cat([clip_patch_features,
                               image_ids_permuted_box_level.view(-1, 1)],
                              dim=-1).detach()))
        return losses, queues_update

    def get_loss(self, clip_images, clip_model, image_info, features, roi_head):
        losses = dict()
        queue_update = dict()
        device = clip_images.tensor.device
        sampled_instances, normed_boxes, spanned_boxes \
            = self.checkboard_sampling.sample(clip_images.image_sizes)
        sampled_instances = [inst.to(device) for inst in sampled_instances]
        pseudo_words = roi_head.get_pseudo_words(sampled_instances, features)
        box_ids = torch.cat([inst.box_ids for inst in sampled_instances])
        loss_kd, queue_kd = self.kd_clip_contrast(pseudo_words,
                                                  box_ids,
                                                  normed_boxes, spanned_boxes,
                                                  clip_images,
                                                  clip_model, image_info)
        losses.update(loss_kd)
        queue_update.update(queue_kd)

        self.queues.dequeue_and_enqueue(queue_update)

        return losses
