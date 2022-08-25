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
        if checkboard_cfg.ENABLE:
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

    @torch.no_grad()
    def _bbox_clip_image(self, spanned_boxes, normed_boxes, clip_images,
                         clip_model):
        # TODO: repeat and mask
        device = clip_images.tensor.device
        spanned_boxes = [g.to(device) for g in spanned_boxes]
        normed_boxes = [[g.to(device) for g in img] for img in normed_boxes]
        num_groups_per_image = [g.shape[0] for g in spanned_boxes]
        clip_input_size = self.cfg.INPUT_RESOLUTION
        input_to_clip = roi_align(
            clip_images.tensor, spanned_boxes, (clip_input_size, clip_input_size), 1.0, 2, True)
        input_to_clip = input_to_clip.split(num_groups_per_image, dim=0)
        storage = get_event_storage()
        tik = time()
        attn_masks = multi_apply(get_att_mask,
                                 input_to_clip,
                                 normed_boxes,
                                 num_heads=clip_model.visual.num_heads,
                                 grid_size=clip_input_size // 32)
        storage.put_scalar("contrast_learning_time/generate_attn_mask",
                           np.float32(time() - tik))
        attn_masks = torch.cat(attn_masks, dim=0)
        clip_img_features, clip_img_tokens = clip_model.encode_image(
            repeated_crops, normalize=True, return_image_tokens=True, attn_masks=attn_masks)

        return clip_img_features.float(), clip_img_tokens.float()

    def kd_clip_contrast(self,
                         pseudo_words, normed_boxes, spanned_boxes,
                         clip_images,
                         clip_model, image_info):
        device = pseudo_words.device
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[0]))
        positions = bbox_xyxy_to_cxcywh(torch.cat([g for img in normed_boxes for g in img], dim=0))
        position_embeddings = self.positional_embed(positions)
        pseudo_words = pseudo_words + position_embeddings

        nun_permutations = self.cfg.CHECKBOARD.MAX_PERMUTATIONS
        group_split = [g.shape[0] for img in normed_boxes for g in img]
        word_masks = [self._drop_word(pseudo_words).split(group_split, dim=0)
                      for _ in range(nun_permutations)]
        word_sequences = pseudo_words.split(group_split, dim=0)
        permutations_per_group = [pseudo_permutations(g.shape[0],
                                                      min(math.factorial(g.shape[0]),
                                                          nun_permutations))
                                  for img in normed_boxes for g in img]
        word_sequences_permuted = []
        group_ids_permuted = []
        for group_id, (word_seq, perms) in enumerate(zip(word_sequences, permutations_per_group)):
            for idx, perm in enumerate(perms):
                group_ids_permuted.append(group_id)
                word_seq_flat = word_seq[perm].flatten(0, 1)
                word_mask_flat = word_masks[idx][group_id].flatten(0, 1)
                word_sequences_permuted.append(word_seq_flat[word_mask_flat])

        context_length = max(seq.shape[0] for seq in word_sequences_permuted)
        tok = time()
        clip_model.eval()
        with autocast():
            # TODO: get local image tokens
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text(
                word_sequences,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features, clip_word_tokens = \
                clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                              text_pe=True, normalize=True,
                                              return_word_tokens=True)
            clip_text_features = clip_text_features.float()
            clip_image_features, clip_image_tokens = self._bbox_clip_image(spanned_boxes, clip_images,
                                                                           clip_model)  # need to repeat for perms?
        tik = time()
        storage.put_scalar("contrast_learning_time/clip_model_forward",
                           np.float32(tik-tok))
        global_clip_image_features = self.queues.get_queue('clip_image_features')
        global_clip_text_features = self.queues.get_queue('clip_text_features')
        num_queries = clip_text_features.shape[0]
        assert clip_image_features.shape[0] == num_queries
        label_mask = seq_ids[None] == seq_ids[:, None]
        label_mask.fill_diagonal_(False)
        # mask same synced_img
        img_ids = [torch.tensor(sum(b) * [img_id])
                   for b, img_id in zip(seqs_split_split_by_origin,
                                        image_info.keys())]
        img_ids = torch.cat(img_ids).to(device)
        global_text_feature_img_ids = global_clip_text_features[..., -1]
        global_image_feature_img_ids = global_clip_image_features[..., -1]

        # text features as queries
        image_keys = torch.cat([clip_image_features, global_clip_image_features[..., :-1]], dim=0)
        similarity_matrix_0 = self.ce_temp * clip_text_features @ image_keys.T
        similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
        if global_image_feature_img_ids.shape[0] > 0:
            img_id_mask_0 = img_ids[:, None] == global_image_feature_img_ids[None]
            similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
        # image features as queries
        text_keys = torch.cat([clip_text_features, global_clip_text_features[..., :-1]], dim=0)
        similarity_matrix_1 = self.ce_temp * clip_image_features @ text_keys.T
        similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')
        if global_text_feature_img_ids.shape[0] > 0:
            img_id_mask_1 = img_ids[:, None] == global_text_feature_img_ids[None]
            similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')

        label = torch.arange(num_queries).to(device)

        loss = 0.5 * F.cross_entropy(similarity_matrix_0, label) \
               + 0.5 * F.cross_entropy(similarity_matrix_1, label)
        losses = dict(contrast_loss=loss * self.cfg.CONTRAST_LOSS_WEIGHT)
        # Enqueue
        queues_update = dict(clip_text_features=torch.cat([clip_text_features,
                                                      img_ids.view(-1, 1)], dim=-1).detach(),
                             clip_image_features=torch.cat([clip_image_features,
                                                      img_ids.view(-1, 1)], dim=-1).detach()
                             )

        if self.checkboard_cfg.LOCAL_CORRESPONDENCE:
            tik = time()
            preds_split_by_batch = [n.shape[0] for n in normed_boxes]
            img_ids = [torch.tensor(b * [img_id])
                       for b, img_id in zip(preds_split_by_batch,
                                            image_info.keys())]
            img_ids = torch.cat(img_ids).to(device)
            normed_boxes = torch.cat(normed_boxes, dim=0).split(preds_split_by_perms, dim=0)
            clip_patch_features = F.normalize(roi_align(
                clip_image_tokens, normed_boxes, (1, 1),
                float(clip_image_tokens.shape[-1]), 2, True)[..., 0, 0], dim=-1)
            num_words_per_pred = [wm.sum(-1).tolist() for wm in word_masks]
            clip_word_features = [tk.split(spl) for (tk, spl)
                                  in zip(clip_word_tokens, num_words_per_pred)]
            clip_word_features = F.normalize(torch.stack([feat.mean(0).float()
                                                          for feats in clip_word_features
                                                          for feat in feats], dim=0), dim=-1)
            tok = time()
            storage.put_scalar("contrast_learning_time/prepare_dense_features",
                               np.float32(tok - tik))

            start_id = 0
            box_ids = []
            for g in group_info:
                for ori in g['box_ids']:
                    box_ids_per_ori = [torch.tensor(perm, dtype=torch.float32)
                                       for perm in ori]   # avoid overflow
                    try:
                        box_ids_per_ori = torch.cat(box_ids_per_ori) + start_id
                    except RuntimeError:
                        print(box_ids_per_ori, start_id)
                        exit()
                    start_id += (box_ids_per_ori.max().item() + 1)
                    box_ids.append(box_ids_per_ori)
            box_ids = torch.cat(box_ids).to(device)
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
                img_id_mask_0 = img_ids[:, None] == global_patch_feature_img_ids[None]
                similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
            # image features as queries
            text_keys = torch.cat([clip_word_features, global_clip_word_features[..., :-1]])
            similarity_matrix_1 = self.token_temp * clip_patch_features @ text_keys.T
            if global_word_feature_img_ids.shape[0] > 0:
                img_id_mask_1 = img_ids[:, None] == global_word_feature_img_ids[None]
                similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')
            labels = torch.arange(num_queries, device=device)
            label_mask = box_ids[None] == box_ids[:, None]
            label_mask.fill_diagonal_(False)

            similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
            similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')

            loss = F.cross_entropy(similarity_matrix_0, labels) * 0.5 \
                   + F.cross_entropy(similarity_matrix_1, labels) * 0.5
            losses.update(token_loss=loss * self.cfg.TOKEN_LOSS_WEIGHT)

            queues_update.update(clip_word_features=torch.cat([clip_word_features,
                                                               img_ids.view(-1, 1)], dim=-1).detach(),
                                 clip_patch_features=torch.cat([clip_patch_features,
                                                                img_ids.view(-1, 1)], dim=-1).detach())
        return losses, queues_update

    @torch.no_grad()
    def get_caption_features(self, captions, device, clip_model):
        num_captions_per_image = [len(cap) for cap in captions]
        if sum(num_captions_per_image) == 0:
            return None, num_captions_per_image
        all_captions = [cap for caps in captions for cap in caps]
        tokens = CLIP.tokenize_dynamic(all_captions, truncate=True).to(device)
        caption_features = clip_model.encode_text(tokens, normalize=True).float()
        return caption_features, num_captions_per_image

    def get_loss(self, clip_images, clip_model, image_info, features, roi_head):
        losses = dict()
        queue_update = dict()
        if self.checkboard_cfg.ENABLE:
            sampled_instances, normed_boxes, spanned_boxes \
                = self.checkboard_sampling.sample(image_info.keys())
            pseudo_words = roi_head.get_pseudo_words(sampled_instances, features)
            loss_kd, queue_kd = self.kd_clip_contrast(pseudo_words, normed_boxes, spanned_boxes,
                                                      clip_images,
                                                      clip_model, image_info)
            losses.update(loss_kd)
            queue_update.update(queue_kd)

        self.queues.dequeue_and_enqueue(queue_update)

        return losses
