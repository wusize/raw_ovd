from .context_modelling import ContextModelling, process_single_image_groups
import time
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from .utils import (multi_apply,
                    bbox_xyxy_to_cxcywh)
from detectron2.utils.events import get_event_storage
import numpy as np
from time import time
from torch.cuda.amp import autocast


class ContextModellingV6(ContextModelling):
    def __init__(self, *args, **kwargs):
        super(ContextModellingV6, self).__init__(*args, **kwargs)
        box_types = ["top left", "top", "top right",
                     "left", "center", "right",
                     "bottom left", "bottom", "bottom right"]
        box_type_info = torch.load("datasets/metadata/box_type_embeddings.pth")   # hard code this
        assert box_types == box_type_info['box_types']
        prompt_length = self.cfg.PROMPT_LENGTH
        box_type_embeddings = box_type_info['embeddings']
        valid = box_type_info['valid']
        saved_length = valid.shape[1]
        assert prompt_length >= saved_length
        embedding_template = torch.zeros(len(box_types), prompt_length, self.word_embed_dim)
        valid_template = torch.zeros(len(box_types), prompt_length, 1)

        embedding_template[:, :saved_length] = box_type_embeddings
        valid_template[:, :saved_length, 0] = valid

        self.register_buffer('box_type_embeddings', embedding_template, persistent=False)
        self.register_buffer('box_type_valid', valid_template, persistent=False)

    def _add_prompting(self, pseudo_words, word_masks, positional_embeddings, box_types=None):

        prompts = self.prompt(pseudo_words)
        if not self.cfg.USE_BOX_POSITION:
            positional_embeddings = positional_embeddings * 0.0
        prompts = prompts + positional_embeddings
        box_type_embeddings = self.box_type_embeddings[box_types]
        box_type_valid = self.box_type_valid[box_types]
        prompts = prompts * (1.0 - box_type_valid) + box_type_embeddings * box_type_valid

        prompt_masks = torch.ones_like(prompts[..., 0]) > 0.0
        pseudo_words = torch.cat([prompts, pseudo_words], dim=1)
        word_masks = torch.cat([prompt_masks, word_masks], dim=1)
        return pseudo_words, word_masks

    def kd_clip_contrast(self,
                         group_info,
                         predictions, clip_images,
                         clip_model,
                         image_info=None,
                         positional_encoder=None):
        pseudo_words = predictions.pop('kd_pseudo_words')
        device = pseudo_words.device
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[0]))
        # Note: perms = seq
        normed_boxes, spanned_boxes, origin_split, group_split, preds_split_by_perms,\
            seqs_split_split_by_origin, seqs_split_by_group = \
            multi_apply(process_single_image_groups, group_info, device=device)
        positions = bbox_xyxy_to_cxcywh(torch.cat(normed_boxes, dim=0))
        position_embeddings = positional_encoder(positions)
        word_masks = self._drop_word(pseudo_words)
        pseudo_words, word_masks = self._add_prompting(pseudo_words, word_masks, position_embeddings,
                                                       box_types=torch.cat([g['box_types'] for g in group_info]))
        # pseudo_words = pseudo_words + position_embeddings
        # word_masks = self._drop_word(pseudo_words)
        start_id = 0
        seq_ids = []
        for g in group_info:
            seq_ids_ = g['seq_ids']
            for seq_id in seq_ids_:
                seq_ids.append(seq_id + start_id)
            start_id += (max(seq_ids_) + 1)
        seq_ids = torch.tensor(seq_ids, dtype=torch.float32).to(device)   # avoid overflow
        normed_boxes_split_by_perms = [normed_boxes_.split(preds_split_by_perms_, dim=0)
                                       for normed_boxes_, preds_split_by_perms_
                                       in zip(normed_boxes, preds_split_by_perms)]
        # torch.cat(normed_boxes).split(preds_split_by_perms, dim=0)
        preds_split_by_perms = [p for b in preds_split_by_perms for p in b]
        word_sequences = pseudo_words.split(preds_split_by_perms, dim=0)
        word_masks = word_masks.split(preds_split_by_perms, dim=0)
        word_sequences = [seq.flatten(0, 1)[wm.flatten(0, 1)] for seq, wm in zip(word_sequences, word_masks)]
        context_length = max([seq.shape[0] for seq in word_sequences])
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
                                                                           seqs_split_by_group,
                                                                           normed_boxes_split_by_perms,
                                                                           clip_model)
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
            # Discard the prompt
            clip_word_features = F.normalize(torch.stack([feat[self.cfg.PROMPT_LENGTH:].mean(0).float()
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