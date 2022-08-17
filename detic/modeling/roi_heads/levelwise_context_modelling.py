import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from .utils import (multi_apply,
                    bbox_xyxy_to_cxcywh)
from detectron2.utils.events import get_event_storage
import numpy as np
from time import time
from torch.cuda.amp import autocast
from .context_modelling import ContextModelling, process_single_image_groups
from detectron2.modeling.poolers import convert_boxes_to_pooler_format


class LevelWiseContextModelling(ContextModelling):

    def get_multilevel_pseudo_words(self, instances, features, roi_head):
        num_levels = len(roi_head.box_pooler.level_poolers)
        multilevel_pseudo_words = []
        rois = convert_boxes_to_pooler_format(
                    [x.proposal_boxes[x.sample_types == 1] for x in instances]
                )
        if self.cfg.MERGE_MULTILEVEL:
            target_shape = features[1].shape[2:]  # resize to level2  1/8
            resized_x = torch.stack(
                [F.interpolate(x_, size=target_shape,
                               mode="bilinear",
                               align_corners=False) for x_ in features], dim=0)  # stack at dim 0
            resized_x = resized_x.sum(0)
            box_features = roi_head.box_predictor.level_poolers[1](resized_x, rois)
            box_features = roi_head.box_head(box_features)
            input_box_features = roi_head.box_predictor.pre_forward(box_features)
            pseudo_words = roi_head.box_predictor.pred_words(input_box_features)
            multilevel_pseudo_words.append(pseudo_words)
        else:
            for i in range(num_levels):
                box_features = roi_head.box_pooler.level_poolers[i](
                    features[i], rois
                )
                box_features = roi_head.box_head(box_features)
                input_box_features = roi_head.box_predictor.pre_forward(box_features)
                pseudo_words = roi_head.box_predictor.pred_words(input_box_features)
                multilevel_pseudo_words.append(pseudo_words)

        return torch.stack(multilevel_pseudo_words, dim=0)

    def kd_clip_contrast(self,
                         group_infos, clip_images, image_info, features, roi_head):
        clip_model = roi_head.box_predictor.clip
        clip_model.eval()
        pseudo_words = self.get_multilevel_pseudo_words([g['instances'] for g in group_infos],
                                                        features, roi_head)
        device = pseudo_words.device
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[1]))
        # Note: perms = seq
        normed_boxes, spanned_boxes, origin_split, group_split, preds_split_by_perms,\
            seqs_split_split_by_origin, seqs_split_by_group = \
            multi_apply(process_single_image_groups, [g['checkborad_group_info']
                                                      for g in group_infos], device=device)
        positions = bbox_xyxy_to_cxcywh(torch.cat(normed_boxes, dim=0))
        position_embeddings = self.positional_embed(positions)
        pseudo_words = pseudo_words + position_embeddings[None]
        start_id = 0
        seq_ids = []
        group_infos_ = [g['checkborad_group_info'] for g in group_infos]
        for g in group_infos_:
            seq_ids_ = g['seq_ids']
            for seq_id in seq_ids_:
                seq_ids.append(seq_id + start_id)
            start_id += (max(seq_ids_) + 1)
        seq_ids = torch.tensor(seq_ids, dtype=torch.float32).to(device)   # avoid overflow
        normed_boxes_split_by_perms = [normed_boxes_.split(preds_split_by_perms_, dim=0)
                                       for normed_boxes_, preds_split_by_perms_
                                       in zip(normed_boxes, preds_split_by_perms)]
        img_ids = [torch.tensor(len(b) * [img_id])
                   for b, img_id in zip(preds_split_by_perms,
                                        image_info.keys())]
        img_ids = torch.cat(img_ids).to(device).float()
        # torch.cat(normed_boxes).split(preds_split_by_perms, dim=0)
        preds_split_by_perms = [p for b in preds_split_by_perms for p in b]

        # TODO: get multilevel feautures
        num_levels = pseudo_words.shape[0]
        pseudo_words = pseudo_words.flatten(0, 1)
        multilevel_preds_split_by_perms = preds_split_by_perms * num_levels

        word_masks = self._drop_word(pseudo_words)
        word_sequences = pseudo_words.split(multilevel_preds_split_by_perms, dim=0)
        word_masks = word_masks.split(multilevel_preds_split_by_perms, dim=0)
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
            clip_image_features = clip_image_features.repeat(num_levels, 1)    # replicate the image features

        tik = time()
        storage.put_scalar("contrast_learning_time/clip_model_forward",
                           np.float32(tik-tok))
        global_clip_image_features = self.queues.get_queue('clip_image_features')
        global_clip_text_features = self.queues.get_queue('clip_text_features')
        assert clip_image_features.shape[0] == clip_text_features.shape[0]
        num_queries = clip_image_features.shape[0]
        seq_ids = seq_ids.repeat(num_levels)     # replicate the seq ids
        img_ids = img_ids.repeat(num_levels)
        label_mask = seq_ids[None] == seq_ids[:, None]
        label_mask.fill_diagonal_(False)        # before replicate
        # mask same synced_img
        global_text_feature_img_ids = global_clip_text_features[..., -1]
        global_image_feature_img_ids = global_clip_image_features[..., -1]

        # text features as queries
        image_keys = torch.cat([clip_image_features, global_clip_image_features[..., :-1]], dim=0)
        similarity_matrix_0 = self.ce_temp * clip_text_features @ image_keys.T
        similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
        if global_image_feature_img_ids.shape[0] > 0:
            img_id_mask_0 = img_ids[:, None] == global_image_feature_img_ids[None]
            if similarity_matrix_0[:, num_queries:].shape != img_id_mask_0.shape:   # TODO: fix it
                print(f'bug emerges: {similarity_matrix_0.shape}, {img_id_mask_0.shape}', flush=True)
                return self.kd_jump_over_error(pseudo_words, {}, {})
            similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')

        # image features as queries
        text_keys = torch.cat([clip_text_features, global_clip_text_features[..., :-1]], dim=0)
        similarity_matrix_1 = self.ce_temp * clip_image_features @ text_keys.T
        similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')
        if global_text_feature_img_ids.shape[0] > 0:
            img_id_mask_1 = img_ids[:, None] == global_text_feature_img_ids[None]
            if similarity_matrix_1[:, num_queries:].shape != img_id_mask_1.shape:   # TODO: fix it
                print(f'bug emerges: {similarity_matrix_1.shape}, {img_id_mask_1.shape}', flush=True)
                return self.kd_jump_over_error(pseudo_words, {}, {})
            similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')

        label = torch.arange(num_queries).to(device)
        loss = 0.5 * F.cross_entropy(similarity_matrix_0, label) \
               + 0.5 * F.cross_entropy(similarity_matrix_1, label)
        losses = dict(contrast_loss=loss * self.cfg.CONTRAST_LOSS_WEIGHT)
        # Enqueue
        queues_update = dict(clip_text_features=torch.cat([clip_text_features,
                                                           img_ids.view(-1, 1)],
                                                          dim=-1).detach(),
                             clip_image_features=torch.cat([clip_image_features[:num_queries//num_levels],
                                                            img_ids[:num_queries//num_levels].view(-1, 1)],
                                                           dim=-1).detach()
                             )
        # todo multi-level supervision at the word level
        if self.checkboard_cfg.LOCAL_CORRESPONDENCE:
            tik = time()
            preds_split_by_batch = [n.shape[0] for n in normed_boxes]
            img_ids = [torch.tensor(b * [img_id])
                       for b, img_id in zip(preds_split_by_batch,
                                            image_info.keys())]
            img_ids = torch.cat(img_ids).to(device).repeat(num_levels)
            normed_boxes = torch.cat(normed_boxes, dim=0).split(preds_split_by_perms, dim=0)
            clip_patch_features = F.normalize(roi_align(
                clip_image_tokens, normed_boxes, (1, 1),
                float(clip_image_tokens.shape[-1]), 2, True)[..., 0, 0], dim=-1)
            clip_patch_features = clip_patch_features.repeat(num_levels, 1)
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
            for g in group_infos_:
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
            box_ids = torch.cat(box_ids).to(device).repeat(num_levels)
            global_clip_word_features = self.queues.get_queue('clip_word_features')
            global_clip_patch_features = self.queues.get_queue('clip_patch_features')

            global_word_feature_img_ids = global_clip_word_features[..., -1]
            global_patch_feature_img_ids = global_clip_patch_features[..., -1]

            assert clip_patch_features.shape[0] == clip_word_features.shape[0]
            num_queries = clip_patch_features.shape[0]
            # text features as queries
            image_keys = torch.cat([clip_patch_features, global_clip_patch_features[..., :-1]])
            similarity_matrix_0 = self.token_temp * clip_word_features @ image_keys.T
            if global_patch_feature_img_ids.shape[0] > 0:
                img_id_mask_0 = img_ids[:, None] == global_patch_feature_img_ids[None]
                if similarity_matrix_0[:, num_queries:].shape != img_id_mask_0.shape:  # TODO: fix it
                    print(f'bug emerges: {similarity_matrix_0.shape}, {img_id_mask_0.shape}', flush=True)
                    return self.kd_jump_over_error(pseudo_words, losses, queues_update)
                similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
            # image features as queries
            text_keys = torch.cat([clip_word_features, global_clip_word_features[..., :-1]])
            similarity_matrix_1 = self.token_temp * clip_patch_features @ text_keys.T
            if global_word_feature_img_ids.shape[0] > 0:
                img_id_mask_1 = img_ids[:, None] == global_word_feature_img_ids[None]
                if similarity_matrix_1[:, num_queries:].shape != img_id_mask_1.shape:  # TODO: fix it
                    print(f'bug emerges: {similarity_matrix_1.shape}, {img_id_mask_1.shape}', flush=True)
                    return self.kd_jump_over_error(pseudo_words, losses, queues_update)
                similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')
            labels = torch.arange(num_queries, device=device)

            label_mask = box_ids[None] == box_ids[:, None]
            label_mask.fill_diagonal_(False)

            similarity_matrix_0[:, :clip_patch_features.shape[0]][label_mask] = float('-inf')
            similarity_matrix_1[:, :clip_word_features.shape[0]][label_mask] = float('-inf')

            loss = F.cross_entropy(similarity_matrix_0, labels) * 0.5 \
                   + F.cross_entropy(similarity_matrix_1, labels) * 0.5
            losses.update(token_loss=loss * self.cfg.TOKEN_LOSS_WEIGHT)

            queues_update.update(clip_word_features=torch.cat([clip_word_features,
                                                               img_ids.view(-1, 1)],
                                                              dim=-1).detach(),
                                 clip_patch_features=torch.cat([clip_patch_features[:num_queries//num_levels],
                                                                img_ids[:num_queries//num_levels].view(-1, 1)],
                                                               dim=-1).detach())
        return losses, queues_update

    def get_loss(self, group_infos, clip_images, image_info, features, roi_head):
        losses = dict()
        queue_update = dict()
        if self.checkboard_cfg.ENABLE:
            loss_kd, queue_kd = self.kd_clip_contrast(group_infos, clip_images, image_info, features, roi_head)
            losses.update(loss_kd)
            queue_update.update(queue_kd)

        # caption not currently supported

        self.queues.dequeue_and_enqueue(queue_update)

        return losses
