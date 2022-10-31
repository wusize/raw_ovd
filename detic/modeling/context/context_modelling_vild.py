import torch
from .utils import multi_apply
from detectron2.utils.events import get_event_storage
import numpy as np
from .context_modelling_cache import CacheV2ContextModelling
from .context_modelling import process_single_image_groups
from time import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision.ops import roi_align


class ViLDContextModelling(CacheV2ContextModelling):
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
        # add start and end token mask
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask

    def kd_clip_contrast(self,
                         group_info,
                         predictions, clip_images,
                         clip_model,
                         image_info=None):
        image_ids = [im['image_id'] for im in image_info]
        pseudo_words = predictions.pop('kd_pseudo_words')
        device = pseudo_words.device
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[0]))
        # Note: perms = seq
        normed_boxes, spanned_boxes, origin_split, group_split, preds_split_by_perms,\
            seqs_split_split_by_origin, seqs_split_by_group = \
            multi_apply(process_single_image_groups, group_info, device=device)
        clip_model.eval()

        normed_boxes_split_by_perms = [normed_boxes_.split(preds_split_by_perms_, dim=0)
                                       for normed_boxes_, preds_split_by_perms_
                                       in zip(normed_boxes, preds_split_by_perms)]
        preds_split_by_perms = [p for b in preds_split_by_perms for p in b]

        tok = time()
        with autocast():
            # TODO: get local image tokens
            valid_mask = self._drop_word(pseudo_words.half())
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text_tensor(
                pseudo_words.half(), valid_mask)  # add start and stop token
            clip_word_features = \
                clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                              text_pe=True, normalize=True).float()
            clip_patch_features = self._bbox_clip_image(spanned_boxes, clip_images,
                                                        seqs_split_by_group,
                                                        normed_boxes_split_by_perms,
                                                        clip_model)
        tik = time()
        storage.put_scalar("contrast_learning_time/clip_model_forward",
                           np.float32(tik-tok))
        tik = time()
        preds_split_by_batch = [n.shape[0] for n in normed_boxes]
        img_ids = [torch.tensor(b * [img_id])
                   for b, img_id in zip(preds_split_by_batch,
                                        image_ids)]
        img_ids = torch.cat(img_ids).to(device)
        tok = time()
        storage.put_scalar("contrast_learning_time/prepare_dense_features",
                           np.float32(tok - tik))

        start_id = 0
        box_ids = []
        for g in group_info:
            for ori in g['box_ids']:
                box_ids_per_ori = [torch.tensor(perm, dtype=torch.float32)
                                   for perm in ori]  # avoid overflow
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
        losses = dict(vild_loss=loss * self.cfg.TOKEN_LOSS_WEIGHT)

        queues_update = dict(clip_word_features=torch.cat([clip_word_features,
                                                           img_ids.view(-1, 1)], dim=-1).detach(),
                             clip_patch_features=torch.cat([clip_patch_features,
                                                            img_ids.view(-1, 1)], dim=-1).detach())
        return losses, queues_update

    @staticmethod
    def denormalize_boxes(spanned_box, normalized_boxes):
        spanned_wh = spanned_box[2:] - spanned_box[:2]
        normalized_boxes = normalized_boxes.view(-1, 2)
        boxes = normalized_boxes * spanned_wh[None]
        boxes = spanned_box[None, :2] + boxes
        return boxes.view(-1, 4)

    @torch.no_grad()
    def _bbox_clip_image(self, spanned_boxes, clip_images,
                         seqs_split_by_group,
                         normed_boxes_split_by_perms,
                         clip_model):
        denormed_boxes_list = []
        for spanned_boxes_per_image, repeat_num_per_image, normed_boxes_per_image \
                in zip(spanned_boxes, seqs_split_by_group, normed_boxes_split_by_perms):
            repeated_spanned_boxes = []
            for num, spanned_box in zip(repeat_num_per_image, spanned_boxes_per_image):
                repeated_spanned_boxes.extend([spanned_box] * num)
            denormed_boxes = list(map(self.denormalize_boxes, repeated_spanned_boxes,
                                      normed_boxes_per_image))
            denormed_boxes_list.append(torch.cat(denormed_boxes))

        clip_input_size = self.cfg.INPUT_RESOLUTION
        input_to_clip = roi_align(
            clip_images.tensor, denormed_boxes_list, (clip_input_size, clip_input_size), 1.0, 2, True)
        clip_img_features = clip_model.encode_image(
            input_to_clip, normalize=True, return_image_tokens=False)

        return clip_img_features.float()
