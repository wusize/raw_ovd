import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops.boxes import box_iou, box_area
from detectron2.utils.events import get_event_storage
import numpy as np
from time import time
from torch.cuda.amp import autocast
from .utils import multi_apply, bbox_xyxy_to_cxcywh
from .context_modelling import ContextModelling, process_single_image_groups
from detic.data.datasets.coco_zeroshot import categories_unseen

novel_cat_ids = [cat['id'] for cat in categories_unseen]


def denormalize_boxes(normed_boxes, spanned_box):
    normed_boxes = normed_boxes.view(-1, 2, 2)
    xy0 = spanned_box[:2]
    delta_xy = normed_boxes * (spanned_box[2:] - xy0).view(1, 1, 2)

    boxes = xy0.view(1, 1, 2) + delta_xy

    return boxes.view(-1, 4)


class ContextModellingV3(ContextModelling):
    def analyze_single_image_samples(self, image_info, group_info):
        reference_gt = image_info['reference_gt']
        original_size = image_info['original_size']
        target_size = reference_gt.image_size
        scale2original_area = (original_size[0] * original_size[1]) / (target_size[0] * target_size[1])

        normed_boxes = [group for ori in group_info['normed_boxes'] for group in ori]
        num_permutations_per_group = [len(g) for g in normed_boxes]
        spanned_boxes = [group for ori in group_info['spanned_boxes'] for group in ori]

        boxes_without_permutation = [denormalize_boxes(nb[0], sb)
                                     for nb, sb in zip(normed_boxes, spanned_boxes)]
        areas_without_permutation = torch.stack([box_area(group)[0] * scale2original_area
                                                 for group in boxes_without_permutation], dim=0)

        box_size_type = torch.ones_like(areas_without_permutation).int()
        box_size_type[areas_without_permutation > 96 ** 2] = 2     # large
        box_size_type[areas_without_permutation < 32 ** 2] = 0  # large

        has_base, has_novel, is_background = self.match_with_gts(boxes_without_permutation, reference_gt)

        return has_base, has_novel, is_background, box_size_type, num_permutations_per_group

    def match_with_gts(self, boxes_without_permutation, reference_gt):
        num_boxes_per_group = [g.shape[0] for g in boxes_without_permutation]
        boxes = torch.cat(boxes_without_permutation).to(reference_gt.proposal_boxes.device)
        iou_with_gts = box_iou(boxes, reference_gt.proposal_boxes.tensor)
        ious, matched_gt_idx = iou_with_gts.max(-1)
        is_matched2novel = reference_gt.is_novel[matched_gt_idx]

        # assign matching results
        is_background = ious < self.cfg.MATCH_IOU_THR
        is_novel = torch.logical_and(is_background.logical_not(),
                                     is_matched2novel > 0)
        is_base = torch.logical_and(is_background.logical_not(),
                                    is_matched2novel < 1)

        has_novel = is_novel.split(num_boxes_per_group, dim=0)
        has_novel = torch.stack([h.sum() > 0 for h in has_novel], dim=0)
        has_base = is_base.split(num_boxes_per_group, dim=0)
        has_base = torch.stack([h.sum() > 0 for h in has_base], dim=0)

        is_background = torch.logical_and(has_novel.logical_not(),
                                          has_base.logical_not())

        return has_base, has_novel, is_background

    @torch.no_grad()
    def debug(self, image_info, group_info, similarity_matrix_0, similarity_matrix_1):
        has_base, has_novel, is_background, box_size_type, num_permutations_per_group = \
            multi_apply(self.analyze_single_image_samples, list(image_info.values()), group_info)
        has_base = torch.cat(has_base)
        has_novel = torch.cat(has_novel)
        is_background = torch.cat(is_background)
        box_size_type = torch.cat(box_size_type)
        num_permutations_per_group = [num for img in num_permutations_per_group for num in img]
        num_queries = sum(num_permutations_per_group)
        assert num_queries == similarity_matrix_0.shape[0]
        assert num_queries == similarity_matrix_1.shape[0]
        similarity_matrix_0 = similarity_matrix_0[:, :num_queries]
        label = torch.arange(num_queries).to(has_base.device)

        loss = 0.5 * F.cross_entropy(similarity_matrix_0, label, reduction='none') \
               + 0.5 * F.cross_entropy(similarity_matrix_1, label, reduction='none')

        loss_split_by_perms = loss.split(num_permutations_per_group, dim=0)
        loss = torch.stack([l.mean() for l in loss_split_by_perms], dim=0)

        novel_ave_loss = loss[has_novel].mean().cpu().numpy() if has_novel.sum() > 0 else np.float(0.0)
        base_ave_loss = loss[has_base].mean().cpu().numpy() if has_base.sum() > 0 else np.float(0.0)
        background_ave_loss = loss[is_background].mean().cpu().numpy() if is_background.sum() > 0 else np.float(0.0)

        storage = get_event_storage()
        storage.put_scalar('loss_statistics/box_type/novel_ave_loss', novel_ave_loss)
        storage.put_scalar('loss_statistics/box_type/base_ave_loss', base_ave_loss)
        storage.put_scalar('loss_statistics/box_type/background_ave_loss', background_ave_loss)

        # storage.put_histogram('loss_statistics/histogram', loss.cpu())

        num_novel = has_novel.sum().cpu().numpy()
        num_base = has_base.sum().cpu().numpy()
        num_background = is_background.sum().cpu().numpy()


        storage.put_scalar('num_boxes_statistics/box_type/num_novel', num_novel)
        storage.put_scalar('num_boxes_statistics/box_type/num_base', num_base)
        storage.put_scalar('num_boxes_statistics/box_type/num_background', num_background)

        is_small = box_size_type == 0
        is_medium = box_size_type == 1
        is_large = box_size_type == 2

        small_ave_loss = loss[is_small].mean().cpu().numpy() if is_small.sum() > 0 else np.float(0.0)
        medium_ave_loss = loss[is_medium].mean().cpu().numpy() if is_medium.sum() > 0 else np.float(0.0)
        large_ave_loss = loss[is_large].mean().cpu().numpy() if is_large.sum() > 0 else np.float(0.0)


        storage.put_scalar('loss_statistics/box_size/small_ave_loss', small_ave_loss)
        storage.put_scalar('loss_statistics/box_size/medium_ave_loss', medium_ave_loss)
        storage.put_scalar('loss_statistics/box_size/large_ave_loss', large_ave_loss)

        num_small = is_small.sum().cpu().numpy()
        num_medium = is_medium.sum().cpu().numpy()
        num_large = is_large.sum().cpu().numpy()

        storage.put_scalar('num_boxes_statistics/box_size/num_small', num_small)
        storage.put_scalar('num_boxes_statistics/box_size/num_medium', num_medium)
        storage.put_scalar('num_boxes_statistics/box_size/num_large', num_large)

    def kd_clip_contrast(self,
                         group_info,
                         predictions, clip_images,
                         clip_model,
                         image_info=None):
        pseudo_words = predictions.pop('kd_pseudo_words')
        device = pseudo_words.device
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[0]))
        # Note: perms = seq
        normed_boxes, spanned_boxes, origin_split, group_split, preds_split_by_perms,\
            seqs_split_split_by_origin, seqs_split_by_group = \
            multi_apply(process_single_image_groups, group_info, device=device)
        # positions = bbox_xyxy_to_cxcywh(torch.cat(normed_boxes, dim=0))
        # position_embeddings = self.positional_embed(positions)
        # pseudo_words = pseudo_words + position_embeddings
        word_masks = self._drop_word(pseudo_words)
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

        self.debug(image_info, group_info, similarity_matrix_0, similarity_matrix_1)

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
