import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .stochastic_sampling import StochasticSampling
from torchvision.ops import roi_align, nms, box_iou
from .utils import (multi_apply, get_normed_boxes,
                    bbox_xyxy_to_cxcywh, repeat_crops_and_get_att_mask,
                    scale)
from detectron2.structures import Instances, Boxes
from .queues import Queues
from detectron2.utils.events import get_event_storage
import numpy as np
from time import time
from torch.cuda.amp import autocast
from detic.modeling import clip as CLIP
from detectron2.structures.masks import PolygonMasks
from pycocotools.coco import COCO
from detic.data.datasets.coco_zeroshot import categories_unseen


def perm_generator(seq):
    seen = set()
    length = len(seq)
    while True:
        perm = tuple(random.sample(seq, length))
        if perm not in seen:
            seen.add(perm)
            yield perm


def pseudo_permutations(seq_length, num_permutation):
    rand_perms = perm_generator(list(range(seq_length)))
    return [list(next(rand_perms)) for _ in range(num_permutation)]


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
    if sum(seq_level_group_split) != sum(seq_level_origin_split):
        print(f'{seq_level_group_split}, {seq_level_origin_split}', flush=True)

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
    # TODO: record base_novel_bg proportions
    def __init__(self, cfg, num_words, word_embed_dim, word_dropout, sigmoid=True):
        super(ContextModelling, self).__init__()
        self.num_words_per_pred = num_words
        self.word_embed_dim = word_embed_dim
        self.word_dropout = word_dropout
        self.cfg = cfg


        # TODO add gts
        gt_path = cfg.ALL_GTS
        gt_coco = COCO(gt_path)
        images = {}
        # id_map = {v: i for i, v in enumerate(sorted(list(gt_coco.cats.keys())))}
        if cfg.DATASET == 'COCO':
            unseen_cat_ids = [cat['id'] for cat in categories_unseen]
        else:
            raise NotImplementedError
        all_gt_is_unseen = []
        for img_id, anns in gt_coco.imgToAnns.items():
            gt_boxes = torch.tensor([ann['bbox'] for ann in anns])
            gt_boxes[:, 2:] = gt_boxes[:, 2:] + gt_boxes[:, :2]
            # gt_classes = torch.tensor([id_map[ann['category_id']] for ann in anns]).long()
            gt_is_unseen = torch.tensor([1 if ann['category_id'] in unseen_cat_ids else 0
                                         for ann in anns]).float()
            all_gt_is_unseen.append(gt_is_unseen)

            img_info = gt_coco.imgs[img_id]
            image_size = (img_info['height'], img_info['width'])

            images[img_id] = dict(gt_boxes=gt_boxes, gt_is_unseen=gt_is_unseen,
                                  image_size=image_size)
        unseen_ratio = torch.cat(all_gt_is_unseen).mean()
        print(f'Ratio of novel boxes: {unseen_ratio}', flush=True)

        self.images = images

        checkboard_cfg = cfg.CHECKBOARD
        if sigmoid:
            self.get_objectness = torch.sigmoid
        else:
            self.get_objectness = identity_func
        if checkboard_cfg.ENABLE:
            self.checkboard_sampling = StochasticSampling(
                max_groups=checkboard_cfg.MAX_GROUPS,
                max_permutations=checkboard_cfg.MAX_PERMUTATIONS,
                alpha=checkboard_cfg.ALPHA,
                cut_off_thr=checkboard_cfg.CUT_OFF_THR,
                interval=checkboard_cfg.INTERVAL,
                base_probability=checkboard_cfg.BASE_PROBABILITY
            )
        self.checkboard_cfg = checkboard_cfg
        self.caption_cfg = cfg.CAPTION
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

    # preprocess topk proposals
    def preprocess_proposals(self, proposals, shape_ratio_thr, area_ratio_thr, objectness_thr, nms_thr):
        image_area = proposals.image_size[0] * proposals.image_size[1]

        topk_proposal_boxes = proposals.proposal_boxes
        size_of_boxes = topk_proposal_boxes.tensor[..., 2:] - \
                        topk_proposal_boxes.tensor[..., :2]
        boxes_shape_ratio = size_of_boxes[..., 0] / (size_of_boxes[..., 1] + 1e-12)

        assert shape_ratio_thr < 1.0

        valid_shape_ratio = torch.logical_and(shape_ratio_thr < boxes_shape_ratio,
                                              boxes_shape_ratio < (1.0 / shape_ratio_thr))
        valid_area = topk_proposal_boxes.area() > (area_ratio_thr * image_area)
        valid_object_score = self.get_objectness(proposals.objectness_logits) > objectness_thr
        valid_shape = torch.logical_and(valid_shape_ratio, valid_area)

        all_valid = torch.logical_and(valid_shape, valid_object_score)
        if all_valid.sum() < 1:
            all_valid[proposals.objectness_logits.argmax()] = True

        proposals = proposals[all_valid]

        nms_kept = nms(proposals.proposal_boxes.tensor,
                       scores=proposals.objectness_logits,
                       iou_threshold=nms_thr)
        nmsed_proposals = proposals[nms_kept]

        return nmsed_proposals

    def _checkboard_sampling(self, topk_proposals, mask_on=False):
        if not self.checkboard_cfg.ENABLE:
            return topk_proposals[:0], None
        device = topk_proposals.proposal_boxes.device
        if len(topk_proposals) == 0:
            h, w = topk_proposals.image_size
            image_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)
            topk_proposals = Instances(image_size=topk_proposals.image_size,
                                       proposal_boxes=Boxes(image_box.view(-1, 4)),
                                       objectness_logits=-torch.ones(1, device=device),
                                       gt_classes=-torch.ones(1, device=device, dtype=torch.int64),
                                       gt_boxes=Boxes(torch.zeros(1, 4, device=device)),
                                       sample_types=-torch.ones(1, device=device).int())
            if mask_on:
                topk_proposals.set('gt_masks',
                                   PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]), ], ]))
        nmsed_proposals = self.preprocess_proposals(topk_proposals,
                                               self.cfg.SHAPE_RATIO_THR,
                                               self.checkboard_cfg.AREA_RATIO_THR,
                                               self.cfg.OBJECTNESS_THR,
                                               self.checkboard_cfg.NMS_THR)
        nmsed_proposals.sample_types[:] = 1    # clip_kd_samples: 1
        func = self.checkboard_sampling.sample
        boxes = nmsed_proposals.proposal_boxes.tensor.tolist()
        groups_per_proposal, normed_boxes, spanned_boxes, box_ids = \
            multi_apply(func, boxes,
                        [nmsed_proposals.image_size] * len(nmsed_proposals))
        new_boxes = torch.cat([c for p in groups_per_proposal
                               for g in p for c in g], dim=0).to(device)
        num_added = len(new_boxes)
        added_instances = Instances(image_size=nmsed_proposals.image_size,
                                    proposal_boxes=Boxes(new_boxes),
                                    objectness_logits=-torch.ones(num_added, device=device),
                                    gt_classes=-torch.ones(num_added, device=device,
                                                           dtype=torch.int64),
                                    gt_boxes=Boxes(torch.zeros(num_added, 4, device=device)),
                                    sample_types=torch.ones(num_added, device=device).int())   # clip_kd: 1
        if mask_on:
            added_instances.set('gt_masks',
                                PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])]] * num_added))

        return added_instances, dict(normed_boxes=normed_boxes,
                                     spanned_boxes=spanned_boxes,
                                     sampled_instances=added_instances,
                                     box_ids=box_ids)

    def _caption_sampling(self, topk_proposals, mask_on=False):
        if not self.caption_cfg.ENABLE:
            return topk_proposals[:0], None
        if len(topk_proposals) == 0:
            h, w = topk_proposals.image_size
            device = topk_proposals.gt_classes.device
            spanned_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)
            topk_proposals = Instances(image_size=topk_proposals.image_size,
                                       proposal_boxes=Boxes(spanned_box.view(-1, 4)),
                                       objectness_logits=-torch.ones(1, device=device),
                                       gt_classes=-torch.ones(1, device=device, dtype=torch.int64),
                                       gt_boxes=Boxes(torch.zeros(1, 4, device=device)),
                                       sample_types=-torch.ones(1, device=device).int())
            if mask_on:
                topk_proposals.set('gt_masks',
                                   PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]), ], ]))
        nmsed_proposals = self.preprocess_proposals(topk_proposals,
                                               self.cfg.SHAPE_RATIO_THR,
                                               self.caption_cfg.AREA_RATIO_THR,
                                               self.cfg.OBJECTNESS_THR,
                                               self.caption_cfg.NMS_THR)
        nmsed_proposals.sample_types[:] = 2     # caption_samples: 2
        num_proposals = len(nmsed_proposals)
        sampled_ids = list(range(num_proposals)) \
            if num_proposals < self.caption_cfg.MAX_NUM \
            else random.sample(list(range(num_proposals)),
                               k=self.caption_cfg.MAX_NUM)
        kept_proposals = nmsed_proposals[sampled_ids]
        h, w = kept_proposals.image_size
        device = kept_proposals.gt_classes.device
        spanned_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)
        normed_boxes = get_normed_boxes(kept_proposals.proposal_boxes.tensor,
                                        spanned_box)
        if self.caption_cfg.ADD_IMAGE_BOX:
            added_instance = Instances(image_size=kept_proposals.image_size,
                                       proposal_boxes=Boxes(spanned_box.view(-1, 4)),
                                       objectness_logits=-torch.ones(1, device=device),
                                       gt_classes=-torch.ones(1, device=device, dtype=torch.int64),
                                       gt_boxes=Boxes(torch.zeros(1, 4, device=device)),
                                       sample_types=2 * torch.ones(1, device=device).int())
            if mask_on:
                added_instance.set('gt_masks',
                                   PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]), ], ]))
            kept_proposals = Instances.cat([kept_proposals, added_instance])

        return kept_proposals, normed_boxes

    def _sample_topk_proposals(self, proposals_per_image, mask_on=False):
        num = min(len(proposals_per_image), self.cfg.TOPK)
        topk_objectness_logits, topk_inds = proposals_per_image.objectness_logits.topk(num)
        num_added = len(topk_objectness_logits)
        topk_proposals = Instances(image_size=proposals_per_image.image_size,
                                   proposal_boxes=proposals_per_image.proposal_boxes[topk_inds],
                                   objectness_logits=topk_objectness_logits,
                                   gt_classes=-torch.ones_like(topk_inds),
                                   gt_boxes=Boxes(torch.zeros(num_added, 4,
                                                              device=topk_objectness_logits.device,
                                                              dtype=topk_objectness_logits.dtype)),
                                   sample_types=-torch.ones(num_added,
                                                            device=topk_objectness_logits.device).int())
        if mask_on:
            topk_proposals.set('gt_masks',
                               PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])]] * num_added))

        return topk_proposals

    def sample_topk_proposals(self, proposals_per_image, mask_on=False):
        return self._sample_topk_proposals(proposals_per_image, mask_on)

    # TODO: input topk proposals
    def sample(self, proposals_per_image, mask_on=False, image_id=None):
        if image_id is not None:
            gt_ious_scores = self.get_topk_average_scores(proposals_per_image, image_id)
        else:
            gt_ious_scores = None
        topk_proposals = self._sample_topk_proposals(proposals_per_image, mask_on)
        added_instances, info = self.sample_on_topk(topk_proposals, mask_on)
        info.update(gt_ious_scores=gt_ious_scores)

        return added_instances, info

    @torch.no_grad()
    def get_topk_average_scores(self, topk_proposals, image_id):
        proposal_boxes = topk_proposals.proposal_boxes.tensor
        image_size = topk_proposals.image_size
        image = self.images[image_id]
        device = proposal_boxes.device
        gt_boxes = image['gt_boxes'].to(device)
        ori_image_size = image['image_size']
        gt_is_unseen = image['gt_is_unseen'].to(device)

        proposal_boxes = scale(proposal_boxes,
                               ori_image_size[1] / image_size[1],
                               ori_image_size[0] / image_size[0])

        ious = box_iou(gt_boxes, proposal_boxes)
        mathed_preds = ious > 0.5
        proposal_scores = topk_proposals.objectness_logits.sigmoid()
        average_scores_per_gt = (mathed_preds.float() * proposal_scores).sum(-1) / (mathed_preds.sum(-1) + 1e-12)
        max_ious_per_gt = ious.max(-1).values

        return dict(base_scores=average_scores_per_gt[gt_is_unseen < 1.0].cpu().numpy(),
                    novel_scores=average_scores_per_gt[gt_is_unseen > 0.0].cpu().numpy(),
                    base_ious=max_ious_per_gt[gt_is_unseen < 1.0].cpu().numpy(),
                    novel_ious=max_ious_per_gt[gt_is_unseen > 0.0].cpu().numpy(),
                    )

    def sample_on_topk(self, topk_proposals, mask_on=False):
        checkborad_instances, checkborad_group_info = self._checkboard_sampling(topk_proposals, mask_on)
        caption_instances, caption_normed_boxes = self._caption_sampling(topk_proposals, mask_on)

        return Instances.cat([checkborad_instances, caption_instances]), \
               dict(checkborad_group_info=checkborad_group_info,
                    caption_normed_boxes=caption_normed_boxes)

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
    def _bbox_clip_image(self, spanned_boxes, clip_images,
                         seqs_split_by_group,
                         normed_boxes_split_by_perms,
                         clip_model):
        # TODO: repeat and mask
        num_groups_per_image = [b.shape[0] for b in spanned_boxes]
        clip_input_size = self.cfg.INPUT_RESOLUTION
        input_to_clip = roi_align(
            clip_images.tensor, spanned_boxes, (clip_input_size, clip_input_size), 1.0, 2, True)
        input_to_clip = input_to_clip.split(num_groups_per_image, dim=0)
        storage = get_event_storage()
        tik = time()
        repeated_crops, attn_masks = multi_apply(repeat_crops_and_get_att_mask,
                                                 input_to_clip, seqs_split_by_group,
                                                 normed_boxes_split_by_perms,
                                                 num_heads=clip_model.visual.num_heads,
                                                 grid_size=clip_input_size // 32)
        storage.put_scalar("contrast_learning_time/generate_attn_mask",
                           np.float32(time() - tik))
        repeated_crops = torch.cat(repeated_crops, dim=0)
        attn_masks = torch.cat(attn_masks, dim=0)
        clip_img_features, clip_img_tokens = clip_model.encode_image(
            repeated_crops, normalize=True, return_image_tokens=True, attn_masks=attn_masks)

        return clip_img_features.float(), clip_img_tokens.float()

    def record_class_proportions(self, sampled_instances, image_ids, name):
        num_bg = 0
        num_novel = 0
        num_base = 0
        cnt = 0
        thr = 0.2
        for img_id, inst in zip(image_ids, sampled_instances):
            if img_id not in self.images:
                continue
            image_size = inst.image_size
            gt = self.images[img_id]
            gt_image_size = gt['image_size']
            sampled_boxes = inst.proposal_boxes.tensor
            sampled_boxes = scale(sampled_boxes,
                                  gt_image_size[1] / image_size[1],
                                  gt_image_size[0] / image_size[0])
            device = sampled_boxes.device
            gt_boxes = gt['gt_boxes'].to(device)
            gt_is_unseen = gt['gt_is_unseen'].to(device)
            ious = box_iou(sampled_boxes, gt_boxes)
            ious, matched_gts = ious.max(1)
            is_unseen = gt_is_unseen[matched_gts]
            is_unseen = is_unseen[ious >= thr]
            num_bg += (ious < thr).sum().item()
            num_novel += (is_unseen > 0.0).sum().item()
            num_base += (is_unseen < 1.0).sum().item()
            cnt += len(sampled_boxes)

        cnt += 1e-12
        storage = get_event_storage()
        storage.put_scalar(f"{name}/background", np.float32(num_bg / cnt))
        storage.put_scalar(f"{name}/novel", np.float32(num_novel / cnt))
        storage.put_scalar(f"{name}/base", np.float32(num_base / cnt))

    @staticmethod
    def _record_gradient(grad):
        val = grad.norm()
        storage = get_event_storage()
        storage.put_scalar("gradients/contrastive", val.cpu().numpy())

    def kd_clip_contrast(self,
                         group_info,
                         predictions, clip_images,
                         clip_model,
                         image_info=None):
        self.record_class_proportions([g['sampled_instances'] for g in group_info],
                                      list(image_info.keys()), 'sampled_proportions')
        pseudo_words = predictions.pop('kd_pseudo_words')
        pseudo_words.register_hook(self._record_gradient)
        device = pseudo_words.device
        storage = get_event_storage()
        storage.put_scalar("num_proposals/contrast_proposals", np.float32(pseudo_words.shape[0]))
        # Note: perms = seq
        normed_boxes, spanned_boxes, origin_split, group_split, preds_split_by_perms,\
            seqs_split_split_by_origin, seqs_split_by_group = \
            multi_apply(process_single_image_groups, group_info, device=device)
        positions = bbox_xyxy_to_cxcywh(torch.cat(normed_boxes, dim=0))
        position_embeddings = self.positional_embed(positions)
        pseudo_words = pseudo_words + position_embeddings
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
        img_ids = [torch.tensor(len(b) * [img_id])
                   for b, img_id in zip(preds_split_by_perms,
                                        image_info.keys())]
        img_ids = torch.cat(img_ids).to(device).float()
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

    def caption_contrast(self, caption_normed_boxes, predictions, clip_model, image_info):
        clip_model.eval()
        batch_size = len(caption_normed_boxes)
        caption_pseudo_words = predictions.pop('caption_pseudo_words')
        device = caption_pseudo_words.device
        all_clip_caption_features, num_captions_per_image = self.get_caption_features([v['captions']
                                                                                       for v in image_info.values()],
                                                                                      device,
                                                                                      clip_model)

        num_boxes_per_image = [b.shape[0] for b in caption_normed_boxes]
        positions = bbox_xyxy_to_cxcywh(torch.cat(caption_normed_boxes, dim=0))
        position_embeddings = self.positional_embed(positions)
        permutations_per_image = [pseudo_permutations(n, min(math.factorial(n),
                                                             self.caption_cfg.MAX_PERMUTATIONS))
                                  for n in num_boxes_per_image]
        if self.caption_cfg.ADD_IMAGE_BOX:
            position_embeddings = position_embeddings.split(num_boxes_per_image, dim=0)
            zero_pes = [torch.zeros_like(position_embeddings[0][:1]) for _ in caption_normed_boxes]
            position_embeddings = torch.cat([torch.cat([pe, ze], dim=0)
                                             for pe, ze in zip(position_embeddings, zero_pes)], dim=0)
            num_boxes_per_image = [n+1 for n in num_boxes_per_image]
            permutations_per_image = [[perm + [len(perm)] for perm in b] for b in permutations_per_image]
        num_perms_per_image = [len(b) for b in permutations_per_image]
        caption_pseudo_words = (caption_pseudo_words + position_embeddings).split(num_boxes_per_image, dim=0)
        caption_pseudo_words = [[ws[perm] for perm in perms]
                                for ws, perms in zip(caption_pseudo_words, permutations_per_image)]
        caption_pseudo_sequences = [perm for b in caption_pseudo_words for perm in b]
        words_split = [perm.shape[0] for b in caption_pseudo_words for perm in b]
        words_mask = self._drop_word(torch.cat(caption_pseudo_sequences, dim=0)).split(words_split,
                                                                                       dim=0)
        caption_pseudo_sequences = [seq.flatten(0, 1)[wm.view(-1)]
                                    for seq, wm in zip(caption_pseudo_sequences,  words_mask)]
        context_length = max([seq.shape[0] for seq in caption_pseudo_sequences])
        with autocast():
            # TODO: get local image tokens
            pseudo_text, end_token_ids = clip_model.prepare_pseudo_text(
                caption_pseudo_sequences,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features = \
                clip_model.encode_pseudo_text(pseudo_text, end_token_ids,
                                              text_pe=True, normalize=True,
                                              return_word_tokens=False)
            clip_text_features = clip_text_features.float()

        if all_clip_caption_features is None:
            caption_valid = torch.zeros(batch_size, device=device)
            clip_caption_features = torch.zeros(batch_size, 512, device=device)
            caption_img_ids = torch.tensor(list(image_info.keys()), device=device,
                                           dtype=torch.float32)
        else:
            caption_valid = []
            clip_caption_features = all_clip_caption_features.split(num_captions_per_image, dim=0)
            clip_caption_features_list = []
            caption_img_ids = []
            max_caps = self.caption_cfg.CAPS_PER_IMG
            for img_id, num_cap, cap_feat in zip(image_info.keys(),
                                                 num_captions_per_image, clip_caption_features):
                assert num_cap == cap_feat.shape[0]
                if num_cap > 0:
                    num_samples = min(max_caps, num_cap)
                    caption_valid.append(torch.ones(num_samples, device=device))
                    sampled_ids = random.sample(range(num_cap), num_samples)
                    clip_caption_features_list.append(cap_feat[sampled_ids])
                    caption_img_ids.append(img_id * torch.ones(num_samples, device=device))
                else:
                    clip_caption_features_list.append(torch.zeros(1, 512, device=device))
                    caption_valid.append(torch.zeros(1, device=device))
                    caption_img_ids.append(img_id * torch.ones(1, device=device))
            caption_valid = torch.cat(caption_valid)
            clip_caption_features = torch.cat(clip_caption_features_list)
            caption_img_ids = torch.cat(caption_img_ids)
        pred_image_ids = torch.tensor([k for num_perms, k in zip(num_perms_per_image,
                                                                 image_info.keys()) for _ in range(num_perms)],
                                      device=device)
        num_preds = clip_text_features.shape[0]
        assert sum(num_perms_per_image) == num_preds
        assert pred_image_ids.shape[0] == num_preds

        # pred_text as queries
        global_clip_caption_features = self.queues.get_queue('clip_caption_features')
        keys_caption = torch.cat([clip_caption_features, global_clip_caption_features[..., :-1]])
        similarity_matrix_0 = self.bce_temp * clip_text_features @ keys_caption.T + self.bce_bias
        key_img_ids = torch.cat([caption_img_ids, global_clip_caption_features[..., -1]])
        label_matrix = (pred_image_ids[:, None] == key_img_ids[None]).float()

        loss_weights = torch.ones_like(similarity_matrix_0)
        loss_weights[label_matrix > 0.0] = self.cfg.BCE_POS_WEIGHT    # pos weight
        loss_weights[:, :clip_caption_features.shape[0]][:, caption_valid < 1.0] = 0.0

        loss = F.binary_cross_entropy_with_logits(similarity_matrix_0, label_matrix, reduction='none')
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-12)

        if all_clip_caption_features is None:
            clip_caption_features_update = -torch.ones(1, 512 + 1, device=device)
        else:
            all_cap_image_ids = [img_id for img_id, num_cap in zip(image_info.keys(), num_captions_per_image)
                                 for _ in range(num_cap)]
            all_cap_image_ids = torch.tensor(all_cap_image_ids,
                                             device=device, dtype=torch.float32).view(-1, 1)
            clip_caption_features_update = torch.cat([all_clip_caption_features,
                                                      all_cap_image_ids], dim=-1)

        queue_update = dict(clip_caption_features=clip_caption_features_update)

        return dict(caption_loss=loss * self.cfg.CAPTION_LOSS_WEIGHT), queue_update

    def get_loss(self, group_infos, predictions, clip_images, clip_model, image_info):
        losses = dict()
        queue_update = dict()
        if self.checkboard_cfg.ENABLE:
            loss_kd, queue_kd = self.kd_clip_contrast([g['checkborad_group_info'] for g in group_infos],
                                                      predictions, clip_images,
                                                      clip_model, image_info)
            losses.update(loss_kd)
            queue_update.update(queue_kd)

        if self.caption_cfg.ENABLE:
            loss_caption, queue_caption = self.caption_contrast([g['caption_normed_boxes'] for g in group_infos],
                                                                predictions, clip_model, image_info)
            losses.update(loss_caption)
            queue_update.update(queue_caption)

        self.queues.dequeue_and_enqueue(queue_update)

        return losses

    def kd_jump_over_error(self, pseudo_words, losses, queues_update):
        device = pseudo_words.device
        for k in ['clip_text_features', 'clip_image_features']:
            if k not in queues_update:
                queues_update[k] = -torch.ones(1, 513).to(device)
        if 'contrast_loss' in losses:
            losses['contrast_loss'] = losses['contrast_loss'] * 0.0
        else:
            losses['contrast_loss'] = pseudo_words[0, 0, 0] * 0.0

        if self.checkboard_cfg.LOCAL_CORRESPONDENCE:
            for k in ['clip_word_features', 'clip_patch_features']:
                if k not in queues_update:
                    queues_update[k] = -torch.ones(1, 513).to(device)
            if 'token_loss' in losses:
                losses['token_loss'] = losses['token_loss'] * 0.0
            else:
                losses['token_loss'] = pseudo_words[0, 0, 0] * 0.0

        return losses, queues_update
