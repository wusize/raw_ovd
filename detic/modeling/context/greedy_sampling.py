import random
import torch
from detic.modeling.context.baseline import get_enclosing_box
import math
from detic.modeling.context.context_modelling import pseudo_permutations
from detic.modeling.context.baseline import get_normed_boxes
from .utils import multi_apply


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def max_giou_with_single_group(boxes, grouped_boxes):
    gious = bbox_overlaps(boxes, grouped_boxes, mode='giou')
    return gious.max(-1).values


def greedy_grouping(boxes, giou_thr):
    groups = []
    # random initialization
    while len(boxes) > 0:
        if len(groups) == 0:
            center_id = random.choice(range(len(boxes)))
        else:  # obtain center box
            max_gious = torch.cat([max_giou_with_single_group(boxes, g) for g in groups],
                                  dim=-1).max(-1).values
            center_id = max_gious.argmin().item()
        center_box = boxes[center_id:center_id+1]
        boxes = boxes[list(range(center_id)) + list(range(center_id + 1, len(boxes)))]  # squeeze out center box
        giou_with_boxes = bbox_overlaps(boxes, center_box, mode='giou')[:, 0]
        sampled_boxes = boxes[giou_with_boxes > giou_thr]
        grouped_boxes = torch.cat([center_box, sampled_boxes], dim=0)
        groups.append(grouped_boxes)
        boxes = boxes[giou_with_boxes <= giou_thr]

    return groups


def sparse_dense_grouping(center_boxes, dense_boxes, giou_thr, max_num, num_perms):
    return multi_apply(_sparse_dense_grouping_single, center_boxes,
                       dense_boxes=dense_boxes, giou_thr=giou_thr, max_num=max_num,
                       num_perms=num_perms
                       )


def _sparse_dense_grouping_single(center_box, dense_boxes, giou_thr, max_num, num_perms):
    gious = bbox_overlaps(center_box[None], dense_boxes, mode='giou')
    valid_candidates = torch.where(gious >= giou_thr)[0].tolist()
    if len(valid_candidates) > max_num:
        valid_candidates = random.sample(valid_candidates, k=max_num)

    sampled_boxes = dense_boxes[valid_candidates]
    grouped_boxes = torch.cat([center_box[None], sampled_boxes], dim=0)
    spanned_box = get_enclosing_box(grouped_boxes)
    normed_boxes = get_normed_boxes(grouped_boxes, spanned_box)

    num_regions = len(grouped_boxes)

    perms = pseudo_permutations(num_regions, min(math.factorial(num_regions), num_perms))
    permed_normed_boxes = [normed_boxes[p] for p in perms]

    return torch.cat([grouped_boxes[p] for p in perms]), [spanned_box, ], [permed_normed_boxes, ], perms


if __name__ == '__main__':
    boxes = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2], [1, 1, 2, 2],
                          [3, 2, 4, 3], [4, 2, 5, 3]])

    groups = greedy_grouping(boxes, giou_thr=-0.5)
