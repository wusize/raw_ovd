import torch
from functools import partial
from six.moves import map, zip
from torchvision.ops import nms


def get_potional_indices(mask):
    # note: the mask has considered the start and end token

    # append a zero start
    mask = torch.cat([torch.zeros_like(mask[:, :1]),
                      mask], dim=-1)
    pe_indices = (mask > 0.0).cumsum(-1)[:, :-1]

    return pe_indices


def get_normed_boxes(boxes, spanned_box):
    spanned_box_shape = spanned_box[2:] - spanned_box[:2]
    boxes = boxes.view(-1, 2, 2) - spanned_box[:2].view(1, 1, 2)
    boxes = boxes / (spanned_box_shape.view(1, 1, 2) + 1e-12)

    return boxes.view(-1, 4)


# preprocess topk proposals
def preprocess_proposals(proposals, shape_ratio_thr, area_ratio_thr, objectness_thr, nms_thr):
    image_area = proposals.image_size[0] * proposals.image_size[1]

    topk_proposal_boxes = proposals.proposal_boxes
    size_of_boxes = topk_proposal_boxes.tensor[..., 2:] - \
                    topk_proposal_boxes.tensor[..., :2]
    boxes_shape_ratio = size_of_boxes[..., 0] / (size_of_boxes[..., 1] + 1e-12)

    assert shape_ratio_thr < 1.0

    valid_shape_ratio = torch.logical_and(shape_ratio_thr < boxes_shape_ratio,
                                          boxes_shape_ratio < (1.0 / shape_ratio_thr))
    valid_area = topk_proposal_boxes.area() > (area_ratio_thr * image_area)
    valid_object_score = proposals.objectness_logits.sigmoid() > objectness_thr
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


# repeat crops and get attention masks
def repeat_crops_and_get_att_mask(crops, repeat_nums, normed_boxes, num_heads, grid_size=7, use_attn_mask=True):
    repeated_crops = torch.cat([crop[None].repeat(repeat_num, 1, 1, 1)
                                for crop, repeat_num in zip(crops, repeat_nums)], dim=0)
    if use_attn_mask:
        boxes_split_by_seqs = [n.shape[0] for n in normed_boxes]
        normed_boxes = torch.cat(normed_boxes)
        masks_per_box = get_att_mask_by_matrix(normed_boxes, grid_size)
        masks_split_by_seqs = masks_per_box.split(boxes_split_by_seqs, dim=0)
        masks_split_by_seqs = [ms.sum(0) for ms in masks_split_by_seqs]
        masks_split_by_seqs = torch.stack(masks_split_by_seqs, dim=0)
        mask_flatten = masks_split_by_seqs.flatten(-2, -1)
        mask_flatten = torch.cat([torch.ones_like(mask_flatten[..., :1]),
                                  mask_flatten], dim=-1)
        attn_mask = mask_flatten[..., None] * mask_flatten[:, None, :]
        attn_mask = torch.where(attn_mask > 0.0, 0.0, float('-inf'))
        attn_mask[:, range(grid_size ** 2 + 1), range(grid_size ** 2 + 1)] = 0.0
        attn_mask = attn_mask[:, None].repeat(1, num_heads, 1, 1)
        attn_mask = attn_mask.flatten(0, 1)
    else:
        attn_mask = None

    return repeated_crops, attn_mask


def get_att_mask_by_matrix(normed_boxes, grid_size):
        boxes = normed_boxes * (grid_size - 1) + 0.5
        boxes = boxes.view(-1, 2, 2)
        num_boxes = boxes.shape[0]
        boxes[:, 0] = boxes[:, 0].floor()
        boxes[:, 1] = boxes[:, 1].ceil()
        boxes = boxes.long()
        x_range_pairs = boxes[..., 0]
        y_range_pairs = boxes[..., 1]
        x_mask, y_mask = single_direction_mask(torch.cat([x_range_pairs,
                                                                y_range_pairs], dim=0),
                                               grid_size).split([num_boxes] * 2, dim=0)
        mask = torch.logical_and(y_mask.view(num_boxes, grid_size, 1),
                                 x_mask.view(num_boxes, 1, grid_size))

        return mask


def single_direction_mask(range_pairs, grid_size):
        num_pairs = range_pairs.shape[0]
        device = range_pairs.device
        ref_matrix = torch.arange(
            grid_size).view(1, -1).repeat(num_pairs, 1).to(device)
        beg = range_pairs[:, 0:1].repeat(1, grid_size)
        end = range_pairs[:, 1:2].repeat(1, grid_size)
        mask = ref_matrix.ge(beg) & ref_matrix.lt(end)

        return mask


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)
