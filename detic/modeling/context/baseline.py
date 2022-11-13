import random
from detic.modeling.context.context_modelling import pseudo_permutations
from detic.modeling.context.context_modelling import ContextModelling
from detic.modeling.context.stochastic_sampling import StochasticSampling
from detic.modeling.context.context_modelling_cache import CacheV2ContextModelling
import torch
from detectron2.structures import Instances, PolygonMasks, Boxes
import math
import numpy as np
from detectron2.utils.events import get_event_storage


class GridSampling(StochasticSampling):
    def sample(self,  box, image_size):
        img_h, img_w = image_size
        x1, y1, x2, y2 = img_w / 3.0, img_h / 3.0, img_w / 1.5, img_h / 1.5
        box = [x1, y1, x2, y2]
        return super(GridSampling, self).sample(box, image_size)


class OVRCNNContextModelling(ContextModelling):
    def __init__(self, cfg, num_words, word_embed_dim, word_dropout, sigmoid=True):
        super(OVRCNNContextModelling, self).__init__(cfg, num_words, word_embed_dim, word_dropout, sigmoid)
        self.checkboard_sampling = GridSampling(
            max_groups=1,
            max_permutations=2,
            alpha=1.0,
            cut_off_thr=0.3,
            interval=0.0,
            base_probability=1.0
        )


class CaptionLikeContextModelling(CacheV2ContextModelling):

    def _checkboard_sampling(self, topk_proposals, mask_on=False, image_info=None):
        if not self.checkboard_cfg.ENABLE:
            return topk_proposals[:0], None
        device = topk_proposals.proposal_boxes.device
        h, w = topk_proposals.image_size
        image_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)
        if len(topk_proposals) == 0:
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
        if image_info is not None:
            storage = get_event_storage()
            iter = storage.iter
            # print(f'Device: {comm.get_rank()}, iter: {iter}', flush=True)
            if iter > self.cfg.START_CACHE:
                nmsed_proposals = self.boxes_cache.update(image_info, nmsed_proposals,
                                                          self.cfg.OBJECTNESS_THR)
        num_nmsed = len(nmsed_proposals)
        if num_nmsed > 9:
            kept_ids = random.sample(range(num_nmsed), k=9)
            nmsed_proposals = nmsed_proposals[kept_ids]
            num_nmsed = 9
        perms = pseudo_permutations(num_nmsed, min(math.factorial(num_nmsed),
                                                   self.checkboard_cfg.MAX_PERMUTATIONS))
        new_boxes = torch.cat([nmsed_proposals[p].proposal_boxes.tensor for p in perms],
                              dim=0).to(device)

        nmsed_proposals.proposal_boxes.scale(1/w, 1/h)
        spanned_boxes = [[image_box]]
        normed_boxes = [[[nmsed_proposals[p].proposal_boxes.tensor for p in perms], ], ]
        box_ids = [perms, ]

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
                                     box_ids=box_ids)


def get_enclosing_box(boxes):
    # Nx4
    x_y_min = boxes[:, :2].min(dim=0).values
    x_y_max = boxes[:, :2].max(dim=0).values

    return torch.cat([x_y_min, x_y_max])


def get_normed_boxes(boxes, spanned_box):
    spanned_box_shape = spanned_box[2:] - spanned_box[:2]
    boxes = boxes.view(-1, 2, 2) - spanned_box[:2].view(1, 1, 2)
    boxes = boxes / (spanned_box_shape.view(1, 1, 2) + 1e-12)

    return boxes.reshape(-1, 4)


class CaptionLikeV2ContextModelling(CaptionLikeContextModelling):
    def _checkboard_sampling(self, topk_proposals, mask_on=False, image_info=None):
        if not self.checkboard_cfg.ENABLE:
            return topk_proposals[:0], None
        device = topk_proposals.proposal_boxes.device
        h, w = topk_proposals.image_size
        image_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)
        if len(topk_proposals) == 0:
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
        if image_info is not None:
            storage = get_event_storage()
            iter = storage.iter
            # print(f'Device: {comm.get_rank()}, iter: {iter}', flush=True)
            if iter > self.cfg.START_CACHE:
                nmsed_proposals = self.boxes_cache.update(image_info, nmsed_proposals,
                                                          self.cfg.OBJECTNESS_THR)
        num_nmsed = len(nmsed_proposals)
        if num_nmsed > 9:
            kept_ids = random.sample(range(num_nmsed), k=9)
            nmsed_proposals = nmsed_proposals[kept_ids]
            num_nmsed = 9
        perms = pseudo_permutations(num_nmsed, min(math.factorial(num_nmsed),
                                                   self.checkboard_cfg.MAX_PERMUTATIONS))
        new_boxes = torch.cat([nmsed_proposals[p].proposal_boxes.tensor for p in perms],
                              dim=0).to(device)

        max_box = get_enclosing_box(new_boxes)
        normed_nmsed_boxes = get_normed_boxes(nmsed_proposals.proposal_boxes.tensor,
                                              max_box)

        spanned_boxes = [[max_box]]
        normed_boxes = [[[normed_nmsed_boxes[p] for p in perms], ], ]
        box_ids = [perms, ]

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
                                     box_ids=box_ids)
