import torch
from .utils import multi_apply
from detectron2.structures import Instances, Boxes
import numpy as np
from detectron2.structures.masks import PolygonMasks
from .context_modelling import ContextModelling
from .queues import BoxesCache


class CacheContextModelling(ContextModelling):
    def __init__(self, *args, **kwargs):
        super(CacheContextModelling, self).__init__(*args, **kwargs)
        self.boxes_cache = BoxesCache(self.cfg.ANN_PATH, self.cfg.TOPK)

    # TODO: input topk proposals
    def sample(self, proposals_per_image, mask_on=False, image_info=None, **kwargs):
        topk_proposals = self._sample_topk_proposals(proposals_per_image, mask_on)
        return self.sample_on_topk(topk_proposals, mask_on, image_info)

    def sample_on_topk(self, topk_proposals, mask_on=False, image_info=None):
        checkborad_instances, checkborad_group_info = self._checkboard_sampling(topk_proposals, mask_on, image_info)
        caption_instances, caption_normed_boxes = self._caption_sampling(topk_proposals, mask_on)

        return Instances.cat([checkborad_instances, caption_instances]), \
               dict(checkborad_group_info=checkborad_group_info,
                    caption_normed_boxes=caption_normed_boxes)

    def _checkboard_sampling(self, topk_proposals, mask_on=False, image_info=None):
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

        if image_info is not None:
            nmsed_proposals = self.boxes_cache.update(image_info, nmsed_proposals, self.checkboard_cfg.NMS_THR,
                                                      self.cfg.OBJECTNESS_THR)
        # TODO: merge with cached samples by nms
        # name: "kd_proposals"

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
                                     box_ids=box_ids)
