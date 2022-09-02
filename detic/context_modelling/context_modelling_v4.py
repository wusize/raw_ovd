import torch
import numpy as np
from .utils import multi_apply
from .context_modelling import ContextModelling
from detectron2.structures import Instances, Boxes, PolygonMasks
from detectron2.modeling.poolers import assign_boxes_to_levels
from torchvision.ops import nms


class ContextModellingV4(ContextModelling):
    # TODO: multi-level sampling strategy
    # preprocess topk proposals
    @staticmethod
    def _deal_with_empty_proposals(empty_proposals, mask_on):
        h, w = empty_proposals.image_size
        device = empty_proposals.proposal_boxes.device

        image_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)
        topk_proposals = Instances(image_size=empty_proposals.image_size,
                                   proposal_boxes=Boxes(image_box.view(-1, 4)),
                                   objectness_logits=-torch.ones(1, device=device),
                                   gt_classes=-torch.ones(1, device=device, dtype=torch.int64),
                                   gt_boxes=Boxes(torch.zeros(1, 4, device=device)),
                                   sample_types=-torch.ones(1, device=device).int())
        if mask_on:
            topk_proposals.set('gt_masks',
                               PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]), ], ]))

        return topk_proposals

    def _checkboard_sampling(self, topk_proposals, mask_on=False):
        if not self.checkboard_cfg.ENABLE:
            return topk_proposals[:0], None
        device = topk_proposals.proposal_boxes.device
        if len(topk_proposals) == 0:
            topk_proposals = self._deal_with_empty_proposals(topk_proposals, mask_on)

        num_levels = len(self.checkboard_cfg.AREA_THR)

        level_assignment = assign_boxes_to_levels([topk_proposals.proposal_boxes],
                                                  0, num_levels - 1,
                                                  self.checkboard_cfg.AREA_THR[-2], num_levels - 2)
        # TODO: sample at each level
        multi_level_instances = []
        for i in range(num_levels):
            max_num = self.checkboard_cfg.TOPK[i]
            lvl_proposals = topk_proposals[level_assignment == i]
            if len(lvl_proposals) == 0:
                continue
            shape_ratio_thr = self.checkboard_cfg.SHAPE_RATIO_THR[i]
            area_ratio_thr = self.checkboard_cfg.AREA_RATIO_THR
            objectness_thr = self.checkboard_cfg.OBJECTNESS_THR[i]
            nms_thr = self.checkboard_cfg.NMS_THR[i]
            lvl_proposals = self.preprocess_proposals(lvl_proposals,
                                                      shape_ratio_thr,
                                                      area_ratio_thr,
                                                      objectness_thr,
                                                      nms_thr)
            if len(lvl_proposals) > max_num:
                _, topk_indices = lvl_proposals.objectness_logits.topk(max_num)
                lvl_proposals = lvl_proposals[topk_indices]
            lvl_proposals.sample_types[:] = 1    # clip_kd_samples: 1
            multi_level_instances.append(lvl_proposals)
        sampled_proposals = Instances.cat(multi_level_instances)
        nms_kept = nms(sampled_proposals.proposal_boxes.tensor,
                       scores=sampled_proposals.objectness_logits,
                       iou_threshold=max(self.checkboard_cfg.NMS_THR))
        sampled_proposals = sampled_proposals[nms_kept]
        func = self.checkboard_sampling.sample
        boxes = sampled_proposals.proposal_boxes.tensor.tolist()
        groups_per_proposal, normed_boxes, spanned_boxes, box_ids = \
            multi_apply(func, boxes,
                        [sampled_proposals.image_size] * len(sampled_proposals))
        new_boxes = torch.cat([c for p in groups_per_proposal
                               for g in p for c in g], dim=0).to(device)
        num_added = len(new_boxes)
        added_instances = Instances(image_size=sampled_proposals.image_size,
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
