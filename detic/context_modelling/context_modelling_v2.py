import torch
from detectron2.structures import Instances, Boxes
from pycocotools.coco import COCO
import numpy as np
from detectron2.modeling.proposal_generator.proposal_utils import \
    add_ground_truth_to_proposals_single_image
from detectron2.structures.masks import PolygonMasks
from .context_modelling import ContextModelling
from detic.data.datasets.coco_zeroshot import categories_unseen

novel_cat_ids = [cat['id'] for cat in categories_unseen]


class ContextModellingV2(ContextModelling):
    def __init__(self, cfg, num_words, word_embed_dim, word_dropout, sigmoid=True):
        super(ContextModellingV2, self).__init__(cfg, num_words, word_embed_dim, word_dropout, sigmoid)
        pr_coco = COCO(self.cfg.PROPOSALS)
        proposals = {}
        is_unseen = []
        for img_id, anns in pr_coco.imgToAnns.items():
            boxes = torch.tensor([ann['bbox'] for ann in anns])
            is_unseen.extend([1.0 if ann['category_id'] in novel_cat_ids else 0.0 for ann in anns])
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            img_info = pr_coco.imgs[img_id]
            image_size = (img_info['height'], img_info['width'])
            proposals[img_id] = Instances(image_size=image_size, gt_boxes=Boxes(boxes))
        print(f'Possible novel instances: {sum(is_unseen) / len(is_unseen)}', flush=True)

        self.proposals = proposals

    def _get_proposals(self, image_id, topk_proposals):
        tar_image_size = topk_proposals.image_size
        device = topk_proposals.proposal_boxes.device
        if image_id in self.proposals:
            gt_proposals = self.proposals[image_id]
            ori_image_size = gt_proposals.image_size
            boxes = gt_proposals.gt_boxes
            boxes.scale(scale_x=tar_image_size[1] / ori_image_size[1],
                        scale_y=tar_image_size[0] / ori_image_size[0])
            gt_proposals = Instances(image_size=tar_image_size,
                                     gt_boxes=boxes).to(device)
            topk_proposals = add_ground_truth_to_proposals_single_image(gt_proposals, topk_proposals)

        return topk_proposals

    # TODO: input topk proposals
    def sample(self, image_id, proposals, mask_on=False):
        return super(ContextModellingV2, self).sample(self._get_proposals(image_id, proposals), mask_on)
