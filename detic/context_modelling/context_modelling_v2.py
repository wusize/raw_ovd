import torch
from detectron2.structures import Instances, Boxes
from pycocotools.coco import COCO
import numpy as np
from detectron2.modeling.proposal_generator.proposal_utils import \
    add_ground_truth_to_proposals_single_image
from detectron2.structures.masks import PolygonMasks
from .context_modelling import ContextModelling


class ContextModellingV2(ContextModelling):
    def __init__(self, cfg, num_words, word_embed_dim, word_dropout, sigmoid=True):
        super(ContextModellingV2, self).__init__(cfg, num_words, word_embed_dim, word_dropout, sigmoid)
        pr_coco = COCO(self.cfg.PROPOSALS)
        proposals = {}
        for img_id, anns in pr_coco.imgToAnns.items():
            boxes = torch.tensor([ann['bbox'] for ann in anns])
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
            img_info = pr_coco.imgs[img_id]
            image_size = (img_info['height'], img_info['width'])
            proposals[img_id] = Instances(image_size=image_size, proposal_boxes=Boxes(boxes))

        self.proposals = proposals

    def _get_proposals(self, image_id, tar_image_size):
        if image_id in self.proposals:
            proposals = self.proposals[image_id]
            ori_image_size = proposals.image_size
            boxes = proposals.proposal_boxes
            boxes.scale(scale_x=tar_image_size[1] / ori_image_size[1],
                        scale_y=tar_image_size[0] / ori_image_size[0])
            num_boxes = len(boxes)
            proposals = Instances(image_size=tar_image_size,
                                  proposal_boxes=boxes,
                                  objectness_logits=torch.ones(num_boxes),  # ~0.7
                                  gt_classes=-torch.ones(num_boxes),
                                  gt_boxes=boxes,
                                  sample_types=-torch.ones(num_boxes).int()
                                  )
        else:
            proposals = Instances(image_size=tar_image_size,
                                  proposal_boxes=Boxes(torch.zeros(0, 4)),
                                  objectness_logits=torch.ones(0),  # ~0.7
                                  gt_classes=-torch.ones(0),
                                  gt_boxes=Boxes(torch.zeros(0, 4)),
                                  sample_types=-torch.ones(0).int()
                                  )

        return proposals

    # TODO: input topk proposals
    def sample(self, image_id, gt_instances, mask_on=False):
        proposals = self._get_proposals(image_id, gt_instances.image_size
                                        ).to(gt_instances.gt_boxes.device)
        if self.cfg.ADD_GT:
            proposals = add_ground_truth_to_proposals_single_image(gt=gt_instances, proposals=proposals)
        if mask_on:
            proposals.set('gt_masks',
                          PolygonMasks([[np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])]] * len(proposals)))

        return self.sample_on_topk(proposals, mask_on)
