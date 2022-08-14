from .context_modelling import ContextModelling
import torch
from detectron2.structures import Instances, Boxes


class OfflineContextModelling(ContextModelling):
    def __init__(self, **kwargs):
        super(OfflineContextModelling, self).__init__(**kwargs)
        self.instances_predictions = torch.load(kwargs.get('cfg').INSTANCES_PREDICTIONS)
        image_ids2list_ids = {}
        for list_id, inst in enumerate(self.instances_predictions):
            image_ids2list_ids[inst['image_id']] = list_id
        self.image_ids2list_ids = image_ids2list_ids

    def sample(self, gts, image_id, mask_on):
        device = gts.gt_boxes.device
        list_id = self.image_ids2list_ids[image_id]
        proposals = self.instances_predictions[list_id]['proposals']
        num_proposals = len(proposals)
        num_gts = len(gts)
        proposal_boxes = proposals.proposal_boxes
        proposal_boxes.scale(gts.image_size[1] / proposals.image_size[1],
                             gts.image_size[0] / proposals.image_size[0])
        proposals_with_gt = Instances(image_size=gts.image_size,
                                      proposal_boxes=Boxes.cat([
                                          gts.gt_boxes,
                                          proposal_boxes.to(device)
                                      ]),
                                      objectness_logits=torch.cat([
                                          100.0 * torch.ones(num_gts, device=device),
                                          proposals.objectness_logits.to(device)
                                      ], dim=0),
                                      gt_classes=torch.cat([
                                          gts.gt_classes,
                                          -torch.ones(num_proposals, device=device).long()
                                      ], dim=0)
                                      )

        return super(OfflineContextModelling, self).sample(proposals_with_gt, mask_on)
