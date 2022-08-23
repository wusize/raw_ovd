from .context_modelling import ContextModelling
from torchvision.ops import box_iou


class ContextModellingV2(ContextModelling):
    def get_loss(self, group_infos, clip_images, clip_model, image_info, roi_head, features, *args, **kwargs):
        sampled_instances = [g['sampled_instances'] for g in group_infos]
        predictions = roi_head.get_pseudo_words(sampled_instances, features, *args, **kwargs)
        return super(ContextModellingV2, self).get_loss(group_infos, predictions,
                                                        clip_images, clip_model, image_info)

    # TODO: input topk proposals
    def sample(self, proposals_per_image, mask_on=False, targets=None):
        topk_proposals = self._sample_topk_proposals(proposals_per_image, mask_on, targets=targets)
        added_instances, info = self.sample_on_topk(topk_proposals, mask_on)
        info.update(gt_ious_scores=None)

        return added_instances, info

    def _sample_topk_proposals(self, proposals_per_image, mask_on=False, targets=None):
        topk_proposals = super(ContextModellingV2, self)._sample_topk_proposals(proposals_per_image,
                                                                                mask_on)
        if targets is not None and len(targets) > 0:
            # the targets are all base annotations
            gt_boxes = targets.gt_boxes
            ious_with_gts = box_iou(topk_proposals.proposal_boxes, gt_boxes)
            ious_with_gts = ious_with_gts.max(-1).values

            valid = ious_with_gts < self.cfg.MAX_IOU_WITH_GT

            if valid.sum() == 0:
                topk_proposals = topk_proposals[:1]
            else:
                topk_proposals = topk_proposals[valid]

        return topk_proposals
