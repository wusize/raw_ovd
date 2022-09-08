import torch
from torchvision.ops import box_iou
from .context_modelling_fix_gt import ContextModellingFixGT


class ContextModellingDetachGT(ContextModellingFixGT):

    def use_gt_embeddings(self, pseudo_words, word_masks, group_info):
        proposals = [g['proposals'] for g in group_info]
        gts = [g['gts'] for g in group_info]
        num_proposals_per_image = [len(p) for p in proposals]
        pseudo_words = pseudo_words.split(num_proposals_per_image, dim=0)
        word_masks = word_masks.split(num_proposals_per_image, dim=0)
        new_pseudo_words = []
        new_word_masks = []
        for gt, prs, wds, wd_ms in zip(gts, proposals, pseudo_words, word_masks):
            new_wds = torch.zeros_like(wds)
            if len(gt) > 0:
                # match
                ious = box_iou(prs.proposal_boxes.tensor, gt.gt_boxes.tensor)
                matched_iou, matched_gt = ious.max(-1)
                is_pos = matched_iou > self.cfg.GT_THR
                positive = torch.where(is_pos)[0]
                negative = torch.where(is_pos.logical_not())[0]
                new_wds[positive] = wds[positive].detach()    # detach positive ones
                new_wds[negative] = wds[negative]
                new_wd_ms = wd_ms
            else:
                new_wds = wds
                new_wd_ms = wd_ms
            new_pseudo_words.append(new_wds)
            new_word_masks.append(new_wd_ms)

        return torch.cat(new_pseudo_words), torch.cat(new_word_masks)
