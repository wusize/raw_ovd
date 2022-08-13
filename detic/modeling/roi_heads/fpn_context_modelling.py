import torch
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from .utils import multi_apply
from detectron2.utils.events import get_event_storage
from .context_modelling import ContextModelling
from detectron2.modeling.poolers import assign_boxes_to_levels
import torch.nn as nn


class FPNContextModelling(nn.Module):
    def __init__(self, cfg, num_words, word_embed_dim, word_dropout, sigmoid=True):
        super(FPNContextModelling, self).__init__()
        self.num_levels = cfg.CONTEXT_MODELLING.NUM_LEVELS
        self.cfg = cfg
        fpn_context_modellings = []
        for i in range(self.num_levels):
            fpn_context_modellings.append(ContextModelling(cfg.CONTEXT_MODELLING, num_words,
                                                           word_embed_dim,
                                                           word_dropout, sigmoid))
        self.fpn_context_modellings = nn.ModuleList(fpn_context_modellings)

    def sample(self, region_proposals, roi_head):
        # step1: split region proposals into different levels
        num_proposals_per_image = [len(p) for p in region_proposals]
        level_assignments = assign_boxes_to_levels(
            [p.proposal_boxes for p in region_proposals],
            roi_head.box_pooler.min_level,
            roi_head.box_pooler.max_level,
            roi_head.box_pooler.canonical_box_size,
            roi_head.box_pooler.canonical_level
        )
        storage = get_event_storage()
        for i in range(self.num_levels):
            cnt = (level_assignments == i).sum().cpu().numpy()
            storage.put_scalar(f"level_statistics/level_{i}",
                               cnt)

        level_assignments = level_assignments.split(num_proposals_per_image, dim=0)
        region_proposals_per_level = [[p[levels == i]
                                       for p, levels in zip(region_proposals, level_assignments)]
                                      for i in range(self.num_levels)]
        # step2: sample at all levels
        multilevel_samples = [multi_apply(self.fpn_context_modellings[i].sample,
                                          region_proposals_per_level[i]
                                          )
                              for i in range(self.num_levels)]

        return multilevel_samples

    def get_loss(self, multilevel_samples, roi_head,
                 clip_model, clip_images, image_info, features):
        losses = dict()
        for i in range(self.num_levels):
            sampled_instances, group_infos = multilevel_samples[i]
            # extract box feature at all levels

            box_features = roi_head.box_pooler.level_poolers[i](
                features[i], convert_boxes_to_pooler_format(
                    [x.proposal_boxes for x in sampled_instances]
                )
            )
            box_features = roi_head.box_head(box_features)
            input_box_features = roi_head.box_predictor.pre_forward(box_features)
            pseudo_words = roi_head.box_predictor.pred_words(input_box_features)
            sample_types = torch.cat([p.sample_types for p in sampled_instances], dim=0)

            predictions = dict(kd_pseudo_words=pseudo_words[sample_types == 1],
                               caption_pseudo_words=pseudo_words[sample_types == 2])
            level_losses = self.fpn_context_modellings[i].get_loss(group_infos, predictions,
                                                                   clip_images, clip_model, image_info)
            loss_weight = self.cfg.CONTEXT_MODELLING.WEIGHTS_PER_LEVEL[i]
            for k, v in level_losses.items():
                losses[k + f'_{i}'] = v * loss_weight

        return losses
