from .context_modelling import ContextModelling


class ContextModellingV2(ContextModelling):
    def get_loss(self, group_infos, clip_images, clip_model, image_info, roi_head, features, *args, **kwargs):
        sampled_instances = [g['sampled_instances'] for g in group_infos]
        predictions = roi_head.get_pseudo_words(sampled_instances, features, *args, **kwargs)
        return super(ContextModellingV2, self).get_loss(group_infos, predictions,
                                                        clip_images, clip_model, image_info)
