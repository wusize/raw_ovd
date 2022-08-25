import torch
from .context_modelling import ContextModelling


class ContextModellingV2(ContextModelling):

    def sample(self, image_sizes):
        sampled_instances, normed_boxes, spanned_boxes \
            = self.checkboard_sampling.sample(image_sizes)
        box_ids = torch.cat([inst.box_ids for inst in sampled_instances])
        return dict(box_ids=box_ids,
                    sampled_instances=sampled_instances,
                    normed_boxes=normed_boxes,
                    spanned_boxes=spanned_boxes)

    def get_loss(self, group_info, clip_images, clip_model, image_info):
        losses = dict()
        queue_update = dict()
        loss_kd, queue_kd = self.kd_clip_contrast(**group_info,
                                                  clip_images=clip_images,
                                                  clip_model=clip_model,
                                                  image_info=image_info)
        losses.update(loss_kd)
        queue_update.update(queue_kd)

        self.queues.dequeue_and_enqueue(queue_update)

        return losses
