import torch
import torch.nn as nn
from detectron2.utils.events import get_event_storage
import numpy as np
from torchvision.ops import nms
import detectron2.utils.comm as comm
from detectron2.structures import Boxes, Instances
import json


class Queues(nn.Module):
    def __init__(self, queue_cfg):
        super(Queues, self).__init__()
        self.queue_cfg = queue_cfg
        self._init_queues()

    def _init_queues(self):
        attr_names = self.queue_cfg.NAMES
        queue_lengths = self.queue_cfg.LENGTHS
        for n in attr_names:
            self.register_buffer(n, torch.zeros(0, 512 + 1),
                                 persistent=False)
        self.queue_lengths = {n: queue_lengths[i] for i, n in enumerate(attr_names)}

    def _debug_queues(self):
        attr_names = self.queue_cfg.NAMES
        storage = get_event_storage()
        for n in attr_names:
            val = getattr(self, n)
            storage.put_scalar(f"queues/{n}", np.float32(val.shape[0]))

    @torch.no_grad()
    def dequeue_and_enqueue(self, queue_update):
        for k, feat in queue_update.items():
            queue_length = self.queue_lengths[k]
            valid = feat[:, -1] >= 0    # valid label
            if valid.sum() == 0:
                continue
            feat = feat[valid]
            feat = feat[:queue_length]
            in_length = feat.shape[0]
            queue_value = getattr(self, k)
            current_length = queue_value.shape[0]
            kept_length = min(queue_length - in_length, current_length)

            queue_value.data = torch.cat([feat, queue_value[:kept_length]])

        self._debug_queues()

    @torch.no_grad()
    def get_queue(self, key):
        value = getattr(self, key)
        return value


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


# TODO: a queue that save proposals for all images
class BoxesCache(nn.Module):
    def __init__(self, json_path, num_proposals):
        super(BoxesCache, self).__init__()
        with open(json_path, 'r') as f:
            images_info = json.load(f)['images']
        num_images = len(images_info)
        self.image_id2ordered_id = {info['id']: ordered_id for ordered_id, info in enumerate(images_info)}
        boxes = torch.zeros(num_images, num_proposals, 5)   # [x1, y1, x2, y2, s]
        self.register_buffer("boxes", boxes, persistent=False)
        self.num_proposals = num_proposals

    @torch.no_grad()
    def update(self, image_id, proposals, nms_thr, score_thr):
        # TODO: pull cached boxes from all devices
        ordered_id = self.image_id2ordered_id[image_id]
        image_boxes_cache = self.boxes[ordered_id]

        proposal_boxes = proposals.proposal_boxes.tensor
        proposal_scores = proposals.objectness_logits.sigmoid()

        merged_boxes = torch.cat([image_boxes_cache[:, :4], proposal_boxes])
        merged_scores = torch.cat([image_boxes_cache[:, 4], proposal_scores])

        score_kept = merged_scores > score_thr
        if score_kept.sum() == 0:
            score_kept = [0]

        merged_boxes = merged_boxes[score_kept]
        merged_scores = merged_scores[score_kept]

        nmsed_kept = nms(merged_boxes, merged_scores, nms_thr)

        kept_boxes = merged_boxes[nmsed_kept]
        kept_scores = merged_scores[nmsed_kept]

        num_update = min(self.num_proposals, len(kept_boxes))
        device = kept_scores.device
        update_cache_to_sync = torch.zeros(self.num_proposals, 6)     # [x,y,x,y,s,order_id]
        update_cache_to_sync[:, -1] = float(ordered_id)    # ordered_id
        update_cache = torch.cat([kept_boxes, kept_scores[:, None]], dim=1)[:num_update]
        update_cache_to_sync[:num_update, :5] = update_cache

        # sync for updates from other devices
        update_cache = comm.all_gather(update_cache_to_sync)
        for update_cache_ in update_cache:
            ordered_id_ = int(update_cache_[0, -1].item())
            self.boxes[ordered_id_] = update_cache_[:, :5].to(device)  # update

        kept_logits = inverse_sigmoid(kept_scores)

        return Instances(image_size=proposals.image_size,
                         proposal_boxes=Boxes(kept_boxes),
                         objectness_logits=kept_logits)
