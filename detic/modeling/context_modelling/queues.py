import torch
import torch.nn as nn
from detectron2.utils.events import get_event_storage
import numpy as np


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
            valid = feat[:, -1] > 0
            if valid.sum() == 0:
                continue
            feat = feat[valid]
            queue_length = self.queue_lengths[k]
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
