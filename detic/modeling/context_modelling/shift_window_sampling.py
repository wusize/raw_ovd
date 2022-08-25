import random
import torch
import sys
sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.modeling.utils import multi_apply
from detic.modeling.context_modelling.stochastic_sampling \
    import StochasticSampling
from detectron2.structures import Boxes, Instances


def ij_generator(i_range, j_range):
    seen = set()
    while True:
        ij = (random.choice(range(i_range)),
              random.choice(range(j_range)))
        if ij not in seen:
            seen.add(ij)
            yield ij


def random_2d_choices(i_range, j_range, num):
    generator = ij_generator(i_range, j_range)
    return [next(generator) for _ in range(num)]


class ShiftWindowSampling:
    def __init__(self, cfg):
        self.split_per_level = cfg.SPLIT_PER_LEVEL   # [16, 8, 4, 2]
        self.step_per_level = cfg.STEP_PER_LEVEL     # [4, 2, 1, 1]
        self.num_samples_per_step = cfg.NUM_SAMPLES_PER_STEP   # [2, 2, 1, 1]

        # cfg.MAX_GROUPS_PER_SAMPLE = [1, 1, 1, 1]
        # cfg.NUM_SAMPLES_PER_STEP = [2, 2, 1, 1]
        self.group_sampler = StochasticSampling(
            max_groups_per_sample=cfg.MAX_GROUPS_PER_SAMPLE,
            alpha=cfg.ALPHA,
            interval=cfg.INTERVAL,
            base_probability=cfg.BASE_PROBABILITY
        )

    @staticmethod
    def _get_grid_templates(split, image_size):
        h, w = image_size
        box_h, box_w = h / split, w / split
        x0 = torch.arange(split).float() * box_w
        y0 = torch.arange(split).float() * box_h

        grid_x0, grid_y0 = torch.meshgrid(x0, y0, indexing='xy')

        boxes = torch.stack([grid_x0, grid_y0, grid_x0 + box_w, grid_y0 + box_h],
                            dim=-1)

        return boxes

    @staticmethod
    def _box_ij2box_num(box_ijs, split, box_id_start):
        assert box_ijs.min() >= 0
        return box_ijs[:, 0] * split + box_ijs[:, 1] + box_id_start

    def _sample_at_single_level(self, split, step, box_id_start, level, image_size):
        assert split % step == 0
        boxes = self._get_grid_templates(split, image_size)
        num_blocks = split // step
        groups_per_level, normed_boxes_per_level, spanned_boxes_per_level, box_ids_per_level \
            = [], [], [], []
        for i in range(num_blocks):
            for j in range(num_blocks):
                ij_samples = random_2d_choices(step, step, self.num_samples_per_step[level])
                for s in ij_samples:
                    i_sample = i * step + s[0]
                    j_sample = j * step + s[1]
                    roi = boxes[i_sample, j_sample]
                    roi_index = torch.tensor([i_sample, j_sample])
                    groups, normed_boxes, spanned_boxes, box_ids \
                        = self.group_sampler.sample(roi, image_size, level)
                    box_ids = [self._box_ij2box_num(g + roi_index,
                                                    split, box_id_start)
                               for g in box_ids]
                    groups_per_level.extend(groups)
                    normed_boxes_per_level.extend(normed_boxes)
                    spanned_boxes_per_level.extend(spanned_boxes)
                    box_ids_per_level.extend(box_ids)

        # convert to boxes
        boxes_tensor = torch.cat(groups_per_level, dim=0)
        box_ids_tensor = torch.cat(box_ids_per_level, dim=0)
        sampled_instances = Instances(image_size=image_size,
                                      boxes=Boxes(boxes_tensor),
                                      levels=torch.ones_like(box_ids_tensor) * level,
                                      box_ids=box_ids_tensor)

        return sampled_instances, normed_boxes_per_level, spanned_boxes_per_level

    def sample_on_single_image(self, image_size):
        box_id_starts = [0]
        for i in range(len(self.split_per_level) - 1):
            split = self.split_per_level[i]
            box_id_starts.append(box_id_starts[-1] + split ** 2)

        sampled_instances, normed_boxes, spanned_boxes \
            = multi_apply(self._sample_at_single_level,
                          self.split_per_level,
                          self.step_per_level, box_id_starts,
                          range(len(self.split_per_level)),
                          image_size=image_size)

        sampled_instances = Instances.cat(sampled_instances)
        normed_boxes = [g for lvl in normed_boxes for g in lvl]
        spanned_boxes = torch.stack([g for lvl in spanned_boxes for g in lvl], dim=0)

        return sampled_instances, normed_boxes, spanned_boxes

    @staticmethod
    def _modify_box_ids(sampled_instances):
        new_instances = []
        box_id_start = 0
        for inst in sampled_instances:
            inst.box_ids = inst.box_ids + box_id_start
            box_id_start += inst.box_ids.max().item()
            new_instances.append(inst)

        return new_instances

    def sample(self, image_sizes):
        sampled_instances, normed_boxes, spanned_boxes = multi_apply(self.sample_on_single_image,
                                                                     image_sizes)
        # normed_boxes = [g for img in normed_boxes for g in img]
        # spanned_boxes = [g for img in spanned_boxes for g in img]
        sampled_instances = self._modify_box_ids(sampled_instances)

        return sampled_instances, normed_boxes, spanned_boxes


if __name__ == '__main__':
    from time import time
    from detectron2.config import CfgNode as CN
    cfg = CN()
    cfg.SPLIT_PER_LEVEL = [16, 8, 4, 2]
    cfg.STEP_PER_LEVEL = [8, 4, 2, 1]       # NUM_STEPS = [4, 4, 4, 2]
    cfg.MAX_GROUPS_PER_SAMPLE = [1, 1, 1, 1]
    cfg.NUM_SAMPLES_PER_STEP = [8, 4, 2, 1]
    cfg.MAX_PERMUTATIONS = 2
    cfg.ALPHA = 2.0
    cfg.INTERVAL = -0.2
    cfg.BASE_PROBABILITY = [0.3, 0.3, 0.3, 0.3]

    shift_window_sampler = ShiftWindowSampling(cfg)

    tik = time()
    sampled_instances, normed_boxes, spanned_boxes = shift_window_sampler.sample(
        [(800, 1333), (1200, 800)])
    tok = time()
    print(time() - tik)
