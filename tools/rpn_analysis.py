import json
from glob import glob
import os
import torch
from pycocotools.coco import COCO
from torchvision.ops import box_iou, nms
from tqdm import tqdm


class COCORPNAnalysis:
    def __init__(self, gt_json_file, proposals_path,
                out_path,
                score_thrs=[0.85], iou_thrs=[0.5], nms_thrs=[0.7], topks=[300]):
        self.gt_coco = COCO(gt_json_file)
        self.cat_ids = sorted(self.gt_coco.cats.keys())
        self.cat_names = [self.gt_coco.cats[cat_id]['name'] for cat_id in self.cat_ids]
        self.proposal_json_files = sorted(glob(f'{proposals_path}/*.json'))
        self.out_path = out_path
        self.score_thrs = score_thrs
        self.iou_thrs = iou_thrs
        self.nms_thrs = nms_thrs
        self.topks = topks
        self.classes_unseen = ['airplane', 'bus', 'cat', 'dog', 'cow',
                          'elephant', 'umbrella', 'tie', 'snowboard',
                          'skateboard', 'cup', 'knife', 'cake', 'couch',
                          'keyboard', 'sink', 'scissors']
        self.classes_seen = ['person', 'bicycle', 'car',
                        'motorcycle', 'train', 'truck',
                        'boat', 'bench', 'bird', 'horse',
                        'sheep', 'bear', 'zebra', 'giraffe',
                        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis',
                        'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet',
                        'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster', 'refrigerator',
                        'book', 'clock', 'vase', 'toothbrush']

    def recall_analysis(self, score_thr, iou_thr, nms_thr, topk):
        classes_recall = {k: [] for k in self.classes_seen}
        unseen_recall = {k: [] for k in self.classes_unseen}
        classes_recall.update(unseen_recall)
        for proposal_json in tqdm(self.proposal_json_files):
            with open(proposal_json, 'r') as f:
                proposals = json.load(f)

            image_id = proposals['image_id']
            gts = self.gt_coco.imgToAnns[image_id]
            if len(gts) == 0:
                continue
            gt_boxes = torch.tensor([gt['bbox'] for gt in gts])
            gt_boxes[:, 2:] = gt_boxes[:, 2:] + gt_boxes[:, :2]
            gt_names = [self.gt_coco.cats[gt['category_id']]['name'] for gt in gts]

            proposal_boxes = torch.tensor(proposals['proposals'])
            proposal_boxes = self.rescale_boxes(proposal_boxes, proposals['image_size'],
                                                [proposals['height'], proposals['width']])
            proposal_scores = torch.tensor(proposals['objectness_scores'])

            num = min(topk, len(proposal_scores))
            topk_scores, topk_inds = proposal_scores.topk(num)
            topk_boxes = proposal_boxes[topk_inds]

            valid = topk_scores > score_thr
            if valid.sum() > 0:
                valid_boxes = topk_boxes[valid]
                valid_scores = topk_scores[valid]

                nms_kept = nms(valid_boxes, valid_scores, iou_threshold=nms_thr)
                nmsed_boxes = valid_boxes[nms_kept]

                iou_matrix = box_iou(gt_boxes, nmsed_boxes)

                max_ious = iou_matrix.max(-1)[0]

                detected = (max_ious > iou_thr).float().tolist()
            else:
                detected = [0.0] * len(gts)

            for det, name in zip(detected, gt_names):
                if name in classes_recall:
                    classes_recall[name].append(det)

        text_to_write = 'Seen Classes\n'
        for cls in self.classes_seen:
            recall = sum(classes_recall[cls]) / (len(classes_recall[cls]) + 1e-12)
            text_to_write += f'{cls}: {recall}\n'

        text_to_write += 'Unseen Classes\n'
        for cls in self.classes_unseen:
            recall = sum(classes_recall[cls]) / (len(classes_recall[cls]) + 1e-12)
            text_to_write += f'{cls}: {recall}\n'

        with open(os.path.join(self.out_path,
                               '%.2f_%.2f_%.2f_%d' % (score_thr, iou_thr, nms_thr, topk)), 'w') as f:
            f.writelines(text_to_write)

    def analyze(self):
        for score_thr, iou_thr, nms_thr, topk in zip(
                self.score_thrs, self.iou_thrs, self.nms_thrs, self.topks):
            self.recall_analysis(score_thr, iou_thr, nms_thr, topk)

    @staticmethod
    def rescale_boxes(boxes, image_size, ori_image_size):
        h, w = image_size
        ori_h, ori_w = ori_image_size
        scale = torch.tensor([ori_w / w, ori_h / h,
                              ori_w / w, ori_h / h])
        return boxes * scale[None]


if __name__ == '__main__':
    gt_json_file = 'datasets/coco/annotations/instances_val2017.json'
    proposals_path = 'output/save_debug/'
    out_path = 'output/'
    rpn_analysis = COCORPNAnalysis(gt_json_file, proposals_path,
                    out_path)
    rpn_analysis.analyze()
