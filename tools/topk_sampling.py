from detectron2.data import detection_utils, transforms
from detectron2.config import get_cfg
from pycocotools.coco import COCO
from detectron2.structures import BoxMode
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
import cv2
import os
import sys
import torch
from tqdm import tqdm
sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.context_modelling.context_modelling_v4 import ContextModellingV4 as ContextModelling
from detic.config import add_detic_config
from detic.data.datasets.coco_zeroshot import categories_unseen

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_root", default="'datasets/coco/train2017'", type=str)
parser.add_argument("--output_root", type=str)
parser.add_argument("--json_path", type=str)
parser.add_argument('--proposal_path', type=str)
args = parser.parse_args()

novel_cat_ids = [cat['id'] for cat in categories_unseen]

def visualizer(image, topk_proposals, out_name):
    for box in topk_proposals.proposal_boxes.tensor:
        x0, y0, x1, y1 = box.long().tolist()
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 5)
    cv2.imwrite(f'{out_name}.jpg', image)

cfg = get_cfg()
add_detic_config(cfg)
cfg.INPUT.MIN_SIZE_TRAIN = (800,)
cfg.INPUT.RANDOM_FLIP = "none"
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.SHAPE_RATIO_THR = [0.25] * 4
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.NMS_THR = [0.10] * 4
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.AREA_THR = [-1, 112, 224, -1]
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.OBJECTNESS_THR = [0.5, 0.5, 0.5, 0.5]
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.AREA_RATIO_THR = 0.002
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.TOPK = [10, 15, 15, 15]
cfg.CONTEXT_MODELLING_V4.ENABLE = True
cfg.CONTEXT_MODELLING_V4.CHECKBOARD.ENABLE = True
augs = detection_utils.build_augmentation(cfg, True)
augs = transforms.AugmentationList(augs)
sampler = ContextModelling(cfg.CONTEXT_MODELLING_V4, 4, 512, 0.5, sigmoid=True)

output_root = args.output_root
os.makedirs(output_root, exist_ok=True)
image_root = args.image_root
json_path = args.json_path
coco = COCO(json_path)
proposal_path = args.proposal_path
instances = torch.load(proposal_path)
img2instances = {inst['image_id']: inst['proposals'] for inst in instances}


for img_id, img_info in tqdm(coco.imgs.items()):
    file_name = img_info['file_name']
    image_path = os.path.join(image_root, file_name)
    out_dir = os.path.join(output_root, file_name.split('.')[0])
    image = detection_utils.read_image(image_path, format='BGR')
    aug_input = transforms.AugInput(image, sem_seg=None)
    trans = augs(aug_input)
    image = aug_input.image
    image_shape = image.shape[:2]
    instances = img2instances[img_id]
    img2ann = coco.imgToAnns[img_id]
    annos = []
    is_novel = []
    for obj in img2ann:
        if obj.get("iscrowd", 0) == 0:
            obj['bbox_mode'] = BoxMode.XYWH_ABS
            is_novel.append(1.0 if obj['category_id'] in novel_cat_ids else 0.0)
            annos.append(
                detection_utils.transform_instance_annotations(
                    obj, trans, image_shape, keypoint_hflip_indices=None
                )
            )
    is_novel = torch.tensor(is_novel)
    gt_instances = detection_utils.annotations_to_instances(
        annos, image_shape, mask_format="polygon"
    )
    instances = add_ground_truth_to_proposals([gt_instances[is_novel < 1.0]], [instances])[0]   # only add base classes

    topk_proposals = sampler.sample_topk_proposals(instances)
    nmsed_proposals = sampler.multilevel_process(topk_proposals)
    novel_instances = gt_instances[is_novel > 0.0]
    for box in novel_instances.gt_boxes.tensor:
        x0, y0, x1, y1 = box.long().tolist()
        image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 5)
    visualizer(image, nmsed_proposals, out_dir)
