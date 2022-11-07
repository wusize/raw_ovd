import json
import argparse
from pycocotools.coco import COCO

categories_seen = [
    {'id': 1, 'name': 'person'},
    {'id': 2, 'name': 'bicycle'},
    {'id': 3, 'name': 'car'},
    {'id': 4, 'name': 'motorcycle'},
    {'id': 7, 'name': 'train'},
    {'id': 8, 'name': 'truck'},
    {'id': 9, 'name': 'boat'},
    {'id': 15, 'name': 'bench'},
    {'id': 16, 'name': 'bird'},
    {'id': 19, 'name': 'horse'},
    {'id': 20, 'name': 'sheep'},
    {'id': 23, 'name': 'bear'},
    {'id': 24, 'name': 'zebra'},
    {'id': 25, 'name': 'giraffe'},
    {'id': 27, 'name': 'backpack'},
    {'id': 31, 'name': 'handbag'},
    {'id': 33, 'name': 'suitcase'},
    {'id': 34, 'name': 'frisbee'},
    {'id': 35, 'name': 'skis'},
    {'id': 38, 'name': 'kite'},
    {'id': 42, 'name': 'surfboard'},
    {'id': 44, 'name': 'bottle'},
    {'id': 48, 'name': 'fork'},
    {'id': 50, 'name': 'spoon'},
    {'id': 51, 'name': 'bowl'},
    {'id': 52, 'name': 'banana'},
    {'id': 53, 'name': 'apple'},
    {'id': 54, 'name': 'sandwich'},
    {'id': 55, 'name': 'orange'},
    {'id': 56, 'name': 'broccoli'},
    {'id': 57, 'name': 'carrot'},
    {'id': 59, 'name': 'pizza'},
    {'id': 60, 'name': 'donut'},
    {'id': 62, 'name': 'chair'},
    {'id': 65, 'name': 'bed'},
    {'id': 70, 'name': 'toilet'},
    {'id': 72, 'name': 'tv'},
    {'id': 73, 'name': 'laptop'},
    {'id': 74, 'name': 'mouse'},
    {'id': 75, 'name': 'remote'},
    {'id': 78, 'name': 'microwave'},
    {'id': 79, 'name': 'oven'},
    {'id': 80, 'name': 'toaster'},
    {'id': 82, 'name': 'refrigerator'},
    {'id': 84, 'name': 'book'},
    {'id': 85, 'name': 'clock'},
    {'id': 86, 'name': 'vase'},
    {'id': 90, 'name': 'toothbrush'},
]
categories_unseen = [
    {'id': 5, 'name': 'airplane'},
    {'id': 6, 'name': 'bus'},
    {'id': 17, 'name': 'cat'},
    {'id': 18, 'name': 'dog'},
    {'id': 21, 'name': 'cow'},
    {'id': 22, 'name': 'elephant'},
    {'id': 28, 'name': 'umbrella'},
    {'id': 32, 'name': 'tie'},
    {'id': 36, 'name': 'snowboard'},
    {'id': 41, 'name': 'skateboard'},
    {'id': 47, 'name': 'cup'},
    {'id': 49, 'name': 'knife'},
    {'id': 61, 'name': 'cake'},
    {'id': 63, 'name': 'couch'},
    {'id': 76, 'name': 'keyboard'},
    {'id': 81, 'name': 'sink'},
    {'id': 87, 'name': 'scissors'},
]

novel_cat_ids = [cat['id'] for cat in categories_unseen]
base_cat_ids = [cat['id'] for cat in categories_seen]

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="datasets/coco/annotations/instances_val2017.json", type=str)
parser.add_argument("--out_path", default="datasets/coco/wusize/instances_val2017_cooccur.json")
args = parser.parse_args()


with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

coco = COCO(args.json_path)

kept_image_ids = []
for image_id, anns in coco.imgToAnns.items():
    has_base = False
    has_novel = False
    for ann in anns:
        if ann['category_id'] in base_cat_ids:
            has_base = True
        if ann['category_id'] in novel_cat_ids:
            has_novel = True
    if has_novel and has_base:
        kept_image_ids.append(image_id)

images = [coco.imgs[img_id] for img_id in kept_image_ids]
annotations = []
for img_id in kept_image_ids:
    anns = coco.imgToAnns[img_id]
    for ann in anns:
        if ann['category_id'] in base_cat_ids:
            annotations.append(ann)
        elif ann['category_id'] in novel_cat_ids:
            annotations.append(ann)

json_coco['annotations'] = annotations
json_coco['images'] = images

with open(args.out_path, 'w') as f:
    json.dump(json_coco, f)
