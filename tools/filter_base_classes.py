import json
import argparse
import sys

sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
from detic.data.datasets.coco_zeroshot import categories_unseen

novel_cat_ids = [cat['id'] for cat in categories_unseen]

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="datasets/coco/annotations/instances_val2017.json", type=str)
parser.add_argument("--out_path", default="datasets/coco/zero-shot/instances_val2017.json")
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

annotations = []

for ann in json_coco['annotations']:
    if ann['category_id'] in novel_cat_ids:
        annotations.append(ann)

json_coco['annotations'] = annotations

with open(args.out_path, 'w') as f:
    json.dump(json_coco, f)
