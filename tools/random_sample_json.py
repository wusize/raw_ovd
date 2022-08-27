import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="datasets/coco/annotations/instances_val2017.json", type=str)
parser.add_argument("--out_path", default="datasets/coco/zero-shot/instances_val2017.json")
parser.add_argument("--num_samples", default=5000, type=int)
args = parser.parse_args()

with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

annotations = []
image_id2image = {}
for img in json_coco['images']:
    image_id2image[img['id']] = img
sampled_image_ids = random.choices(list(image_id2image.keys()), k=args.num_samples)
json_coco['images'] = [image_id2image[image_id] for image_id in sampled_image_ids]
annotations = []
for ann in json_coco['annotations']:
    if ann['image_id'] in sampled_image_ids:
        annotations.append(ann)

json_coco['annotations'] = annotations

output_path = args.out_path.replace('.json', f'_{args.num_samples}.json')
with open(output_path, 'w') as f:
    json.dump(json_coco, f)
