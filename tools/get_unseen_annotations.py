import json
from pycocotools.coco import COCO
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

unseen_ids = [cat['id'] for cat in categories_unseen]

json_path = 'datasets/coco/annotations/instances_train2017.json'
save_path = 'datasets/coco/annotations/instances_train2017_rare.json'
coco = COCO(json_path)
valid_images = []
valid_anns = []
for img_id, anns in coco.imgToAnns.items():
    anns_ = []
    for ann in anns:
        if ann['category_id'] in unseen_ids:
            anns_.append(ann)
    if len(anns_) > 0:
        valid_anns.extend(anns_)
        valid_images.append(coco.imgs[img_id])
valid_cats = []
for idx in unseen_ids:
    valid_cats.append(coco.cats[idx])

with open(save_path, 'w') as f:
    json.dump(dict(annotations=valid_anns,
                   images=valid_images,
                   categories=valid_cats), f, indent=4, sort_keys=True)
