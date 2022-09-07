from pycocotools.coco import COCO
from tqdm import tqdm
import json
coco_path = 'datasets/coco/annotations/instances_train2017.json'
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

coco_rare_cat_ids = [cat['id'] for cat in categories_unseen]
coco = COCO(coco_path)
annotations = []
images = []
for img_id, anns in tqdm(coco.imgToAnns.items()):
    has_rare = False
    for ann in anns:
        if ann['category_id'] in coco_rare_cat_ids:
            has_rare = True
            break
    if has_rare:
        annotations.extend(anns)
        images.append(coco.imgs[img_id])
with open('output/instances_train2017_rare_anns_and_imgs.json', 'w') as f:
    json.dump(dict(images=images,
                   annotations=annotations,
                   categories=list(coco.cats.values())), f)
del coco
lvis_path = 'datasets/lvis/annotations/lvis_v1_train.json'
lvis = COCO(lvis_path)

annotations = []
images = []
for img_id, anns in tqdm(lvis.imgToAnns.items()):
    has_rare = False
    for ann in anns:
        cat_id = ann['category_id']
        if lvis.cats[cat_id]['frequency'] == 'r':
            has_rare = True
            break
    if has_rare:
        annotations.extend(anns)
        images.append(lvis.imgs[img_id])
with open('output/lvis_v1_train_rare_anns_and_imgs.json', 'w') as f:
    json.dump(dict(images=images,
                   annotations=annotations,
                   categories=list(lvis.cats.values())), f)
