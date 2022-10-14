import torch
import pickle
from tqdm import tqdm
import numpy as np
from detectron2.structures.boxes import BoxMode


pth_file = 'instances_predictions.pth'
instances_predictions = torch.load(pth_file)

num_images = len(instances_predictions)
ids = []
boxes = []
objectness_logits = []
clip_image_features = []
bbox_mode = BoxMode.XYWH_ABS

for data in tqdm(instances_predictions):
    image_id = data['image_id']
    instances = data['instances']
    boxes_per_image = np.array([inst['bbox'] for inst in instances],
                               dtype=np.float32)
    scores_per_image = np.array([inst['score'] for inst in instances],
                                dtype=np.float32)
    clip_image_features_per_image = np.array([inst['clip_image_feature'] for inst in instances],
                                             dtype=np.float32)

    ids.append(image_id)
    boxes.append(boxes_per_image)
    objectness_logits.append(scores_per_image)
    clip_image_features.append(clip_image_features_per_image)

with open('proposals_with_clip.pkl', 'wb') as f:
    pickle.dump(dict(ids=ids,
                     boxes=boxes,
                     objectness_logits=objectness_logits,
                     clip_image_features=clip_image_features,
                     bbox_mode=bbox_mode), f)
