import os
import torch.nn.functional as F
import numpy as np


def process_proposals(batched_inputs, images, proposals):
    image_proposals = []
    for input, real_image_size, p in zip(batched_inputs, images.image_sizes, proposals):
        output = dict(file_name=os.path.basename(input['file_name']),
                      height=input['height'],
                      width=input['width'],
                      image_id=input['image_id'],
                      image_size=real_image_size,
                      proposals=p.proposal_boxes.tensor.cpu().numpy().tolist(),
                      objectness_scores=p.objectness_logits.sigmoid().cpu().numpy().tolist())
        image_proposals.append(output)
    return image_proposals


def save_features(features, image_names, out_path):
    os.makedirs(out_path, exist_ok=True)
    tar_shape = list(features.values())[0].shape[2:]
    for k, v in features.items():
        v = v.max(1, keepdim=True).values.cpu()
        features[k] = F.interpolate(v, size=tar_shape,
                                    mode='bilinear',
                                    align_corners=False)[:, 0].numpy()
    for i, image_name in enumerate(image_names):
        name = os.path.basename(image_name)
        name = name.split('.')[0]
        folder = os.path.join(out_path, name)
        os.makedirs(folder, exist_ok=True)
        for k, feats in features.items():
            np.save(os.path.join(folder, k+'.npy'), feats[i])
