# Copyright (c) Facebook, Inc. and its affiliates.
import json
import argparse
from PIL import Image
import numpy as np
import os
import threading
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/cc3m/train_image_info_tags.json')
    parser.add_argument('--save_image_path', default='datasets/cc3m/training/')
    parser.add_argument('--num_threads', default=12)
    args = parser.parse_args()
    if not os.path.exists(args.save_image_path):
        os.makedirs(args.save_image_path)
    with open(args.ann, 'r') as f:
        images_info = json.load(f)
    images = images_info['images']

    num_images = len(images)
    print(f'Total number of images: {num_images}', flush=True)
    num_images_per_thread = num_images // args.num_threads
    num_threads = math.ceil(num_images / num_images_per_thread)

    def _read_images(start_id):
        valid_images = []
        end_id = min(start_id + num_images_per_thread, len(images))
        for i in range(start_id, end_id):
            img_info = images[i]
            os.system(
                f"wget {img_info['path']} -t 1 -T 4 -O {args.save_image_path}/{img_info['file_name']}")
            try:
                img = Image.open(
                    open('{}/{}.jpg'.format(args.save_image_path, i + 1), "rb"))
                img = np.asarray(img.convert("RGB"))
                h, w = img.shape[:2]
            except:
                continue
            img_info.update(height=h, width=w)
            valid_images.append(img_info)

        data = {'images': valid_images}
        out = args.ann.replace('.json', f'_{start_id}.json')
        print('Saving to', out)
        with open(out, 'w') as f:
            json.dump(data, f)
    threads = []
    for thr in range(num_threads):
        t = threading.Thread(target=_read_images, name='read_images',
                             args={thr * num_images_per_thread})
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    del images
    valid_images = []
    for thr in range(num_threads):
        start_id = thr * num_images_per_thread
        out = args.ann.replace('.json', f'_{start_id}.json')
        with open(out, 'r') as f:
            part_data = json.load(f)
        valid_images.extend(part_data['images'])
        os.system(f'rm {out}')
    images_info['images'] = valid_images

    with open(args.ann.replace('.json', f'_valid.json'), 'w') as f:
        json.dump(images_info, f, indent=4, sort_keys=True)
