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
    parser.add_argument('--ann', default='datasets/cc3m/Train_GCC-training.tsv')
    parser.add_argument('--save_image_path', default='datasets/cc3m/training/')
    parser.add_argument('--num_threads', default=12)
    parser.add_argument('--cat_info', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--out_path', default='datasets/cc3m/train_image_info.json')
    args = parser.parse_args()
    categories = json.load(open(args.cat_info, 'r'))['categories']
    if not os.path.exists(args.save_image_path):
        os.makedirs(args.save_image_path)
    with open(args.ann, 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    num_lines = len(lines)
    num_lines_per_thread = num_lines // args.num_threads
    num_threads = math.ceil(num_lines / num_lines_per_thread)

    def _read_images(start_id):
        images = []
        end_id = min(start_id + num_lines_per_thread, len(lines))
        for i in range(start_id, end_id):
            cap, path = lines[i][:-1].split('\t')
            os.system(
                f"wget {path} -t 1 -T 1 -O {args.save_image_path}/{i + 1}.jpg")
            try:
                img = Image.open(
                    open('{}/{}.jpg'.format(args.save_image_path, i + 1), "rb"))
                img = np.asarray(img.convert("RGB"))
                h, w = img.shape[:2]
            except:
                continue
            image_info = {
                'id': i + 1,
                'file_name': '{}.jpg'.format(i + 1),
                'height': h,
                'width': w,
                'captions': [cap],
            }
            images.append(image_info)

        data = {'images': images}
        out = args.out_path.replace('.json', f'_{start_id}.json')
        print('Saving to', out)
        with open(out, 'w') as f:
            json.dump(data, f)
    threads = []
    for thr in range(num_threads):
        t = threading.Thread(target=_read_images, name='read_images',
                             args={thr * num_lines_per_thread})
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    images = []
    for thr in range(num_threads):
        start_id = thr * num_lines_per_thread
        out = args.out_path.replace('.json', f'_{start_id}.json')
        with open(out, 'r') as f:
            part_data = json.load(f)
        images.extend(part_data['images'])
        os.system(f'rm {out}')
    data = {'categories': categories, 'images': images, 'annotations': []}

    print('Saving to', args.out_path)
    with open(args.out_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
