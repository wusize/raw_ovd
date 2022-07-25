# Copyright (c) Facebook, Inc. and its affiliates.
import json
import argparse
import os
import threading
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/cc3m/train_image_info_tags.json')
    parser.add_argument('--save_image_path', default='datasets/cc3m/training/')
    parser.add_argument('--num_threads', default=12, type=int)
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
        end_id = min(start_id + num_images_per_thread, len(images))
        for i in range(start_id, end_id):
            img_info = images[i]
            os.system(
                f"wget {img_info['path']} -t 1 -T 10 -O {args.save_image_path}/{img_info['file_name']}")

    threads = []
    for thr in range(num_threads):
        t = threading.Thread(target=_read_images, name='read_images',
                             args={thr * num_images_per_thread})
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
