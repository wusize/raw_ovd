# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import logging
import os
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["custom_load_coco_json"]


def custom_load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    assert dataset_name is not None
    meta = MetadataCatalog.get(dataset_name)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
        if "coco" not in dataset_name:
            logger.warning(
                    """Category ids in annotations are not in [1, #categories]! 
                    We'll apply a mapping for you."""
            )
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] if img_id in coco_api.imgToAnns
            else [] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        # TODO: add captions
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]

        record['pos_category_ids'] = [
            id_map[x] for x in img_dict.get("pos_category_ids", [])]

        record['captions'] = img_dict.get('captions', [])
        record['ann_type'] = img_dict.get('ann_type', 'only_caption')

        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts


def custom_register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: custom_load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
