from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 1, 'name': 'carton board box'},
    {'id': 2, 'name': 'white square tag'},
]


def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(2)}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS_ = dict(shelf_train=("shelf_v0/10", "shelf_v0/annotations/annotations_train_coco.json"),
                           shelf_val=("shelf_v0/10", "shelf_v0/annotations/annotations_val_coco.json"))

for key, (image_root, json_file) in _PREDEFINED_SPLITS_.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
