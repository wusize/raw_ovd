# Dependencies
detectron2
# Data
## format
```
├─ raw_ovd
    ├─datasets
        ├─ coco
            │-- val
            │-- annotations
            │-- zero-shot
                │-- instances_val2017_all_2_oriorder.json
        ├─ shelf_v0
            │-- annotations
            │-- 10
    ├─models
        │-- ViT-B-32.pt
        │-- coco_kd_best_34.0.pth
        │-- lvis_kd_best_22.8.pth
```
## Obtain data and cpts
For shelf_v0, log into 84
```
cd /mnt/lustreold/share_data/wusize/shelf_v0
```

For checkpoints, download from
[Jinsheng](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/js20_connect_hku_hk/EingbMkSjIZKu8PObuGte_wBTPGOeV5M88C_Xq34qewiNQ?e=JtHRYv)


# Run
## Train on shelf_v0
```
python train_net.py --num-gpus 8 --config-file configs/applications/res50_fpn_shelfv0_2x_kd_handcraft_ensemble_12epochs.yaml \
MODEL.WEIGHTS path/to/the/model/pretrained/on/lvis
```
## Inference
### Infer
```
python demo/demo.py --config-file configs/test/infer_shelf_v0.yaml \
--input path/to/the/images --output path/to/the/output/directories \
--opts MODEL.WEIGHTS path/to/the/model
```
### Infer with reference
```
python demo/demo_with_reference.py --config-file configs/test/infer_with_reference.yaml \
--input path/to/the/images --reference path/to/image/or/folder/of/images;text description \
--output path/to/the/output/directories --opts MODEL.WEIGHTS path/to/the/model
```
