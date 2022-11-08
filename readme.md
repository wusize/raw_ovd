# Dependencies
detectron2
# Data
```
├─ raw_ovd
    ├─datasets
        ├─ coco
            │-- val
            │-- annotations
            │-- zero-shot
                │-- instances_val2017_all_2_oriorder.json
        ├─ objects365
            ├─  val
                │-- patch0
                │-- ...
                │-- patch43
            ├─  annotations
                │-- zhiyuan_objv2_val.json
    ├─models
        │-- ViT-B-32.pt
        │-- coco_kd_best_34.0.pth
        │-- lvis_kd_best_22.8.pth
  
```
# Run
## COCO
```
python train_net.py --num-gpus 8 --config-file configs/test/infer_coco.yaml \\
 --eval-only MODEL.WEIGHTS models/coco_kd_best_34.0.pth
```
## objects365
```
python train_net.py --num-gpus 8 --config-file \\ 
configs/transfer/learned_prompt2objects365v2.yaml --eval-only \\
MODEL.WEIGHTS models/lvis_kd_best_22.8.pth
```
