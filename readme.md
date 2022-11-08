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
## Obtain data and cpts
For object365
```
wget https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json
```
For COCO
[google drive](https://drive.google.com/file/d/1K4T0Q-rhzl09RkhKsur6xWqs31HSx3fB/view?usp=sharing)
For cpts
[Jinsheng](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/js20_connect_hku_hk/EingbMkSjIZKu8PObuGte_wBTPGOeV5M88C_Xq34qewiNQ?e=JtHRYv)


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
