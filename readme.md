# Dependencies
detectron2
# Data
## format
```
├─ raw_ovd
    ├─datasets
        ├─ lvis
            │-- lvis_v1_train_norare.json
            |-- lvis_v1_val.json
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
        │-- ViT-L-14.pt
        │-- res50_fpn_soco_star_400_ensemble.pkl
        │-- coco_kd_best_34.0.pth
        │-- lvis_kd_best_22.8.pth
```
## Obtain data and cpts
**For object365**
```
wget https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json
```
**For COCO**

Obtain the json files from
[google drive](https://drive.google.com/file/d/1K4T0Q-rhzl09RkhKsur6xWqs31HSx3fB/view?usp=sharing).


**For LVIS**

Obtain `lvis_v1_train_norare.json` from [google drive](https://drive.google.com/file/d/1ahmCUXyFAQqnlMb-ZDDSQUMnIosYqhu5/view?usp=drive_link).

**For checkpoints**

**SOCO Pretrain**

Obtain the checkpoints  [res50_fpn_soco_star_400.pkl](https://drive.google.com/file/d/1rIW9IXjWEnFZa4klZuZ5WNSchRYaOC0x/view?usp=drive_link)
and [res50_fpn_soco_star_400_ensemble.pkl](https://drive.google.com/file/d/16-u1lj13T8TQswEx_-f5p1w8bRPA7Ifb/view?usp=drive_link). 

Obtain the checkpoints of baron from
[Jinsheng](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/js20_connect_hku_hk/EingbMkSjIZKu8PObuGte_wBTPGOeV5M88C_Xq34qewiNQ?e=JtHRYv) 
to reproduce the results in paper. Note there is typo on the name of file `lvis_kd_best_22.8.pth`, it should be 22.6.


# Run
## COCO
**Test**

```
python train_net.py --num-gpus 8 --config-file configs/test/infer_coco.yaml \
 --eval-only MODEL.WEIGHTS PATH/TO/CPTS/coco_kd_best_34.0.pth
```

**Train**
```
python train_net.py --num-gpus 8 --config-file configs/test/infer_coco.yaml \
MODEL.WEIGHTS PATH/TO/CPTS/res50_fpn_soco_star_400.pkl
```

## LVIS

**Test**

```
python train_net.py --num-gpus 8 \
--config-file configs/test/res50_fpn_lvis_2x_kd_prompt_ensemble_keeplast_inf_mask.yaml \
 --eval-only MODEL.WEIGHTS PATH/TO/CPTS/lvis_kd_best_22.8.pth
```

**Train**
```
python train_net.py --num-gpus 8 \
--config-file configs/test/res50_fpn_lvis_2x_kd_prompt_ensemble_keeplast_inf_mask.yaml \
MODEL.WEIGHTS PATH/TO/CPTS/res50_fpn_soco_star_400_ensemble.pkl
```

## objects365
```
python train_net.py --num-gpus 8 --config-file \
configs/transfer/learned_prompt2objects365v2.yaml --eval-only \
MODEL.WEIGHTS models/lvis_kd_best_22.8.pth
```
## Train the reduced sampling strategy

Obtain the 

```
python train_net.py --num-gpus 8 --config-file configs/sampling/reduce_num_of_regions.yaml \\
MODEL.WEIGHTS path/to/the/soco_pretrained_model.pkl
```
