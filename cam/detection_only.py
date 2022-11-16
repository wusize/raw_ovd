import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import optim
from cam.gradcam import GradCAM, GradCamPlusPlus
sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
from cam.detectron2_gradcam import Detectron2GradCAM
from detectron2.data import MetadataCatalog
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    # cfg.freeze()
    default_setup(cfg, args)

    return cfg

args = default_argument_parser()
args = args.parse_args()

cfg = setup(args)

# plt.rcParams["figure.figsize"] = (30,10)
img_names = [
             "000000007977.jpg", "000000009483.jpg", "000000017029.jpg",
             "000000025394.jpg", "000000029984.jpg", "000000030494.jpg",
             "000000035682.jpg",
             # "000000053624.jpg", "000000117525.jpg",
             # "000000134886.jpg", "000000157138.jpg", "000000166426.jpg",
             # "000000186282.jpg", "000000212573.jpg", "000000225184.jpg",
             ]

layer_name = "backbone.bottom_up.res5.2.conv3"
instances = [
             "skateboard", "keyboard", "dog",
             "cup", "umbrella", "dog",
             "cake",
             # "elephant", "dog",
             # "airplane", "cake", "knife",
             # "keyboard", "umbrella", "dog"
            ]



if __name__ == "__main__":
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
