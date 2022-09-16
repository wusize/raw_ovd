import os
import threading
from argparse import ArgumentParser

parser = ArgumentParser(description='COCO Error Analysis Tool')
parser.add_argument('--partition', type=str)
parser.add_argument('--quota_type', type=str, default="")
parser.add_argument('--config_file', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--cpus', type=int, default=40)

args = parser.parse_args()

PARTITION = args.partition
RESCALE_TEMPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
ENSEMBLE_FACTORS = [0.6, 0.67, 0.75, 0.8]
CPUS_PER_TASK = args.cpus
QUOTA = args.quota_type
CONFIG_FILE = args.config_file
MODEL_PATH = args.model_path
OUT_DIR = args.out_dir


def th(rescale_temp, ensemble_factor):
    print(f"RESCALE_TEMP{rescale_temp}_ENSEMBLE_FACTOR{ensemble_factor}", flush=True)
    srun_cfg = f"srun -p {PARTITION} --gres=gpu:8 -n1 --cpus-per-task={CPUS_PER_TASK} -J test -N 1 "
    if QUOTA != "":
        srun_cfg += f"--quota={QUOTA} "
    python_cfg = f"python train_net.py --num-gpus 8 --config-file {CONFIG_FILE} --eval-only " \
                 f"MODEL.WEIGHTS {MODEL_PATH} " \
                 f"MODEL.ROI_BOX_HEAD.RESCALE_TEMP {rescale_temp} " \
                 f"MODEL.ROI_BOX_HEAD.ENSEMBLE_FACTOR {ensemble_factor} " \
                 f"OUTPUT_DIR {OUT_DIR}/RESCALE_TEMP{rescale_temp}_ENSEMBLE_FACTOR{ensemble_factor}"

    cmd = srun_cfg + python_cfg
    os.system(cmd)

threads = []

for temp in RESCALE_TEMPS:
    for factor in ENSEMBLE_FACTORS:
        t = threading.Thread(target=th, name='th', args={temp, factor})
        threads.append(t)
        t.start()

for t in threads:
    t.join()
