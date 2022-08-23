#!/bin/bash
set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
DIST_URL="tcp://$MASTER_NODE:12399"
SOCKET_NAME=$(ip r | grep default | awk '{print $5}')
export GLOO_SOCKET_IFNAME=$SOCKET_NAME


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:8 \
    --ntasks=16 \
    --ntasks-per-node=8 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train_net.py --num-gpus 8 --num-machines 2 --machine-rank "$SLURM_NODEID"  \
       --dist-url "$DIST_URL" --config-file ${CONFIG} ${PY_ARGS}
