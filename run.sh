CONFIG=$1
GPUS=${GPUS:-8}
SLEEP=${SLEEP:-0h}
ARGS=${ARGS:-""}
sudo sleep ${GPUS}
sudo python train_net.py --num-gpus ${GPUS} --config-files ${CONFIG} ${ARGS}