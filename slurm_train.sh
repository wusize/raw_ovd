#!/bin/bash
#SBATCH -p YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=40
#SBATCH --time 4320
#SBATCH -o "output/slurm-%j.out"

srun multi-node_run.sh $@