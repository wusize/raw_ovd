#!/bin/bash
#SBATCH -p pat_dev
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time 4320
#SBATCH -o "output/slurm-%j.out"

srun bash multi-node_run.sh $@
