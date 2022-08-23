#!/bin/bash
#SBATCH -p mm_research
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=496G
#SBATCH --time 4320
#SBATCH -o "slurm-output/slurm-%j.out"

srun --quota=spot multi-node_run.sh $@