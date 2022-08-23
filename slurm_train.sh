#!/bin/bash
#SBATCH -p mm_research
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=16
#SBATCH --quotatype=spot
#SBATCH --cpus-per-task=5
#SBATCH --time 4320
#SBATCH -o "output/slurm-%j.out"

srun multi-node_run.sh $@