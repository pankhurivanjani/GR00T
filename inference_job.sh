#!/bin/bash
#SBATCH --job-name=off_gr00t
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 20:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/miniconda3/envs/groot
python /hkfs/work/workspace/scratch/vb0283-fastslowtac/GR00T/scripts/inference_vis.py

