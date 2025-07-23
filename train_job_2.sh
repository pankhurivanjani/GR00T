#!/bin/bash
#SBATCH --job-name=gr00t_offline_train
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /home/ka/ka_stud/ka_ulgrl/miniconda3/etc/profile.d/conda.sh
conda activate /home/ka/ka_stud/ka_ulgrl/miniconda3/envs/gr00t
python /home/ka/ka_stud/ka_ulgrl/policies/Isaac-GR00T/scripts/gr00t_finetune_2.py
