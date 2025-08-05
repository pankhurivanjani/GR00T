#!/bin/bash
#SBATCH --job-name=task_1
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL    
#SBATCH --mail-user=ulgrl@student.kit.edu


source /home/ka/ka_stud/ka_ulgrl/miniconda3/etc/profile.d/conda.sh
conda activate /home/ka/ka_stud/ka_ulgrl/miniconda3/envs/gr00t
python /home/ka/ka_stud/ka_ulgrl/policies/Isaac-GR00T/scripts/gr00t_finetune_2.py
