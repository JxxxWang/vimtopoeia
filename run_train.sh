#!/bin/bash
#SBATCH --job-name=vimtopoeia_ast
#SBATCH --output=vimtopoeia_ast_%j.out
#SBATCH --error=vimtopoeia_ast_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8            
#SBATCH --mem=64G                     
#SBATCH --time=12:00:00              

module purge

# === 1. 环境准备 ===
echo "Job started on $(hostname) at $(date)"
echo "Loading modules..."

# module load cuda/11.8 python/3.9
# conda init
# conda activate venv

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate venv
cd /scratch/hw3140/roundtwo/vimtopoeia

export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/hw3140/ast_model_local

echo "Starting training..."

python -u model_training/train.py

echo "Job finished at $(date)"