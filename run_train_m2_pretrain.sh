#!/bin/bash
#SBATCH --job-name=m2_pretrain
#SBATCH --output=sbatch/m2_pretrain_%j.out
#SBATCH --error=sbatch/m2_pretrain_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4            
#SBATCH --mem=64G                     
#SBATCH --time=24:00:00      

# M2 Phase 1 Pre-training Script - HPC ONLY
# Usage: sbatch run_train_m2_pretrain.sh

echo "Running M2 Pre-training in HPC mode with SLURM..."

module purge

echo "Job started on $(hostname) at $(date)"
echo "Loading modules..."

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate venv
cd /scratch/hw3140/roundtwo/vimtopoeia

export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/hw3140/ast_model_local

# HPC paths
TRAIN_H5="/scratch/hw3140/roundtwo/vimtopoeia/surge_train.h5"
VAL_H5="/scratch/hw3140/roundtwo/vimtopoeia/surge_val.h5"
TEST_H5="/scratch/hw3140/roundtwo/vimtopoeia/surge_test.h5"
MIT_IR_DIR="/scratch/hw3140/vimtopoeia/datasets/mit_ir_survey"
VOCAL_DIR="/scratch/hw3140/vimtopoeia/datasets/vimsketch_synth_vocals"
CHECKPOINTS_DIR="./M2/pretrain/checkpoints"

echo "Starting M2 Phase 1 pre-training on HPC..."
echo "Train H5: $TRAIN_H5"
echo "Val H5: $VAL_H5"
echo "MIT IR: $MIT_IR_DIR"
echo "Vocals: $VOCAL_DIR"
echo "Checkpoints: $CHECKPOINTS_DIR"

python -u M2/pretrain/train_phase1.py \
    --train_h5 "$TRAIN_H5" \
    --val_h5 "$VAL_H5" \
    --test_h5 "$TEST_H5" \
    --mit_ir_dir "$MIT_IR_DIR" \
    --vocal_dir "$VOCAL_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --batch_size 64 \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --num_workers 4

echo "Job finished at $(date)"
