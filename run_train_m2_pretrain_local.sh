#!/bin/bash

# M2 Phase 1 Pre-training Script - LOCAL TESTING ONLY
# Usage: bash run_train_m2_pretrain_local.sh

echo "Running M2 Pre-training in LOCAL mode (MOCK TEST)..."

# Activate local environment
source .venv/bin/activate

# Local paths (adjust these to your actual local paths)
TRAIN_H5="/Users/wanghuixi/vimtopoeia_m1/data/surge_train_40k.h5"
VAL_H5="/Users/wanghuixi/vimtopoeia_m1/data/surge_validation.h5"
TEST_H5="/Users/wanghuixi/vimtopoeia_m1/data/surge_test.h5"
MIT_IR_DIR="./MIT_IR"
VOCAL_DIR="./vimsketch_synth/vocal"
CHECKPOINTS_DIR="./M2/pretrain/checkpoints"

echo "Starting M2 Phase 1 pre-training locally (mock test)..."
echo "Train H5: $TRAIN_H5"
echo "Val H5: $VAL_H5"
echo "MIT IR: $MIT_IR_DIR"
echo "Vocals: $VOCAL_DIR"
echo "Checkpoints: $CHECKPOINTS_DIR"

python M2/pretrain/train_phase1.py \
    --train_h5 "$TRAIN_H5" \
    --val_h5 "$VAL_H5" \
    --test_h5 "$TEST_H5" \
    --mit_ir_dir "$MIT_IR_DIR" \
    --vocal_dir "$VOCAL_DIR" \
    --checkpoints_dir "$CHECKPOINTS_DIR" \
    --batch_size 4 \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --num_workers 2

echo "Training finished at $(date)"
