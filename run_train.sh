#!/bin/bash
#SBATCH --job-name=vimtopoeia_ast
#SBATCH --output=sbatch/vimtopoeia_ast_%j.out
#SBATCH --error=sbatch/vimtopoeia_ast_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4            
#SBATCH --mem=64G                     
#SBATCH --time=12:00:00      

# Usage: 
#   For HPC (SLURM):  sbatch run_train.sh
#   For local:        bash run_train.sh local

MODE=${1:-hpc}  # Default to 'hpc' if no argument provided

if [ "$MODE" == "hpc" ]; then
    echo "Running in HPC mode with SLURM..."
    
        

    module purge
    
    echo "Job started on $(hostname) at $(date)"
    echo "Loading modules..."
    
    source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
    conda activate venv
    cd /scratch/hw3140/roundtwo/vimtopoeia
    
    export HF_HUB_OFFLINE=1
    export HF_HOME=/scratch/hw3140/ast_model_local
    
    # HPC paths
    H5_PATH="./dataset_4k_pair.h5"
    IR_DIR="/scratch/hw3140/vimtopoeia/datasets/vimsketch_synth_vocals"
    CHECKPOINTS_DIR="./checkpoints/v4"
    AST_MODEL_PATH="/scratch/hw3140/ast_model_local"
    
    echo "Starting training on HPC..."
    python -u model_training/train.py \
        --h5_path "$H5_PATH" \
        --ir_dir "$IR_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --ast_model_path "$AST_MODEL_PATH"
    
    echo "Job finished at $(date)"

elif [ "$MODE" == "local" ]; then
    echo "Running in LOCAL mode..."
    
    # Activate local environment
    source .venv/bin/activate
    
    # Local paths
    H5_PATH="./dataset_4k_pair.h5"
    IR_DIR="/Users/wanghuixi/vimtopoeia/V1_outputs/v1_input_vocals"  # Adjust to your local IR directory
    CHECKPOINTS_DIR="./model_training/checkpoints"
    AST_MODEL_PATH="/Users/wanghuixi/ast_model_local"  # Adjust to your local AST model path
    
    echo "Starting training locally..."
    python model_training/train.py \
        --h5_path "$H5_PATH" \
        --ir_dir "$IR_DIR" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --ast_model_path "$AST_MODEL_PATH"
    
    echo "Training finished at $(date)"

else
    echo "Unknown mode: $MODE"
    echo "Usage: bash run_train.sh [hpc|local]"
    exit 1
fi