#!/bin/bash

# ============================================================
# Usage:
#   sbatch slurm_train.sh cnn #run cnn
#   sbatch slurm_train.sh cnn #run transformer
#   sbatch slurm_train.sh cnn #run svm
#   sbatch slurm_train.sh cnn #run bi_lstm

# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Default task if none provided as argument


echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# Run from the code directory
cd "$SLURM_SUBMIT_DIR"

source venv/bin/activate

# Run training
uv run python train.py --model "$TASK"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
