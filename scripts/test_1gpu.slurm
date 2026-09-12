#!/bin/bash
#SBATCH --job-name=c4test1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu-51,gpu-56,gpu-05
#SBATCH --output=logs/test_1gpu.out
#SBATCH --error=logs/test_1gpu.out

# Single-GPU functional test (paper: 128 frames; spatial grid: pinned T1 contract).
# One process, no DDP. Confirms a clean training step without OOM.
set -e
REPO=/path/to/CONNECT-4
cd "$REPO"; mkdir -p logs
source ~/.bashrc
conda activate mica
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[test] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date +%H:%M:%S)"
python -m training.train --config configs/connect4.yaml
