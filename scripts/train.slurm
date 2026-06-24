#!/bin/bash
#SBATCH --job-name=connect4
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu-51,gpu-56,gpu-05  # known nodes whose advertised 4th GPU is dead
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/connect4_%j.out
#SBATCH --error=logs/connect4_%j.err

# CONNECT-4 training launcher — 4x A100-40GB, DistributedDataParallel.
set -e

REPO=/path/to/CONNECT-4
cd "$REPO"
mkdir -p logs checkpoints outputs

source ~/.bashrc
conda activate mica            # torch 2.8 + torch-geometric + monai + transformers

export PYTHONPATH="$REPO:$PYTHONPATH"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM
# torchrun picks a free port; --standalone keeps it single-node.

# Notes:
#  * uses the PRECOMPUTED feature pipeline (no BrainIAC checkpoint needed at run time).
#  * perceptual loss falls back to a model-free proxy unless SLIM-Brain weights are set:
#      export CONNECT4_SLIMBRAIN=/path/to/slimbrain.pt   # optional
#  * if you hit OOM, lower data.num_frames (paper = 128) or models.dit.input_size.

echo "[slurm] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Launch via torch.distributed.run (uses the active env's torch; avoids PATH torchrun shims)
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    -m training.train --config configs/connect4.yaml
