#!/usr/bin/env bash
#SBATCH --job-name=connect4-train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${CONNECT4_CONFIG:?Set CONNECT4_CONFIG to an absolute YAML configuration path}"
: "${CONNECT4_PYTHON:?Set CONNECT4_PYTHON to the isolated Python executable}"
: "${CONNECT4_RUNTIME_IDENTITY:?Set CONNECT4_RUNTIME_IDENTITY to the qualified runtime JSON}"

if [[ "${CONNECT4_CONFIG}" != /* || "${CONNECT4_PYTHON}" != /* || "${CONNECT4_RUNTIME_IDENTITY}" != /* ]]; then
  echo "Configured inputs must be absolute paths" >&2
  exit 64
fi
if [[ ! -r "${CONNECT4_CONFIG}" || ! -x "${CONNECT4_PYTHON}" || ! -r "${CONNECT4_RUNTIME_IDENTITY}" ]]; then
  echo "Configured YAML or Python runtime is unavailable" >&2
  exit 65
fi
if [[ "$(hostname)" == *login* ]]; then
  echo "Training on a login node is forbidden" >&2
  exit 66
fi

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${CONNECT4_PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  -m training.train --config "${CONNECT4_CONFIG}"
