#!/bin/bash
#SBATCH --job-name=reg_mni
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --array=0-47
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reg_%A_%a.out
#SBATCH --error=logs/reg_%A_%a.err

# Offline native-fMRI -> MNI registration, parallelised across the cohort.
# Each array task registers a 100-subject slice; transforms cache to
# /path/to/data/mni_xfm/<sid>.npy . Already-done subjects skip.
set -e
REPO=/path/to/CONNECT-4
cd "$REPO"; mkdir -p logs
source ~/.bashrc
conda activate mica
export PYTHONPATH="$REPO:$PYTHONPATH"
export OMP_NUM_THREADS=4

START=$((SLURM_ARRAY_TASK_ID * 100))
STOP=$((START + 100))
python -m preprocessing.register_fmri_mni --start "$START" --stop "$STOP"
