#!/bin/bash
#SBATCH --job-name=reg_miss
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/regmiss_%A_%a.out
#SBATCH --error=logs/regmiss_%A_%a.err

# Register the valid training scans that the first array missed (1114 of them).
set -e
REPO=/path/to/CONNECT-4
cd "$REPO"; mkdir -p logs
source ~/.bashrc
conda activate mica
export PYTHONPATH="$REPO:$PYTHONPATH"
export OMP_NUM_THREADS=4

START=$((SLURM_ARRAY_TASK_ID * 100))
STOP=$((START + 100))
python -m preprocessing.register_fmri_mni \
    --list /path/to/data/mni_xfm_missing.txt \
    --start "$START" --stop "$STOP"
