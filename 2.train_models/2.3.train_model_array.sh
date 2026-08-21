#!/usr/bin/env bash

# Submit one architecture at a time, for example:
# sbatch 2.3.train_model_array.sh UNet
# sbatch 2.3.train_model_array.sh wGAN
# sbatch 2.3.train_model_array.sh UNeXt

#SBATCH --job-name=model_train
#SBATCH --array=0-24%5

# change to appropriate partition and qos for your cluster

#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a100-40gb:1

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail


# ---------------------------------------------------------------------------
# Project / environment
# ---------------------------------------------------------------------------

# change to project path on directory
cd /path/to/project/2.train_models/nbconverted

# Activate environment as appropriate, e.g.
# source .venv/bin/activate


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

ARCHITECTURE="${1:-UNet}"
case "$ARCHITECTURE" in
    UNet|wGAN|UNeXt) ;;
    *)
        echo "Unsupported architecture: ${ARCHITECTURE}. Expected UNet, wGAN, or UNeXt." >&2
        exit 2
        ;;
esac

INPUT_CHANNEL="OrigBrightfield"
ON_HPC=True
SUBSET_TRAINING=False

TARGET_CHANNELS=(
    "OrigDNA"
    "OrigRNA"
    "OrigAGP"
    "OrigMito"
    "OrigER"
)

CONFLUENCES=(
    1000
    2000
    4000
    8000
    12000
)

N_CONFLUENCES=${#CONFLUENCES[@]}

target_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFLUENCES ))
confluence_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFLUENCES ))

TARGET_CHANNEL="${TARGET_CHANNELS[$target_idx]}"
CONFLUENCE="${CONFLUENCES[$confluence_idx]}"


# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------

echo "============================================================"
echo "JOB_ID:          ${SLURM_JOB_ID}"
echo "ARRAY_JOB_ID:    ${SLURM_ARRAY_JOB_ID}"
echo "ARRAY_TASK_ID:   ${SLURM_ARRAY_TASK_ID}"
echo "HOST:            $(hostname)"
echo
echo "ARCHITECTURE:    ${ARCHITECTURE}"
echo "SUBSET_TRAINING: ${SUBSET_TRAINING}"
echo "INPUT_CHANNEL:   ${INPUT_CHANNEL}"
echo "TARGET_CHANNEL:  ${TARGET_CHANNEL}"
echo "CONFLUENCE:      ${CONFLUENCE}"
echo "============================================================"


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

ON_HPC="$ON_HPC" \
SUBSET_TRAINING="$SUBSET_TRAINING" \
ARCHITECTURE="$ARCHITECTURE" \
INPUT_CHANNEL="$INPUT_CHANNEL" \
TARGET_CHANNEL="$TARGET_CHANNEL" \
CONFLUENCE="$CONFLUENCE" \
python 2.2.train_unet.py
