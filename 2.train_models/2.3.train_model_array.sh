#!/usr/bin/env bash

# Train one model for every target-channel/seeding-density combination by
# making multiple slurm job submissions, each training a single model.
#
# Submit one architecture at a time, for example (from 2.train_models/):
# sbatch 2.3.train_model_array.sh UNet
# sbatch 2.3.train_model_array.sh wGAN
# sbatch 2.3.train_model_array.sh UNeXt

# Use a recognizable base name for every task in the array.
#SBATCH --job-name=model_train
# Create task IDs 0-24 and run no more than five tasks concurrently.
# Change the upper bound of the array and the concurrency limit as permitted by the cluster. 
#SBATCH --array=0-24%5

# Select the cluster's A100 GPU partition and its normal-priority GPU QoS.
# Change these cluster-specific values when running elsewhere.
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
# Limit each array task to 24 hours on one node with eight CPU tasks and one
# 80 GB A100 GPU.
# All architectures except UNeXt fit on 40GB GPUs, using the 80GB here to accommodate UNeXt.
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a100-80gb:1

# Write separate stdout and stderr files for each array task. %x is the job
# name, %A the parent array job ID, and %a the array task ID.
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Exit on the first failed command (-e), reject unset variables (-u), and
# propagate pipeline failures (pipefail) instead of continuing silently.
set -euo pipefail


# ---------------------------------------------------------------------------
# Project / environment
# ---------------------------------------------------------------------------

# Resolve the converted training script relative to this submission script so
# the job works regardless of the directory from which sbatch was invoked.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/nbconverted"

# Activate the project environment here when it is not initialized by the
# cluster module/profile configuration, for example:
# source .venv/bin/activate


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# The training script uses this value to construct the selected generator and
# trainer. Accept only architectures implemented by 2.2.train_unet.py.
ARCHITECTURE="${1:-UNet}"
case "$ARCHITECTURE" in
    UNet|wGAN|UNeXt) ;;
    *)
        echo "Unsupported architecture: ${ARCHITECTURE}. Expected UNet, wGAN, or UNeXt." >&2
        exit 2
        ;;
esac

# Name of the source microscopy channel supplied to the model.
INPUT_CHANNEL="OrigBrightfield"

# Tell the Python script to use HPC paths and file-based MLflow tracking rather
# than the local MLflow server. TRAIN_ROOT or SCRATCH can further override the
# storage location in the Python script.
ON_HPC=True

# Run full production training: 2,900 samples, 300 epochs, and batch size 32.
# Set to True only for the shorter 300-sample, 30-epoch smoke-test mode.
SUBSET_TRAINING=False

# Fluorescence channels to predict from INPUT_CHANNEL. One model is trained for
# each target in combination with every confluence listed below.
TARGET_CHANNELS=(
    "OrigDNA"
    "OrigRNA"
    "OrigAGP"
    "OrigMito"
    "OrigER"
)

# Cell seeding densities used to filter the training and held-out datasets.
CONFLUENCES=(
    1000
    2000
    4000
    8000
    12000
)

# Decode the one-dimensional Slurm task ID into the Cartesian product of
# TARGET_CHANNELS and CONFLUENCES. Integer division selects the target channel;
# modulo selects the confluence. With five values in each array, IDs 0-4 train
# OrigDNA and IDs 20-24 train OrigER.
N_CONFLUENCES=${#CONFLUENCES[@]}

target_idx=$(( SLURM_ARRAY_TASK_ID / N_CONFLUENCES ))
confluence_idx=$(( SLURM_ARRAY_TASK_ID % N_CONFLUENCES ))

TARGET_CHANNEL="${TARGET_CHANNELS[$target_idx]}"
CONFLUENCE="${CONFLUENCES[$confluence_idx]}"


# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------

# Record both Slurm metadata and model parameters in each task's log so that a
# failed or completed run can be associated with its exact configuration.
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

# Export this task's configuration only for the training process. The Python
# script validates these environment variables, filters data by CONFLUENCE,
# builds INPUT_CHANNEL -> TARGET_CHANNEL datasets, and starts the selected
# architecture's training run.
ON_HPC="$ON_HPC" \
SUBSET_TRAINING="$SUBSET_TRAINING" \
ARCHITECTURE="$ARCHITECTURE" \
INPUT_CHANNEL="$INPUT_CHANNEL" \
TARGET_CHANNEL="$TARGET_CHANNEL" \
CONFLUENCE="$CONFLUENCE" \
python 2.2.train_unet.py
