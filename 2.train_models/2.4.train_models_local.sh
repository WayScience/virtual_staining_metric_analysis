#!/usr/bin/env bash

# Run training of 3 architectures sequentially on a configurable local condition.
# Before running this script, please ensure you have started a local mlflow logging server.
# This can be done in any terminal, with a virtual environment where mlflow is installed,
# by running `mlflow server --host 127.0.0.1 --port 5000`
# Please ensure that the server continues to run while training.

# Note that script 2.2.train_unet.py is hard-coded to log to http://127.0.0.1:5000,
# Should you need to change the host or port, please modify the script accordingly.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash 2.4.train_models_local.sh [OPTIONS]

Options:
  --subset                 Run 300 samples for 30 epochs (default).
  --full                   Run 2,900 samples for 300 epochs.
  --input-channel CHANNEL  Input channel (default: OrigBrightfield).
  --target-channel CHANNEL Target channel (default: OrigDNA).
  --confluence VALUE       Seeding density/condition (default: 1000).
  -h, --help               Show this help message.
EOF
}

# Default to the short local test train on subsetted data and smaller epoch number.
# The options below enables the switch to the full run.
SUBSET_TRAINING=True
INPUT_CHANNEL="OrigBrightfield"
TARGET_CHANNEL="OrigDNA"
CONFLUENCE=1000

# This launcher supports the local-tracking mode of the notebook only; 
# the HPC launcher configures its own paths and file-based MLflow tracking separately.
ON_HPC=False

# Parse options from left to right, consuming a value after each value-bearing
# option and rejecting missing values or unknown options.
while (( $# > 0 )); do
    case "$1" in
        --subset)
        # explicitly call script default to run the short local test train on subsetted data
            SUBSET_TRAINING=True
            shift
            ;;
        --full)
        # override the default to run the full production training on all data
            SUBSET_TRAINING=False
            shift
            ;;
        # The following three options require a value after the option name.
        # allows overriding the default input channel, target channel, and confluence for local training.
        # changing of target channel and confluence is useful for training select models,
        # without having to run the full grid.
        --input-channel|--target-channel|--confluence)
            if (( $# < 2 )); then
                echo "Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            case "$1" in
                --input-channel) INPUT_CHANNEL="$2" ;;
                --target-channel) TARGET_CHANNEL="$2" ;;
                --confluence) CONFLUENCE="$2" ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$CONFLUENCE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Confluence must be a positive integer, but received: ${CONFLUENCE}" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/nbconverted"

ARCHITECTURES=(
    "UNet"
    "wGAN"
    "UNeXt"
)

for architecture in "${ARCHITECTURES[@]}"; do
    echo "============================================================"
    echo "ARCHITECTURE:    ${architecture}"
    echo "SUBSET_TRAINING: ${SUBSET_TRAINING}"
    echo "INPUT_CHANNEL:   ${INPUT_CHANNEL}"
    echo "TARGET_CHANNEL:  ${TARGET_CHANNEL}"
    echo "CONFLUENCE:      ${CONFLUENCE}"
    echo "============================================================"

    ON_HPC="$ON_HPC" \
    SUBSET_TRAINING="$SUBSET_TRAINING" \
    ARCHITECTURE="$architecture" \
    INPUT_CHANNEL="$INPUT_CHANNEL" \
    TARGET_CHANNEL="$TARGET_CHANNEL" \
    CONFLUENCE="$CONFLUENCE" \
    python 2.2.train_unet.py
done
