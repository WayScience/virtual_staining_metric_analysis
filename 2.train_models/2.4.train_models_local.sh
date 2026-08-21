#!/usr/bin/env bash

# Run the three architectures sequentially on a configurable local condition.
# Start the local MLflow server expected by 2.2.train_unet.py before running.

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

SUBSET_TRAINING=True
INPUT_CHANNEL="OrigBrightfield"
TARGET_CHANNEL="OrigDNA"
CONFLUENCE=1000
ON_HPC=False

while (( $# > 0 )); do
    case "$1" in
        --subset)
            SUBSET_TRAINING=True
            shift
            ;;
        --full)
            SUBSET_TRAINING=False
            shift
            ;;
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
