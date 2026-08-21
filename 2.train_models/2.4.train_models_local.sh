#!/usr/bin/env bash

# Run the three example architectures sequentially on the local configuration.
# Start the local MLflow server expected by 2.2.train_unet.py before running.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/nbconverted"

INPUT_CHANNEL="OrigBrightfield"
TARGET_CHANNEL="OrigDNA"
CONFLUENCE=1000
ON_HPC=False

ARCHITECTURES=(
    "UNet"
    "wGAN"
    "UNeXt"
)

for architecture in "${ARCHITECTURES[@]}"; do
    echo "============================================================"
    echo "ARCHITECTURE:    ${architecture}"
    echo "INPUT_CHANNEL:   ${INPUT_CHANNEL}"
    echo "TARGET_CHANNEL:  ${TARGET_CHANNEL}"
    echo "CONFLUENCE:      ${CONFLUENCE}"
    echo "============================================================"

    ON_HPC="$ON_HPC" \
    ARCHITECTURE="$architecture" \
    INPUT_CHANNEL="$INPUT_CHANNEL" \
    TARGET_CHANNEL="$TARGET_CHANNEL" \
    CONFLUENCE="$CONFLUENCE" \
    python 2.2.train_unet.py
done
