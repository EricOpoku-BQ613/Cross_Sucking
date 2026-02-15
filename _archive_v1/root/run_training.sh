#!/bin/bash
# Quick training launcher for Linux/Mac
# Usage: ./run_training.sh [config_name]
# Example: ./run_training.sh train_binary_v3_intravideo

if [ -z "$1" ]; then
    echo "Usage: ./run_training.sh [config_name]"
    echo "Example: ./run_training.sh train_binary_v3_intravideo"
    echo ""
    echo "Available configs:"
    echo "  - train_binary_v3_intravideo          [RECOMMENDED]"
    echo "  - train_binary_v3_intravideo_boosted  [30% tail]"
    echo "  - train_binary_baseline_old_split     [For comparison]"
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_PATH="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

echo "======================================================================"
echo "STARTING TRAINING"
echo "======================================================================"
echo "Config: $CONFIG_PATH"
echo ""

# Use venv python if available, otherwise system python
if [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

$PYTHON scripts/train_supervised.py --config "$CONFIG_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "TRAINING FAILED WITH ERROR CODE: $?"
    echo "======================================================================"
fi
