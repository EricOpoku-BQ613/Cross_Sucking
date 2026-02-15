#!/bin/bash
# setup_runpod.sh
# Run once after starting a RunPod instance to set up the environment.
# Assumes: code is in /workspace/cross_sucking, clips are in /workspace/data/clips_v4
#
# Usage:
#   bash scripts/setup_runpod.sh

set -e

echo "===== RunPod setup for cross-sucking project ====="

# 1. Install PyTorch with CUDA 12.1 (matches most RunPod images)
#    Adjust cu121 -> cu124 if your RunPod image uses CUDA 12.4
echo "[1/4] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# 2. Install project dependencies
echo "[2/4] Installing requirements..."
pip install -r requirements.txt -q

# 3. Install the project package in editable mode
echo "[3/4] Installing project package..."
pip install -e . -q

# 4. Verify GPU
echo "[4/4] GPU check..."
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

echo ""
echo "===== Setup complete ====="
echo ""
echo "To start training:"
echo "  python -u scripts/train_supervised.py --config configs/train_binary_v4_mvit_runpod.yaml"
echo ""
echo "NOTE: Update clip_dir in config if your clips are not at data/processed/clips_v4"
