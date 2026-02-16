#!/bin/bash
# setup_isaac.sh
# One-time environment setup on UTK ISAAC-NG.
# Run this from the repo root BEFORE submitting the SLURM job.
#
# Usage:
#   cd /path/to/cross_sucking   # wherever you cloned the repo
#   bash scripts/setup_isaac.sh
#
# What it does:
#   1. Loads CUDA and conda modules
#   2. Creates conda env "cross_sucking" with Python 3.10
#   3. Installs PyTorch 2.1 + CUDA 12.1
#   4. Installs project requirements
#   5. Installs the project package in editable mode

set -e

echo "===== ISAAC-NG setup for cross-sucking project ====="
echo "Working dir: $(pwd)"
echo ""

# ---------------------------------------------------------------------------
# 1. Load system modules (verified on ISAAC-NG 2026-02)
# ---------------------------------------------------------------------------
echo "[1/5] Loading modules..."
module purge
module load cuda/12.1.1-binary
module load anaconda3/2024.06

echo "CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo ""

# ---------------------------------------------------------------------------
# 2. Create conda environment
#    Skip if already exists
# ---------------------------------------------------------------------------
echo "[2/5] Creating conda environment..."
if conda env list | grep -q "cross_sucking"; then
    echo "  Env 'cross_sucking' already exists — skipping creation."
else
    conda create -n cross_sucking python=3.10 -y
    echo "  Created env 'cross_sucking'."
fi

# Activate — use 'conda activate' not 'source activate' for reliable env switching
eval "$(conda shell.bash hook)"
conda activate cross_sucking
echo "  Active env: $CONDA_DEFAULT_ENV"
echo "  Python:     $(which python)"
echo ""

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.1 (verbose — show errors if any)
# ---------------------------------------------------------------------------
echo "[3/5] Installing PyTorch..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify torch installed before continuing
python -c "import torch; print('  torch OK:', torch.__version__)"

# Pin numpy < 2.0 (torch 2.1 is incompatible with numpy 2.x)
pip install "numpy<2.0"
python -c "import numpy; print('  numpy OK:', numpy.__version__)"

# ---------------------------------------------------------------------------
# 4. Install project requirements (no torch — already installed above)
# ---------------------------------------------------------------------------
echo "[4/5] Installing requirements..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 5. Install project package
# ---------------------------------------------------------------------------
echo "[5/5] Installing project package..."
pip install -e .

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "===== Verification ====="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
    print('CUDA version:', torch.version.cuda)
"

echo ""
echo "===== Setup complete ====="
echo ""
echo "Next steps:"
echo "  1. Verify clips exist:  ls /lustre/isaac24/scratch/eopoku2/clips_v4/ | head -5"
echo "  2. Submit job:          sbatch scripts/train_isaac.slurm"
echo "  3. Monitor job:         squeue -u \$USER"
echo "  4. Watch output:        tail -f /lustre/isaac24/scratch/eopoku2/runs/slurm_*.log"
