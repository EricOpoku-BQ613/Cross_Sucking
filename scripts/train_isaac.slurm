#!/bin/bash
#SBATCH --job-name=cross_sucking_mvit
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/slurm_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/slurm_%j.log
#
# ---- GPU & resource request ----
# ISAAC-NG partitions (use whichever you have access to):
#   campus-gpu   -- campus allocation (most common)
#   gpu          -- general GPU partition
#   isaac-gpu    -- ISAAC-specific (check with: sinfo -s)
#
#SBATCH --partition=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10         # 8 workers + 2 headroom
#SBATCH --gres=gpu:1               # 1 GPU (A100, V100, or whatever is available)
#SBATCH --mem=64G                  # 64GB RAM — Lustre I/O + num_workers=8 needs ~32GB
#SBATCH --time=08:00:00            # 8 hours: 50 epochs should finish in 2-4hr on A100
#
# ---- Email notifications ----
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eopoku2@utk.edu
#
# ---- Notes ----
# If you get a V100 (16GB VRAM), reduce batch_size to 8:
#   Edit configs/train_binary_v4_mvit_isaac.yaml: batch_size: 8
# Or pass batch_size override on CLI if supported.
#
# To submit:   sbatch scripts/train_isaac.slurm
# To cancel:   scancel <job_id>
# To monitor:  squeue -u $USER
# To check GPU: squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R %b"

set -e

echo "===== Job info ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "Working:  $(pwd)"
echo ""

# ---------------------------------------------------------------------------
# 1. Load modules (must match what was used in setup_isaac.sh)
# ---------------------------------------------------------------------------
module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2023.09-0

# ---------------------------------------------------------------------------
# 2. Activate conda environment
# ---------------------------------------------------------------------------
source activate cross_sucking
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---------------------------------------------------------------------------
# 3. Verify GPU
# ---------------------------------------------------------------------------
echo ""
echo "===== GPU info ====="
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU: ', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# ---------------------------------------------------------------------------
# 4. Verify data
# ---------------------------------------------------------------------------
echo ""
echo "===== Data check ====="
CLIP_DIR="/lustre/isaac24/scratch/eopoku2/clips_v4"
echo "Clips dir: $CLIP_DIR"
echo "Clip count: $(ls $CLIP_DIR/*.mp4 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
# 5. Create output directory
# ---------------------------------------------------------------------------
mkdir -p /lustre/isaac24/scratch/eopoku2/runs

# ---------------------------------------------------------------------------
# 6. Change to repo directory
#    Adjust REPO_DIR to wherever you cloned the code on ISAAC
# ---------------------------------------------------------------------------
REPO_DIR="/lustre/isaac24/scratch/eopoku2/cross_sucking"
# OR if cloned to home: REPO_DIR="$HOME/cross_sucking"
cd $REPO_DIR
echo ""
echo "Repo dir: $REPO_DIR"

# ---------------------------------------------------------------------------
# 7. Run training
# ---------------------------------------------------------------------------
echo ""
echo "===== Training start: $(date) ====="
python -u scripts/train_supervised.py \
    --config configs/train_binary_v4_mvit_isaac.yaml

echo ""
echo "===== Training done: $(date) ====="
