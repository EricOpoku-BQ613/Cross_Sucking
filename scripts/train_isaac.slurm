#!/bin/bash
#SBATCH --job-name=cross_sucking_mvit
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/slurm_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/slurm_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eopoku2@utk.edu
#
# NOTE: Before first submission, create the output dir on the login node:
#   mkdir -p /lustre/isaac24/scratch/eopoku2/runs
#
# To submit:   sbatch scripts/train_isaac.slurm
# To cancel:   scancel <job_id>
# To monitor:  squeue -u $USER
# To watch:    tail -f /lustre/isaac24/scratch/eopoku2/runs/slurm_<job_id>.log

set -e

echo "===== Job info ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo ""

# ---------------------------------------------------------------------------
# 1. Activate conda (miniforge)
# ---------------------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cross_sucking
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---------------------------------------------------------------------------
# 2. Verify GPU
# ---------------------------------------------------------------------------
echo ""
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU: ', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# ---------------------------------------------------------------------------
# 3. Verify data
# ---------------------------------------------------------------------------
CLIP_DIR="/lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4"
echo ""
echo "Clip count: $(ls $CLIP_DIR/*.mp4 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
# 4. Change to repo directory
# ---------------------------------------------------------------------------
REPO_DIR="/lustre/isaac24/scratch/eopoku2/cross_sucking"
cd $REPO_DIR
echo "Repo:     $REPO_DIR"

# ---------------------------------------------------------------------------
# 5. Run training — auto-resumes if last.ckpt exists
# ---------------------------------------------------------------------------
LAST_CKPT="/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4_mvit/last.ckpt"

echo ""
echo "===== Training start: $(date) ====="
if [ -f "$LAST_CKPT" ]; then
    echo "Resuming from $LAST_CKPT"
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4_mvit_isaac.yaml \
        --resume "$LAST_CKPT"
else
    echo "No checkpoint — starting fresh"
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4_mvit_isaac.yaml
fi

echo ""
echo "===== Training done: $(date) ====="
