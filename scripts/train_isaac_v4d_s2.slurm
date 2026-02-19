#!/bin/bash
#SBATCH --job-name=cs_v4d_s2
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/slurm_v4d_s2_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/slurm_v4d_s2_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eopoku2@utk.edu
#
# E1d Stage 2: unfrozen backbone at very low LR (2e-6), loads Stage 1 weights
# MUST run after Stage 1 completes.

set -e
echo "===== Job info ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate cross_sucking

REPO_DIR="/lustre/isaac24/scratch/eopoku2/cross_sucking"
cd $REPO_DIR

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

S1_BEST="/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4d_s1_mvit/best.ckpt"
LAST_CKPT="/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4d_s2_mvit/last.ckpt"

if [ ! -f "$S1_BEST" ]; then
    echo "ERROR: Stage 1 best.ckpt not found at $S1_BEST"
    echo "Run train_isaac_v4d_s1.slurm first and wait for it to complete."
    exit 1
fi

echo "===== Stage 2 training start: $(date) ====="
if [ -f "$LAST_CKPT" ]; then
    # Resuming Stage 2 (restores optimizer from Stage 2 last.ckpt)
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4d_s2_mvit_isaac.yaml \
        --resume "$LAST_CKPT"
else
    # Fresh Stage 2 start (loads Stage 1 model weights, fresh optimizer)
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4d_s2_mvit_isaac.yaml \
        --weights-path "$S1_BEST"
fi
echo "===== Stage 2 done: $(date) ====="
