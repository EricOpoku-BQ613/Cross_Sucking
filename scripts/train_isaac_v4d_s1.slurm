#!/bin/bash
#SBATCH --job-name=cs_v4d_s1
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/slurm_v4d_s1_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/slurm_v4d_s1_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eopoku2@utk.edu
#
# E1d Stage 1: frozen backbone, head calibration only (25 epochs, fast)
# After this completes, submit train_isaac_v4d_s2.slurm

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
echo "Clip count: $(ls /lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4/*.mp4 2>/dev/null | wc -l)"

LAST_CKPT="/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4d_s1_mvit/last.ckpt"
echo "===== Stage 1 training start: $(date) ====="
if [ -f "$LAST_CKPT" ]; then
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4d_s1_mvit_isaac.yaml \
        --resume "$LAST_CKPT"
else
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4d_s1_mvit_isaac.yaml
fi
echo "===== Stage 1 done: $(date) ====="
echo "Next: sbatch scripts/train_isaac_v4d_s2.slurm"
