#!/bin/bash
#SBATCH -A acf-utk0011
#SBATCH -p campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -J crosssuck_train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

cd ~/projects/cross_sucking
source ~/miniforge3/etc/profile.d/conda.sh
conda activate crosssuck

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/train_supervised.py --config configs/train_binary_v3_intravideo.yaml
