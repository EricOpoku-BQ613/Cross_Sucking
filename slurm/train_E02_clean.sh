#!/bin/bash
#SBATCH --job-name=E02_aug_clean
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/E02_clean_%j.out
#SBATCH --error=logs/E02_clean_%j.err

# Required when using cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd ~/projects/cross_sucking

source ~/miniforge3/etc/profile.d/conda.sh
conda activate cross_sucking_py310

# Optional resume checkpoint (leave empty for fresh run)
RESUME=""

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
  RESUME_ARGS+=(--resume "$RESUME")
fi

python scripts/train_supervised.py \
  --config configs/E02_simple_augmentation_final.yaml \
  ${RESUME_ARGS[@]}

