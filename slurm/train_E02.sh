#!/bin/bash
#SBATCH --job-name=E02_simple_aug
#SBATCH --account=tug2106
#SBATCH --partition=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/E02_%j.out
#SBATCH --error=logs/E02_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd ~/projects/cross_sucking
source ~/miniforge3/etc/profile.d/conda.sh
conda activate crossssuck

python scripts/train_supervised.py --config configs/E02_simple_augmentation_final.yaml
