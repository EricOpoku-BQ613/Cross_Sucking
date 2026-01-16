#!/bin/bash
#SBATCH --job-name=E02_cpu_smoke
#SBATCH --account=acf-utk00
#SBATCH --partition=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/E02_cpu_%j.out
#SBATCH --error=logs/E02_cpu_%j.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd ~/projects/cross_sucking
source ~/miniforge3/etc/profile.d/conda.sh
conda activate crosssuck

# force CPU if your script supports it; otherwise set in config device: "cpu"
python scripts/train_supervised.py \
  --config configs/E02_simple_augmentation_final.yaml \
  --output runs/ablation_E02_cpu
