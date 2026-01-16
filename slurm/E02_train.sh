#!/bin/bash
#SBATCH --job-name=E02_simple_aug
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/E02_%j.out
#SBATCH --error=logs/E02_%j.err

set -euo pipefail

mkdir -p logs

# go to where you submitted the job from (your project dir)
cd "$SLURM_SUBMIT_DIR"

# conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cross_sucking_py310

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== ENV CHECK ==="
which python
python -V

echo "=== CUDA CHECK ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "=== TRAINING START ==="
python scripts/train_supervised.py --config configs/E02_simple_augmentation_final.yaml
