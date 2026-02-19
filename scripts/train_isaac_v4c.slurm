#!/bin/bash
#SBATCH --job-name=cs_mvit_v4c
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/slurm_v4c_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/slurm_v4c_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eopoku2@utk.edu
#
# E1c: balanced_sampling only (no class weights) + all E1b regularization improvements
#      Hypothesis: E1 collapsed to ear because LR 2e-5 was too high mid-warmup.
#      E1b collapsed to tail because inv_sqrt stacked on balanced_sampling + focal.
#      E1c: single correction (balanced_sampling) + better LR/dropout/wd.
#
# To submit:  sbatch scripts/train_isaac_v4c.slurm
# To cancel:  scancel <job_id>
# To monitor: squeue -u $USER
# To watch:   tail -f /lustre/isaac24/scratch/eopoku2/runs/slurm_v4c_<job_id>.log

set -e

echo "===== Job info ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo ""

# ---------------------------------------------------------------------------
# 1. Activate conda
# ---------------------------------------------------------------------------
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate cross_sucking
echo "Python:  $(which python)"

# ---------------------------------------------------------------------------
# 1b. Ensure torch is installed
# ---------------------------------------------------------------------------
if ! python -c "import torch" 2>/dev/null; then
    echo "torch not found — installing now..."
    REPO_DIR="/lustre/isaac24/scratch/eopoku2/cross_sucking"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121 -q
    pip install "numpy<2.0" -q
    pip install -r "$REPO_DIR/requirements.txt" -q
    pip install -e "$REPO_DIR" -q
fi
python -c "import numpy; v=numpy.__version__; assert int(v.split('.')[0])<2, f'numpy {v} too new'" 2>/dev/null \
    || pip install "numpy<2.0" -q
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
# 5. Run training
# ---------------------------------------------------------------------------
LAST_CKPT="/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4c_mvit/last.ckpt"

echo ""
echo "===== Training start: $(date) ====="
if [ -f "$LAST_CKPT" ]; then
    echo "Resuming from $LAST_CKPT"
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4c_mvit_isaac.yaml \
        --resume "$LAST_CKPT"
else
    echo "No checkpoint — starting fresh"
    python -u scripts/train_supervised.py \
        --config configs/train_binary_v4c_mvit_isaac.yaml
fi

echo ""
echo "===== Training done: $(date) ====="
