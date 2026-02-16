#!/bin/bash
#SBATCH --job-name=cross_sucking_feral
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/feral_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/feral_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eopoku2@utk.edu
#
# FERAL (V-JEPA2) training on cross-sucking clips.
# Prerequisite steps (run once before submitting this job):
#
#   1. Resize clips to 256x256:
#      sbatch scripts/resize_clips_feral.sh
#
#   2. Prepare FERAL JSON annotations (from repo root, cross_sucking conda env):
#      conda activate cross_sucking
#      python scripts/prepare_feral_json.py \
#          --clip-dir /lustre/isaac24/scratch/eopoku2/clips_feral \
#          --output   data/manifests/feral_annotations.json
#
#   3. Clone FERAL repo into scratch (once):
#      cd /lustre/isaac24/scratch/eopoku2
#      git clone https://github.com/Skovorp/feral.git feral_repo
#      conda activate cross_sucking
#      pip install -r /lustre/isaac24/scratch/eopoku2/feral_repo/requirements.txt
#
# To submit:  sbatch scripts/train_feral.slurm
# To watch:   tail -f /lustre/isaac24/scratch/eopoku2/runs/feral_<job_id>.log

set -e

mkdir -p /lustre/isaac24/scratch/eopoku2/runs

echo "===== FERAL Job info ====="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo ""

# ---------------------------------------------------------------------------
# 1. Activate conda
# ---------------------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate cross_sucking
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ---------------------------------------------------------------------------
# 2. GPU info
# ---------------------------------------------------------------------------
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
CLIP_DIR="/lustre/isaac24/scratch/eopoku2/clips_feral"
ANNO_JSON="/lustre/isaac24/scratch/eopoku2/cross_sucking/data/manifests/feral_annotations.json"
echo ""
echo "Clips (256x256): $(ls $CLIP_DIR/*.mp4 2>/dev/null | wc -l)"
echo "Annotations:     $ANNO_JSON"
if [ ! -f "$ANNO_JSON" ]; then
    echo "ERROR: feral_annotations.json not found. Run prepare_feral_json.py first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Change to FERAL repo
# ---------------------------------------------------------------------------
FERAL_DIR="/lustre/isaac24/scratch/eopoku2/feral_repo"
REPO_DIR="/lustre/isaac24/scratch/eopoku2/cross_sucking"
cd $FERAL_DIR
echo "FERAL dir: $FERAL_DIR"

# ---------------------------------------------------------------------------
# 5. Run FERAL training
#    Wandb offline — compute nodes may not have internet
# ---------------------------------------------------------------------------
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

echo ""
echo "===== FERAL training start: $(date) ====="

python run.py \
    "$CLIP_DIR" \
    "$ANNO_JSON" \
    --config "$REPO_DIR/configs/feral_cross_sucking.yaml"

echo ""
echo "===== FERAL training done: $(date) ====="
echo ""
echo "Results saved to: $FERAL_DIR/answers/"
echo "Checkpoints in:   $FERAL_DIR/  (look for *.pt files)"
