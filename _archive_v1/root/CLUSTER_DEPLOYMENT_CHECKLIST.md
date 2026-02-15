# Cluster Deployment Checklist

**Status:** Ready for ISAAC cluster deployment
**Date:** 2026-01-05
**GPU:** Tesla V100S-PCIE-32GB (32GB VRAM)

---

## Step 1: Clean Up Project (Before Upload)

### Run cleanup script to remove unnecessary files:

```bash
# DRY RUN (see what will be removed, no files deleted)
python cleanup_project.py

# EXECUTE (actually delete files)
python cleanup_project.py --execute
```

### What gets removed:
- Old training runs (runs/)
- Python cache (__pycache__, *.pyc)
- Old reports (data/reports/, analysis/)
- Debug scripts (breakdown_splits.py, check_corrupted.py, etc.)
- Temporary manifests (old splits)
- Virtual environment (venv/, .venv/) - will recreate on cluster
- Git files (.git/, .gitignore) - optional
- Jupyter checkpoints
- Windows batch files (run_training.bat)

### What gets kept:
- All source code (src/, scripts/)
- New intra-video manifests (train/val_intravideo_*.csv)
- Test manifests (test.csv, test_collapsed.csv)
- All 3 experiment configs
- Documentation (TRAINING_PLAN.md, READY_FOR_CLUSTER.md, FIXED_AND_READY.md)
- Raw annotations (data/annotations/)
- Linux training script (run_training.sh)

### Expected space saved:
Approximately 1-3 GB (mostly venv and old runs/)

---

## Step 2: Upload to ISAAC Cluster

### Transfer cleaned project:

```bash
# From your local machine
scp -r cross_sucking/ <your_username>@isaac.orc.gmu.edu:~/

# Or use rsync for faster transfer
rsync -avz --progress cross_sucking/ <your_username>@isaac.orc.gmu.edu:~/cross_sucking/
```

---

## Step 3: Setup on Cluster

### SSH into ISAAC:

```bash
ssh <your_username>@isaac.orc.gmu.edu
cd ~/cross_sucking
```

### Create virtual environment:

```bash
module load python/3.9
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies:

```bash
pip install --upgrade pip
pip install -e .
```

### Verify GPU access:

```bash
nvidia-smi
```

Expected: Tesla V100S-PCIE-32GB, 32GB VRAM

---

## Step 4: Update Configs for Linux

### CRITICAL: Change num_workers in all 3 configs

Edit the following files:
- configs/train_binary_v3_intravideo.yaml
- configs/train_binary_v3_intravideo_boosted.yaml
- configs/train_binary_baseline_old_split.yaml

Change:
```yaml
data:
  num_workers: 0  # Windows setting
```

To:
```yaml
data:
  num_workers: 4  # Linux setting (or 8 for V100S)
```

---

## Step 5: Create SLURM Job Scripts

### Experiment 1: Main (Intra-Video Split, 14% tail)

Create `job_exp1.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=cross_suck_exp1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:V100S:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err

module load python/3.9
source venv/bin/activate

echo "Starting Experiment 1: Intra-Video Split (14% tail)"
python scripts/train_supervised.py --config configs/train_binary_v3_intravideo.yaml

echo "Experiment 1 Complete"
```

### Experiment 2: Boosted (Intra-Video Split, 30% tail)

Create `job_exp2.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=cross_suck_exp2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:V100S:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err

module load python/3.9
source venv/bin/activate

echo "Starting Experiment 2: Intra-Video Split (30% tail)"
python scripts/train_supervised.py --config configs/train_binary_v3_intravideo_boosted.yaml

echo "Experiment 2 Complete"
```

### Experiment 3: Baseline (Old Split, for comparison)

Create `job_exp3.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=cross_suck_exp3
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:V100S:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err

module load python/3.9
source venv/bin/activate

echo "Starting Experiment 3: Baseline (Old Split)"
python scripts/train_supervised.py --config configs/train_binary_baseline_old_split.yaml

echo "Experiment 3 Complete"
```

---

## Step 6: Submit Jobs

### Create logs directory:

```bash
mkdir -p logs
```

### Submit all 3 jobs in parallel:

```bash
sbatch job_exp1.slurm
sbatch job_exp2.slurm
sbatch job_exp3.slurm
```

### Monitor job status:

```bash
squeue -u $USER
```

### Check job logs (live):

```bash
tail -f logs/exp1_<job_id>.out
tail -f logs/exp2_<job_id>.out
tail -f logs/exp3_<job_id>.out
```

---

## Step 7: Monitor Training Progress

### Check training history:

```bash
# Experiment 1
tail -f runs/sup_binary_v3_intravideo/training_history.json

# Experiment 2
tail -f runs/sup_binary_v3_intravideo_boosted/training_history.json

# Experiment 3
tail -f runs/sup_binary_baseline_old_split/training_history.json
```

### Key metrics to watch:
- `val_tail_recall` - Should increase from ~20% to 35-50%
- `val_tail_f1` - Model selection criterion
- `val_loss` - Should decrease
- `train_loss` vs `val_loss` - Should be similar (no huge gap)

### Expected training time:
- On V100S: ~2-3 hours per experiment (30 epochs)
- ~4-6 minutes per epoch

---

## Step 8: After Training - Evaluate on Test Set

### For each experiment, run inference with TTA:

```bash
# Experiment 1
python scripts/inference_aggregated.py \
    --checkpoint runs/sup_binary_v3_intravideo/best.ckpt \
    --manifest data/manifests/test.csv \
    --output runs/sup_binary_v3_intravideo/test_predictions.csv \
    --config configs/train_binary_v3_intravideo.yaml \
    --clips-per-event 5 \
    --aggregation mean

# Experiment 2
python scripts/inference_aggregated.py \
    --checkpoint runs/sup_binary_v3_intravideo_boosted/best.ckpt \
    --manifest data/manifests/test.csv \
    --output runs/sup_binary_v3_intravideo_boosted/test_predictions.csv \
    --config configs/train_binary_v3_intravideo_boosted.yaml \
    --clips-per-event 5 \
    --aggregation mean

# Experiment 3
python scripts/inference_aggregated.py \
    --checkpoint runs/sup_binary_baseline_old_split/best.ckpt \
    --manifest data/manifests/test.csv \
    --output runs/sup_binary_baseline_old_split/test_predictions.csv \
    --config configs/train_binary_baseline_old_split.yaml \
    --clips-per-event 5 \
    --aggregation mean
```

### Compute test metrics:

```bash
# Experiment 1
python scripts/eval_metrics.py \
    --predictions runs/sup_binary_v3_intravideo/test_predictions.csv \
    --output runs/sup_binary_v3_intravideo/test_metrics.json

# Experiment 2
python scripts/eval_metrics.py \
    --predictions runs/sup_binary_v3_intravideo_boosted/test_predictions.csv \
    --output runs/sup_binary_v3_intravideo_boosted/test_metrics.json

# Experiment 3
python scripts/eval_metrics.py \
    --predictions runs/sup_binary_baseline_old_split/test_predictions.csv \
    --output runs/sup_binary_baseline_old_split/test_metrics.json
```

---

## Step 9: Compare Results

### Run comparison script:

```bash
python scripts/compare_experiments.py
```

### Expected results table:

| Metric | Baseline (Old Split) | Exp1 (14% tail) | Exp2 (30% tail) |
|--------|---------------------|-----------------|-----------------|
| Test Tail Recall | 17% | **>35%** | **>35%** |
| Test ROC AUC | 0.35 | **>0.60** | **>0.60** |
| Prob Clustering | Many | **None** | **None** |
| Test Tail F1 | 0.13 | **>0.30** | **>0.30** |

### Success criteria:
- Tail recall improves from 17% to >35% (2x improvement)
- ROC AUC improves from 0.35 to >0.60 (better than random)
- No probability clustering (no video confound)
- Training completes without errors

---

## Step 10: Download Results

### Download trained models and metrics:

```bash
# From your local machine
scp -r <your_username>@isaac.orc.gmu.edu:~/cross_sucking/runs/ ./results/
```

---

## Troubleshooting

### Out of Memory Error
Reduce batch size in configs:
```yaml
batch_size: 2  # was 4
```

### Job killed unexpectedly
Request more memory or time:
```bash
#SBATCH --mem=64GB  # was 32GB
#SBATCH --time=8:00:00  # was 4:00:00
```

### Training too slow
Increase num_workers:
```yaml
num_workers: 8  # was 4
```

### Tail recall not improving
Try:
1. Use boosted split (Experiment 2)
2. Increase TTA clips: `--clips-per-event 10`
3. Train longer: change `epochs: 40` in config

---

## Final Pre-Upload Checklist

Before uploading to cluster:

- [ ] Ran `python cleanup_project.py` (dry run)
- [ ] Reviewed what will be removed
- [ ] Ran `python cleanup_project.py --execute` (actually deleted files)
- [ ] Verified essential files still exist:
  - [ ] src/ directory
  - [ ] scripts/ directory
  - [ ] configs/ (all 3 YAML files)
  - [ ] data/manifests/ (intra-video splits + test sets)
  - [ ] data/annotations/
  - [ ] run_training.sh
  - [ ] Documentation files

After upload to cluster:

- [ ] Created virtual environment
- [ ] Installed dependencies
- [ ] Changed num_workers from 0 to 4 in all configs
- [ ] Created SLURM job scripts
- [ ] Created logs/ directory
- [ ] Verified GPU access (nvidia-smi)

---

## Quick Reference Commands

```bash
# Clean project
python cleanup_project.py --execute

# Upload to cluster
rsync -avz cross_sucking/ <user>@isaac.orc.gmu.edu:~/cross_sucking/

# Setup on cluster
ssh <user>@isaac.orc.gmu.edu
cd ~/cross_sucking
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Update configs (change num_workers: 0 to num_workers: 4)

# Submit jobs
sbatch job_exp1.slurm
sbatch job_exp2.slurm
sbatch job_exp3.slurm

# Monitor
squeue -u $USER
tail -f logs/exp1_*.out

# After training
python scripts/compare_experiments.py

# Download results
scp -r <user>@isaac.orc.gmu.edu:~/cross_sucking/runs/ ./results/
```

---

You're ready for cluster deployment! Good luck!
