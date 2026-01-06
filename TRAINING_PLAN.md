# Cross-Sucking Detection: Training Plan
## Intra-Video Split Experiments

**Date:** 2026-01-05
**Status:** Ready for cluster training
**Goal:** Eliminate video confounding and improve test set generalization

---

## 🎯 Problem Summary

**Video Confounding Diagnosed:**
- OLD split: 0 videos overlap between train/val
- Model learned VIDEO features (lighting, camera angle) instead of BEHAVIOR
- **Evidence:**
  - Validation tail recall: 60%
  - **Test tail recall: 17%** (catastrophic failure on unseen videos)
  - Probability clustering: 16 values appearing 3+ times
  - ROC AUC: 0.35 (worse than random)

**Root Cause:** Model memorized which videos contain tail events based on camera-specific features.

---

## ✅ Solution: Intra-Video Split

**New splits created:**
- **76 out of 87 videos** appear in BOTH train and val
- **23 videos** have tail events in BOTH splits
- Forces model to distinguish behavior WITHIN same video contexts

**Files:**
- Standard: `train_intravideo_20260105_162028.csv` (770 events, 14.5% tail)
- Boosted: `train_intravideo_20260105_162114.csv` (940 events, 30% tail, oversampled)

---

## 🚀 Experiments to Run on Cluster

### Experiment 1: Baseline (Old Split) - FOR COMPARISON
**Config:** `configs/train_binary_baseline_old_split.yaml`
```bash
python scripts/train_supervised.py --config configs/train_binary_baseline_old_split.yaml
```

**Expected:**
- ✅ High validation metrics (~60% tail recall)
- ❌ Low test metrics (~17% tail recall)
- ❌ Probability clustering

**Purpose:** Demonstrate the confounding problem

---

### Experiment 2: Intra-Video Split (Standard)
**Config:** `configs/train_binary_v3_intravideo.yaml`
```bash
python scripts/train_supervised.py --config configs/train_binary_v3_intravideo.yaml
```

**Key settings:**
- Intra-video split (76 videos overlap)
- Heavy augmentation (TrivialAugment)
- Dropout: 0.5
- LR: 0.00005 (lower)
- Weight decay: 0.05 (higher)
- Label smoothing: 0.15
- **Metric:** `f1_tail` (select best by tail performance)
- **TTA:** 5 clips per event during validation

**Expected:**
- ⚠️ Lower validation metrics initially (more honest evaluation)
- ✅ **Much better test metrics** (30-50% tail recall)
- ✅ No probability clustering
- ✅ Better calibration

---

### Experiment 3: Intra-Video Split (Boosted)
**Config:** `configs/train_binary_v3_intravideo_boosted.yaml`
```bash
python scripts/train_supervised.py --config configs/train_binary_v3_intravideo_boosted.yaml
```

**Difference from Exp 2:**
- Uses **boosted split** with 30% tail (oversampled 170 train, 40 val)
- Same heavy augmentation and regularization

**Expected:**
- ✅ Better tail recall than Experiment 2
- ⚠️ Potential overfitting to tail events (monitor)

---

## 📊 Evaluation Protocol

### 1. During Training
Monitor these metrics in `training_history.json`:
- `val_tail_recall` - Critical metric
- `val_tail_f1` - Model selection criterion
- `val_macro_f1` - Overall balance
- `train_loss` vs `val_loss` - Check for overfitting

**Red flags:**
- Train loss << Val loss (overfitting)
- Validation tail recall plateaus < 40%

### 2. After Training
Run inference on test set with TTA:

```bash
# For each experiment
python scripts/inference_aggregated.py \
    --checkpoint runs/sup_binary_v3_intravideo/best.ckpt \
    --manifest data/manifests/test.csv \
    --output runs/sup_binary_v3_intravideo/test_predictions.csv \
    --config configs/train_binary_v3_intravideo.yaml \
    --clips-per-event 5 \
    --aggregation mean
```

### 3. Compute Test Metrics
```bash
python scripts/eval_metrics.py \
    --predictions runs/sup_binary_v3_intravideo/test_predictions.csv \
    --output runs/sup_binary_v3_intravideo/test_metrics.json
```

### 4. Check for Video Confounding (Diagnostic)
```bash
python scripts/check_video_confound.py \
    --train data/manifests/train_intravideo_20260105_162028.csv \
    --val data/manifests/val_intravideo_20260105_162028.csv \
    --predictions runs/sup_binary_v3_intravideo/predictions_agg.csv
```

**Look for:**
- ✅ No probability clustering (diverse predictions)
- ✅ Similar predictions for events from different videos
- ❌ Identical predictions only if events are truly similar

---

## 📈 Expected Results Comparison

| Metric | Baseline (Old) | Intra-Video | Intra-Video Boosted |
|--------|---------------|-------------|---------------------|
| **Val Tail Recall** | 60% | 35-45% | 45-55% |
| **Test Tail Recall** | **17%** ❌ | **35-50%** ✅ | **40-55%** ✅ |
| Prob Clustering | Many | None | None |
| ROC AUC (test) | 0.35 | 0.65-0.75 | 0.70-0.80 |

**Key insight:** Validation will look "worse" but test will be MUCH better!

---

## 🔧 Training Infrastructure

### Requirements
- GPU with 8GB+ VRAM (for batch_size=4)
- Python 3.12
- PyTorch with CUDA
- Dependencies in `requirements.txt`

### Cluster Setup
```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Verify dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Parallel Execution
Run all 3 experiments in parallel (if you have multiple GPUs):

```bash
# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 python scripts/train_supervised.py \
    --config configs/train_binary_baseline_old_split.yaml &

# GPU 1: Standard intra-video
CUDA_VISIBLE_DEVICES=1 python scripts/train_supervised.py \
    --config configs/train_binary_v3_intravideo.yaml &

# GPU 2: Boosted intra-video
CUDA_VISIBLE_DEVICES=2 python scripts/train_supervised.py \
    --config configs/train_binary_v3_intravideo_boosted.yaml &

wait
echo "All experiments complete!"
```

---

## 📁 Output Structure

Each experiment creates:
```
runs/<experiment_name>/
├── best.ckpt                    # Best model by f1_tail
├── last.ckpt                    # Final epoch
├── training_history.json        # Loss/metrics per epoch
├── label_map.json              # Class mapping
├── manifests_used.txt          # Manifests used
├── predictions_agg.csv         # Val predictions (with TTA)
└── metrics/
    └── metrics.json            # Final validation metrics
```

---

## ✅ Success Criteria

**Experiment is successful if:**
1. ✅ Test tail recall > 35% (vs 17% baseline)
2. ✅ No probability clustering in test predictions
3. ✅ ROC AUC > 0.60 (vs 0.35 baseline)
4. ✅ Training converges smoothly (no NaN, no collapse)

**Best experiment wins if:**
- Highest test tail F1
- Good calibration (ECE < 0.2)
- Robust to different random seeds

---

## 🐛 Troubleshooting

### If training fails with OOM:
- Reduce `batch_size` from 4 to 2
- Reduce `num_workers` from 4 to 2
- Disable AMP: set `amp: false`

### If model doesn't converge:
- Check learning rate schedule (should start at 0.00005)
- Verify balanced sampling is working (check logs)
- Reduce augmentation strength if loss oscillates

### If tail recall is still low:
- Try boosted split
- Increase `clips_per_event` to 7 or 10 for TTA
- Consider ensemble of multiple runs

---

## 📞 Next Steps After Results

1. **Compare all 3 experiments** - create table of test metrics
2. **Analyze failure cases** - which tail events are still missed?
3. **Error analysis** - are failures video-specific or behavior-specific?
4. **Consider ensemble** - combine predictions from multiple models
5. **Deploy best model** - create inference pipeline for new videos

---

## 📝 Notes

- All configs use **TTA (5 clips)** during validation for stable metrics
- All configs use **balanced sampling** to handle 14% tail imbalance
- All configs use **focal loss** with gamma=2.0 for minority class focus
- **Selection metric:** `f1_tail` (not `macro_f1`) - prioritizes tail detection

**TTA is already configured in all configs** - no additional setup needed!
