# Cross-Sucking Behavior Classification — Lab Progress Report
**Date:** February 17, 2026
**Presenter:** Emmanuel Opoku
**Status:** E1 Baseline Complete → E1b Retraining Submitted

---

## 1. Project Overview

**Goal:** Automatically detect and classify cross-sucking behavior in group-housed dairy calves from 4K surveillance video, using a deep learning video classifier.

**Behaviors of interest:**
- **Ear-sucking** — a calf sucking on the ear of a pen-mate (1,651 events, 88%)
- **Tail-sucking** — a calf sucking on the tail of a pen-mate (225 events, 12%)

**Significance:** Cross-sucking is a welfare indicator in intensive calf rearing. Manual annotation is labor-intensive and subjective; automated detection would enable large-scale, objective welfare monitoring.

---

## 2. Dataset

### 2.1 Source Data
- **6 groups** of calves, recorded on Day 1 and Day 4 post-grouping
- **12 video folders** (Group 1–6 × Day 1, Day 4) on E: drive
- Videos: 4K (3840×2160) @ 15 fps, ~30-minute segments per camera per session
- **4 cameras per group**; behavior visible only on specific indoor cameras (group-dependent)

### 2.2 Annotation Manifest
- **1,897 labeled events** from `final_manifest.xlsx` (Kinematic Cohorts → Group N)
- Annotated fields: group, day, initiator/receiver calf ID, start/end time, duration, behavior, ended-by, pen location
- Duration statistics: mean = 6.9 s, median = 4 s, 75th percentile = 8 s, max = 272 s

### 2.3 Class Distribution
| Behavior | Events | % | Notes |
|----------|--------|---|-------|
| Ear-sucking | 1,651 | 87.0% | Primary class |
| Tail-sucking | 225 | 11.9% | Minority class — key challenge |
| Teat-sucking | 12 | 0.6% | Dropped (too few) |
| Other | 9 | 0.5% | Dropped (ambiguous) |
| **Binary total** | **1,876** | — | Ear vs. tail |

**Imbalance ratio: 7.3:1** (ear:tail)

---

## 3. Data Pipeline

### 3.1 Camera-Corrected Video Linking
Each group records on multiple cameras; cross-sucking is only visible from specific indoor viewpoints. A critical early finding was that the initial linking code selected the wrong camera for several groups.

**Correct camera mapping (confirmed by visual inspection):**
| Group | Primary Camera | Secondary |
|-------|---------------|-----------|
| 1 | Cam 8 | Cam 9 |
| 2 | Cam 16 | Cam 12 |
| 3 | Cam 12 | Cam 14 |
| 4 | Cam 3 | Cam 5, 10 |
| 5 | Cam 1 | Cam 9 |
| 6 | Cam 12 | Cam 14 |

Scripts written to re-link all 1,897 events to the correct camera file (`scripts/relink_cameras.py`) and to verify camera selection by extracting the same event frame from all cameras simultaneously (`scripts/verify_cameras.py`).

### 3.2 Clip Extraction
- **Method:** FFmpeg H.264 CRF 20 (switched from OpenCV mp4v after size analysis)
- **Size comparison:** OpenCV mp4v at 4K → ~367 GB; FFmpeg H.264 CRF 20 → ~60 GB
- **Output:** `data/processed/clips_v4/` on ISAAC-NG Lustre scratch (1,897 clips)
- Each clip is a padded segment around the annotated event, at native 4K resolution

### 3.3 Bout-Grouped Splits
Standard random event-level splitting would cause temporal leakage: 45% of inter-event gaps in the same video are < 30 s, meaning adjacent events look nearly identical.

**Solution — bout-grouped intra-video splits:**
1. Events within 30 s of each other in the same video are grouped into a "bout"
2. Entire bouts are assigned to train or val together (prevents leakage)
3. The OOD test set uses Groups 4 and 5 exclusively (cameras never seen in training)

| Split | Events | Ear | Tail | Cameras |
|-------|--------|-----|------|---------|
| Train | 1,374 | ~1,207 | ~167 | 8, 16, 12, 12 (Grps 1,2,3,6) |
| Val | 361 | 323 | 38 | Same as train (intra-video) |
| Test OOD | 141 | 129 | 12 | 3, 1 (Grps 4, 5 — unseen) |

---

## 4. Model: E1 Baseline

### 4.1 Architecture
- **Backbone:** MViTv2-S (Multiscale Vision Transformer, Small) — 34.2M parameters
- **Pre-training:** Kinetics-400 (600K video clips, 400 action categories)
- **Head:** Linear classifier (768-d → 2 classes)
- **Input:** 16 frames sampled at 12 fps (= 1.33 s temporal window), center-cropped to 224×224

### 4.2 Training Configuration (E1 — `train_binary_v4_mvit_isaac.yaml`)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for transformers |
| Learning rate | 2e-5 | Conservative for pretrained model |
| Warmup epochs | 5 | Linear warmup from 5e-7 |
| LR schedule | Cosine annealing (T=50) | Smooth decay |
| Loss | Focal (γ=2) | Penalizes easy negatives |
| Label smoothing | 0.05 | Regularization |
| Class weights | None | Balanced sampler active |
| Batch size | 16 | A100/V100S-32GB |
| Balanced sampling | Yes | 50/50 ear:tail batches |
| Augmentation | TrivialAugment + Mixup (α=0.4) | |
| Early stopping | Patience = 15 on val macro_F1 | |

**Hardware:** Tesla V100S-PCIE-32GB on ISAAC-NG HPC (UTK)

### 4.3 Training Issues Resolved

During E1 training, three distinct bugs were encountered and fixed:

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| PyTorch AMP API mismatch | `AttributeError: module 'torch.amp' has no attribute 'GradScaler'` | ISAAC uses PyTorch 2.1.2; `torch.amp.GradScaler` is 2.4+ API | Replaced with `torch.cuda.amp.GradScaler` |
| CUDA OOM during validation | `CUDA out of memory` after epoch 1 forward pass | Validation ran without `torch.no_grad()`, building full computation graph (~10 GB extra) on top of 23 GB training state | Added `torch.no_grad()` context for val-only passes |
| All-tail class collapse | Model predicted 100% tail from epoch 2, train F1_tail ≈ 0.19 | Original config had `use_class_weights: true` (6.85× on tail) ON TOP of balanced sampler + focal loss — triple correction | Disabled class weights; switched `metric_key` from `f1_tail` to `macro_f1` |

*Note on the all-tail collapse:* The `metric_key = f1_tail` exacerbated the bug because a degenerate all-tail classifier achieves tail recall = 1.0, giving F1_tail ≈ 0.19 (= 2 × 0.105 × 1.0 / 1.105), which the early-stopping logic treated as the best checkpoint. Switching to `macro_f1` correctly penalizes one-class collapse.

---

## 5. E1 Results

### 5.1 Validation Set (n = 361; 323 ear, 38 tail)

| Metric | Value |
|--------|-------|
| Accuracy | 80.6% |
| Macro F1 | **0.538** (epoch 3 best) |
| Ear recall | 87.6% (283/323) |
| Tail recall | **21.1%** (8/38) |
| Tail F1 | 0.186 |
| Tail precision | 16.7% |

Confusion matrix (val):

|  | Pred Ear | Pred Tail |
|--|----------|-----------|
| **True Ear** | 283 | 40 |
| **True Tail** | 30 | 8 |

### 5.2 Out-of-Distribution Test — Groups 4 & 5 (n = 141; 129 ear, 12 tail)

| Metric | Value |
|--------|-------|
| Accuracy | 66.7% |
| Macro F1 | **0.454** |
| Ear F1 | 0.795 |
| Ear recall | 70.5% (91/129) |
| Ear precision | 91.0% |
| Tail F1 | 0.113 |
| Tail recall | **25.0%** (3/12) |
| Tail precision | 7.3% |

Confusion matrix (OOD):

|  | Pred Ear | Pred Tail |
|--|----------|-----------|
| **True Ear** | 91 | 38 |
| **True Tail** | 9 | 3 |

---

## 6. Learning Curve Analysis

*(See* `results/E1_v4_mvit/learning_curves.png`*)*

Four observations from the training curves:

**1. Rapid overfitting.** Train macro F1 climbed steadily from 0.50 → 0.87 over 18 epochs. Val macro F1 peaked at **0.538 (epoch 3)** and then stagnated in the 0.49–0.53 range for the remaining 15 epochs. The train–val gap grew from 0.004 at epoch 1 to **0.34 at epoch 17** — a classic overfitting signature.

**2. Tail recall collapse.** This is the most critical finding. Tail recall on validation peaked at **24% (epoch 2)**, then fell sharply to 3% by epoch 8, and never recovered beyond 16% for the rest of training. The model learned to predict ear for almost everything within the first few epochs and then continued to improve on ear while abandoning tail.

**3. Model peaked mid-warmup.** The LR schedule used a 5-epoch linear warmup (4.4e-6 → 2e-5). The best checkpoint (epoch 3) occurred at LR = 1.22e-5, while the backbone was still warming up and the head had not converged. This means the model never had a chance to learn properly at full LR before overfitting set in.

**4. Val loss lower than train loss (epochs 1–5).** Early in training, val loss was lower than train loss, indicating that balanced sampling made the training distribution harder (50/50 ear:tail) compared to the natural val distribution (89/11 split). The loss curves cross around epoch 12 after which standard overfitting is visible.

---

## 7. Threshold Calibration Analysis

After training, probability thresholds were swept from 0.10 to 0.90 on both val and OOD sets to determine whether the E1 checkpoint had headroom that a lower classification threshold could unlock.

**Finding: Threshold calibration offers no meaningful improvement on OOD data.**

| Threshold | OOD Tail Recall | OOD Ear F1 | OOD Macro F1 |
|-----------|----------------|------------|--------------|
| 0.10–0.40 | 100% | 0.00 | 0.078 |
| 0.45 | 75% | 0.32 | 0.231 |
| **0.50** | **25%** | **0.79** | **0.454** |
| 0.55+ | 0% | 0.96 | 0.478 |

The transition between threshold 0.45 and 0.50 is a cliff: at 0.45, nearly all samples are called tail (too many false positives); at 0.50, only 3 of 12 tail events are caught. There is no threshold that simultaneously achieves acceptable tail recall and ear precision. This indicates the model's raw probability outputs are not reliably separating the two classes for unseen camera viewpoints — the learned features have not generalized beyond the training cameras.

**Conclusion: retraining required.** Post-hoc threshold adjustment is not a viable improvement path for E1.

---

## 8. Challenges

### 8.1 Minority Class (Tail) Learning
The 7.3:1 ear:tail imbalance is the dominant challenge. Even with balanced sampling (50/50 per batch), focal loss (γ=2), and MixUp augmentation, the model collapsed toward ear prediction. Balanced sampling helps the gradient but does not prevent the model from using high-confidence ear features to minimize overall loss.

### 8.2 Camera Generalization (OOD Gap)
Val accuracy (80.6%) vs. OOD accuracy (66.7%) represents a 14-point drop when evaluated on cameras never seen during training. Groups 4 and 5 use different camera angles, lighting conditions, and pen configurations than Groups 1, 2, 3, and 6. This camera-specific feature learning is a fundamental challenge for deployment.

### 8.3 Short Temporal Window vs. Long Events
The model receives only 16 frames at 12 fps = **1.33 seconds** of video per clip. The average event duration is **6.9 seconds** (median 4 s). The training clips are randomly jittered within each event, so the model sees only a fragment of each behavior. This is a known limitation of fixed-length clip models applied to variable-duration events.

### 8.4 Overfitting with Small Tail Support
Only 167 tail events in the training split. With a 34.2M-parameter transformer, the model has far more capacity than the data can constrain, particularly for the minority class. The train–val gap of 0.34 in macro F1 by epoch 17 confirms the model memorizes training examples.

### 8.5 Warmup-Peak Alignment
The best checkpoint appeared at epoch 3, mid-warmup at LR = 1.22e-5 — before the backbone was fully activated. This is a training stability problem rather than a model capacity problem: the model briefly reaches a reasonable balance between classes then shifts to ear-dominated predictions as LR increases toward 2e-5.

---

## 9. Improvement Plan: E1b

Based on the diagnostic findings, the following targeted changes have been implemented for the next training run (`train_binary_v4b_mvit_isaac.yaml`), submitted to ISAAC-NG:

| Change | E1 Value | E1b Value | Addresses |
|--------|---------|----------|-----------|
| Class weights | None | **inv_sqrt (√7.3 ≈ 2.71×)** | Tail collapse |
| Learning rate | 2e-5 | **1e-5** | Warmup-peak alignment |
| Warmup epochs | 5 | **8** | Model peaked mid-warmup |
| Early stop patience | 15 | **20** | More room to recover |
| Output directory | `…/v4_mvit` | `…/v4b_mvit` | Preserve E1 for comparison |

**Rationale for `inv_sqrt` class weights:** The original `inv_freq` weighting (6.85×) caused all-tail collapse because it over-corrected on top of an already balanced sampler and focal loss. The square-root transformation (2.71×) provides a gentler signal — enough to keep tail in the gradient without destabilizing ear predictions. This follows the approach used in the FERAL paper for long-tail video classification.

**Expected behavior:** Tail recall should be 0.30–0.50 by epoch 8 (end of warmup) if the class weights are effective. If recall again collapses, the next step would be to disable balanced sampling and use only `inv_freq` weights (single correction rather than stacked corrections).

---

## 10. Roadmap

```
Phase           Status
─────────────────────────────────────────────────────────────────
Data pipeline   COMPLETE  (1,897 events, 12 folders, v4 splits)
E1 baseline     COMPLETE  (val macro_F1=0.538, OOD=0.454)
E1 analysis     COMPLETE  (curves, threshold sweep, diagnostics)
E1b retrain     RUNNING   (submitted to ISAAC-NG, 24h window)
E1b eval        PENDING   (run eval_model.py + generate_predictions.py)
E2: FERAL       PENDING   (requires E1b to establish strong baseline)
E3: Spatial     PENDING   (200 annotated events with bounding boxes)
```

**Milestone target:** E1b should achieve val macro_F1 ≥ 0.60 and val tail recall ≥ 0.40. If achieved, it constitutes a sufficient supervised baseline for QE proposal comparison against FERAL (E2).

---

## Appendix A: Infrastructure Notes

- **Cluster:** ISAAC-NG (UTK), account `acf-utk0011`, partition `campus-gpu`
- **GPU:** Tesla V100S-PCIE-32GB (assigned); A100-40/80GB also available in queue
- **Data location:** `/lustre/isaac24/scratch/eopoku2/clips_v4/` (~60 GB, 1,897 clips)
- **Run artifacts:** `/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4_mvit/`
- **Conda env:** `cross_sucking` (PyTorch 2.1.2 + cu121, NumPy < 2.0)
- **Key lesson:** Always use `--num-workers ≤ 4` for 4K clips on ISAAC; 8 workers causes OOM in DataLoader worker processes due to peak 4K frame decode memory

## Appendix B: Key Files

| File | Purpose |
|------|---------|
| `data/manifests/MASTER_LINKED_v4.csv` | All 1,897 camera-corrected linked events |
| `data/manifests/train_v4.csv` | 1,374 training events (bout-grouped) |
| `data/manifests/val_v4.csv` | 361 validation events |
| `data/manifests/test_ood_v4.csv` | 141 OOD test events (Groups 4+5) |
| `configs/train_binary_v4_mvit_isaac.yaml` | E1 config |
| `configs/train_binary_v4b_mvit_isaac.yaml` | E1b config (current run) |
| `scripts/train_isaac_v4b.slurm` | SLURM job for E1b (24h, campus-gpu) |
| `scripts/eval_model.py` | Checkpoint evaluation (metrics + confusion matrix) |
| `scripts/generate_predictions.py` | Per-sample probabilities + threshold sweep |
| `results/E1_v4_mvit/` | All E1 artifacts (local copy) |
