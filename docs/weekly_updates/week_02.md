# Week 2: Model Architecture, Training Pipeline, and Experiment Design

**Date**: February 10–16, 2026
**Focus**: Architecture selection, training pipeline implementation, local training analysis, infrastructure deployment, and FERAL experiment design.

---

## 1. Overview

Building on the data foundation established in Week 1, this week focused on four parallel threads: (1) selecting and justifying a video backbone architecture, (2) implementing and validating a complete supervised training pipeline, (3) diagnosing training failure modes observed in local runs, and (4) preparing a multi-experiment infrastructure across two compute platforms (UTK ISAAC-NG HPC and RunPod cloud GPU). A secondary contribution emerged from the discovery of the FERAL system (Sun et al., 2025), which introduced a second experiment track based on the V-JEPA2 video foundation model.

---

## 2. Architecture Selection

### 2.1 Candidate Models Considered

Four spatiotemporal architectures from the TorchVision model zoo were evaluated as candidate backbones for the binary cross-sucking classifier:

| Model | Params | Pretrain | Input | Key Characteristic |
|---|---|---|---|---|
| R3D-18 | 33.4M | Kinetics-400 | 16×112×112 | Lightweight 3D ResNet; fast to train |
| X3D-S | 3.8M | Kinetics-400 | 13×160×160 | Efficient mobile-scale 3D CNN |
| SlowFast R50 | 34.4M | Kinetics-400 | dual-pathway | Dual-rate temporal fusion |
| **MViTv2-S** | **34.2M** | **Kinetics-400** | **16×224×224** | **Multiscale Vision Transformer** |

### 2.2 Rationale for MViTv2-S

MViTv2-S (Li et al., 2022) was selected as the primary backbone for the following reasons:

**Self-attention over temporal context.** Cross-sucking is a subtle, low-motion behavior in which the key discriminative signal is anatomical location (ear vs. tail) rather than gross motion. CNN-based architectures such as R3D and X3D extract local motion features via 3D convolutions, but lack the long-range spatial attention that allows the model to reason simultaneously about the relative positions of both calves and the specific body part being contacted. MViTv2's multi-scale attention blocks aggregate spatial tokens across the full clip, which is more appropriate for this task.

**Multiscale pooling.** MViTv2 hierarchically reduces the spatial resolution of attention keys and values while preserving semantic content, achieving efficient computation at 34M parameters — comparable to R3D-18 in size but with substantially richer representational capacity for spatial reasoning.

**Kinetics-400 pretraining.** Although Kinetics-400 consists primarily of human actions, the low-level spatiotemporal features learned — motion saliency, object persistence, spatial co-occurrence — transfer well to animal behavior. This is consistent with findings in related livestock video analysis work (e.g., Kholiavchenko et al., 2024).

**Feature dimensionality.** MViTv2-S produces 768-dimensional clip features, which provides sufficient expressive capacity for the binary classification head while remaining tractable for fine-tuning on a dataset of 1,374 training clips.

X3D-S was considered for its efficiency but was rejected due to its very small capacity (3.8M parameters), which is likely insufficient for 4K surveillance footage where the relevant anatomical region may occupy only a small fraction of the frame. SlowFast was not selected because the dual-pathway design increases memory substantially and the added slow-pathway temporal resolution does not provide clear benefit for distinguishing static anatomical targets.

### 2.3 Model Specification

```
Backbone:     MViTv2-S (torchvision.models.video.mvit_v2_s)
Pretrained:   Kinetics-400
Input shape:  (B, C, T, H, W) = (B, 3, 16, 224, 224)
Features:     768-dimensional global average-pooled representation
Head:         Linear(768, 2) with Dropout(p=0.3)
Parameters:   34.2M total; ~34.0M trainable (all layers fine-tuned)
Mixed prec.:  AMP (torch.cuda.amp) with float16 forward, float32 gradients
```

---

## 3. Training Pipeline Implementation

### 3.1 Data Loading

The `LabeledEventDataset` class (src/data/datasets.py) supports two operating modes:

**Source-video mode**: Reads directly from the original 4K surveillance files on the E: drive by seeking to the event's temporal offset. This mode is used for local development.

**Clip-dir mode**: Loads from pre-extracted `.mp4` files in `data/processed/clips_v4/`, identified by the naming convention `evt_{event_idx:04d}_{behavior}.mp4`. This mode is used for remote training (ISAAC, RunPod) where the source videos are not available. All 1,374 training clips and 361 validation clips were verified present before the first remote training run.

Both modes apply identical temporal sampling: 16 frames are uniformly sampled from the decoded clip at a target rate of 12 fps, then resized and normalized to (224, 224) with ImageNet statistics.

### 3.2 Temporal Sampling and Augmentation

Spatial augmentations are applied per-frame in `src/data/transforms.py`:

- **Training**: RandomResizedCrop(224, scale=(0.7, 1.0)), RandomHorizontalFlip, TrivialAugmentWide
- **Validation/Test**: CenterCrop(224)

A critical implementation issue was discovered during the first training run on the local machine:

**Out-of-memory error in transforms.py.** The augmentation pipeline converted each 4K decoded frame to a float32 NumPy array before applying per-frame transforms:

```python
# Buggy original — 94.9 MiB per frame for 3840×2160×3 float32
arr = (frame_cpu.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
```

For a single 4K frame at float32, this allocates a 3840×2160×3×4 = 94.9 MiB intermediate array. With 4 DataLoader workers each processing up to 4 frames simultaneously, peak RAM usage exceeded system limits.

The fix converted to uint8 on the GPU tensor before calling numpy, reducing per-frame allocation from 94.9 MiB to 23.7 MiB — a 4× reduction:

```python
# Fixed — 23.7 MiB per frame (uint8, no intermediate float)
arr = (frame_cpu.clamp(0.0, 1.0) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
```

Additionally, `num_workers` was reduced from 4 to 2 for local training. For ISAAC (A100, 64GB RAM) the default of 8 workers is appropriate.

### 3.3 Class Imbalance Handling

The training set contains 1,199 ear events and 175 tail events — a 6.85:1 ratio. Three complementary mechanisms address this imbalance:

**Balanced sampling.** The `WeightedRandomSampler` oversamples tail events such that, in expectation, each training batch contains equal numbers of ear and tail samples.

**Focal loss.** The focal loss (Lin et al., 2017) downweights easy negative examples by a factor of $(1 - p_t)^\gamma$ where $\gamma = 2.0$. This concentrates gradient signal on hard examples, preventing the model from trivially achieving low loss by always predicting ear.

**Class-weighted loss.** When `use_class_weights: true`, per-class weights of magnitude $w_{\text{tail}} = N_{\text{ear}} / N_{\text{tail}} = 6.85$ are applied to the focal loss. This explicitly penalizes misclassification of tail events 6.85× more than ear events.

**Mixup augmentation.** Mixup (Zhang et al., 2018) interpolates between pairs of training samples and their labels: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$, where $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha = 0.4$. This prevents the model from committing hard decision boundaries, which is particularly important for the tail class which has few training examples.

### 3.4 Optimization

| Parameter | Local config | RunPod/ISAAC config | Justification |
|---|---|---|---|
| Optimizer | AdamW | AdamW | Weight decay decoupling beneficial for transformer |
| Learning rate | 5×10⁻⁵ | **2×10⁻⁵** | Halved after observing tail collapse (§4.2) |
| Weight decay | 0.05 | 0.05 | Standard for MViTv2 fine-tuning |
| Grad clip | 1.0 | 1.0 | Prevents exploding gradients in attention layers |
| Scheduler | cosine_warmup | cosine_warmup | Linear warmup prevents early LR spike |
| Warmup epochs | 3 | **5** | Longer warmup at lower LR |
| Warmup start LR | 1×10⁻⁶ | 5×10⁻⁷ | Scale proportionally with base LR |
| Min LR | 5×10⁻⁷ | 5×10⁻⁷ | Cosine floor |
| Total epochs | 30 (local) | **50** (remote) | More compute available on GPU servers |

### 3.5 Trainer Implementation

The `Trainer` class (src/training/trainer.py) provides a complete training loop with the following components:

**Checkpoint management.** After each validation epoch, two checkpoints are saved to `out_dir/`:
- `best.ckpt`: The checkpoint with the highest value of the monitoring metric (f1_tail).
- `last.ckpt`: The most recent epoch checkpoint, used for resuming interrupted runs.

Checkpoint payloads include model weights, optimizer state, AMP scaler state, scheduler state, `best_metric`, and the epoch index, enabling exact resumption of training state.

**Resume mechanism.** Training can be resumed by passing `--resume /path/to/last.ckpt` to `train_supervised.py`. The `_load_checkpoint()` method restores all stateful components and returns `start_epoch = saved_epoch + 1`. This is particularly important on ISAAC where the 12-hour job time limit may interrupt long runs; the SLURM script automatically detects `last.ckpt` and passes `--resume` accordingly.

**Early stopping.** Patience-based early stopping monitors `f1_tail` with a patience of 15 epochs and a minimum delta of 0.001. The criterion is applied to the validation set. Early stopping fires only when `f1_tail` fails to improve, not when macro F1 stagnates — this prevents the model from being declared converged when it has abandoned the tail class (see §4.2).

**Per-sample loss export.** After the final training epoch, `_export_per_sample_loss()` records the cross-entropy loss for each training and validation sample (with `reduction='none'`) and saves a CSV containing sample index, loss, predicted label, true label, and correctness. These per-sample diagnostics are intended to support noisy label detection and curriculum learning in future work.

**History logging.** After each epoch, the trainer records a `train_val_gap = macro_F1_train − macro_F1_val` metric to detect overfitting. Values persistently above 0.10–0.15 indicate that regularization should be increased.

---

## 4. Local Training Analysis

### 4.1 Experimental Setup

A preliminary training run was conducted on the local workstation (Windows 11, NVIDIA GPU, 2 DataLoader workers) using the local configuration (`configs/train_binary_v4_mvit.yaml`): LR=5×10⁻⁵, batch_size=4, no class weights, macro_f1 monitoring, 3-epoch warmup.

### 4.2 Observation: Tail Class Collapse

The per-class validation metrics over epochs 2–8 revealed a pattern of tail class collapse:

| Epoch | Val macro_F1 | Val F1 (ear) | Val F1 (tail) | Val Recall (tail) |
|---|---|---|---|---|
| 2 | 0.52 | 0.85 | 0.18 | 0.21 |
| 3 | 0.55 | 0.87 | 0.23 | 0.26 |
| 4 | 0.56 | 0.88 | 0.24 | 0.27 |
| **5** | **0.57** | **0.88** | **0.24** | **0.29** |
| 6 | 0.55 | 0.87 | 0.22 | 0.22 |
| 7 | 0.53 | 0.88 | 0.18 | 0.14 |
| 8 | 0.50 | 0.88 | **0.11** | **0.08** |

The model reached its best tail F1 of 0.24 at epoch 5 and subsequently collapsed: by epoch 8, tail recall had fallen to 0.08, meaning the model correctly identified only 8% of tail events. Ear performance remained stable throughout (F1 ≈ 0.88), confirming that the collapse was specific to the minority class.

### 4.3 Root Cause Analysis

Three concurrent factors contributed to tail collapse:

**1. Learning rate too high.** A base LR of 5×10⁻⁵ is aggressive for a transformer backbone. Transformers are particularly sensitive to learning rate due to the quadratic attention mechanism and the relatively small distance between local minima. After the warmup phase, the cosine schedule decreases LR slowly, meaning the model may over-shoot the tail-class optimum early in training. This was corrected by halving the LR to 2×10⁻⁵ in the remote configuration.

**2. No explicit class weights.** While the focal loss and balanced sampler provide implicit minority-class emphasis, they do not explicitly penalize tail misclassification in the loss gradient. Without `use_class_weights: true`, the 6.85:1 imbalance in the underlying data distribution continues to bias gradients toward ear. The remote configuration enables explicit class weights.

**3. Monitoring metric masked collapse.** The early stopping criterion monitored `macro_f1`, which is the arithmetic mean of ear and tail F1. Because ear F1 was stable at 0.88 while tail F1 fell, macro F1 declined only modestly (0.57 → 0.50). This delayed early stopping and allowed the model to continue training into a regime where tail was essentially ignored. The monitoring metric was changed to `f1_tail` in all remote configurations, ensuring early stopping fires specifically when tail performance deteriorates.

### 4.4 Interventions Applied

The following changes were applied to the remote training configurations (RunPod and ISAAC) based on this analysis:

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `optim.lr` | 5×10⁻⁵ | **2×10⁻⁵** | Reduce overshoot |
| `loss.use_class_weights` | false | **true** | Explicit 6.85× tail penalty |
| `loss.label_smoothing` | 0.1 | **0.05** | Less smoothing → preserve tail gradient |
| `train.metric_key` | macro_f1 | **f1_tail** | Monitor actual failure mode |
| `train.early_stopping_patience` | 10 | **15** | More patience for slower convergence |
| `augmentation.use_mixup` | false | **true** (α=0.4) | Prevent hard decision boundary on tail |
| `sched.warmup_epochs` | 3 | **5** | Longer warmup at lower LR |
| `data.num_workers` | 4 | **8** (ISAAC) / **4** (RunPod) | Match platform CPU count |

---

## 5. Infrastructure Preparation

### 5.1 RunPod Deployment (Experiment E1 — pending data upload)

A RunPod spot instance with an RTX 4090 GPU (24GB VRAM, 8 vCPU, ~$0.50/hr) was provisioned for continuous availability. The RTX 4090 was selected over alternatives (RTX 4000 Ada, A5000) based on VRAM-to-cost ratio: 24GB VRAM supports batch_size=8 with MViTv2-S under AMP, completing 50 epochs in approximately 1–1.5 hours (~$0.60 total).

The pre-extracted 4K clips (71.2 GB, 1,897 files) are being transferred to RunPod network storage from the local Windows machine. The clips are stored in `clips_v4/` with naming convention `evt_{event_idx:04d}_{behavior}.mp4`. SSH key authentication was configured for direct SCP transfer.

The RunPod environment is configured with `configs/train_binary_v4_mvit_runpod.yaml`, which uses `clip_dir` mode in `LabeledEventDataset` to bypass the need for source video access.

### 5.2 ISAAC-NG HPC Deployment (Experiment E1 — queued)

The code repository was cloned to `/lustre/isaac24/scratch/eopoku2/cross_sucking/` on the UTK ISAAC-NG cluster. Pre-extracted 4K clips (1,897 files) were transferred from the local Windows machine to `/lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4/` via Globus (note: Globus created a nested directory; config corrected to reflect the actual path).

The conda environment `cross_sucking` was established with PyTorch 2.1.2 + CUDA 12.1. A NumPy version conflict was identified and resolved: PyTorch 2.1 requires NumPy < 2.0; the system had installed NumPy 2.x, which caused the warning `Failed to initialize NumPy: _ARRAY_API not found`. This was corrected with `pip install "numpy<2.0"`.

The SLURM job `scripts/train_isaac.slurm` was submitted to the `campus-gpu` partition with the following resource request:

```
Account:         acf-utk0011
Partition:       campus-gpu
QoS:             campus-gpu
Nodes:           1 (1 GPU, 8 CPU, 64GB RAM)
Wall time:       12:00:00
GRES:            gpu:1
```

The job includes an auto-resume mechanism: at startup, the SLURM script checks for the existence of `last.ckpt` in the output directory. If found, training resumes from that checkpoint; otherwise it starts from epoch 0. This ensures that job resubmission after timeout or preemption is idempotent.

As of writing, the ISAAC job is in `PD` (pending) state, queued behind higher-priority jobs on the `campus-gpu` partition.

**Key ISAAC configuration parameters** (`configs/train_binary_v4_mvit_isaac.yaml`):

| Parameter | Value | Notes |
|---|---|---|
| `data.clip_dir` | `/lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4` | Lustre scratch, Globus nested copy |
| `data.batch_size` | 16 | A100 40/80GB with AMP |
| `data.num_workers` | 8 | 8 CPUs allocated per job |
| `train.out_dir` | `/lustre/isaac24/scratch/eopoku2/runs/sup_binary_v4_mvit` | On Lustre scratch for fast I/O |
| `train.metric_key` | f1_tail | Monitor tail-class F1 |
| `train.epochs` | 50 | Full run expected in ~2–4hr on A100 |

---

## 6. FERAL Integration Plan (Experiment E2)

### 6.1 Motivation

The FERAL system (Skovorodnikov, Zhao et al., 2025 — bioRxiv:2025.11.16.688666) was identified as a state-of-the-art video behavior recognition framework with direct relevance to the cross-sucking task. FERAL addresses the same problem class — supervised fine-tuning of a video foundation model on small annotated datasets of animal behavior — and reports surpassing Google's VideoPrism model using only 25% of the training data on mouse social interaction benchmarks (CalMS21).

Three technical properties make FERAL particularly suitable for cross-sucking classification:

**V-JEPA2 backbone (Meta FAIR).** FERAL adopts V-JEPA2 (Bardes et al., 2024), a ViT-L architecture (330M parameters) trained self-supervised on over one million hours of unlabeled video via masked joint embedding prediction. The self-supervised pre-training objective produces representations that generalize more broadly than supervised Kinetics-400 pretraining, which is biased toward human action categories. Importantly, V-JEPA2 has not been exposed to cross-sucking or livestock footage, making domain adaptation a natural next step (see §7.2).

**Attention-based pooling head.** Standard video classifiers apply global average pooling over the spatiotemporal token sequence before the linear classification head, discarding spatial structure. FERAL replaces this with 64 learnable query embeddings that cross-attend to the encoder output, extracting temporally-aligned features for frame-level prediction. For our clip-level task, predictions from all frames within a clip are aggregated by majority vote.

**sqrt class weighting.** FERAL uses $w_c = \sqrt{N_{\text{neg}} / N_{\text{pos}}}$ as the per-class loss weight, which is a softer weighting than the full inverse frequency used in our MViTv2 configuration. For our dataset: $w_{\text{tail}} = \sqrt{1651 / 225} \approx 2.71$. This softer weighting may improve tail precision by reducing the tendency to over-predict tail, which is a failure mode of aggressive class weighting.

### 6.2 Architecture Comparison

| Component | E1: MViTv2-S | E2: FERAL / V-JEPA2 |
|---|---|---|
| Backbone | MViTv2-S, 34M params | V-JEPA2 ViT-L, 330M params |
| Pretraining | Supervised, Kinetics-400 | Self-supervised, 1M+ hrs video |
| Fine-tune scope | All layers | Last 12 of 24 transformer layers |
| Pooling head | Global avg pool → Linear | 64-query attention pool → BN → Dropout → Linear |
| Input resolution | 224×224 | 256×256 |
| Input frames | 16 | 64 (chunk_length) |
| Class weighting | Focal + 6.85× explicit | sqrt(1651/225) = 2.71× |
| Label smoothing | 0.05 | 0.10 |
| Mixup alpha | 0.4 | 0.4 |
| Epochs | 50 | 10 |

### 6.3 Data Preparation for FERAL

FERAL requires videos at 256×256 resolution and annotations in a specific JSON format. Two preparation scripts were implemented:

**`scripts/resize_clips_feral.sh`**: A SLURM CPU job that processes all 1,897 clips from `clips_v4/clips_v4/` into `clips_feral/` using FFmpeg with center-crop scaling:

```bash
-vf "scale=256:256:force_original_aspect_ratio=increase,crop=256:256"
```

The center-crop strategy is preferred over direct squash rescaling to avoid aspect ratio distortion of the calf anatomy. The output uses H.264 CRF 23 with fast preset; estimated output size is approximately 2–4 GB (compared to 71 GB for the 4K clips).

**`scripts/prepare_feral_json.py`**: Converts the three split CSV manifests (train_v4.csv, val_v4.csv, test_ood_v4.csv) to the FERAL JSON format:

```json
{
  "splits": {
    "train": ["evt_0001_ear.mp4", "evt_0004_tail.mp4", ...],
    "val":   [...],
    "test":  [...]
  },
  "labels": {
    "evt_0001_ear.mp4":  [0, 0, 0, ..., 0],
    "evt_0004_tail.mp4": [1, 1, 1, ..., 1]
  }
}
```

Since cross-sucking events receive a single clip-level label, all frames within a clip are assigned the same label integer (ear=0, tail=1). Frame counts are obtained via `ffprobe` for each clip; if unavailable, a fallback of 90 frames is used (6 seconds × 15 fps, consistent with our extraction parameters).

**`configs/feral_cross_sucking.yaml`**: A FERAL configuration file adapted from the published default (`default_vjepa.yaml`), with the following cross-sucking-specific modifications:

```yaml
model_name: "facebook/vjepa2-vitl-fpc32-256-diving48"
model:
  class_weights: inv_freq_sqrt
  freeze_encoder_layers: 14      # Fine-tune last 12 of 24 layers
data:
  chunk_length: 64
  chunk_shift: 32                # 50% temporal overlap
training:
  train_bs: 4                    # V-JEPA2 ViT-L: ~330M params
  lr: 4.0e-5
  patience: 5                    # Early stopping on val mAP
mixup_alpha: 0.4
```

### 6.4 Execution Sequence

The FERAL experiment requires four sequential steps, the first two of which are preparatory (no GPU required):

1. `sbatch scripts/resize_clips_feral.sh` — CPU job, ~30–60 min on `campus` partition
2. `python scripts/prepare_feral_json.py` — CPU, runs in < 5 min
3. Clone FERAL repository and install dependencies
4. `sbatch scripts/train_feral.slurm` — GPU job, ~2–4 hr on `campus-gpu`

---

## 7. Experiment Roadmap

### 7.1 Three-Experiment Plan

The complete experiment plan for this project is structured in three tiers of increasing model capacity and data utilization:

| Exp | Name | Model | Pretrain | Data | Key Hypothesis |
|---|---|---|---|---|---|
| **E1** | MViTv2-S Baseline | MViTv2-S (34M) | Kinetics-400 supervised | 1,374 clips, 4K→224px | Multiscale attention + focal loss sufficient for binary classification |
| **E2** | FERAL / V-JEPA2 Generic | V-JEPA2 ViT-L (330M) | Self-supervised, 1M+ hrs | 1,374 clips, 4K→256px | Larger foundation model outperforms supervised baseline with small labeled data |
| **E3** | FERAL + Domain Adaptation | V-JEPA2 ViT-L (330M) | SSL on 960hr farm video, then fine-tuned | 1,374 clips, 4K→256px | Domain-adapted backbone improves tail recall by reducing distribution shift |

### 7.2 Experiment E3: Domain-Adaptive Pre-training

Approximately 960 hours of unlabeled surveillance footage from the same recording setup (N884A6 NVR, 4K @ 15fps, six groups) is available and has not been annotated. This footage captures the same animals, environments, lighting conditions, and camera angles as the labeled data.

V-JEPA2 was pre-trained on general internet video and has no exposure to livestock barn environments. We hypothesize that intermediate self-supervised pre-training on the 960-hour farm corpus — using the V-JEPA2 masked joint embedding objective on in-domain video — will produce feature representations more tightly aligned with the visual statistics of cross-sucking footage, reducing the distribution shift penalty in the final fine-tuning stage.

The domain adaptation pipeline will involve:
1. Extracting short unlabeled clips from the 960-hour corpus (sampling strategy TBD)
2. Continuing V-JEPA2 self-supervised training on the farm clips (masked prediction)
3. Fine-tuning the domain-adapted backbone on the labeled cross-sucking data using the FERAL pipeline

This constitutes a methodological contribution to the QE proposal: the use of large-scale unlabeled farm video for domain adaptation of video foundation models in livestock behavior recognition.

### 7.3 Evaluation Protocol

All three experiments will be evaluated using identical metrics on the same test sets:

**Primary metric**: F1 score for the tail class (F1\_tail). Tail is the minority and clinically relevant class; the ear class is comparatively easy to detect.

**Secondary metrics**:
- Macro-averaged F1 (mean of F1\_ear and F1\_tail)
- Per-class precision and recall
- Area under the ROC curve (AUC) for the tail class
- Confusion matrix

**Test sets**:
- **Intra-distribution validation** (n=361): Groups 1–3, 6; same cameras as training.
- **OOD test** (n=141): Groups 4–5, Cameras 3 and 1; never seen during training. This set specifically tests generalization to new camera viewpoints and new animals.

---

## 8. Issues and Resolutions

| Issue | Root Cause | Resolution |
|---|---|---|
| OOM in transforms.py | float32 numpy conversion of 4K frame (94.9 MiB/frame) | Convert to uint8 on tensor before numpy call |
| Tail class collapse (epoch 5→8) | LR=5e-5 too high + no class weights + wrong monitoring metric | LR halved, class weights enabled, metric_key=f1_tail |
| Training output buffered (0 bytes) | Python stdout buffering in background process | Use `python -u` (unbuffered) flag |
| SLURM account error | Account must be lowercase: `acf-utk0011` not `ACF-UTK0011` | Corrected in SLURM header |
| SLURM QoS error | `qos=campus-gpu` required for GPU jobs on ISAAC | Added `#SBATCH --qos=campus-gpu` |
| NumPy 2.x incompatibility | System numpy version > 2.0 conflicts with PyTorch 2.1 | `pip install "numpy<2.0"` |
| Globus nested directory | Globus copied folder into destination creating `clips_v4/clips_v4/` | Updated `clip_dir` in config to reflect actual path |
| SSH permission denied (RunPod) | Campus firewall blocks port 48282; authorized_keys malformed | Mobile hotspot bypass; fix authorized_keys via web terminal |

---

## 9. Next Steps

1. **Monitor ISAAC job**: When E1 starts running, validate per-epoch metrics in `slurm_*.log`. Expect tail F1 to peak around epoch 10–15 with the improved configuration. If collapse recurs, consider enabling two-stage fine-tuning (freeze backbone for first 5 epochs).

2. **Complete RunPod data upload**: Transfer remaining clips. Submit E1 on RunPod in parallel for comparison under identical training conditions.

3. **Execute FERAL prereqs on ISAAC** (no GPU needed):
   - Submit `resize_clips_feral.sh` to `campus` partition
   - Run `prepare_feral_json.py` to generate `feral_annotations.json`
   - Clone FERAL repo and install dependencies

4. **Submit E2 (FERAL)**: Once prereqs are complete, submit `train_feral.slurm`.

5. **Plan E3 (domain adaptation)**: Define the sampling strategy for the 960-hour unlabeled corpus. Key decisions: clip duration (5–10s), sampling interval (every N minutes), and whether to include all cameras or only indoor cameras.

6. **Per-sample loss analysis**: After E1 completes, analyze the exported per-sample loss CSV to identify potential label noise or consistently hard examples.

---

## References

- Li, Y., Wu, C.-Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., & Feichtenhofer, C. (2022). MViTv2: Improved multiscale vision transformers for classification and detection. *CVPR 2022*.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017*.
- Skovorodnikov, P., Zhao, Y., Sun, J., et al. (2025). FERAL: A video-understanding system for direct video-to-behavior mapping. *bioRxiv 2025.11.16.688666*.
- Bardes, A., et al. (2024). V-JEPA 2: Self-supervised video models that understand the physical world. *Meta FAIR Technical Report*.
- Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *ICLR 2018*.
- Kholiavchenko, M., et al. (2024). KABR: In-situ dataset for Kenyan animal behavior recognition from drone videos. *WACV 2024*.
