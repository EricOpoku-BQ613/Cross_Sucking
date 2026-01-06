# Cross-Sucking Detection Pipeline

**Role-Aware Interaction Tubes for Livestock Welfare Monitoring**

---

## 🎯 Overview

Automated detection of cross-sucking behavior in dairy calves using computer vision, with focus on:
- **WHO**: Identifying initiator vs receiver roles
- **WHAT**: Classifying target body region (ear/tail/teat/other)
- **WHEN**: Precise temporal event localization

### Key Innovation: Role-Conditioned Graph Attention Network (RCGAN)
Unlike existing approaches that treat both animals symmetrically, RCGAN explicitly models the **asymmetric dynamics** between initiator and receiver—leveraging the behavioral insight that initiators control event termination.

---

## 📁 Project Structure

```
cross_sucking/
├── configs/                    # Configuration files
│   ├── base.yaml              # Shared settings
│   ├── paths.yaml             # Data paths (gitignored)
│   ├── pretrain.yaml          # SSL pretraining
│   ├── detect.yaml            # Object detection
│   ├── track.yaml             # Multi-object tracking
│   ├── tubes.yaml             # Tube generation
│   ├── model.yaml             # RCGAN model
│   └── eval.yaml              # Evaluation
│
├── data/
│   ├── raw/                   # Symlinks to original data
│   ├── annotations/           # Labels + mapping rules
│   ├── manifests/             # Generated manifests
│   ├── interim/               # Intermediate artifacts
│   └── processed/             # Training-ready data
│
├── src/
│   ├── cli/                   # Command-line tools
│   ├── datasets/              # PyTorch datasets
│   ├── models/                # Model architectures
│   ├── eval/                  # Evaluation metrics
│   ├── utils/                 # Utilities
│   └── viz/                   # Visualization
│
├── notebooks/                 # Jupyter notebooks
├── runs/                      # Experiment outputs
├── docs/                      # Documentation
├── tests/                     # Unit tests
└── scripts/                   # Shell scripts
```

---

## 📊 Data Overview

### Labeled Data
| Group | Day | Cameras | Events |
|-------|-----|---------|--------|
| 1 | 1 | 7, 8, 9, 10 | 316 |
| 1 | 4 | 7, 8, 9, 10 | 407 |
| 2 | 4 | 10, 11, 12, 116 | 237 |
| 3 | 4 | 8, 12, 14, 16 | 333+ |

### Unlabeled Data (~1,680 hours)
- Groups 2-6, Week 2
- 4 cameras per group
- Used for SSL pretraining

### Behavior Distribution
- **Ear**: 87.1% (1,703 events)
- **Tail**: 11.7% (228 events)
- **Teat**: 0.6% (12 events) ← Rare but critical

---

## 🚀 Quick Start

### 1. Setup
```bash
# Clone repository
git clone https://github.com/your-username/cross_sucking.git
cd cross_sucking

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e ".[all]"
```

### 2. Configure Paths
```bash
# Copy and edit paths config
cp configs/paths.yaml.example configs/paths.yaml
# Edit paths.yaml with your data locations
```

### 3. Build Manifests
```bash
# Scan all videos and create manifest
cs-manifest build --config configs/paths.yaml

# Clean and normalize annotations
cs-clean data/annotations/interactions.xlsx
```

### 4. Run Pipeline
```bash
# Stage 1: SSL Pretraining (optional but recommended)
python -m src.cli.pretrain_ssl --config configs/pretrain.yaml

# Stage 2: Detection
cs-detect --config configs/detect.yaml

# Stage 3: Tracking
cs-track --config configs/track.yaml

# Stage 4: Build Tubes
cs-tubes --config configs/tubes.yaml

# Stage 5: Train Interaction Model
cs-train --config configs/model.yaml

# Stage 6: Evaluate
cs-eval --config configs/eval.yaml
```

---

## 📈 Pipeline Stages

```
Stage 0: Data Foundation (Week 1-2)
├── Build video manifest
├── Clean annotations  
├── Link events to videos
└── Create train/val/test splits

Stage 1: Foundation Encoder (Week 3-4)
├── SSL pretrain on unlabeled data
├── Fine-tune on labeled events
└── Extract embeddings

Stage 2: Detection + Tracking (Week 5-6)
├── YOLOv8 calf detection
├── ByteTrack multi-object tracking
└── Generate tracklets

Stage 3: Interaction Tubes (Week 7-8)
├── Pairwise tube proposals
├── RCGAN training
└── Role + target classification

Stage 4: Temporal Refinement (Week 9)
├── Boundary-aware scoring
└── Temporal NMS

Stage 5: Uncertainty + Active Learning (Week 10)
├── MC-Dropout calibration
├── Abstain policy
└── Sample selection for annotation
```

---

## 🎯 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| mAP@0.3 | Temporal detection (loose) | 0.55-0.65 |
| mAP@0.5 | Temporal detection (strict) | 0.45-0.55 |
| Role Accuracy | Initiator vs receiver | 0.75-0.85 |
| Target F1 | Ear/tail/teat/other | 0.50-0.60 |
| Teat F1 | Rare class | 0.35-0.45 |

---

## 🛠️ CLI Commands

| Command | Description |
|---------|-------------|
| `cs-manifest build` | Build video manifest |
| `cs-manifest verify` | Verify manifest integrity |
| `cs-clean` | Clean annotations |
| `cs-detect` | Run object detection |
| `cs-track` | Run multi-object tracking |
| `cs-tubes` | Generate interaction tubes |
| `cs-train` | Train interaction model |
| `cs-eval` | Evaluate model |

---

## 📚 References

- [Agriculture-Vision Workshop](https://www.agriculture-vision.com/)
- [VideoMAE](https://arxiv.org/abs/2203.12602)
- [ByteTrack](https://arxiv.org/abs/2110.06864)
- [ActionFormer](https://arxiv.org/abs/2202.07925)

---

## 📝 License

MIT License

---

## 👥 Contributors

- Your Name
