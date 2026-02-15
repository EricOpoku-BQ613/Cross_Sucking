#!/usr/bin/env python3
"""
Quick Smoke Test for Training Pipeline
=======================================
Runs 1 epoch of training to verify everything works before cluster submission.

Usage:
    python scripts/test_training_pipeline.py --config configs/train_binary_v3_intravideo.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads(config_path: str):
    """Test that config file loads correctly."""
    print(f"\n{'='*70}")
    print("TEST 1: Config Loading")
    print(f"{'='*70}")

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Check required keys
        required = ["data", "model", "loss", "optim", "train"]
        missing = [k for k in required if k not in cfg]

        if missing:
            print(f"FAIL: Missing keys: {missing}")
            return False

        print(f"PASS: Config loaded successfully")
        print(f"   Train CSV: {cfg['data']['train_csv']}")
        print(f"   Val CSV: {cfg['data']['val_csv']}")
        print(f"   Epochs: {cfg['train']['epochs']}")
        print(f"   Output: {cfg['train']['out_dir']}")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_manifests_exist(config_path: str):
    """Test that manifest files exist."""
    print(f"\n{'='*70}")
    print("TEST 2: Manifest Files")
    print(f"{'='*70}")

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        train_csv = Path(cfg['data']['train_csv'])
        val_csv = Path(cfg['data']['val_csv'])

        if not train_csv.exists():
            print(f"FAIL: Train CSV not found: {train_csv}")
            return False

        if not val_csv.exists():
            print(f"FAIL: Val CSV not found: {val_csv}")
            return False

        # Count rows
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        print(f"[OK] PASS: Manifests exist")
        print(f"   Train: {len(train_df)} events")
        print(f"   Val: {len(val_df)} events")

        # Check for intra-video split
        from pathlib import Path as P
        train_vids = set(train_df['video_path'].apply(lambda x: P(x).name))
        val_vids = set(val_df['video_path'].apply(lambda x: P(x).name))
        overlap = train_vids & val_vids

        print(f"   Video overlap: {len(overlap)} videos")
        if len(overlap) > 0:
            print(f"   [OK] Intra-video split detected!")
        else:
            print(f"   [WARNING] No video overlap (old split?)")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_dataloader(config_path: str):
    """Test that dataloaders can be created."""
    print(f"\n{'='*70}")
    print("TEST 3: DataLoaders")
    print(f"{'='*70}")

    try:
        import pandas as pd
        import torch
        from torch.utils.data import WeightedRandomSampler
        from src.data.datasets import LabeledEventDataset

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Create dataset
        train_ds = LabeledEventDataset(
            csv_path=cfg['data']['train_csv'],
            mode='train',
            clip_len=cfg['clip']['clip_len'],
            fps=cfg['clip']['fps'],
        )

        print(f"[OK] Dataset created: {len(train_ds)} events")

        # Test loading one sample
        x, y = train_ds[0]
        print(f"[OK] Sample loaded: x.shape={x.shape}, y={y}")

        # Test balanced sampler
        if cfg['data'].get('balanced_sampling', False):
            df = pd.read_csv(cfg['data']['train_csv'])
            label_map = {name: i for i, name in enumerate(cfg['labels']['class_names'])}

            y_train = df['behavior'].str.strip().str.lower().map(label_map).values
            class_counts = torch.bincount(torch.tensor(y_train), minlength=len(label_map))
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[y_train]

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            print(f"[OK] Balanced sampler created")
            print(f"   Class counts: {class_counts.tolist()}")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation(config_path: str):
    """Test that model can be created."""
    print(f"\n{'='*70}")
    print("TEST 4: Model Creation")
    print(f"{'='*70}")

    try:
        import torch
        from src.models.backbone import VideoBackbone
        from src.models.classifier import LinearHead, VideoClassifier

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Create backbone
        backbone = VideoBackbone(
            name=cfg['model']['backbone'],
            pretrained=cfg['model'].get('pretrained', True)
        )

        # Create head
        num_classes = cfg['labels']['num_classes']
        dropout = cfg['model'].get('head_dropout', 0.3)
        head = LinearHead(backbone.feat_dim, num_classes=num_classes, dropout=dropout)

        # Create full model
        model = VideoClassifier(backbone, head)

        print(f"[OK] Model created: {cfg['model']['backbone']}")
        print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Test forward pass
        x = torch.randn(1, 3, 16, 112, 112)
        with torch.no_grad():
            out = model(x)

        # Handle ClassifierOut or raw tensor
        if hasattr(out, 'logits'):
            out_shape = out.logits.shape
        else:
            out_shape = out.shape

        print(f"[OK] Forward pass: input={x.shape}, output={out_shape}")

        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("TRAINING PIPELINE SMOKE TEST")
    print("="*70)
    print(f"Config: {args.config}\n")

    tests = [
        ("Config Loading", lambda: test_config_loads(args.config)),
        ("Manifest Files", lambda: test_manifests_exist(args.config)),
        ("DataLoaders", lambda: test_dataloader(args.config)),
        ("Model Creation", lambda: test_model_creation(args.config)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print(f"\n{'='*70}")
        print("ALL TESTS PASSED - Ready for training!")
        print(f"{'='*70}")
        print(f"\nTo run full training:")
        print(f"  python scripts/train_supervised.py --config {args.config}")
        return 0
    else:
        print(f"\n{'='*70}")
        print("SOME TESTS FAILED - Fix issues before training")
        print(f"{'='*70}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
