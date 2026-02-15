"""
Smoke test: validate v4 dataloaders and model forward pass.
Run from project root: python scripts/smoke_dataloaders.py
"""
import sys
import torch
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import LabeledEventDataset
from src.models.backbone import VideoBackbone
from src.models.classifier import LinearHead, VideoClassifier


def main():
    train_csv = "data/manifests/train_v4.csv"
    val_csv = "data/manifests/val_v4.csv"

    # --- Supervised dataloaders ---
    print("=" * 50)
    print("[SMOKE] Loading train dataset (v4)...")
    ds_train = LabeledEventDataset(train_csv, mode="train", clip_len=16, fps=12)
    print(f"  Train samples: {len(ds_train)}")

    print("[SMOKE] Loading val dataset (v4)...")
    ds_val = LabeledEventDataset(val_csv, mode="eval", clip_len=16, fps=12)
    print(f"  Val samples: {len(ds_val)}")

    # Fetch one sample from each
    print("[SMOKE] Fetching train sample 0...")
    x_train, y_train = ds_train[0]
    print(f"  x shape: {x_train.shape}, y: {y_train}")

    print("[SMOKE] Fetching val sample 0...")
    x_val, y_val = ds_val[0]
    print(f"  x shape: {x_val.shape}, y: {y_val}")

    # --- Batch loading ---
    print("[SMOKE] Testing batch loading (batch_size=2)...")
    dl = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=False, num_workers=0)
    xb, yb = next(iter(dl))
    print(f"  Batch x: {xb.shape}, y: {yb.shape}, labels: {yb.tolist()}")

    # --- Model forward pass ---
    print("[SMOKE] Testing model forward pass...")
    backbone = VideoBackbone(name="r3d_18", pretrained=False)
    head = LinearHead(backbone.feat_dim, num_classes=2, dropout=0.3)
    model = VideoClassifier(backbone, head)
    model.eval()

    with torch.no_grad():
        out = model(xb)
    print(f"  Output logits shape: {out.logits.shape}")
    print(f"  Output logits: {out.logits}")

    # --- GPU test ---
    if torch.cuda.is_available():
        print("[SMOKE] Testing GPU transfer...")
        model_gpu = model.cuda()
        xb_gpu = xb.cuda()
        with torch.no_grad():
            out_gpu = model_gpu(xb_gpu)
        print(f"  GPU output: {out_gpu.logits.shape} on {out_gpu.logits.device}")
        print("[GPU OK]")
    else:
        print("[SMOKE] No GPU available, skipping GPU test")

    print("=" * 50)
    print("[SMOKE] All checks passed!")


if __name__ == "__main__":
    main()
