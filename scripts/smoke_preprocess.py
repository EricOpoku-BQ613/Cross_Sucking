# scripts/smoke_preprocess.py
from pathlib import Path
import pandas as pd
import torch

from src.data.datasets import LabeledVideoDataset

def main():
    csv_path = Path("data/manifests/train.csv")
    df = pd.read_csv(csv_path).head(20)

    # temporary csv for smoke
    tmp = Path("data/reports/preprocess_debug/train_head20.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)

    ds = LabeledVideoDataset(tmp, mode="train", clip_len=16, fps=12)

    ok, fail = 0, 0
    for i in range(len(ds)):
        try:
            x, y = ds[i]
            assert isinstance(x, torch.Tensor)
            print(f"[OK] {i}: x={tuple(x.shape)} y={y.item()} min={x.min().item():.3f} max={x.max().item():.3f}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {i}: {e}")
            fail += 1

    print(f"\nDone. ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
