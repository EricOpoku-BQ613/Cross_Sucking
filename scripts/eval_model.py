"""
scripts/eval_model.py
Run evaluation on a saved checkpoint against any split CSV.

Usage:
    python scripts/eval_model.py \
        --checkpoint /path/to/best.ckpt \
        --config     configs/train_binary_v4_mvit_isaac.yaml \
        --test-csv   data/manifests/test_ood_v4.csv \
        --split-name ood_test

Outputs:
    - Metrics printed to stdout
    - JSON saved alongside the checkpoint as eval_<split_name>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.datasets import LabeledEventDataset
from src.models.backbone import VideoBackbone
from src.models.classifier import LinearHead, VideoClassifier


# ── helpers ────────────────────────────────────────────────────────────────

def normalize_behavior(s: str) -> str:
    return str(s).strip().lower()


def build_label_map(class_names):
    names = [normalize_behavior(x) for x in class_names]
    return {n: i for i, n in enumerate(names)}


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def confusion_matrix(pred, target, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(target, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def metrics_from_cm(cm, class_names, eps=1e-8):
    cmf = cm.float()
    tp = torch.diag(cmf)
    fp = cmf.sum(0) - tp
    fn = cmf.sum(1) - tp
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)

    out = {
        "accuracy":        (tp.sum() / (cmf.sum() + eps)).item(),
        "macro_f1":        f1.mean().item(),
        "macro_recall":    rec.mean().item(),
        "macro_precision": prec.mean().item(),
        "confusion_matrix": cm.tolist(),
    }
    for i, name in enumerate(class_names):
        out[f"f1_{name}"]        = f1[i].item()
        out[f"recall_{name}"]    = rec[i].item()
        out[f"precision_{name}"] = prec[i].item()
        out[f"support_{name}"]   = int(cmf.sum(1)[i].item())
    return out


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best.ckpt or last.ckpt")
    ap.add_argument("--config",     required=True, help="YAML config used for training")
    ap.add_argument("--test-csv",   required=True, help="CSV manifest to evaluate on")
    ap.add_argument("--split-name", default="test", help="Label for output file (e.g. ood_test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers",type=int, default=4)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    class_names = cfg["labels"]["class_names"]
    num_classes = int(cfg["labels"]["num_classes"])
    label_map   = build_label_map(class_names)
    print(f"[Label map] {label_map}")

    # ── dataset ──────────────────────────────────────────────────────────
    import pandas as pd, tempfile, os

    df = pd.read_csv(args.test_csv)
    df["behavior"] = df["behavior"].astype(str).map(normalize_behavior)
    keep = set(label_map.keys())
    df = df[df["behavior"].isin(keep)].copy()
    print(f"[Test CSV] {len(df)} samples after filtering | "
          f"{df['behavior'].value_counts().to_dict()}")

    # write filtered CSV to a temp file so dataset can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()

    clip_dir = cfg["data"].get("clip_dir", None)
    ds = LabeledEventDataset(
        csv_path=tmp.name,
        label_map=label_map,
        clip_len=int(cfg["clip"]["clip_len"]),
        fps=float(cfg["clip"]["fps"]),
        clip_dir=clip_dir,
        train=False,       # validation/center-crop transforms
    )
    os.unlink(tmp.name)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[DataLoader] {len(ds)} samples, {len(loader)} batches")

    # ── model ────────────────────────────────────────────────────────────
    backbone = VideoBackbone(
        model_name=cfg["model"]["backbone"],
        pretrained=False,      # weights come from checkpoint
    )
    head = LinearHead(
        in_features=backbone.out_features,
        num_classes=num_classes,
        dropout=float(cfg["model"].get("head_dropout", 0.0)),
    )
    model = VideoClassifier(backbone, head).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    trained_epoch = ckpt.get("epoch", "?")
    best_metric   = ckpt.get("best_metric", "?")
    print(f"[Checkpoint] epoch={trained_epoch}  best_train_metric={best_metric}")

    # ── inference ────────────────────────────────────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        for step, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y)
            all_probs.append(probs.cpu())

            if (step + 1) % 10 == 0:
                print(f"  [{step+1}/{len(loader)}] batches done")

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs  = torch.cat(all_probs)

    # ── metrics ──────────────────────────────────────────────────────────
    cm      = confusion_matrix(all_preds, all_labels, num_classes)
    results = metrics_from_cm(cm, class_names)

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS — {args.split_name}")
    print("="*60)
    print(f"  Samples    : {len(all_labels)}")
    print(f"  Accuracy   : {results['accuracy']:.4f}")
    print(f"  Macro F1   : {results['macro_f1']:.4f}")
    print(f"  ┌────────────────────────────────────")
    for name in class_names:
        sup = results[f"support_{name}"]
        f1  = results[f"f1_{name}"]
        rec = results[f"recall_{name}"]
        pre = results[f"precision_{name}"]
        print(f"  │ {name.upper():6s}  F1={f1:.4f}  Recall={rec:.4f}  Prec={pre:.4f}  (n={sup})")
    print(f"  └────────────────────────────────────")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm.tolist()):
        print(f"    {class_names[i]:6s}: {row}")

    # ── save ─────────────────────────────────────────────────────────────
    out_dir  = Path(args.checkpoint).parent
    out_path = out_dir / f"eval_{args.split_name}.json"
    payload  = {
        "split":       args.split_name,
        "checkpoint":  args.checkpoint,
        "epoch":       trained_epoch,
        "n_samples":   len(all_labels),
        "metrics":     results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
