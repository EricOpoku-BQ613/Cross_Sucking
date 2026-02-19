# scripts/generate_predictions.py
"""
Generate predictions CSV from a trained model for evaluation.

Usage:
    python scripts/generate_predictions.py \
        --checkpoint runs/sup_binary_r3d18_boosted/best.ckpt \
        --manifest data/manifests/val_boosted_20251230_191708.csv \
        --output runs/sup_binary_r3d18_boosted/predictions_val.csv \
        --config configs/train_binary.yaml

For OOD evaluation (include teat/other samples):
    python scripts/generate_predictions.py \
        --checkpoint runs/sup_binary_r3d18_boosted/best.ckpt \
        --manifest data/manifests/smoke_ood_confirmed.csv \
        --output runs/sup_binary_r3d18_boosted/predictions_ood.csv \
        --config configs/train_binary.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:
    raise ImportError("Please `pip install pyyaml`")

from src.data.datasets import LabeledEventDataset
from src.models.backbone import VideoBackbone
from src.models.classifier import LinearHead, MLPHead, VideoClassifier


def load_cfg(path: str | Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_behavior(s: str) -> str:
    return str(s).strip().lower()


def build_model(cfg: Dict, num_classes: int, device: torch.device) -> VideoClassifier:
    """Build model from config."""
    backbone = VideoBackbone(
        name=cfg["model"]["backbone"],
        pretrained=False,  # Will load weights from checkpoint
        dropout=float(cfg["model"].get("backbone_dropout", 0.0)),
        freeze_backbone=False,
    )

    head_name = cfg["model"].get("head", "linear")
    if head_name == "linear":
        head = LinearHead(
            backbone.feat_dim, 
            num_classes=num_classes, 
            dropout=float(cfg["model"].get("head_dropout", 0.2))
        )
    elif head_name == "mlp":
        head = MLPHead(
            backbone.feat_dim,
            num_classes=num_classes,
            hidden_dim=int(cfg["model"].get("head_hidden_dim", 512)),
            dropout=float(cfg["model"].get("head_dropout", 0.3)),
        )
    else:
        raise ValueError(f"Unknown head: {head_name}")

    model = VideoClassifier(backbone, head).to(device)
    return model


def generate_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    label_map: Dict[str, int],
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference and collect predictions with probabilities.
    
    Returns DataFrame with columns:
        clip_id, true_label, pred_label, prob_<class1>, prob_<class2>, ...
    """
    model.eval()
    
    # Reverse label map: idx -> name
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(label_map))]
    
    all_records = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Dataset returns (video, label) or (video, label, clip_id)
            if len(batch) == 2:
                videos, labels = batch
                clip_ids = [f"clip_{i}" for i in range(len(labels))]
            else:
                videos, labels, clip_ids = batch
            
            videos = videos.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(videos)
            if hasattr(logits, "logits"):
                logits = logits.logits
            
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Collect results
            for i in range(len(labels)):
                record = {
                    "clip_id": clip_ids[i] if isinstance(clip_ids[i], str) else f"clip_{clip_ids[i]}",
                    "true_label": idx_to_label.get(labels[i].item(), "unknown"),
                    "pred_label": idx_to_label[preds[i].item()],
                }
                
                # Add probability for each class
                for j, name in enumerate(class_names):
                    record[f"prob_{name}"] = probs[i, j].item()
                
                # Also add raw logits (useful for energy-based OOD)
                for j, name in enumerate(class_names):
                    record[f"logit_{name}"] = logits[i, j].item()
                
                # Max softmax probability (MSP) for convenience
                record["msp"] = probs[i].max().item()
                
                all_records.append(record)
    
    return pd.DataFrame(all_records)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions CSV from trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (best.ckpt)")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Path to manifest CSV (val/test/ood)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for predictions CSV")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config YAML")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="DataLoader workers (max 4 for 4K clips on ISAAC; 8 OOMs workers)")
    parser.add_argument("--filter-labels", type=str, nargs="*", default=None,
                       help="Only include these labels (e.g., --filter-labels ear tail)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_cfg(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # Label map from config
    class_names = cfg["labels"]["class_names"]
    num_classes = len(class_names)
    label_map = {normalize_behavior(n): i for i, n in enumerate(class_names)}
    print(f"[Label map] {label_map}")
    
    # Load checkpoint
    print(f"[Loading checkpoint] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"[Checkpoint] epoch={ckpt.get('epoch', '?')} "
          f"val_macro_f1={ckpt.get('val_stats', {}).get('macro_f1', '?')}")
    
    # Build model
    model = build_model(cfg, num_classes, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Model loaded] {cfg['model']['backbone']} + {cfg['model'].get('head', 'linear')}")
    
    # Load manifest
    manifest_df = pd.read_csv(args.manifest)
    print(f"[Manifest] {args.manifest} ({len(manifest_df)} rows)")
    
    # Optionally filter labels
    if args.filter_labels:
        keep = set(normalize_behavior(l) for l in args.filter_labels)
        manifest_df["behavior"] = manifest_df["behavior"].map(normalize_behavior)
        before = len(manifest_df)
        manifest_df = manifest_df[manifest_df["behavior"].isin(keep)]
        print(f"[Filtered] {before} -> {len(manifest_df)} (kept: {keep})")
    
    # Save filtered manifest temporarily
    temp_manifest = Path(args.output).parent / "temp_manifest.csv"
    temp_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(temp_manifest, index=False)
    
    # Create dataset
    clip_dir = cfg["data"].get("clip_dir", None)
    dataset = LabeledEventDataset(
        str(temp_manifest),
        mode="eval",
        clip_len=int(cfg["clip"]["clip_len"]),
        fps=int(cfg["clip"]["fps"]),
        clip_dir=clip_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Generate predictions
    print(f"\n[Generating predictions]...")
    predictions_df = generate_predictions(model, dataloader, label_map, device)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    
    # Cleanup temp file
    if temp_manifest.exists():
        temp_manifest.unlink()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[OK] Saved predictions to {output_path}")
    print(f"{'='*60}")
    print(f"Samples: {len(predictions_df)}")
    print(f"True label distribution:")
    print(predictions_df["true_label"].value_counts().to_string())
    print(f"\nPredicted label distribution:")
    print(predictions_df["pred_label"].value_counts().to_string())
    
    # Quick accuracy check
    correct = (predictions_df["true_label"] == predictions_df["pred_label"]).sum()
    acc = correct / len(predictions_df)
    print(f"\nAccuracy: {acc:.4f} ({correct}/{len(predictions_df)})")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for label in predictions_df["true_label"].unique():
        mask = predictions_df["true_label"] == label
        cls_correct = (predictions_df.loc[mask, "pred_label"] == label).sum()
        cls_total = mask.sum()
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        print(f"  {label}: {cls_acc:.4f} ({cls_correct}/{cls_total})")

    # Threshold calibration sweep (binary: ear vs tail)
    if "prob_tail" in predictions_df.columns and "prob_ear" in predictions_df.columns:
        print(f"\n{'='*60}")
        print("THRESHOLD CALIBRATION SWEEP (binary ear vs tail)")
        print(f"{'='*60}")
        import numpy as np
        thresholds = np.arange(0.10, 0.91, 0.05)
        best_thresh, best_macro_f1 = 0.5, 0.0
        print(f"{'Thresh':>7}  {'TailPrec':>9}  {'TailRec':>8}  {'TailF1':>7}  {'EarF1':>7}  {'MacroF1':>8}")
        for t in thresholds:
            preds_t = predictions_df["prob_tail"].apply(lambda p: "tail" if p >= t else "ear")
            true = predictions_df["true_label"]
            tp_tail = ((preds_t == "tail") & (true == "tail")).sum()
            fp_tail = ((preds_t == "tail") & (true == "ear")).sum()
            fn_tail = ((preds_t == "ear") & (true == "tail")).sum()
            tp_ear  = ((preds_t == "ear") & (true == "ear")).sum()
            fp_ear  = ((preds_t == "ear") & (true == "tail")).sum()
            fn_ear  = ((preds_t == "tail") & (true == "ear")).sum()
            prec_tail = tp_tail / (tp_tail + fp_tail + 1e-8)
            rec_tail  = tp_tail / (tp_tail + fn_tail + 1e-8)
            f1_tail   = 2 * prec_tail * rec_tail / (prec_tail + rec_tail + 1e-8)
            prec_ear  = tp_ear  / (tp_ear  + fp_ear  + 1e-8)
            rec_ear   = tp_ear  / (tp_ear  + fn_ear  + 1e-8)
            f1_ear    = 2 * prec_ear  * rec_ear  / (prec_ear  + rec_ear  + 1e-8)
            macro_f1  = (f1_tail + f1_ear) / 2
            marker = " <-- best" if macro_f1 > best_macro_f1 else ""
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_thresh = t
            print(f"  {t:.2f}   {prec_tail:9.4f}  {rec_tail:8.4f}  {f1_tail:7.4f}  {f1_ear:7.4f}  {macro_f1:8.4f}{marker}")
        print(f"\nBest threshold: {best_thresh:.2f}  -> macro_f1={best_macro_f1:.4f}")


if __name__ == "__main__":
    main()