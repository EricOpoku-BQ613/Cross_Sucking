# scripts/train_supervised.py
"""
Supervised training script with:
- WeightedRandomSampler for balanced batches
- TrivialAugment (from FERAL paper) for regularization
- Optional MixUp
- Fixed class weighting (disabled by default when using sampler)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

try:
    import yaml
except Exception as e:
    raise ImportError("Please `pip install pyyaml`") from e

import pandas as pd

from src.data.datasets import LabeledEventDataset
from src.models.backbone import VideoBackbone
from src.models.classifier import LinearHead, MLPHead, VideoClassifier
from src.training.losses import FocalLoss, compute_class_weights
from src.training.trainer import Trainer, TrainConfig

# Import augmentations - will be created in src/data/
try:
    from src.data.video_augmentations import VideoTrivialAugment, VideoMixUp, mixup_criterion
    HAS_AUGMENT = True
except ImportError:
    HAS_AUGMENT = False
    print("[WARNING] video_augmentations not found - training without augmentation")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    import math

    name = cfg.get("sched", {}).get("name", "none")
    if name == "none":
        return None
    if name == "cosine":
        t_max = int(cfg["sched"].get("t_max", cfg["train"]["epochs"]))
        min_lr = float(cfg["sched"].get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    if name == "cosine_warmup":
        t_max = int(cfg["sched"].get("t_max", cfg["train"]["epochs"]))
        min_lr = float(cfg["sched"].get("min_lr", 1e-6))
        warmup_epochs = int(cfg["sched"].get("warmup_epochs", 3))
        warmup_start_lr = float(cfg["sched"].get("warmup_start_lr", 1e-6))
        base_lr = float(cfg["optim"]["lr"])

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return warmup_start_lr / base_lr + (1 - warmup_start_lr / base_lr) * (epoch / warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, t_max - warmup_epochs)
            return min_lr / base_lr + 0.5 * (1 - min_lr / base_lr) * (1 + math.cos(math.pi * progress))

        print(f"[Scheduler] cosine_warmup: {warmup_epochs} warmup epochs, {t_max} total, min_lr={min_lr}")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    raise ValueError(f"Unknown scheduler: {name}")


def make_optimizer(cfg: Dict[str, Any], params):
    name = cfg["optim"]["name"].lower()
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"].get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def normalize_behavior(s: str) -> str:
    return str(s).strip().lower()


def build_label_map(class_names: List[str]) -> Dict[str, int]:
    names = [normalize_behavior(x) for x in class_names]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class_names after normalization: {class_names}")
    return {n: i for i, n in enumerate(names)}


def filter_manifest(in_csv: str | Path, out_csv: Path, keep_labels: List[str]) -> None:
    df = pd.read_csv(in_csv)
    if "behavior" not in df.columns:
        raise ValueError(f"{in_csv} missing required column: behavior")
    df["behavior"] = df["behavior"].map(normalize_behavior)
    keep = set(map(normalize_behavior, keep_labels))
    df2 = df[df["behavior"].isin(keep)].copy()
    if len(df2) == 0:
        raise RuntimeError(f"Filtering produced empty manifest: {in_csv} -> keep={sorted(list(keep))}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_csv, index=False)

    vc = df2["behavior"].value_counts().to_dict()
    print(f"[Filtered manifest] {out_csv} counts={vc} rows={len(df2)}")


def compute_weights_from_csv(train_csv: str | Path, label_map: Dict[str, int], num_classes: int, method: str) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    y = []
    for b in df["behavior"].astype(str).map(normalize_behavior).tolist():
        y.append(label_map.get(b, -1))
    y = torch.tensor(y, dtype=torch.long)
    counts = torch.zeros((num_classes,), dtype=torch.long)
    for k in range(num_classes):
        counts[k] = (y == k).sum()

    w = compute_class_weights(counts, method=method, normalize=True)
    print(f"[Class counts] {counts.tolist()}")
    print(f"[Class weights/{method}] {w.tolist()}")
    return w


def build_weighted_sampler(
    train_csv: str | Path, 
    label_map: Dict[str, int], 
    num_classes: int
) -> WeightedRandomSampler:
    """
    Build WeightedRandomSampler for class-balanced batch sampling.
    
    This ensures each batch has roughly equal representation of all classes,
    which is critical for imbalanced datasets like ear (600) vs tail (92).
    """
    df = pd.read_csv(train_csv)
    y = df["behavior"].astype(str).str.strip().str.lower().map(label_map).values
    
    # Compute inverse frequency weights per class
    class_counts = np.bincount(y, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    
    # Assign weight to each sample based on its class
    sample_weights = class_weights[y]
    
    print(f"[WeightedRandomSampler] class_counts={class_counts.tolist()}")
    print(f"[WeightedRandomSampler] class_weights={[f'{w:.6f}' for w in class_weights]}")
    print(f"[WeightedRandomSampler] Expected class ratio per batch ~= 1:1")
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,  # Must be True to oversample minority class
    )
    
    return sampler


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper that applies augmentation to an existing dataset.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--resume", type=str, default=None, help="Path to a .ckpt to resume from (e.g., runs/.../last.ckpt)")
    ap.add_argument("--weights-path", type=str, default=None,
                    help="Load model weights only from a checkpoint (no optimizer/scheduler state). "
                         "Use for two-stage fine-tuning: stage-1 trains frozen backbone, "
                         "stage-2 loads stage-1 weights and unfreezes backbone with fresh optimizer.")
    
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 42)))

    # device
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[Device] {device}")

    # labels from config (IMPORTANT)
    class_names = cfg["labels"]["class_names"]
    num_classes = int(cfg["labels"]["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(f"labels.num_classes={num_classes} but class_names has {len(class_names)} items")

    label_map = build_label_map(class_names)
    print(f"[Label map] {label_map}")

    # manifests
    train_csv = Path(cfg["data"]["train_csv"])
    val_csv = Path(cfg["data"]["val_csv"])

    # binary mode: filter train/val to ID only (ear/tail)
    filter_to_id = bool(cfg["data"].get("filter_to_id", False))
    id_labels = cfg["labels"].get("id_labels", class_names)

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    used_train_csv = train_csv
    used_val_csv = val_csv

    if filter_to_id:
        run_manifest_dir = out_dir / "manifests_used"
        used_train_csv = run_manifest_dir / "train_id.csv"
        used_val_csv = run_manifest_dir / "val_id.csv"
        filter_manifest(train_csv, used_train_csv, keep_labels=id_labels)
        filter_manifest(val_csv, used_val_csv, keep_labels=id_labels)

    # datasets
    clip_dir = cfg["data"].get("clip_dir", None)
    if clip_dir:
        print(f"[DataLoader] Clip mode: loading from {clip_dir}")

    ds_train_base = LabeledEventDataset(
        str(used_train_csv),
        mode="train",
        clip_len=int(cfg["clip"]["clip_len"]),
        fps=int(cfg["clip"]["fps"]),
        clip_dir=clip_dir,
    )
    ds_val = LabeledEventDataset(
        str(used_val_csv),
        mode="eval",
        clip_len=int(cfg["clip"]["clip_len"]),
        fps=int(cfg["clip"]["fps"]),
        clip_dir=clip_dir,
    )
    
    # Apply augmentation to training set
    use_augmentation = bool(cfg.get("augmentation", {}).get("use_trivial_augment", True))
    
    if use_augmentation and HAS_AUGMENT:
        print("[Augmentation] Using TrivialAugment for training")
        train_transform = VideoTrivialAugment()
        ds_train = AugmentedDataset(ds_train_base, transform=train_transform)
    else:
        print("[Augmentation] Disabled or not available")
        ds_train = ds_train_base

    # loaders
    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 0))
    pin = (device.type == "cuda")
    
    # Check if we should use balanced sampling
    use_balanced_sampling = bool(cfg["data"].get("balanced_sampling", True))
    
    if use_balanced_sampling:
        # Build WeightedRandomSampler for class-balanced batches
        sampler = build_weighted_sampler(used_train_csv, label_map, num_classes)
        
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=bs,
            sampler=sampler,          # Use sampler for balanced batches
            shuffle=False,            # Must be False when sampler is used
            num_workers=nw,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=(nw > 0),
        )
        print(f"[DataLoader] Using WeightedRandomSampler for balanced batches")
    else:
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=(nw > 0),
        )
        print(f"[DataLoader] Using standard shuffle (no balanced sampling)")
    
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=(nw > 0),
    )

    # model
    backbone = VideoBackbone(
        name=cfg["model"]["backbone"],
        pretrained=bool(cfg["model"].get("pretrained", False)),
        dropout=float(cfg["model"].get("backbone_dropout", 0.0)),
        freeze_backbone=bool(cfg["model"].get("freeze_backbone", False)),
        slowfast_alpha=int(cfg["model"].get("slowfast_alpha", 4)),
    )

    head_name = cfg["model"].get("head", "linear")
    if head_name == "linear":
        head = LinearHead(backbone.feat_dim, num_classes=num_classes, dropout=float(cfg["model"].get("head_dropout", 0.2)))
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

    # loss
    loss_name = cfg["loss"]["name"].lower()
    alpha = None
    
    # IMPORTANT: When using balanced sampling, class weights can cause over-correction!
    # Default to False when balanced_sampling is True
    use_class_weights = bool(cfg["loss"].get("use_class_weights", False))
    
    if use_class_weights:
        if use_balanced_sampling:
            print("[WARNING] Using class weights WITH balanced sampling may cause instability!")
            print("[WARNING] Consider setting use_class_weights: false")
        alpha = compute_weights_from_csv(used_train_csv, label_map, num_classes, method=cfg["loss"].get("weight_method", "inv_sqrt"))
        alpha = alpha.to(device)
    else:
        print("[Loss] No class weights (balanced sampling handles class imbalance)")

    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=float(cfg["loss"].get("label_smoothing", 0.1)))
    elif loss_name == "focal":
        criterion = FocalLoss(
            gamma=float(cfg["loss"].get("gamma", 2.0)),
            alpha=alpha,
            label_smoothing=float(cfg["loss"].get("label_smoothing", 0.1)),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    # optim + sched
    optimizer = make_optimizer(cfg, model.parameters())
    scheduler = make_scheduler(cfg, optimizer)

    # save label map + manifests used for reproducibility
    (out_dir / "label_map.json").write_text(__import__("json").dumps(label_map, indent=2))
    (out_dir / "manifests_used.txt").write_text(f"train={used_train_csv}\nval={used_val_csv}\n")

    # Default to macro_f1 for stable training (not recall_tail which can oscillate)
    metric_key = cfg["train"].get("metric_key", "macro_f1")
    print(f"[Training] Selecting best model by: {metric_key}")

    tcfg = TrainConfig(
        out_dir=out_dir,
        epochs=int(cfg["train"]["epochs"]),
        amp=bool(cfg["train"].get("amp", True)),
        grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
        log_every=int(cfg["train"].get("log_every", 20)),
        save_best=True,
        metric_key=metric_key,
        class_names=class_names,
        accum_steps=int(cfg["train"].get("accum_steps", 1)),
        early_stopping=bool(cfg["train"].get("early_stopping", False)),
        early_stopping_patience=int(cfg["train"].get("early_stopping_patience", 10)),
        early_stopping_min_delta=float(cfg["train"].get("early_stopping_min_delta", 0.001)),
        save_per_sample_loss=bool(cfg["train"].get("save_per_sample_loss", False)),
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        cfg=tcfg,
    )

    # Two-stage fine-tuning: load weights only (fresh optimizer) for stage 2
    if args.weights_path:
        ckpt = torch.load(args.weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[Weights] Loaded model weights from {args.weights_path} (optimizer NOT restored — fresh start)")
        print(f"[Weights] Checkpoint was epoch={ckpt.get('epoch','?')}, "
              f"val_macro_f1={ckpt.get('val_stats',{}).get('macro_f1','?'):.4f}")

    trainer.fit(dl_train, dl_val, resume_path=args.resume)
    print(f"\nDone. Artifacts in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()