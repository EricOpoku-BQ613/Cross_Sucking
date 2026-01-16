# src/training/trainer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# For learning curve plots
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


@dataclass
class TrainConfig:
    out_dir: Path
    epochs: int = 20

    # performance / stability
    amp: bool = True
    grad_clip: float = 1.0
    accum_steps: int = 1
    log_every: int = 20

    # checkpointing
    save_best: bool = True
    metric_key: str = "macro_f1"
    
    # NEW: class names for logging (optional)
    class_names: Optional[List[str]] = None


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
    k = (target >= 0) & (target < num_classes)
    t = target[k].to(torch.int64)
    p = pred[k].to(torch.int64)
    cm.index_put_((t, p), torch.ones_like(t), accumulate=True)
    return cm


def _metrics_from_cm(cm: torch.Tensor, class_names: Optional[List[str]] = None, eps: float = 1e-8) -> Dict[str, float]:
    cmf = cm.to(torch.float32)
    tp = torch.diag(cmf)
    fp = cmf.sum(dim=0) - tp
    fn = cmf.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    out = {
        "acc": (tp.sum() / (cmf.sum() + eps)).item(),
        "macro_f1": f1.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_precision": precision.mean().item(),
    }
    
    # Per-class metrics with readable names
    for i in range(cm.shape[0]):
        name = class_names[i] if class_names and i < len(class_names) else f"c{i}"
        out[f"f1_{name}"] = f1[i].item()
        out[f"recall_{name}"] = recall[i].item()
        out[f"precision_{name}"] = precision[i].item()
        out[f"support_{name}"] = cmf.sum(dim=1)[i].item()
        
        # Keep numeric keys for backwards compatibility
        out[f"f1_c{i}"] = f1[i].item()
        out[f"recall_c{i}"] = recall[i].item()
        out[f"support_c{i}"] = cmf.sum(dim=1)[i].item()
    
    return out


def _extract_logits(model_out):
    return model_out.logits if hasattr(model_out, "logits") else model_out


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        num_classes: int,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.cfg = cfg

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = bool(cfg.amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_metric = float("-inf")
        self.best_path = self.cfg.out_dir / "best.ckpt"
        self.last_path = self.cfg.out_dir / "last.ckpt"
        
        # NEW: History for learning curves
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "val_loss": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "val_tail_recall": [],  # Critical for imbalanced
            "val_tail_f1": [],
        }
        
        # NEW: Track collapse warnings
        self._collapse_warnings = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume_path: Optional[str] = None) -> None:
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        # >>> PUT B HERE <<<
        start_epoch = 1
        if resume_path:
            start_epoch = self._load_checkpoint(Path(resume_path))
            print(f"[RESUME] Starting from epoch {start_epoch}/{self.cfg.epochs}")


        for epoch in range(start_epoch, self.cfg.epochs + 1):
            train_stats = self._run_epoch(train_loader, train=True, epoch=epoch)
            val_stats = self._run_epoch(val_loader, train=False, epoch=epoch)

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats.get(self.cfg.metric_key, 0.0))
                else:
                    self.scheduler.step()

            # Save last
            self._save(self.last_path, epoch, train_stats, val_stats)

            # Save best
            metric = val_stats.get(self.cfg.metric_key, float("-inf"))
            is_best = False
            if self.cfg.save_best and metric > self.best_metric:
                self.best_metric = metric
                self._save(self.best_path, epoch, train_stats, val_stats)
                is_best = True

            lr = self.optimizer.param_groups[0]["lr"]
            
            # Update history
            self._update_history(epoch, lr, train_stats, val_stats)
            
            # Print epoch summary with TAIL METRICS
            self._print_epoch_summary(epoch, lr, train_stats, val_stats, is_best)
            
            # Check for collapse
            self._check_collapse(val_stats, epoch)

        # Save learning curves
        self._save_history()
        if HAS_PLT:
            self._plot_learning_curves()
        
        print("\n" + "="*70)
        print(f"TRAINING COMPLETE - Best {self.cfg.metric_key}: {self.best_metric:.4f}")
        print("="*70)

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> Dict[str, float]:
        self.model.train(mode=train)

        total_loss = 0.0
        total_n = 0
        cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)

        accum_steps = max(1, int(self.cfg.accum_steps))
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # One-time distribution check (first batch only)
            if step == 0:
                with torch.no_grad():
                    uniq_y, cnt_y = torch.unique(y, return_counts=True)
                    dist_str = {int(k): int(v) for k, v in zip(uniq_y.cpu(), cnt_y.cpu())}
                    print(f"[{'TR' if train else 'VA'}] epoch={epoch} y_dist: {dist_str}", end="")

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                logits = _extract_logits(self.model(x))
                loss = self.criterion(logits, y)

            loss_to_backprop = loss / accum_steps

            if train:
                self.scaler.scale(loss_to_backprop).backward()

                do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
                if do_step:
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_n += bs

            preds = torch.argmax(logits, dim=1)
            cm += _confusion_matrix(preds, y, self.num_classes)

            # Prediction distribution check (first batch only)
            if step == 0:
                with torch.no_grad():
                    uniq_p, cnt_p = torch.unique(preds, return_counts=True)
                    dist_str = {int(k): int(v) for k, v in zip(uniq_p.cpu(), cnt_p.cpu())}
                    print(f" | pred_dist: {dist_str}")

            if train and (step % self.cfg.log_every == 0):
                print(f"[TR] epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f}")

        stats = _metrics_from_cm(cm, class_names=self.cfg.class_names)
        stats["loss"] = total_loss / max(1, total_n)
        
        # Store confusion matrix for later analysis
        stats["_cm"] = cm.cpu().numpy().tolist()
        
        return stats

    def _update_history(self, epoch: int, lr: float, train_stats: Dict, val_stats: Dict) -> None:
        """Update training history for learning curves."""
        self.history["epoch"].append(epoch)
        self.history["lr"].append(lr)
        self.history["train_loss"].append(train_stats["loss"])
        self.history["val_loss"].append(val_stats["loss"])
        self.history["train_macro_f1"].append(train_stats["macro_f1"])
        self.history["val_macro_f1"].append(val_stats["macro_f1"])
        
        # Tail metrics (class 1 for binary ear/tail)
        tail_recall = val_stats.get("recall_tail", val_stats.get("recall_c1", 0.0))
        tail_f1 = val_stats.get("f1_tail", val_stats.get("f1_c1", 0.0))
        self.history["val_tail_recall"].append(tail_recall)
        self.history["val_tail_f1"].append(tail_f1)

    def _print_epoch_summary(self, epoch: int, lr: float, train_stats: Dict, val_stats: Dict, is_best: bool) -> None:
        """Print comprehensive epoch summary with tail metrics."""
        best_marker = " ★ BEST" if is_best else ""
        
        # Get tail metrics
        tail_recall = val_stats.get("recall_tail", val_stats.get("recall_c1", 0.0))
        tail_f1 = val_stats.get("f1_tail", val_stats.get("f1_c1", 0.0))
        tail_support = val_stats.get("support_tail", val_stats.get("support_c1", 0))
        
        ear_recall = val_stats.get("recall_ear", val_stats.get("recall_c0", 0.0))
        ear_f1 = val_stats.get("f1_ear", val_stats.get("f1_c0", 0.0))
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch}/{self.cfg.epochs} | lr={lr:.2e}{best_marker}")
        print(f"{'─'*70}")
        print(f"  Loss:      train={train_stats['loss']:.4f}  val={val_stats['loss']:.4f}")
        print(f"  Macro F1:  train={train_stats['macro_f1']:.4f}  val={val_stats['macro_f1']:.4f}")
        print(f"  Accuracy:  val={val_stats['acc']:.4f}")
        print(f"  ┌─────────────────────────────────────────")
        print(f"  │ EAR:   F1={ear_f1:.4f}  Recall={ear_recall:.4f}")
        print(f"  │ TAIL:  F1={tail_f1:.4f}  Recall={tail_recall:.4f}  (n={int(tail_support)})")
        print(f"  └─────────────────────────────────────────")

    def _check_collapse(self, val_stats: Dict, epoch: int) -> None:
        """Warn if model is collapsing to majority class."""
        tail_recall = val_stats.get("recall_tail", val_stats.get("recall_c1", 0.0))
        
        if tail_recall < 0.01:  # Essentially zero
            self._collapse_warnings += 1
            if self._collapse_warnings >= 3:
                print(f"\n⚠️  WARNING: Tail recall ≈ 0 for {self._collapse_warnings} epochs!")
                print(f"    Model may be ignoring minority class. Consider:")
                print(f"    - Increasing class weights")
                print(f"    - Lower learning rate")
                print(f"    - More aggressive focal loss gamma")
        else:
            self._collapse_warnings = 0  # Reset if recovering

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.cfg.out_dir / "training_history.json"
        # Remove non-serializable items
        save_history = {k: v for k, v in self.history.items() if not k.startswith("_")}
        with open(history_path, "w") as f:
            json.dump(save_history, f, indent=2)
        print(f"\n[OK] Saved training history to {history_path}")

    def _plot_learning_curves(self) -> None:
        """Plot and save learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = self.history["epoch"]
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.history["train_loss"], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history["val_loss"], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Macro F1 curves
        ax = axes[0, 1]
        ax.plot(epochs, self.history["train_macro_f1"], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history["val_macro_f1"], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro F1')
        ax.set_title('Macro F1 Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Tail metrics (CRITICAL)
        ax = axes[1, 0]
        ax.plot(epochs, self.history["val_tail_recall"], 'g-', label='Tail Recall', linewidth=2, marker='o')
        ax.plot(epochs, self.history["val_tail_f1"], 'm-', label='Tail F1', linewidth=2, marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Tail Class Metrics (Val) - CRITICAL')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history["lr"], 'k-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.cfg.out_dir / "learning_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved learning curves to {plot_path}")

    def _load_checkpoint(self, ckpt_path: Path) -> int:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # model
        self.model.load_state_dict(ckpt["model"])

        # optimizer / scaler
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            self.scaler.load_state_dict(ckpt["scaler"])

        # scheduler (optional)
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[RESUME] Warning: could not load scheduler state: {e}")

        # best metric so far
        self.best_metric = float(ckpt.get("best_metric", float("-inf")))

        # continue at next epoch
        last_epoch = int(ckpt.get("epoch", 0))
        return last_epoch + 1


    def _save(self, path: Path, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None:
        # Remove non-serializable items before saving
        train_stats_clean = {k: v for k, v in train_stats.items() if not k.startswith("_")}
        val_stats_clean = {k: v for k, v in val_stats.items() if not k.startswith("_")}
        
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
            "train_stats": train_stats_clean,
            "val_stats": val_stats_clean,
            "best_metric": self.best_metric,
            "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in self.cfg.__dict__.items()},
        }
        torch.save(payload, path)
