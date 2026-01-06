# src/training/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss on logits.
    - logits: (B, K)
    - targets: (B,) long in [0, K-1]
    - alpha: optional class weights (K,)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.label_smoothing = float(label_smoothing)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per-sample
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)  # pt = softmax prob of true class
        focal = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = focal * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    counts: torch.Tensor,
    method: str = "inv_sqrt",
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    counts: (K,) counts per class
    method:
      - "inv":     w = 1/(c+eps)
      - "inv_sqrt":w = 1/sqrt(c+eps)   (usually more stable)
      - "effective": effective number weighting (good, but optional)
    """
    counts = counts.to(torch.float32).clamp_min(0)

    if method == "inv":
        w = 1.0 / (counts + eps)
    elif method == "inv_sqrt":
        w = 1.0 / torch.sqrt(counts + eps)
    elif method == "effective":
        # Class-Balanced Loss (Cui et al.)
        beta = 0.999
        eff = 1.0 - torch.pow(torch.tensor(beta), counts)
        w = (1.0 - beta) / (eff + eps)
    else:
        raise ValueError(f"Unknown method: {method}")

    # optional normalize so average weight ~ 1
    if normalize:
        w = w * (w.numel() / (w.sum() + eps))
    return w
