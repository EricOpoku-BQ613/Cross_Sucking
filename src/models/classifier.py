# src/models/classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ClassifierOut:
    logits: torch.Tensor  # (B, num_classes)


class LinearHead(nn.Module):
    """
    Simple, stable baseline head: Dropout + Linear.
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> ClassifierOut:
        return ClassifierOut(logits=self.net(feats))


class MLPHead(nn.Module):
    """
    Slightly stronger head if needed.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> ClassifierOut:
        return ClassifierOut(logits=self.net(feats))


class VideoClassifier(nn.Module):
    """
    Combines backbone-features and a classification head.
    Expects backbone.forward(x).feats -> (B,D)
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> ClassifierOut:
        feats = self.backbone(x).feats
        return self.head(feats)
