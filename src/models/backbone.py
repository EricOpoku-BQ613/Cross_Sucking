# src/models/backbone.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18
except Exception:
    r3d_18 = mc3_18 = r2plus1d_18 = None


BackboneName = Literal["r3d_18", "mc3_18", "r2plus1d_18"]


@dataclass
class BackboneOut:
    feats: torch.Tensor  # (B, D)


class VideoBackbone(nn.Module):
    """
    Torchvision video backbone wrapper that returns a feature vector (B, D).
    Input expected: (B, C, T, H, W)
    """

    def __init__(
        self,
        name: BackboneName = "r3d_18",
        pretrained: bool = False,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if r3d_18 is None:
            raise ImportError(
                "torchvision video models not available. Install torchvision with the correct build."
            )

        if name == "r3d_18":
            m = r3d_18(weights="DEFAULT" if pretrained else None)
        elif name == "mc3_18":
            m = mc3_18(weights="DEFAULT" if pretrained else None)
        elif name == "r2plus1d_18":
            m = r2plus1d_18(weights="DEFAULT" if pretrained else None)
        else:
            raise ValueError(f"Unknown backbone name: {name}")

        feat_dim = m.fc.in_features
        m.fc = nn.Identity()  # backbone now outputs (B, feat_dim)

        self.backbone = m
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> BackboneOut:
        # x: (B,C,T,H,W)
        feats = self.backbone(x)  # (B, D)
        feats = self.dropout(feats)
        return BackboneOut(feats=feats)
