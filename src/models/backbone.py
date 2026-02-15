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

try:
    from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
except Exception:
    mvit_v2_s = MViT_V2_S_Weights = None


BackboneName = Literal[
    "r3d_18", "mc3_18", "r2plus1d_18", "mvit_v2_s",
    "slowfast_r50", "x3d_m",
]


@dataclass
class BackboneOut:
    feats: torch.Tensor  # (B, D)


def _load_pytorchvideo_model(name: str, pretrained: bool):
    """Load a model from facebookresearch/pytorchvideo via torch.hub."""
    return torch.hub.load(
        "facebookresearch/pytorchvideo",
        name,
        pretrained=pretrained,
    )


class VideoBackbone(nn.Module):
    """
    Video backbone wrapper that returns a feature vector (B, D).
    Input expected: (B, C, T, H, W)

    Supports torchvision models (r3d_18, mc3_18, r2plus1d_18, mvit_v2_s)
    and pytorchvideo models (slowfast_r50, x3d_m).
    """

    def __init__(
        self,
        name: BackboneName = "r3d_18",
        pretrained: bool = False,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        slowfast_alpha: int = 2,
    ) -> None:
        super().__init__()
        self.name = name
        self.slowfast_alpha = slowfast_alpha

        # --- torchvision models ---
        if name in ("r3d_18", "mc3_18", "r2plus1d_18"):
            if r3d_18 is None:
                raise ImportError(
                    "torchvision video models not available. Install torchvision with the correct build."
                )
            builder = {"r3d_18": r3d_18, "mc3_18": mc3_18, "r2plus1d_18": r2plus1d_18}[name]
            m = builder(weights="DEFAULT" if pretrained else None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()

        elif name == "mvit_v2_s":
            if mvit_v2_s is None:
                raise ImportError("mvit_v2_s not available. Requires torchvision >= 0.14")
            m = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT if pretrained else None)
            feat_dim = m.head[1].in_features
            m.head = nn.Identity()

        # --- pytorchvideo models ---
        elif name == "x3d_m":
            m = _load_pytorchvideo_model("x3d_m", pretrained=pretrained)
            feat_dim = m.blocks[-1].proj.in_features
            # Remove classification head (proj layer), keep pooling
            m.blocks[-1].proj = nn.Identity()
            m.blocks[-1].dropout = nn.Identity()

        elif name == "slowfast_r50":
            m = _load_pytorchvideo_model("slowfast_r50", pretrained=pretrained)
            feat_dim = m.blocks[-1].proj.in_features
            m.blocks[-1].proj = nn.Identity()
            m.blocks[-1].dropout = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone name: {name}")

        self.backbone = m
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> BackboneOut:
        # x: (B, C, T, H, W)
        if self.name == "slowfast_r50":
            # Split into slow and fast pathways
            # Slow: subsample every alpha-th frame
            # Fast: all frames
            slow = x[:, :, :: self.slowfast_alpha, :, :]
            fast = x
            feats = self.backbone([slow, fast])
        else:
            feats = self.backbone(x)

        feats = self.dropout(feats)
        return BackboneOut(feats=feats)
