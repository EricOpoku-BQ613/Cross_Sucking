# src/data/transforms.py
from __future__ import annotations

from typing import Optional, Callable, Union, Literal

import numpy as np
import torch

# Torchvision is optional
try:
    import torchvision.transforms as T
    from PIL import Image
except Exception:
    T = None
    Image = None


Mode = Literal["train", "eval"]
Preset = Literal["sup", "ssl"]


# -------------------------
# Core helpers
# -------------------------
def to_tensor_clip(frames: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    frames:
      - np.ndarray (T,H,W,C) or (T,H,W) uint8/float
      - torch.Tensor in similar layout
    returns:
      torch.FloatTensor (C,T,H,W) in [0,1]
    """
    if isinstance(frames, torch.Tensor):
        x = frames
    else:
        x = torch.from_numpy(np.asarray(frames))

    # Handle (T,H,W) grayscale -> (T,H,W,1)
    if x.ndim == 3:
        x = x.unsqueeze(-1)

    # (T,H,W,C) -> (C,T,H,W)
    if x.ndim == 4 and x.shape[-1] in (1, 3):
        x = x.permute(3, 0, 1, 2)

    if x.ndim != 4:
        raise ValueError(f"Expected 4D clip (C,T,H,W). Got shape={tuple(x.shape)}")

    # If grayscale (1, T, H, W) -> repeat to 3 channels
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1, 1)

    x = x.contiguous()

    # scale to [0,1] float32
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if x.numel() > 0 and x.max() > 1.5:
        x = x / 255.0

    return x


def normalize_clip(
    x: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    x: (C,T,H,W) float in [0,1]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (C,T,H,W), got {tuple(x.shape)}")
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1, 1)
    return (x - mean_t) / std_t


def apply_per_frame(x: torch.Tensor, frame_tf: Optional[Callable]) -> torch.Tensor:
    """
    Apply a torchvision/PIL transform per-frame.

    Input:
      x: torch.Tensor (C,T,H,W), float in [0,1] on CPU recommended
    Output:
      torch.Tensor (C,T,H,W), float in [0,1]
    """
    if frame_tf is None:
        return x
    if T is None or Image is None:
        return x

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"apply_per_frame expects torch.Tensor, got {type(x)}")
    if x.ndim != 4:
        raise ValueError(f"Expected (C,T,H,W), got {tuple(x.shape)}")

    # IMPORTANT: PIL path is CPU-only. Avoid silently moving GPU->CPU per frame.
    if x.is_cuda:
        raise ValueError(
            "apply_per_frame received a CUDA tensor. Keep transforms on CPU (dataset stage), "
            "then move batch to GPU in the training step."
        )

    # (C,T,H,W) -> (T,C,H,W)
    xt = x.permute(1, 0, 2, 3).contiguous()

    outs = []
    for t in range(xt.shape[0]):
        frame = xt[t]  # (C,H,W) float [0,1]
        frame_cpu = frame.detach().cpu()

        arr = (frame_cpu.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        out = frame_tf(img)  # typically torch.Tensor (C,H,W) in [0,1] after ToTensor()

        if isinstance(out, torch.Tensor):
            out_t = out
        else:
            out_np = np.asarray(out)
            if out_np.ndim == 2:
                out_np = out_np[:, :, None]
            out_t = torch.from_numpy(out_np).permute(2, 0, 1).to(torch.float32) / 255.0

        # If grayscale slipped through, expand to 3ch
        if out_t.shape[0] == 1:
            out_t = out_t.repeat(3, 1, 1)

        outs.append(out_t)

    y = torch.stack(outs, dim=0)             # (T,C,H,W)
    y = y.permute(1, 0, 2, 3).contiguous()   # (C,T,H,W)
    return y.to(dtype=x.dtype)               # keep CPU


# -------------------------
# Frame transform builders
# -------------------------
def build_frame_transform(
    mode: Mode = "train",
    size: int = 224,
    preset: Preset = "sup",
) -> Optional[Callable]:
    """
    mode:   "train" or "eval"
    preset: "sup" (milder, good for supervised) or "ssl" (stronger, for contrastive SSL)

    Tuned for: night/low-light + distance/scale variation + line artifacts,
    while keeping supervised training stable.
    """
    if T is None:
        return None

    is_train = (mode == "train")

    # Deterministic eval
    if not is_train:
        return T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])

    # -------------------------
    # Supervised augmentations
    # -------------------------
    if preset == "sup":
        ops = []

        # (1) Distance / scale variation: conservative crop (preserves context)
        # If RandomResizedCrop exists, prefer it over fixed Resize.
        if hasattr(T, "RandomResizedCrop"):
            ops.append(T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)))
        else:
            ops.append(T.Resize((size, size)))

        # (2) Geometry: small shifts/scale jitter
        if hasattr(T, "RandomAffine"):
            ops.append(T.RandomAffine(
                degrees=3,
                translate=(0.02, 0.02),
                scale=(0.92, 1.06),
            ))

        # (3) Flip: barns/cameras usually symmetric enough
        ops.append(T.RandomHorizontalFlip(p=0.5))

        # (4) Night / illumination robustness
        # Autocontrast + moderate jitter improves low-light generalization
        if hasattr(T, "RandomAutocontrast"):
            ops.append(T.RandomAutocontrast(p=0.2))
        ops.append(T.ColorJitter(
            brightness=0.45,
            contrast=0.45,
            saturation=0.15,
            hue=0.02,
        ))

        # (5) Line/artifact robustness: very mild blur
        if hasattr(T, "GaussianBlur"):
            ops.append(T.GaussianBlur(kernel_size=7, sigma=(0.1, 1.2)))

        # (6) Occasionally remove color reliance (helps night / camera differences)
        if hasattr(T, "RandomGrayscale"):
            ops.append(T.RandomGrayscale(p=0.05))

        # Convert to tensor
        ops.append(T.ToTensor())

        # (7) Minor occlusion robustness (other calves / clutter)
        # Keep low probability to avoid destabilizing rare classes.
        if hasattr(T, "RandomErasing"):
            ops.append(T.RandomErasing(
                p=0.15,
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
                value=0,
            ))

        return T.Compose(ops)

    # -------------------------
    # SSL augmentations (stronger)
    # -------------------------
    ops = [
        # Stronger crop: SSL wants invariance to zoom + viewpoint
        T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.55, contrast=0.55, saturation=0.3, hue=0.06),
    ]

    if hasattr(T, "RandomAutocontrast"):
        ops.append(T.RandomAutocontrast(p=0.3))

    if hasattr(T, "RandomGrayscale"):
        ops.append(T.RandomGrayscale(p=0.15))

    # More blur for SSL (sensor lines / compression / motion)
    if hasattr(T, "GaussianBlur"):
        ops.append(T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)))

    ops.append(T.ToTensor())

    # Stronger erasing for SSL is okay (learns invariance to occlusion)
    if hasattr(T, "RandomErasing"):
        ops.append(T.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0,
        ))

    return T.Compose(ops)
