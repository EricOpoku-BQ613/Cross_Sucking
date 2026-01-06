# src/data/loaders.py
from __future__ import annotations
from torch.utils.data import DataLoader

from src.data.datasets import LabeledEventDataset, SSLUnlabeledVideoDataset


def make_supervised_loader(
    csv_path: str,
    mode: str,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    ds = LabeledEventDataset(csv_path, mode=mode, clip_len=16, fps=12)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == "train"),
        persistent_workers=(num_workers > 0),
    )


def make_ssl_loader(
    csv_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    ds = SSLUnlabeledVideoDataset(csv_path, clip_len=16, fps=12)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
