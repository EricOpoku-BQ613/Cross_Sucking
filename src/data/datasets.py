# src/data/datasets.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .video_io import ClipConfig, read_event_clip_decord
from .transforms import to_tensor_clip, build_frame_transform, apply_per_frame, normalize_clip


LABEL_MAP = {"ear": 0, "tail": 1, "teat": 2, "other": 3}


class LabeledEventDataset(Dataset):
    """
    Supervised dataset that returns a clip tensor and a class label.

    Expected columns in csv:
      - video_path
      - behavior
      - event_offset_sec
      - duration_sec
    """
    def __init__(
        self,
        csv_path: str | Path,
        mode: str = "train",
        clip_len: int = 16,
        fps: int = 12,
        size: int = 224,
        label_col: str = "behavior",
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.mode = mode
        self.label_col = label_col

        self.cfg = ClipConfig(
            clip_len=clip_len,
            fps=fps,
            mode=("train" if mode == "train" else "eval"),
            jitter_sec=(0.7 if mode == "train" else 0.0),
        )

        # Use supervised (milder) preset
        self.frame_tf = build_frame_transform("train" if mode == "train" else "eval", size=size, preset="sup")

        required = ["video_path", self.label_col, "event_offset_sec", "duration_sec"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"{self.csv_path} missing columns: {missing}. Found={list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        video_path = str(row["video_path"])

        label_str = str(row[self.label_col]).strip().lower()
        y = LABEL_MAP.get(label_str, -1)
        if y < 0:
            raise ValueError(f"Unknown {self.label_col}='{label_str}' at row {idx} in {self.csv_path}")

        event_offset_sec = float(row["event_offset_sec"])
        duration_sec = float(row["duration_sec"])

        frames = read_event_clip_decord(
            video_path,
            self.cfg,
            event_offset_sec=event_offset_sec,
            duration_sec=duration_sec,
        )

        x = to_tensor_clip(frames)               # (C,T,H,W) float [0,1] CPU
        x = apply_per_frame(x, self.frame_tf)    # (C,T,H,W) float [0,1] CPU
        x = normalize_clip(x)                    # normalized CPU tensor

        return x, torch.tensor(y, dtype=torch.long)


class SSLUnlabeledVideoDataset(Dataset):
    """
    SSL dataset that returns two augmented views (v1, v2) from an unlabeled video.

    Expected columns in csv:
      - video_path
    """
    def __init__(
        self,
        csv_path: str | Path,
        clip_len: int = 16,
        fps: int = 12,
        size: int = 224,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        if "video_path" not in self.df.columns:
            raise ValueError(f"{self.csv_path} missing 'video_path'. Found={list(self.df.columns)}")

        self.cfg = ClipConfig(clip_len=clip_len, fps=fps, mode="train", jitter_sec=0.0)

        # Stronger SSL preset (two independent draws)
        self.tf1 = build_frame_transform("train", size=size, preset="ssl")
        self.tf2 = build_frame_transform("train", size=size, preset="ssl")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        video_path = str(self.df.iloc[idx]["video_path"])

        # For SSL we sample a generic clip (implementation inside read_event_clip_decord)
        frames = read_event_clip_decord(video_path, self.cfg, event_offset_sec=None, duration_sec=None)

        x = to_tensor_clip(frames)  # (C,T,H,W) float [0,1] CPU

        v1 = normalize_clip(apply_per_frame(x, self.tf1))
        v2 = normalize_clip(apply_per_frame(x, self.tf2))
        return v1, v2


# Backward-compatible alias (scripts expect this name)
LabeledVideoDataset = LabeledEventDataset
