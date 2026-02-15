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

    Two loading modes:
      1. Source-video mode (default): reads from original 4K video on disk.
         Required CSV columns: video_path, behavior, event_offset_sec, duration_sec

      2. Clip mode (clip_dir != None): reads from pre-extracted clip files.
         Clip files must follow naming: evt_{event_idx:04d}_{behavior}.mp4
         Required CSV columns: event_idx, behavior
         Clips missing from disk are silently dropped from the dataset.
    """
    def __init__(
        self,
        csv_path: str | Path,
        mode: str = "train",
        clip_len: int = 16,
        fps: int = 12,
        size: int = 224,
        label_col: str = "behavior",
        clip_dir: Optional[str | Path] = None,
    ):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)
        self.mode = mode
        self.label_col = label_col
        self.clip_dir = Path(clip_dir) if clip_dir is not None else None

        self.cfg = ClipConfig(
            clip_len=clip_len,
            fps=fps,
            mode=("train" if mode == "train" else "eval"),
            jitter_sec=(0.7 if mode == "train" else 0.0),
        )

        # Use supervised (milder) preset
        self.frame_tf = build_frame_transform("train" if mode == "train" else "eval", size=size, preset="sup")

        if self.clip_dir is not None:
            # Clip mode: only need event_idx + label
            required = ["event_idx", self.label_col]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{self.csv_path} missing columns: {missing}. Found={list(df.columns)}")

            # Filter to rows where the clip file actually exists
            def _clip_path(row) -> Path:
                return self.clip_dir / f"evt_{int(row['event_idx']):04d}_{str(row[self.label_col]).strip().lower()}.mp4"

            mask = df.apply(lambda r: _clip_path(r).exists(), axis=1)
            n_before = len(df)
            df = df[mask].reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(f"[ClipDataset] Dropped {n_dropped}/{n_before} rows — clip files not found on disk")
            else:
                print(f"[ClipDataset] All {len(df)} clips found in {self.clip_dir}")
        else:
            # Source-video mode
            required = ["video_path", self.label_col, "event_offset_sec", "duration_sec"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{self.csv_path} missing columns: {missing}. Found={list(df.columns)}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        label_str = str(row[self.label_col]).strip().lower()
        y = LABEL_MAP.get(label_str, -1)
        if y < 0:
            raise ValueError(f"Unknown {self.label_col}='{label_str}' at row {idx} in {self.csv_path}")

        if self.clip_dir is not None:
            # Load from pre-extracted clip (no seeking needed — whole clip is the event)
            clip_path = self.clip_dir / f"evt_{int(row['event_idx']):04d}_{label_str}.mp4"
            frames = read_event_clip_decord(
                str(clip_path),
                self.cfg,
                event_offset_sec=None,
                duration_sec=None,
            )
        else:
            # Load from source video with event seeking
            frames = read_event_clip_decord(
                str(row["video_path"]),
                self.cfg,
                event_offset_sec=float(row["event_offset_sec"]),
                duration_sec=float(row["duration_sec"]),
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
