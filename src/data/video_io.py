# src/data/video_io.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class ClipConfig:
    clip_len: int = 16
    fps: int = 12          # target sampling fps
    mode: str = "train"    # "train" or "eval"
    jitter_sec: float = 0.5  # only used in train

def read_event_clip_decord(
    video_path: str | Path,
    cfg: ClipConfig,
    event_offset_sec: float | None = None,
    duration_sec: float | None = None,
) -> np.ndarray:
    """
    Returns RGB uint8 frames: (T, H, W, 3)
    If event_offset_sec is provided, we center the clip around the event.
    """
    from decord import VideoReader, cpu

    video_path = str(video_path)
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    # FPS
    try:
        src_fps = float(vr.get_avg_fps())
    except Exception:
        src_fps = 0.0

    # If fps is unknown, fallback to uniform frame sampling
    if src_fps <= 0:
        start = 0
        if cfg.mode == "train":
            start = np.random.randint(0, max(1, n - cfg.clip_len + 1))
        else:
            start = max(0, (n - cfg.clip_len) // 2)
        idx = np.arange(start, min(n, start + cfg.clip_len))
        if len(idx) < cfg.clip_len:
            idx = np.pad(idx, (0, cfg.clip_len - len(idx)), mode="edge")
        return vr.get_batch(idx.tolist()).asnumpy()

    # stride to approximate target fps
    stride = max(1, round(src_fps / cfg.fps))
    clip_span_frames = (cfg.clip_len - 1) * stride
    max_start = n - clip_span_frames - 1

    # Determine center frame (event-centered)
    if event_offset_sec is not None:
        # center at the middle of the event when possible
        dur = float(duration_sec) if duration_sec is not None else 0.0
        center_sec = float(event_offset_sec) + 0.5 * max(0.0, dur)

        if cfg.mode == "train" and cfg.jitter_sec > 0:
            center_sec += np.random.uniform(-cfg.jitter_sec, cfg.jitter_sec)

        center_frame = int(round(center_sec * src_fps))
    else:
        # no event info -> random or center
        if cfg.mode == "train":
            center_frame = np.random.randint(0, n)
        else:
            center_frame = n // 2

    # Convert center frame to start frame for the strided clip
    start = center_frame - clip_span_frames // 2
    start = int(np.clip(start, 0, max(0, max_start)))

    idx = start + np.arange(cfg.clip_len) * stride
    idx = np.clip(idx, 0, n - 1)
    return vr.get_batch(idx.tolist()).asnumpy()
