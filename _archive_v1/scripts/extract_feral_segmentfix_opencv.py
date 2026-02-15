#!/usr/bin/env python3
"""
Extract event clips for FERAL with robust segment alignment + OpenCV extraction.

Fixes:
- Segment misalignment (event time not in the declared video segment)
- duration_sec == 0 (forces a minimum duration)
- Mac resource-fork filenames like '._XXXX.mp4'
- Downscale / crop clips so files aren't huge (4K -> 256/512)
- Optional blank/empty-clip detection (--validate)
- Writes extracted_manifest.csv with resolved alignment info for audits

Usage:
  python scripts/extract_feral_segmentfix_opencv.py `
    --manifest data/manifests/train_CLEAN_SEGMENTFIX_NO_DOTUNDERSCORE.csv `
    --outdir feral_test_clips `
    --n 50 --fps 12 --pre 2 --post 2 --min_dur 1.0 --max_shift 3 `
    --size 256 --resize_mode center_crop --validate

Tip:
  If you want to prevent very long clips:
    --max_len_sec 30
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import cv2
import numpy as np


_TS_RE = re.compile(r"_(\d{14})_(\d{14})")


# ----------------------------
# Time parsing helpers
# ----------------------------
def parse_segment_times(video_filename: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    m = _TS_RE.search(str(video_filename))
    if not m:
        return None, None
    try:
        s = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        e = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
        return s, e
    except Exception:
        return None, None


def parse_hms(time_str) -> Optional[Tuple[int, int, int]]:
    if time_str is None or (isinstance(time_str, float) and pd.isna(time_str)):
        return None
    s = str(time_str).strip()
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", s)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def event_datetime_from_start_time(seg_start: datetime, start_time) -> Optional[datetime]:
    hms = parse_hms(start_time)
    if not (seg_start and hms):
        return None

    ev = datetime(seg_start.year, seg_start.month, seg_start.day, hms[0], hms[1], hms[2])

    # midnight boundary heuristic
    if ev < seg_start - timedelta(hours=6):
        ev += timedelta(days=1)
    return ev


def duration_from_times(seg_start: datetime, start_time, end_time) -> Optional[float]:
    if not seg_start:
        return None
    st = event_datetime_from_start_time(seg_start, start_time)
    en = event_datetime_from_start_time(seg_start, end_time)
    if not st or not en:
        return None
    if en < st:
        en += timedelta(days=1)
    return max(0.0, (en - st).total_seconds())


def replace_segment_in_filename(video_filename: str, new_start: datetime, new_end: datetime) -> str:
    base = str(video_filename)
    m = _TS_RE.search(base)
    if not m:
        return base
    new_ts = f"_{new_start.strftime('%Y%m%d%H%M%S')}_{new_end.strftime('%Y%m%d%H%M%S')}"
    return base[:m.start()] + new_ts + base[m.end():]


def replace_filename_in_path(video_path: str, old_filename: str, new_filename: str) -> str:
    p = Path(str(video_path))
    if p.name == str(old_filename):
        return str(p.with_name(new_filename))
    return str(p.parent / new_filename)


def sanitize_resource_fork(path_str: str) -> str:
    """
    If path ends with '\\._NAME.mp4', try '\\NAME.mp4'
    """
    p = Path(path_str)
    if p.name.startswith("._"):
        return str(p.with_name(p.name[2:]))
    return path_str


# ----------------------------
# Video helpers
# ----------------------------
def get_video_info(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    dur = (nframes / fps) if (fps > 0 and nframes > 0) else None
    return dur, fps, nframes, w, h


def _resize_frame(frame: np.ndarray, size: int, mode: str) -> np.ndarray:
    """
    mode:
      - stretch: resize directly to (size, size)
      - letterbox: keep aspect, pad to square
      - center_crop: keep aspect, resize so min side >= size, then crop center square
    """
    if size is None:
        return frame

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    if mode == "stretch":
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

    if mode == "letterbox":
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=resized.dtype)
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    # center_crop (default)
    scale = max(size / w, size / h)  # ensure both >= size after resize
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = max(0, (nh - size) // 2)
    x0 = max(0, (nw - size) // 2)
    return resized[y0:y0+size, x0:x0+size]


def extract_clip_opencv(
    src_path: str,
    out_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: int = 12,
    size: Optional[int] = 256,
    resize_mode: str = "center_crop",
) -> Tuple[bool, int]:
    """
    Extract clip [start_sec, end_sec) using OpenCV, downsample to target_fps,
    and downscale to size x size to keep clip sizes small.

    Returns (ok, frames_written)
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False, 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        cap.release()
        return False, 0

    start_frame = int(start_sec * src_fps)
    end_frame = int(end_sec * src_fps)
    if end_frame <= start_frame:
        cap.release()
        return False, 0

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Output size
    if size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = size, size

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # most reliable on Windows
    writer = cv2.VideoWriter(out_path, fourcc, float(target_fps), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        return False, 0

    step = max(1, int(round(src_fps / float(target_fps))))
    written = 0
    current = start_frame

    while current < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if size is not None:
            frame = _resize_frame(frame, size=size, mode=resize_mode)

        writer.write(frame)
        written += 1
        current += step

    cap.release()
    writer.release()

    # Sanity
    p = Path(out_path)
    if written < 3 or (not p.exists()) or p.stat().st_size < 2048:
        return False, written

    return True, written


def count_frames_opencv(path: str) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def looks_blank_video(
    path: str,
    sample_frames: int = 10,
    var_thresh: float = 5.0,
    diff_thresh: float = 1.0,
) -> bool:
    """
    Stronger sanity test:
      - low grayscale variance => blank-ish
      - frames almost identical => likely wrong extraction / static / decode garbage
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return True
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return True

    idxs = np.linspace(0, max(0, n - 1), num=min(sample_frames, n), dtype=int)
    vars_ = []
    diffs_ = []
    prev = None

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vars_.append(float(np.var(gray)))

        if prev is not None:
            diffs_.append(float(np.mean(np.abs(gray.astype(np.float32) - prev.astype(np.float32)))))
        prev = gray

    cap.release()

    if not vars_:
        return True

    vmean = float(np.mean(vars_))
    dmean = float(np.mean(diffs_)) if diffs_ else 999.0

    # blank if very low texture OR frames nearly identical
    return (vmean < var_thresh) or (dmean < diff_thresh)


# ----------------------------
# Segment resolving
# ----------------------------
@dataclass
class ResolvedEvent:
    src_path: str
    seg_start: datetime
    seg_end: datetime
    offset_sec: float
    duration_sec: float
    used_shift: int


def resolve_event(row: pd.Series, max_shift: int, min_dur: float, debug: bool) -> Optional[ResolvedEvent]:
    video_path = sanitize_resource_fork(str(row.get("video_path", "")))
    video_filename = str(row.get("video_filename", Path(video_path).name))

    seg_start, seg_end = parse_segment_times(video_filename)
    if not seg_start or not seg_end:
        if debug:
            print(f"[WARN] Cannot parse segment from filename: {video_filename}")
        return None

    seg_len = (seg_end - seg_start).total_seconds()
    if seg_len <= 0:
        return None

    # Event datetime from wall-clock start_time
    ev_dt = event_datetime_from_start_time(seg_start, row.get("start_time"))

    # duration
    dur_csv = pd.to_numeric(row.get("duration_sec", np.nan), errors="coerce")
    dur = float(dur_csv) if np.isfinite(dur_csv) else 0.0
    if dur <= 0:
        dur2 = duration_from_times(seg_start, row.get("start_time"), row.get("end_time"))
        dur = float(dur2) if (dur2 is not None and dur2 > 0) else 0.0
    dur = max(min_dur, dur)

    # If we can't parse wall-clock event time, fall back to event_offset_sec
    if not ev_dt:
        off = pd.to_numeric(row.get("event_offset_sec", np.nan), errors="coerce")
        if not np.isfinite(off):
            return None
        off = float(off)
        used_shift = int(off // seg_len) if off >= seg_len else 0
        off2 = off - used_shift * seg_len
        # path may still be wrong; but at least stays in-range
        return ResolvedEvent(video_path, seg_start, seg_end, float(off2), float(dur), used_shift)

    # Try shift candidates: 0, +1, -1, +2, -2, ...
    shifts = [0] + [k for t in range(1, max_shift + 1) for k in (t, -t)]
    for k in shifts:
        cand_start = seg_start + timedelta(seconds=k * seg_len)
        cand_end = seg_end + timedelta(seconds=k * seg_len)

        if not (cand_start <= ev_dt < cand_end):
            continue

        cand_filename = replace_segment_in_filename(video_filename, cand_start, cand_end)
        cand_path = replace_filename_in_path(video_path, video_filename, cand_filename)
        cand_path = sanitize_resource_fork(cand_path)

        if Path(cand_path).exists():
            offset = (ev_dt - cand_start).total_seconds()
            if debug:
                print(f"[RESOLVE] evt={row.get('event_idx')} k={k} offset={offset:.2f} src={Path(cand_path).name}")
            return ResolvedEvent(cand_path, cand_start, cand_end, float(offset), float(dur), k)

    # Fall back to original if present (still might miss)
    if Path(video_path).exists():
        offset = (ev_dt - seg_start).total_seconds()
        used_shift = 0
        if offset < 0:
            offset += seg_len
            used_shift = -1
        elif offset >= seg_len:
            used_shift = int(offset // seg_len)
            offset = offset - used_shift * seg_len
        return ResolvedEvent(video_path, seg_start, seg_end, float(offset), float(dur), used_shift)

    return None


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default="feral_clips")
    ap.add_argument("--json", default="feral_annotations.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--pre", type=float, default=2.0)
    ap.add_argument("--post", type=float, default=2.0)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_shift", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--validate", action="store_true", help="Detect blank/empty clips and drop them")

    # NEW: downscale controls
    ap.add_argument("--size", type=int, default=256, help="Output spatial size (square). Use 0 to disable.")
    ap.add_argument("--resize_mode", type=str, default="center_crop",
                    choices=["center_crop", "letterbox", "stretch"],
                    help="How to convert original frames to size x size.")

    # NEW: optional cap for super long events
    ap.add_argument("--max_len_sec", type=float, default=0.0,
                    help="If >0, cap each extracted clip length to this many seconds (after pre/post applied).")

    # NEW: skip existing
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing clips")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    print(f"Loaded manifest: {args.manifest}")
    print(f"Rows: {len(df)}")

    # Normalize size
    size = None if (args.size is None or args.size <= 0) else int(args.size)

    # Balanced sample without pandas FutureWarning
    if args.n and args.n < len(df) and "behavior" in df.columns:
        per = max(1, args.n // max(1, df["behavior"].nunique()))
        chunks = []
        for b, g in df.groupby("behavior"):
            chunks.append(g.sample(n=min(len(g), per + 1), random_state=42))
        df_s = pd.concat(chunks, ignore_index=True).head(args.n)
    else:
        df_s = df.head(args.n).reset_index(drop=True) if args.n else df

    extracted = []
    failures: List[str] = []
    extracted_rows: List[Dict[str, Any]] = []

    for i, row in df_s.iterrows():
        try:
            evt = int(row.get("event_idx"))
        except Exception:
            failures.append(f"row={i} bad_event_idx")
            continue

        label = str(row.get("behavior", "unknown")).strip().lower()
        label_map = {"ear-sucking": "ear", "tail-sucking": "tail", "teat-sucking": "teat"}
        label = label_map.get(label, label)

        dst = outdir / f"evt_{evt:04d}_{label}.mp4"
        if dst.exists() and (not args.overwrite):
            # still record it for downstream
            mb = dst.stat().st_size / (1024 * 1024)
            print(f"[{i+1}/{len(df_s)}] EXISTS: {dst.name} ({mb:.1f} MB)")
            extracted.append({"path": str(dst), "label": label, "event_idx": evt})
            continue

        resolved = resolve_event(row, max_shift=args.max_shift, min_dur=args.min_dur, debug=args.debug)
        if not resolved:
            failures.append(f"evt={evt} could_not_resolve")
            print(f"[{i+1}/{len(df_s)}] SKIP evt_{evt:04d}: could not resolve")
            continue

        src_path = resolved.src_path

        vid_dur, src_fps, src_nframes, src_w, src_h = get_video_info(src_path)
        if vid_dur is None:
            failures.append(f"evt={evt} cannot_open_video {src_path}")
            print(f"[{i+1}/{len(df_s)}] SKIP evt_{evt:04d}: cannot open video ({Path(src_path).name})")
            continue

        start = max(0.0, resolved.offset_sec - args.pre)
        end = min(float(vid_dur), resolved.offset_sec + resolved.duration_sec + args.post)

        # Optional cap (helps huge events)
        if args.max_len_sec and args.max_len_sec > 0:
            end = min(end, start + float(args.max_len_sec))

        if start >= float(vid_dur) or end <= 0.0 or end <= start:
            failures.append(
                f"evt={evt} out_of_range start={start:.2f} end={end:.2f} dur={vid_dur:.2f} src={Path(src_path).name}"
            )
            print(f"[{i+1}/{len(df_s)}] OUT OF RANGE evt_{evt:04d}: {Path(src_path).name}")
            continue

        ok, frames_written = extract_clip_opencv(
            src_path=str(src_path),
            out_path=str(dst),
            start_sec=float(start),
            end_sec=float(end),
            target_fps=args.fps,
            size=size,
            resize_mode=args.resize_mode,
        )
        if not ok:
            failures.append(f"evt={evt} extract_failed frames={frames_written} src={Path(src_path).name}")
            print(f"[{i+1}/{len(df_s)}] FAILED: {dst.name} (frames={frames_written})")
            continue

        # Optional blank detection
        if args.validate and looks_blank_video(str(dst)):
            failures.append(f"evt={evt} blank_clip_detected {dst.name} src={Path(src_path).name}")
            try:
                dst.unlink()
            except Exception:
                pass
            print(f"[{i+1}/{len(df_s)}] DROP (blank): {dst.name}")
            continue

        mb = dst.stat().st_size / (1024 * 1024)
        print(f"[{i+1}/{len(df_s)}] OK: {dst.name} ({mb:.1f} MB) shift={resolved.used_shift} src={Path(src_path).name}")
        extracted.append({"path": str(dst), "label": label, "event_idx": evt})

        # Save alignment audit row
        extracted_rows.append({
            "event_idx": evt,
            "behavior": label,
            "dst_clip": str(dst),
            "dst_mb": round(mb, 3),
            "src_path": str(src_path),
            "src_name": Path(src_path).name,
            "src_w": src_w,
            "src_h": src_h,
            "src_fps": round(float(src_fps or 0.0), 3),
            "src_duration_sec": round(float(vid_dur), 3),
            "resolved_shift": resolved.used_shift,
            "resolved_offset_sec": round(float(resolved.offset_sec), 3),
            "resolved_event_dur_sec": round(float(resolved.duration_sec), 3),
            "extract_start_sec": round(float(start), 3),
            "extract_end_sec": round(float(end), 3),
            "extract_len_sec": round(float(end - start), 3),
            "out_size": (size if size is not None else "native"),
            "resize_mode": args.resize_mode,
        })

    # failures log
    fail_path = outdir / "failed_extractions.txt"
    fail_path.write_text("\n".join(failures), encoding="utf-8")
    print(f"\nWrote failures list: {fail_path}")

    # extracted manifest
    if extracted_rows:
        out_csv = outdir / "extracted_manifest.csv"
        pd.DataFrame(extracted_rows).to_csv(out_csv, index=False)
        print(f"Wrote extracted manifest: {out_csv}")

    # build FERAL json
    feral = {}
    for item in extracted:
        name = Path(item["path"]).name
        n_frames = count_frames_opencv(item["path"])
        feral[name] = {str(i): item["label"] for i in range(n_frames)}

    json_path = outdir / args.json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(feral, f, indent=2)

    # summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Extracted: {len(extracted)}")
    print(f"Failed:    {len(failures)}")
    if extracted:
        c = Counter([x["label"] for x in extracted])
        print("Class distribution:", dict(c))
    print(f"FERAL JSON: {json_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
