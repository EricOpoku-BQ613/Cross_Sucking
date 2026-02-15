#!/usr/bin/env python3
"""
Extract Event Clips using FFmpeg (High Quality)
================================================

Uses ffmpeg for H.264 encoding at full source resolution.
Much better quality/size ratio than cv2 mp4v codec.

Preserves original 4K resolution for annotation tasks:
  - Bounding box annotation
  - Interaction keypoints
  - Motion reconstruction

Usage:
    python scripts/extract_clips_ffmpeg.py --dry-run
    python scripts/extract_clips_ffmpeg.py --crf 20
    python scripts/extract_clips_ffmpeg.py --scale 1920:-1 --crf 23   # 1080p
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Find ffmpeg
FFMPEG = None
for candidate in [
    "ffmpeg",
    r"C:\Users\eopoku2\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
]:
    try:
        result = subprocess.run([candidate, "-version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            FFMPEG = candidate
            break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        continue


def get_video_info(video_path: str) -> dict:
    """Get video duration, fps, resolution using ffprobe."""
    ffprobe = FFMPEG.replace("ffmpeg", "ffprobe") if FFMPEG else "ffprobe"
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                fps_str = stream.get("r_frame_rate", "0/1")
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0
                return {
                    "width": int(stream.get("width", 0)),
                    "height": int(stream.get("height", 0)),
                    "fps": fps,
                    "duration": float(data.get("format", {}).get("duration", 0)),
                }
    except Exception:
        pass
    return {}


def extract_clip_ffmpeg(
    video_path: str,
    start_sec: float,
    duration_sec: float,
    output_path: str,
    crf: int = 20,
    scale: str = None,
    target_fps: int = None,
) -> dict:
    """Extract a clip using ffmpeg with H.264 encoding."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",                          # Overwrite
        "-ss", f"{start_sec:.3f}",     # Seek (before -i for fast seek)
        "-i", video_path,
        "-t", f"{duration_sec:.3f}",   # Duration
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",         # Compatibility
        "-an",                         # No audio (surveillance footage)
    ]

    if scale:
        cmd.extend(["-vf", f"scale={scale}"])

    if target_fps:
        cmd.extend(["-r", str(target_fps)])

    cmd.append(output_path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr[-500:] if result.stderr else "Unknown error"}

        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        return {
            "success": True,
            "file_size_bytes": file_size,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout (120s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def compute_clip_window(
    event_start: float,
    event_end: float,
    pad_before: float,
    pad_after: float,
    min_duration: float,
    video_duration: float,
) -> dict:
    """Compute clip boundaries with clamping and asymmetric compensation."""
    event_dur = event_end - event_start
    requested_total = event_dur + pad_before + pad_after

    if requested_total < min_duration:
        extra = min_duration - requested_total
        pad_before += extra / 2
        pad_after += extra / 2

    clip_start = event_start - pad_before
    clip_end = event_end + pad_after
    flags = []

    if clip_start < 0:
        lost_before = -clip_start
        clip_start = 0.0
        clip_end = min(clip_end + lost_before, video_duration)
        flags.append(f"start_clamped_lost_{lost_before:.1f}s")

    if clip_end > video_duration:
        lost_after = clip_end - video_duration
        clip_end = video_duration
        clip_start = max(clip_start - lost_after, 0.0)
        flags.append(f"end_clamped_lost_{lost_after:.1f}s")

    actual_pad_before = event_start - clip_start
    actual_pad_after = clip_end - event_end
    clip_duration = clip_end - clip_start

    event_visible = (clip_start <= event_start) and (event_end <= clip_end)
    if not event_visible:
        flags.append("EVENT_OUTSIDE_CLIP")

    return {
        "clip_start": round(clip_start, 2),
        "clip_end": round(clip_end, 2),
        "clip_duration": round(clip_duration, 2),
        "actual_pad_before": round(actual_pad_before, 2),
        "actual_pad_after": round(actual_pad_after, 2),
        "event_offset_in_clip": round(event_start - clip_start, 2),
        "event_end_in_clip": round(event_end - clip_start, 2),
        "event_visible": event_visible,
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract event clips with ffmpeg (high quality)")
    parser.add_argument("--input", default="data/manifests/MASTER_LINKED_v4.csv")
    parser.add_argument("--output-dir", default="data/processed/clips_v4")
    parser.add_argument("--pad-before", type=float, default=3.0)
    parser.add_argument("--pad-after", type=float, default=3.0)
    parser.add_argument("--min-duration", type=float, default=6.0)
    parser.add_argument("--crf", type=int, default=20,
                        help="H.264 quality (lower=better, 18=visually lossless, 23=default)")
    parser.add_argument("--scale", type=str, default=None,
                        help="FFmpeg scale filter (e.g. '1920:-1' for 1080p)")
    parser.add_argument("--target-fps", type=int, default=None,
                        help="Output FPS (default: same as source)")
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not FFMPEG:
        print("ERROR: ffmpeg not found!")
        sys.exit(1)

    print(f"Using ffmpeg: {FFMPEG}")
    print(f"Quality: CRF {args.crf} (lower=better)")
    if args.scale:
        print(f"Scale: {args.scale}")
    else:
        print(f"Scale: original (4K)")

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir

    import pandas as pd
    df = pd.read_csv(input_path)

    if "linked" in df.columns:
        df = df[df["linked"] == True]

    if args.groups:
        group_ids = [int(g) for g in args.groups.split(",")]
        df = df[df["group"].isin(group_ids)]
        print(f"Filtered to groups {group_ids}: {len(df)} events")

    if args.max_events:
        df = df.head(args.max_events)

    print(f"Events to extract: {len(df)}")

    # Pre-scan video durations
    unique_videos = df["video_path"].unique()
    print(f"\nPre-scanning {len(unique_videos)} videos...")
    video_info_cache = {}
    missing = []
    for vp in tqdm(unique_videos, desc="Scanning"):
        info = get_video_info(vp)
        if info and info.get("duration", 0) > 0:
            video_info_cache[vp] = info
        else:
            missing.append(vp)

    if missing:
        print(f"  WARNING: {len(missing)} videos not accessible")
        for vp in missing[:5]:
            print(f"    {vp}")

    # Build extraction plan
    plan = []
    edge_cases = {"missing": 0, "zero_dur": 0, "clamped": 0, "outside": 0, "skipped": 0}

    for _, row in df.iterrows():
        event_idx = int(row["event_idx"])
        behavior = row["behavior"]
        video_path = row["video_path"]
        offset = float(row["event_offset_sec"])
        duration = float(row["duration_sec"])

        info = video_info_cache.get(video_path)
        if not info:
            edge_cases["missing"] += 1
            plan.append({"event_idx": event_idx, "skip": True, "reason": "missing_video"})
            continue

        video_dur = info["duration"]
        if duration == 0:
            edge_cases["zero_dur"] += 1

        window = compute_clip_window(offset, offset + duration,
                                     args.pad_before, args.pad_after,
                                     args.min_duration, video_dur)

        flags_str = "|".join(window["flags"]) if window["flags"] else ""
        if any("clamped" in f for f in window["flags"]):
            edge_cases["clamped"] += 1
        if not window["event_visible"]:
            edge_cases["outside"] += 1

        clip_filename = f"evt_{event_idx:04d}_{behavior}.mp4"
        clip_path = str(output_dir / clip_filename)

        skip = False
        if args.skip_existing and os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            edge_cases["skipped"] += 1
            skip = True

        plan.append({
            "event_idx": event_idx,
            "behavior": behavior,
            "group": int(row["group"]),
            "video_path": video_path,
            "clip_path": clip_path,
            "clip_filename": clip_filename,
            "clip_start_sec": window["clip_start"],
            "clip_end_sec": window["clip_end"],
            "clip_duration": window["clip_duration"],
            "event_offset_in_clip_sec": window["event_offset_in_clip"],
            "event_end_in_clip_sec": window["event_end_in_clip"],
            "event_duration_sec": duration,
            "actual_pad_before": window["actual_pad_before"],
            "actual_pad_after": window["actual_pad_after"],
            "flags": flags_str,
            "skip": skip,
        })

    extractable = [p for p in plan if not p.get("skip")]

    # Report
    print(f"\n{'=' * 60}")
    print("EXTRACTION PLAN")
    print(f"{'=' * 60}")
    print(f"  Total: {len(plan)} | Extractable: {len(extractable)} | Skipped: {edge_cases['skipped']}")
    print(f"  Missing videos: {edge_cases['missing']}")
    print(f"  Zero-duration: {edge_cases['zero_dur']} | Clamped: {edge_cases['clamped']} | Outside: {edge_cases['outside']}")

    if extractable:
        durations = [p["clip_duration"] for p in extractable]
        total_hours = sum(durations) / 3600
        print(f"  Total clip time: {total_hours:.1f} hours")
        print(f"  Clip durations: mean={np.mean(durations):.1f}s, median={np.median(durations):.1f}s, max={np.max(durations):.1f}s")

        # Rough disk estimate
        if args.scale and "1920" in args.scale:
            est_rate = 1.5  # MB/s for 1080p H.264 CRF 20
        else:
            est_rate = 6.0  # MB/s for 4K H.264 CRF 20 (surveillance content)
        est_gb = total_hours * 3600 * est_rate / 1024
        print(f"  Est. disk: ~{est_gb:.0f} GB")

    if args.dry_run:
        print(f"\n  DRY RUN - no files written")
        print(f"{'=' * 60}")
        return

    # Execute
    print(f"\nExtracting {len(extractable)} clips to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    total_size = 0
    start_time = time.time()

    for entry in tqdm(plan, desc="Extracting"):
        if entry.get("skip") or entry.get("reason") == "missing_video":
            continue

        clip_duration = entry["clip_end_sec"] - entry["clip_start_sec"]

        result = extract_clip_ffmpeg(
            video_path=entry["video_path"],
            start_sec=entry["clip_start_sec"],
            duration_sec=clip_duration,
            output_path=entry["clip_path"],
            crf=args.crf,
            scale=args.scale,
            target_fps=args.target_fps,
        )

        entry["extraction"] = result
        if result["success"]:
            successes += 1
            total_size += result.get("file_size_bytes", 0)
        else:
            failures += 1
            if failures <= 5:
                print(f"\n  FAILED evt_{entry['event_idx']:04d}: {result.get('error', '')[:100]}")

    elapsed = time.time() - start_time

    # Save manifest
    manifest_path = output_dir / "extracted_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_idx", "behavior", "group", "clip_path", "clip_filename",
            "clip_start_sec", "clip_end_sec", "event_offset_in_clip_sec",
            "event_end_in_clip_sec", "event_duration_sec",
            "clip_duration", "actual_pad_before", "actual_pad_after",
            "flags", "success",
        ])
        writer.writeheader()
        for entry in plan:
            if entry.get("reason") == "missing_video":
                continue
            extraction = entry.get("extraction", {})
            writer.writerow({
                "event_idx": entry["event_idx"],
                "behavior": entry["behavior"],
                "group": entry["group"],
                "clip_path": entry.get("clip_path", ""),
                "clip_filename": entry.get("clip_filename", ""),
                "clip_start_sec": entry.get("clip_start_sec", 0),
                "clip_end_sec": entry.get("clip_end_sec", 0),
                "event_offset_in_clip_sec": entry.get("event_offset_in_clip_sec", 0),
                "event_end_in_clip_sec": entry.get("event_end_in_clip_sec", 0),
                "event_duration_sec": entry.get("event_duration_sec", 0),
                "clip_duration": entry.get("clip_duration", 0),
                "actual_pad_before": entry.get("actual_pad_before", 0),
                "actual_pad_after": entry.get("actual_pad_after", 0),
                "flags": entry.get("flags", ""),
                "success": extraction.get("success", False),
            })

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Success: {successes}/{len(extractable)}")
    print(f"  Failed:  {failures}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(successes,1):.1f}s/clip)")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
