#!/usr/bin/env python3
"""
Extract Event Clips
===================

Extracts padded video clips for each event from 30-minute source videos.
Produces individual .mp4 files and an extraction manifest.

Edge cases handled:
  - Events near video start/end: asymmetric padding compensation
  - Zero-duration events: guaranteed minimum clip centered on event
  - Video duration clamping: metadata always reflects actual clip boundaries
  - Missing videos: flagged at plan time, not just extraction time
  - Resumability: --skip-existing to avoid re-extracting

Usage:
    python scripts/extract_event_clips.py --dry-run
    python scripts/extract_event_clips.py --groups 1,2 --pad-before 3 --pad-after 3
    python scripts/extract_event_clips.py --skip-existing  # resume after crash
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Video duration cache ─────────────────────────────────────────────────────

_duration_cache = {}


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using cv2, with caching per path."""
    if video_path in _duration_cache:
        return _duration_cache[video_path]

    if not os.path.exists(video_path):
        _duration_cache[video_path] = -1.0  # sentinel for missing
        return -1.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _duration_cache[video_path] = 0.0
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        _duration_cache[video_path] = 0.0
        return 0.0

    dur = frame_count / fps
    _duration_cache[video_path] = dur
    return dur


# ── Clip extraction ──────────────────────────────────────────────────────────


def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    target_fps: int = 12,
    letterbox: tuple = None,
) -> dict:
    """
    Extract a clip from a source video.

    Args:
        video_path: Path to source video
        start_sec: Start time in seconds (already clamped by caller)
        end_sec: End time in seconds (already clamped by caller)
        output_path: Output .mp4 path
        target_fps: Target FPS for output
        letterbox: (width, height) tuple for output resolution, or None for original

    Returns:
        dict with extraction metadata
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": f"Cannot open {video_path}"}

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_fps <= 0:
        cap.release()
        return {"success": False, "error": "Cannot determine FPS"}

    video_duration = total_frames / src_fps

    # Final safety clamp (should already be clamped by caller)
    start_sec = max(0.0, start_sec)
    end_sec = min(end_sec, video_duration)

    if end_sec <= start_sec:
        cap.release()
        return {"success": False, "error": f"Invalid range [{start_sec:.1f}, {end_sec:.1f}]"}

    # Compute output dimensions
    if letterbox:
        out_w, out_h = letterbox
    else:
        out_w, out_h = src_width, src_height

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (out_w, out_h))

    # Time-based frame sampling: read sequentially, skip unneeded frames.
    # Ensures output plays at real speed (e.g., 15fps source → 12fps output)
    # without expensive random seeks.
    clip_duration_sec = end_sec - start_sec
    num_output_frames = int(clip_duration_sec * target_fps)

    # Pre-compute which source frames to keep
    keep_frames = set()
    for i in range(num_output_frames):
        t = start_sec + i / target_fps
        keep_frames.add(int(t * src_fps))

    start_frame = int(start_sec * src_fps)
    max_frame = max(keep_frames) if keep_frames else start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    frames_written = 0

    while current_frame <= max_frame and frames_written < num_output_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in keep_frames:
            if letterbox:
                frame = letterbox_resize(frame, out_w, out_h)
            out.write(frame)
            frames_written += 1
        current_frame += 1

    cap.release()
    out.release()

    clip_duration = frames_written / target_fps if target_fps > 0 else 0
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return {
        "success": True,
        "frames_written": frames_written,
        "clip_duration_sec": round(clip_duration, 2),
        "file_size_bytes": file_size,
        "src_fps": src_fps,
        "out_fps": target_fps,
        "resolution": f"{out_w}x{out_h}",
    }


def letterbox_resize(frame, target_w, target_h):
    """Resize frame with letterboxing to preserve aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


# ── Plan building with edge case handling ────────────────────────────────────


def compute_clip_window(
    event_start: float,
    event_end: float,
    pad_before: float,
    pad_after: float,
    min_duration: float,
    video_duration: float,
) -> dict:
    """
    Compute clip boundaries with proper clamping and asymmetric compensation.

    If padding gets eaten on one side (event near video boundary), we try to
    extend the other side so the total padding is preserved.

    Returns dict with clip_start, clip_end, and edge-case flags.
    """
    event_dur = event_end - event_start
    requested_total = event_dur + pad_before + pad_after

    # Ensure minimum clip duration (for zero-duration events)
    if requested_total < min_duration:
        extra = min_duration - requested_total
        pad_before += extra / 2
        pad_after += extra / 2
        requested_total = min_duration

    # Initial window
    clip_start = event_start - pad_before
    clip_end = event_end + pad_after

    flags = []

    # ── Clamp to video boundaries with asymmetric compensation ──

    # If clip_start goes below 0, shift the lost padding to the end
    if clip_start < 0:
        lost_before = -clip_start
        clip_start = 0.0
        clip_end = min(clip_end + lost_before, video_duration)
        flags.append(f"start_clamped_lost_{lost_before:.1f}s")

    # If clip_end exceeds video, shift the lost padding to the start
    if clip_end > video_duration:
        lost_after = clip_end - video_duration
        clip_end = video_duration
        clip_start = max(clip_start - lost_after, 0.0)
        flags.append(f"end_clamped_lost_{lost_after:.1f}s")

    # Recompute actual padding achieved
    actual_pad_before = event_start - clip_start
    actual_pad_after = clip_end - event_end
    clip_duration = clip_end - clip_start

    # Check if event still falls within clip
    event_visible = (clip_start <= event_start) and (event_end <= clip_end)
    if not event_visible:
        flags.append("EVENT_OUTSIDE_CLIP")

    # Check if we have at least some context on each side
    if actual_pad_before < 1.0 and event_start > 1.0:
        flags.append(f"thin_pre_padding_{actual_pad_before:.1f}s")
    if actual_pad_after < 1.0 and (video_duration - event_end) > 1.0:
        flags.append(f"thin_post_padding_{actual_pad_after:.1f}s")

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


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract event clips")
    parser.add_argument("--input", default="data/manifests/MASTER_LINKED_v3.csv",
                        help="Input linked manifest")
    parser.add_argument("--output-dir", default="data/processed/clips",
                        help="Output directory for clips")
    parser.add_argument("--pad-before", type=float, default=3.0,
                        help="Seconds of padding before event")
    parser.add_argument("--pad-after", type=float, default=3.0,
                        help="Seconds of padding after event")
    parser.add_argument("--min-duration", type=float, default=6.0,
                        help="Minimum clip duration (for zero-duration events)")
    parser.add_argument("--target-fps", type=int, default=12,
                        help="Target FPS for output clips")
    parser.add_argument("--letterbox", type=str, default=None,
                        help="Target resolution WxH (e.g. 512x288)")
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated group IDs to extract (default: all)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events to extract (for testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips that already exist on disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute stats only, don't extract")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir

    print(f"Loading: {input_path}")

    import pandas as pd
    df = pd.read_csv(input_path)

    # Filter to linked events
    if "linked" in df.columns:
        df = df[df["linked"] == True]

    # Filter by groups
    if args.groups:
        group_ids = [int(g) for g in args.groups.split(",")]
        df = df[df["group"].isin(group_ids)]
        print(f"Filtered to groups {group_ids}: {len(df)} events")

    # Limit for testing
    if args.max_events:
        df = df.head(args.max_events)

    print(f"Events to extract: {len(df)}")

    # Parse letterbox
    letterbox = None
    if args.letterbox:
        w, h = args.letterbox.split("x")
        letterbox = (int(w), int(h))

    # ── Pre-scan video durations ──
    unique_videos = df["video_path"].unique()
    print(f"\nPre-scanning {len(unique_videos)} unique videos for duration...")
    missing_videos = []
    for vp in tqdm(unique_videos, desc="Scanning videos"):
        dur = get_video_duration(vp)
        if dur <= 0:
            missing_videos.append(vp)

    if missing_videos:
        print(f"\n  WARNING: {len(missing_videos)} videos not accessible:")
        for vp in missing_videos[:10]:
            print(f"    {vp}")
        if len(missing_videos) > 10:
            print(f"    ... and {len(missing_videos) - 10} more")

    # ── Compute extraction plan with proper clamping ──
    plan = []
    edge_cases = {
        "missing_video": 0,
        "start_clamped": 0,
        "end_clamped": 0,
        "zero_duration_event": 0,
        "event_outside_clip": 0,
        "thin_padding": 0,
        "skipped_existing": 0,
    }

    for _, row in df.iterrows():
        event_idx = int(row["event_idx"])
        behavior = row["behavior"]
        video_path = row["video_path"]
        offset = float(row["event_offset_sec"])
        duration = float(row["duration_sec"])

        video_dur = get_video_duration(video_path)

        # Flag missing/unreadable videos
        if video_dur <= 0:
            edge_cases["missing_video"] += 1
            plan.append({
                "event_idx": event_idx,
                "behavior": behavior,
                "group": int(row["group"]),
                "video_path": video_path,
                "clip_path": "",
                "clip_filename": "",
                "clip_start_sec": 0,
                "clip_end_sec": 0,
                "event_offset_in_clip_sec": 0,
                "event_end_in_clip_sec": 0,
                "event_duration_sec": duration,
                "clip_duration_planned": 0,
                "actual_pad_before": 0,
                "actual_pad_after": 0,
                "flags": "missing_video",
                "skip": True,
            })
            continue

        if duration == 0:
            edge_cases["zero_duration_event"] += 1

        # Compute window with clamping and compensation
        event_start = offset
        event_end = offset + duration

        window = compute_clip_window(
            event_start=event_start,
            event_end=event_end,
            pad_before=args.pad_before,
            pad_after=args.pad_after,
            min_duration=args.min_duration,
            video_duration=video_dur,
        )

        # Track edge cases
        flags_str = "|".join(window["flags"]) if window["flags"] else ""
        if any("start_clamped" in f for f in window["flags"]):
            edge_cases["start_clamped"] += 1
        if any("end_clamped" in f for f in window["flags"]):
            edge_cases["end_clamped"] += 1
        if not window["event_visible"]:
            edge_cases["event_outside_clip"] += 1
        if any("thin_" in f for f in window["flags"]):
            edge_cases["thin_padding"] += 1

        clip_filename = f"evt_{event_idx:04d}_{behavior}.mp4"
        clip_path = str(output_dir / clip_filename)

        # Check if already extracted
        skip = False
        if args.skip_existing and os.path.exists(clip_path):
            fsize = os.path.getsize(clip_path)
            if fsize > 0:
                edge_cases["skipped_existing"] += 1
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
            "event_offset_in_clip_sec": window["event_offset_in_clip"],
            "event_end_in_clip_sec": window["event_end_in_clip"],
            "event_duration_sec": duration,
            "clip_duration_planned": window["clip_duration"],
            "actual_pad_before": window["actual_pad_before"],
            "actual_pad_after": window["actual_pad_after"],
            "flags": flags_str,
            "skip": skip,
        })

    # ── Report plan ──
    extractable = [p for p in plan if not p["skip"]]
    skipped = [p for p in plan if p["skip"]]

    print(f"\n{'=' * 60}")
    print(f"EXTRACTION PLAN")
    print(f"{'=' * 60}")
    print(f"  Total events:       {len(plan)}")
    print(f"  Extractable:        {len(extractable)}")
    print(f"  Skipped (existing): {edge_cases['skipped_existing']}")
    print(f"  Skipped (missing):  {edge_cases['missing_video']}")

    print(f"\n  Edge cases detected:")
    print(f"    Zero-duration events:       {edge_cases['zero_duration_event']}")
    print(f"    Start clamped (near t=0):   {edge_cases['start_clamped']}")
    print(f"    End clamped (near vid end): {edge_cases['end_clamped']}")
    print(f"    Thin padding (< 1s):        {edge_cases['thin_padding']}")
    print(f"    Event outside clip:         {edge_cases['event_outside_clip']}")

    if extractable:
        durations = [p["clip_duration_planned"] for p in extractable]
        pad_befores = [p["actual_pad_before"] for p in extractable]
        pad_afters = [p["actual_pad_after"] for p in extractable]

        print(f"\n  Clip durations:")
        print(f"    Mean:   {np.mean(durations):.1f}s")
        print(f"    Median: {np.median(durations):.1f}s")
        print(f"    Min:    {np.min(durations):.1f}s")
        print(f"    Max:    {np.max(durations):.1f}s")
        print(f"    Total:  {sum(durations) / 3600:.1f} hours")

        print(f"\n  Actual padding achieved:")
        print(f"    Before: mean={np.mean(pad_befores):.1f}s  min={np.min(pad_befores):.1f}s")
        print(f"    After:  mean={np.mean(pad_afters):.1f}s  min={np.min(pad_afters):.1f}s")

        print(f"\n  By behavior:")
        for beh in sorted(set(p["behavior"] for p in extractable)):
            beh_items = [p for p in extractable if p["behavior"] == beh]
            beh_durs = [p["clip_duration_planned"] for p in beh_items]
            beh_flagged = sum(1 for p in beh_items if p["flags"])
            print(f"    {beh:8s}: {len(beh_durs):4d} clips, "
                  f"avg {np.mean(beh_durs):.1f}s, "
                  f"{beh_flagged} edge cases")

        est_gb = sum(durations) / 1024  # ~1 MB/s at 512x288
        print(f"\n  Est. disk (~1 MB/s at 512x288): {est_gb:.1f} GB")

    # ── Dry-run: stop here ──
    if args.dry_run:
        # Show some flagged examples
        flagged = [p for p in plan if p["flags"] and not p["skip"]]
        if flagged:
            print(f"\n  Sample flagged events (first 10):")
            for p in flagged[:10]:
                print(f"    evt_{p['event_idx']:04d} ({p['behavior']:5s}): "
                      f"clip=[{p['clip_start_sec']:.1f}, {p['clip_end_sec']:.1f}] "
                      f"event_in_clip=[{p['event_offset_in_clip_sec']:.1f}, "
                      f"{p['event_end_in_clip_sec']:.1f}] "
                      f"pad=[{p['actual_pad_before']:.1f}, {p['actual_pad_after']:.1f}] "
                      f"flags={p['flags']}")
        print(f"\n{'=' * 60}")
        return

    # ── Execute extraction ──
    print(f"\nExtracting {len(extractable)} clips to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    successes = 0
    failures = 0
    total_size = 0
    start_time = time.time()

    for entry in tqdm(plan, desc="Extracting"):
        if entry["skip"]:
            # Still record skipped entries in results
            entry["extraction"] = {
                "success": entry.get("flags", "") != "missing_video",
                "clip_duration_sec": 0,
                "skipped": True,
            }
            results.append(entry)
            if entry.get("flags") != "missing_video":
                successes += 1  # existing file counts as success
            else:
                failures += 1
            continue

        result = extract_clip(
            video_path=entry["video_path"],
            start_sec=entry["clip_start_sec"],
            end_sec=entry["clip_end_sec"],
            output_path=entry["clip_path"],
            target_fps=args.target_fps,
            letterbox=letterbox,
        )

        entry["extraction"] = result
        if result["success"]:
            successes += 1
            total_size += result.get("file_size_bytes", 0)
        else:
            failures += 1

        results.append(entry)

    elapsed = time.time() - start_time

    # ── Save extraction manifest ──
    manifest_path = output_dir / "extracted_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_idx", "behavior", "group", "clip_path", "clip_filename",
            "clip_start_sec", "clip_end_sec", "event_offset_in_clip_sec",
            "event_end_in_clip_sec", "event_duration_sec",
            "clip_duration_planned", "clip_duration_actual",
            "actual_pad_before", "actual_pad_after",
            "flags", "success",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "event_idx": r["event_idx"],
                "behavior": r["behavior"],
                "group": r["group"],
                "clip_path": r["clip_path"],
                "clip_filename": r["clip_filename"],
                "clip_start_sec": r["clip_start_sec"],
                "clip_end_sec": r["clip_end_sec"],
                "event_offset_in_clip_sec": r["event_offset_in_clip_sec"],
                "event_end_in_clip_sec": r["event_end_in_clip_sec"],
                "event_duration_sec": r["event_duration_sec"],
                "clip_duration_planned": r["clip_duration_planned"],
                "clip_duration_actual": r["extraction"].get("clip_duration_sec", 0),
                "actual_pad_before": r.get("actual_pad_before", 0),
                "actual_pad_after": r.get("actual_pad_after", 0),
                "flags": r.get("flags", ""),
                "success": r["extraction"]["success"],
            })

    # ── Print summary ──
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Success: {successes}/{len(plan)}")
    print(f"  Failed:  {failures}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    if extractable:
        print(f"  Time: {elapsed:.0f}s ({elapsed / max(len(extractable), 1):.1f}s/clip)")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
