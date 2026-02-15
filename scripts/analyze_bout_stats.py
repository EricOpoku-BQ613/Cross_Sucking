#!/usr/bin/env python3
"""
Analyze Bout Statistics
=======================

Deep analysis of event durations, temporal density, and video coverage
to inform optimal clip extraction parameters.

Usage:
    python scripts/analyze_bout_stats.py
    python scripts/analyze_bout_stats.py --input data/manifests/MASTER_LINKED_v3.csv
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_durations(df: pd.DataFrame) -> dict:
    """Analyze event duration distribution."""
    dur = df["duration_sec"]

    # Bin definitions
    bins = [0, 1, 3, 5, 10, 30, 60, float("inf")]
    labels = ["0-1s", "1-3s", "3-5s", "5-10s", "10-30s", "30-60s", "60s+"]
    binned = pd.cut(dur, bins=bins, labels=labels, right=False)

    print("\n=== DURATION DISTRIBUTION ===")
    for label in labels:
        count = (binned == label).sum()
        pct = count / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:8s}  {count:5d}  ({pct:5.1f}%)  {bar}")

    # Stats
    print(f"\n  Mean:     {dur.mean():.1f}s")
    print(f"  Median:   {dur.median():.1f}s")
    print(f"  Std:      {dur.std():.1f}s")
    print(f"  Min:      {dur.min():.0f}s")
    print(f"  Max:      {dur.max():.0f}s")
    print(f"  P25:      {dur.quantile(0.25):.0f}s")
    print(f"  P75:      {dur.quantile(0.75):.0f}s")
    print(f"  P90:      {dur.quantile(0.90):.0f}s")
    print(f"  P95:      {dur.quantile(0.95):.0f}s")

    # Zero-duration events
    zero_dur = (dur == 0).sum()
    print(f"\n  Zero-duration events: {zero_dur} ({zero_dur / len(df) * 100:.1f}%)")

    # By behavior
    print("\n  Duration by behavior:")
    for beh in sorted(df["behavior"].unique()):
        sub = df[df["behavior"] == beh]["duration_sec"]
        print(
            f"    {beh:8s}  n={len(sub):4d}  "
            f"mean={sub.mean():.1f}s  median={sub.median():.1f}s  "
            f"max={sub.max():.0f}s  zero={( sub == 0).sum()}"
        )

    return {
        "histogram": {label: int((binned == label).sum()) for label in labels},
        "stats": {
            "mean": float(dur.mean()),
            "median": float(dur.median()),
            "std": float(dur.std()),
            "min": float(dur.min()),
            "max": float(dur.max()),
            "p25": float(dur.quantile(0.25)),
            "p75": float(dur.quantile(0.75)),
            "p90": float(dur.quantile(0.90)),
            "p95": float(dur.quantile(0.95)),
        },
        "zero_duration_count": int(zero_dur),
        "by_behavior": {
            beh: {
                "count": int(len(sub)),
                "mean": float(sub.mean()),
                "median": float(sub.median()),
            }
            for beh in df["behavior"].unique()
            for sub in [df[df["behavior"] == beh]["duration_sec"]]
        },
    }


def analyze_temporal_density(df: pd.DataFrame) -> dict:
    """Analyze temporal density — events per video, inter-event gaps, clustering."""
    print("\n=== TEMPORAL DENSITY ===")

    # Events per video
    events_per_video = df.groupby("video_filename").size()
    print(f"\n  Unique videos with events: {len(events_per_video)}")
    print(f"  Events per video:")
    print(f"    Mean:   {events_per_video.mean():.1f}")
    print(f"    Median: {events_per_video.median():.0f}")
    print(f"    Max:    {events_per_video.max()}")
    print(f"    Min:    {events_per_video.min()}")

    # Distribution of events per video
    epv_bins = [1, 2, 3, 5, 10, 20, float("inf")]
    epv_labels = ["1", "2", "3-4", "5-9", "10-19", "20+"]
    epv_binned = pd.cut(events_per_video, bins=epv_bins, labels=epv_labels, right=False)
    print(f"\n  Videos by event count:")
    for label in epv_labels:
        count = (epv_binned == label).sum()
        print(f"    {label:6s} events: {count:4d} videos")

    # Inter-event gaps (within same video)
    gaps = []
    overlaps = 0
    clusters = 0  # events within 5s of each other

    for video, group in df.groupby("video_filename"):
        if len(group) < 2:
            continue
        sorted_events = group.sort_values("event_offset_sec")
        offsets = sorted_events["event_offset_sec"].values
        durations = sorted_events["duration_sec"].values

        for i in range(1, len(offsets)):
            prev_end = offsets[i - 1] + durations[i - 1]
            gap = offsets[i] - prev_end
            gaps.append(gap)
            if gap < 0:
                overlaps += 1
            if gap < 5:
                clusters += 1

    if gaps:
        gaps_arr = np.array(gaps)
        print(f"\n  Inter-event gaps (within same video):")
        print(f"    Total gaps analyzed: {len(gaps_arr)}")
        print(f"    Mean gap:   {gaps_arr.mean():.1f}s")
        print(f"    Median gap: {np.median(gaps_arr):.1f}s")
        print(f"    Min gap:    {gaps_arr.min():.1f}s")
        print(f"    Max gap:    {gaps_arr.max():.1f}s")
        print(f"    Overlapping events: {overlaps}")
        print(f"    Clustered (< 5s gap): {clusters}")

        # Gap distribution
        gap_bins = [float("-inf"), 0, 5, 10, 30, 60, 300, float("inf")]
        gap_labels = ["overlap", "0-5s", "5-10s", "10-30s", "30-60s", "1-5min", "5min+"]
        gap_binned = pd.cut(gaps_arr, bins=gap_bins, labels=gap_labels, right=False)
        print(f"\n  Gap distribution:")
        for label in gap_labels:
            count = (gap_binned == label).sum()
            pct = count / len(gaps_arr) * 100
            print(f"    {label:10s}  {count:5d}  ({pct:.1f}%)")

    return {
        "unique_videos_with_events": int(len(events_per_video)),
        "events_per_video": {
            "mean": float(events_per_video.mean()),
            "median": float(events_per_video.median()),
            "max": int(events_per_video.max()),
        },
        "overlapping_events": overlaps,
        "clustered_events_under_5s": clusters,
        "gap_stats": {
            "mean": float(gaps_arr.mean()) if gaps else 0,
            "median": float(np.median(gaps_arr)) if gaps else 0,
        },
    }


def analyze_video_coverage(df: pd.DataFrame) -> dict:
    """Analyze video and camera coverage."""
    print("\n=== VIDEO COVERAGE ===")

    # Events per camera per group
    print(f"\n  Events by group x camera:")
    cam_group = pd.crosstab(df["group"], df["camera_id"])
    print(cam_group.to_string())

    # Videos per group
    print(f"\n  Videos per group:")
    for g, group_df in df.groupby("group"):
        n_videos = group_df["video_filename"].nunique()
        n_events = len(group_df)
        print(f"    Group {g}: {n_videos:3d} videos, {n_events:4d} events")

    return {
        "events_by_group_camera": {
            str(g): {str(c): int(v) for c, v in row.items() if v > 0}
            for g, row in cam_group.iterrows()
        },
    }


def suggest_extraction_params(duration_stats: dict) -> dict:
    """Suggest clip extraction parameters based on statistics."""
    print("\n=== RECOMMENDED EXTRACTION PARAMETERS ===")

    median_dur = duration_stats["stats"]["median"]
    p75_dur = duration_stats["stats"]["p75"]
    p95_dur = duration_stats["stats"]["p95"]

    # Training clip length
    # Need to cover at least median duration
    # At 12fps: 16 frames = 1.33s, 32 frames = 2.67s, 48 frames = 4s, 64 frames = 5.33s
    suggested_clip_len = 48  # covers median (4s) well
    suggested_fps = 12

    print(f"\n  Training clips (for model input):")
    print(f"    clip_len: {suggested_clip_len} frames at {suggested_fps} fps = {suggested_clip_len / suggested_fps:.1f}s")
    print(f"    Covers: median ({median_dur:.0f}s) = YES, p75 ({p75_dur:.0f}s) = {'YES' if suggested_clip_len / suggested_fps >= p75_dur else 'partial'}")
    print(f"    Mode: center on event midpoint with jitter")

    # Extraction padding
    pad_before = 3
    pad_after = 3
    min_clip_dur = 6  # minimum clip duration even for 0-duration events

    print(f"\n  Clip extraction (for .mp4 files):")
    print(f"    pad_before: {pad_before}s")
    print(f"    pad_after:  {pad_after}s")
    print(f"    min_clip_duration: {min_clip_dur}s (for 0-duration events)")
    print(f"    max_clip_duration: {p95_dur + pad_before + pad_after:.0f}s (p95 + padding)")

    # Estimated total extraction size
    avg_clip_dur = median_dur + pad_before + pad_after
    # At 512x288 @ 12fps, ~1 MB per second of H264 video
    est_size_gb = 1897 * avg_clip_dur * 1.0 / 1024
    print(f"\n  Estimated extraction:")
    print(f"    Average clip: {avg_clip_dur:.0f}s")
    print(f"    Total clips: 1897")
    print(f"    Est. disk: ~{est_size_gb:.1f} GB (at 512x288, H264)")

    return {
        "training": {
            "clip_len": suggested_clip_len,
            "fps": suggested_fps,
            "window_sec": suggested_clip_len / suggested_fps,
        },
        "extraction": {
            "pad_before_sec": pad_before,
            "pad_after_sec": pad_after,
            "min_clip_duration_sec": min_clip_dur,
            "target_fps": suggested_fps,
            "target_resolution": "512x288",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze bout statistics")
    parser.add_argument(
        "--input",
        default="data/manifests/MASTER_LINKED_v3.csv",
        help="Input linked manifest CSV",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / args.input

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} events")

    # Filter to linked events only
    if "linked" in df.columns:
        linked = df[df["linked"] == True]
        print(f"Linked: {len(linked)} / {len(df)}")
        df = linked

    report = {"timestamp": datetime.now().isoformat(), "total_events": len(df)}

    report["durations"] = analyze_durations(df)
    report["temporal_density"] = analyze_temporal_density(df)
    report["video_coverage"] = analyze_video_coverage(df)
    report["suggested_params"] = suggest_extraction_params(report["durations"])

    # Save report
    reports_dir = project_root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "bout_statistics_v3.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
