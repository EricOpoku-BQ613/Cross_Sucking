#!/usr/bin/env python3
"""
Camera Verification Tool
========================

For a set of sample events, extracts the same clip from ALL cameras in that
group so you can side-by-side compare and identify which camera was used
for behavioral annotation.

Usage:
    python scripts/verify_cameras.py
    python scripts/verify_cameras.py --events 12,22,57,671,907 --pad 5
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_video_times(filename):
    """Extract start/end times from video filename."""
    pattern = r'N884A6_ch(\d+)_main_(\d{14})_(\d{14})'
    match = re.search(pattern, filename)
    if match:
        try:
            start = datetime.strptime(match.group(2), '%Y%m%d%H%M%S')
            end = datetime.strptime(match.group(3), '%Y%m%d%H%M%S')
            return start, end
        except ValueError:
            return None
    return None


def find_matching_video(videos, event_time_str, event_date_str):
    """Find a video file containing the given event time."""
    # Parse event time
    for fmt in ['%H:%M:%S', '%H:%M']:
        try:
            event_time = datetime.strptime(event_time_str, fmt)
            break
        except ValueError:
            continue
    else:
        return None

    for video in videos:
        times = parse_video_times(video['filename'])
        if not times:
            continue
        video_start, video_end = times

        event_datetime = datetime.combine(video_start.date(), event_time.time())

        if video_start <= event_datetime < video_end:
            offset = (event_datetime - video_start).total_seconds()
            return {
                'video_path': video['path'],
                'video_filename': video['filename'],
                'offset_sec': max(0, offset),
            }

    return None


def extract_clip_simple(video_path, start_sec, end_sec, output_path, target_fps=12):
    """Extract a clip from video. Returns True on success."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_fps <= 0:
        cap.release()
        return False

    video_dur = total_frames / src_fps
    start_sec = max(0, start_sec)
    end_sec = min(end_sec, video_dur)

    if end_sec <= start_sec:
        cap.release()
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (src_w, src_h))

    clip_duration = end_sec - start_sec
    num_frames = int(clip_duration * target_fps)

    keep_frames = set()
    for i in range(num_frames):
        t = start_sec + i / target_fps
        keep_frames.add(int(t * src_fps))

    start_frame = int(start_sec * src_fps)
    max_frame = max(keep_frames) if keep_frames else start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    written = 0

    while current_frame <= max_frame and written < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in keep_frames:
            out.write(frame)
            written += 1
        current_frame += 1

    cap.release()
    out.release()
    return written > 0


def main():
    parser = argparse.ArgumentParser(description="Verify annotation camera per group")
    parser.add_argument("--input", default="data/manifests/MASTER_LINKED_v3.csv")
    parser.add_argument("--manifest", default="data/manifests/video_manifest.json")
    parser.add_argument("--output-dir", default="data/processed/camera_verification")
    parser.add_argument("--events", type=str, default=None,
                        help="Comma-separated event indices (default: auto-select 2 per group)")
    parser.add_argument("--pad", type=float, default=5.0,
                        help="Padding in seconds around event")
    parser.add_argument("--samples-per-group", type=int, default=3,
                        help="Events to sample per group (if --events not given)")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    manifest_path = PROJECT_ROOT / args.manifest
    output_dir = PROJECT_ROOT / args.output_dir

    import pandas as pd
    df = pd.read_csv(input_path)

    # Filter to linked events only
    if "linked" in df.columns:
        df = df[df["linked"] == True]

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Select events to verify
    if args.events:
        event_ids = [int(e) for e in args.events.split(",")]
    else:
        # Auto-select: pick events with duration >= 3s from each group/day
        event_ids = []
        for (group, day), gdf in df.groupby(["group", "day"]):
            # Prefer events with decent duration so behavior is visible
            candidates = gdf[gdf["duration_sec"] >= 3].sort_values("duration_sec", ascending=False)
            if len(candidates) == 0:
                candidates = gdf
            selected = candidates.head(args.samples_per_group)
            event_ids.extend(selected["event_idx"].tolist())

    print(f"Verifying {len(event_ids)} events across all cameras")
    print(f"Output: {output_dir}\n")

    # Build camera lookup from manifest
    # {(group_id, day): {cam_id: [videos]}}
    camera_lookup = {}
    for folder_key, folder_data in manifest.get("labeled", {}).items():
        cameras = folder_data.get("cameras", {})
        # Extract group_id and day from folder_key
        import re as re_mod
        m = re_mod.match(r"group(\d+)_day(\d+)", folder_key)
        if m:
            gid = int(m.group(1))
            did = int(m.group(2))
            key = (gid, did)
            camera_lookup[key] = {}
            for cam_id, cam_data in cameras.items():
                videos = cam_data.get("videos", [])
                # Filter out macOS resource fork files
                videos = [v for v in videos if not v["filename"].startswith("._")]
                if videos:
                    camera_lookup[key][int(cam_id)] = videos

    # Process each event
    results = []
    for eidx in event_ids:
        row = df[df["event_idx"] == eidx]
        if len(row) == 0:
            print(f"  Event {eidx}: NOT FOUND in manifest")
            continue

        row = row.iloc[0]
        group = int(row["group"])
        day = int(row["day"])
        behavior = row["behavior"]
        duration = float(row["duration_sec"])
        offset = float(row["event_offset_sec"])
        start_time = str(row["start_time"])
        date = str(row["date"])
        current_cam = int(row["camera_id"])
        current_video = row["video_filename"]

        print(f"Event {eidx}: Group {group} Day {day}, {behavior} ({duration}s), "
              f"time={start_time}, currently linked to Cam {current_cam}")

        key = (group, day)
        if key not in camera_lookup:
            print(f"  No cameras found for Group {group} Day {day}")
            continue

        cameras = camera_lookup[key]
        event_dir = output_dir / f"evt_{eidx:04d}_{behavior}_g{group}d{day}"

        for cam_id in sorted(cameras.keys()):
            videos = cameras[cam_id]
            match = find_matching_video(videos, start_time, date)

            if not match:
                print(f"  Cam {cam_id:2d}: No matching video for time {start_time}")
                continue

            clip_start = match["offset_sec"] - args.pad
            clip_end = match["offset_sec"] + duration + args.pad

            clip_name = f"cam{cam_id:02d}{'_CURRENT' if cam_id == current_cam else ''}.mp4"
            clip_path = str(event_dir / clip_name)

            success = extract_clip_simple(
                match["video_path"], clip_start, clip_end, clip_path
            )

            marker = " <-- CURRENT" if cam_id == current_cam else ""
            status = "OK" if success else "FAILED"
            print(f"  Cam {cam_id:2d}: {status} -> {clip_name}{marker}")

            results.append({
                "event_idx": eidx,
                "group": group,
                "day": day,
                "behavior": behavior,
                "duration_sec": duration,
                "camera_id": cam_id,
                "is_current_link": cam_id == current_cam,
                "video_file": match["video_filename"],
                "offset_sec": match["offset_sec"],
                "clip_path": clip_path,
                "success": success,
            })

        print()

    # Save results summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "verification_summary.csv"
    if results:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Print camera recommendation
    print("=" * 60)
    print("REVIEW INSTRUCTIONS")
    print("=" * 60)
    print(f"\nClips saved to: {output_dir}")
    print(f"\nFor each event folder, compare the camera clips:")
    print(f"  - The file marked '_CURRENT' is what was used for extraction")
    print(f"  - Find which camera ACTUALLY shows the behavior")
    print(f"  - Report back: 'Group X uses Camera Y'")
    print(f"\nFolders to check:")

    seen_groups = set()
    for eidx in event_ids:
        row_match = df[df["event_idx"] == eidx]
        if len(row_match) > 0:
            r = row_match.iloc[0]
            gd = (int(r["group"]), int(r["day"]))
            if gd not in seen_groups:
                seen_groups.add(gd)
                print(f"  Group {gd[0]} Day {gd[1]}: evt_{eidx:04d}_*/")

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
