#!/usr/bin/env python3
"""
Extract flagged events from an alternative camera for comparison.

Given a list of event indices and a target camera, extracts clips from
that camera into a comparison folder alongside the current Cam 8 clips.

Usage:
    python scripts/extract_alt_camera.py --events 25,26,27,32,33 --cam 9
    python scripts/extract_alt_camera.py --events-file audit_flagged.txt --cam 9 --cam 7
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_video_times(filename):
    pattern = r"N884A6_ch(\d+)_main_(\d{14})_(\d{14})"
    match = re.search(pattern, filename)
    if match:
        try:
            start = datetime.strptime(match.group(2), "%Y%m%d%H%M%S")
            end = datetime.strptime(match.group(3), "%Y%m%d%H%M%S")
            return start, end
        except ValueError:
            return None
    return None


def find_matching_video(videos, start_time_str):
    """Find video file covering the event start time."""
    for fmt in ["%H:%M:%S", "%H:%M"]:
        try:
            event_time = datetime.strptime(start_time_str, fmt)
            break
        except ValueError:
            continue
    else:
        return None

    for video in videos:
        times = parse_video_times(video["filename"])
        if not times:
            continue
        video_start, video_end = times
        event_dt = datetime.combine(video_start.date(), event_time.time())
        if video_start <= event_dt < video_end:
            offset = (event_dt - video_start).total_seconds()
            return {
                "path": video["path"],
                "filename": video["filename"],
                "offset": max(0, offset),
            }
    return None


def extract_clip_ffmpeg(video_path, start_sec, duration_sec, output_path, crf=23):
    """Extract clip using ffmpeg. CRF 23 for comparison clips (smaller files)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.2f}",
        "-i", video_path,
        "-t", f"{duration_sec:.2f}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=str, help="Comma-separated event indices")
    ap.add_argument("--cam", type=int, action="append", required=True,
                    help="Camera(s) to extract from (can repeat: --cam 9 --cam 7)")
    ap.add_argument("--input", default="data/manifests/MASTER_LINKED_v4.csv")
    ap.add_argument("--manifest", default="data/manifests/video_manifest.json")
    ap.add_argument("--output-dir", default="data/processed/audit_comparison")
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    event_ids = [int(e.strip()) for e in args.events.split(",")]
    target_cams = args.cam

    df = pd.read_csv(PROJECT_ROOT / args.input)
    with open(PROJECT_ROOT / args.manifest) as f:
        manifest = json.load(f)

    # Build camera lookup: {(group, day): {cam_id: [videos]}}
    camera_lookup = {}
    for folder_key, folder_data in manifest.get("labeled", {}).items():
        m = re.match(r"group(\d+)_day(\d+)", folder_key)
        if not m:
            continue
        gid, did = int(m.group(1)), int(m.group(2))
        cameras = folder_data.get("cameras", {})
        cam_map = {}
        for cam_id_str, cam_data in cameras.items():
            videos = [v for v in cam_data.get("videos", [])
                      if not v["filename"].startswith("._")]
            if videos:
                cam_map[int(cam_id_str)] = videos
        camera_lookup[(gid, did)] = cam_map

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    no_video_count = 0

    for eidx in sorted(event_ids):
        row = df[df["event_idx"] == eidx]
        if len(row) == 0:
            print(f"  evt_{eidx:04d}: NOT FOUND")
            continue
        row = row.iloc[0]

        group = int(row["group"])
        day = int(row["day"])
        behavior = str(row["behavior"]).strip().lower()
        duration = float(row["duration_sec"])
        start_time = str(row["start_time"])
        current_cam = int(row["camera_id"])

        key = (group, day)
        if key not in camera_lookup:
            print(f"  evt_{eidx:04d}: No camera data for Group {group} Day {day}")
            continue

        cams_available = camera_lookup[key]

        for target_cam in target_cams:
            if target_cam not in cams_available:
                print(f"  evt_{eidx:04d}: Cam {target_cam} not available for Group {group} Day {day}")
                continue

            videos = cams_available[target_cam]
            match = find_matching_video(videos, start_time)

            if not match:
                print(f"  evt_{eidx:04d}: Cam {target_cam} - no video covering {start_time}")
                no_video_count += 1
                continue

            clip_start = max(0, match["offset"] - args.pad)
            clip_dur = duration + 2 * args.pad
            clip_dur = max(clip_dur, 6.0)  # minimum 6s

            out_name = f"evt_{eidx:04d}_{behavior}_cam{target_cam:02d}.mp4"
            out_path = str(output_dir / out_name)

            ok = extract_clip_ffmpeg(match["path"], clip_start, clip_dur, out_path, crf=args.crf)

            if ok:
                success_count += 1
                print(f"  evt_{eidx:04d}: Cam {target_cam} -> OK")
            else:
                fail_count += 1
                print(f"  evt_{eidx:04d}: Cam {target_cam} -> FAILED")

    print(f"\nDone: {success_count} extracted, {fail_count} failed, {no_video_count} no matching video")
    print(f"Clips in: {output_dir}")


if __name__ == "__main__":
    main()
