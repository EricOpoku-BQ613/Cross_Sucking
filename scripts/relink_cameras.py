#!/usr/bin/env python3
"""
Re-link Events to Correct Cameras
==================================

The original linking picked the FIRST camera with a matching time window,
which often selected outdoor/wrong cameras. This script re-links every event
to the best indoor camera using a verified priority order per group.

Outdoor cameras (never used for annotation):
  - Cam 8  is outdoor for Groups 3, 4, 5, 6
  - Cam 10 is outdoor for Groups 1, 2

Camera priority (based on manual verification of extracted clips):
  Group 1: Cam 8 > Cam 9 > Cam 7     (Cam 10 outdoor)
  Group 2: Cam 16 > Cam 12 > Cam 11   (Cam 10 outdoor)
  Group 3: Cam 12 > Cam 14 > Cam 16   (Cam 8 outdoor)
  Group 4: Cam 3 > Cam 5 > Cam 10     (Cam 8 outdoor)  -- needs verification
  Group 5: Cam 1 > Cam 9 > Cam 7      (Cam 8 outdoor)  -- needs verification
  Group 6: Cam 12 > Cam 14 > Cam 2    (Cam 8 outdoor)

Usage:
    python scripts/relink_cameras.py --dry-run
    python scripts/relink_cameras.py
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Outdoor cameras per group (NEVER use for annotation)
OUTDOOR_CAMERAS = {
    1: {10},
    2: {10},
    3: {8},
    4: {8},
    5: {8},
    6: {8},
}

# Preferred camera order per group (first = most likely annotation camera)
CAMERA_PRIORITY = {
    1: [8, 9, 7],
    2: [16, 12, 11],
    3: [12, 14, 16],
    4: [3, 5, 10],       # Tentative — Cam 3 was first-linked and might be correct
    5: [1, 9, 7],         # Tentative — Cam 1 was first-linked
    6: [12, 14, 2],
}


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


def find_video_for_time(videos, event_time_str, event_date_str=None):
    """Find the video file covering the given time."""
    # Parse event time
    event_time = None
    for fmt in ['%H:%M:%S', '%H:%M']:
        try:
            event_time = datetime.strptime(event_time_str, fmt)
            break
        except ValueError:
            continue

    if event_time is None:
        return None

    for video in videos:
        # Skip macOS resource fork files
        if video['filename'].startswith('._'):
            continue

        times = parse_video_times(video['filename'])
        if not times:
            continue

        video_start, video_end = times
        event_datetime = datetime.combine(video_start.date(), event_time.time())

        if video_start <= event_datetime < video_end:
            offset = (event_datetime - video_start).total_seconds()
            return {
                'video_filename': video['filename'],
                'video_path': video['path'],
                'event_offset_sec': max(0, offset),
                'camera_id': int(re.search(r'ch(\d+)', video['filename']).group(1)),
            }

    return None


def main():
    parser = argparse.ArgumentParser(description="Re-link events to correct cameras")
    parser.add_argument("--input", default="data/manifests/MASTER_LINKED_v3.csv")
    parser.add_argument("--manifest", default="data/manifests/video_manifest.json")
    parser.add_argument("--output", default="data/manifests/MASTER_LINKED_v4.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without saving")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    manifest_path = PROJECT_ROOT / args.manifest
    output_path = PROJECT_ROOT / args.output

    print(f"Loading events: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total events: {len(df)}")

    print(f"Loading video manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build camera lookup: {(group_id, day): {cam_id: [videos]}}
    camera_lookup = {}
    for folder_key, folder_data in manifest.get("labeled", {}).items():
        m = re.match(r"group(\d+)_day(\d+)", folder_key)
        if not m:
            continue
        gid, did = int(m.group(1)), int(m.group(2))
        key = (gid, did)
        camera_lookup[key] = {}
        for cam_id_str, cam_data in folder_data.get("cameras", {}).items():
            cam_id = int(cam_id_str)
            videos = cam_data.get("videos", [])
            videos = [v for v in videos if not v["filename"].startswith("._")]
            if videos:
                camera_lookup[key][cam_id] = videos

    # Track changes
    changed = 0
    unchanged = 0
    failed = 0
    unlinked = 0
    changes_by_group = {}

    new_video_paths = []
    new_video_filenames = []
    new_camera_ids = []
    new_offsets = []

    for idx, row in df.iterrows():
        if row.get("linked") != True:
            new_video_paths.append(row.get("video_path", ""))
            new_video_filenames.append(row.get("video_filename", ""))
            new_camera_ids.append(row.get("camera_id", ""))
            new_offsets.append(row.get("event_offset_sec", ""))
            unlinked += 1
            continue

        group = int(row["group"])
        day = int(row["day"])
        start_time = str(row["start_time"])
        old_cam = int(row["camera_id"])
        old_video = row["video_filename"]

        key = (group, day)
        if key not in camera_lookup:
            new_video_paths.append(row["video_path"])
            new_video_filenames.append(row["video_filename"])
            new_camera_ids.append(old_cam)
            new_offsets.append(row["event_offset_sec"])
            failed += 1
            continue

        cameras = camera_lookup[key]
        outdoor = OUTDOOR_CAMERAS.get(group, set())
        priority = CAMERA_PRIORITY.get(group, sorted(cameras.keys()))

        # Try cameras in priority order, skipping outdoor
        found = None
        for cam_id in priority:
            if cam_id in outdoor:
                continue
            if cam_id not in cameras:
                continue
            result = find_video_for_time(cameras[cam_id], start_time)
            if result:
                found = result
                break

        # Fallback: try any non-outdoor camera
        if not found:
            for cam_id in sorted(cameras.keys()):
                if cam_id in outdoor:
                    continue
                result = find_video_for_time(cameras[cam_id], start_time)
                if result:
                    found = result
                    break

        if found:
            new_video_paths.append(found["video_path"])
            new_video_filenames.append(found["video_filename"])
            new_camera_ids.append(found["camera_id"])
            new_offsets.append(found["event_offset_sec"])

            if found["camera_id"] != old_cam or found["video_filename"] != old_video:
                changed += 1
                gkey = f"Group {group}"
                if gkey not in changes_by_group:
                    changes_by_group[gkey] = {"count": 0, "old_cams": set(), "new_cams": set()}
                changes_by_group[gkey]["count"] += 1
                changes_by_group[gkey]["old_cams"].add(old_cam)
                changes_by_group[gkey]["new_cams"].add(found["camera_id"])
            else:
                unchanged += 1
        else:
            # Keep original if no match found
            new_video_paths.append(row["video_path"])
            new_video_filenames.append(row["video_filename"])
            new_camera_ids.append(old_cam)
            new_offsets.append(row["event_offset_sec"])
            failed += 1

    # Apply changes
    df["video_path"] = new_video_paths
    df["video_filename"] = new_video_filenames
    df["camera_id"] = new_camera_ids
    df["event_offset_sec"] = new_offsets

    # Report
    print(f"\n{'=' * 60}")
    print("RE-LINKING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Changed:   {changed}")
    print(f"  Unchanged: {unchanged}")
    print(f"  Failed:    {failed}")
    print(f"  Unlinked:  {unlinked}")

    if changes_by_group:
        print(f"\n  Changes by group:")
        for gkey in sorted(changes_by_group.keys()):
            info = changes_by_group[gkey]
            old = ", ".join(f"Cam {c}" for c in sorted(info["old_cams"]))
            new = ", ".join(f"Cam {c}" for c in sorted(info["new_cams"]))
            print(f"    {gkey}: {info['count']} events  ({old} -> {new})")

    # Camera distribution after re-linking
    linked_df = df[df["linked"] == True]
    print(f"\n  Camera distribution after re-linking:")
    for (group, cam), count in linked_df.groupby(["group", "camera_id"]).size().items():
        print(f"    Group {group}: Cam {int(cam)} -> {count} events")

    if args.dry_run:
        print(f"\n  DRY RUN - no files written")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n  Saved: {output_path}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
