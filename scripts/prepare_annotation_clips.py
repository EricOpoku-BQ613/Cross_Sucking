#!/usr/bin/env python3
"""
Prepare Event Clips for Bounding Box Annotation
================================================

This script:
1. Reads your event manifest CSV
2. Extracts short clips for each event
3. Exports keyframes as images (easier to annotate)
4. Creates annotation task files for CVAT/Label Studio

Usage:
    python scripts/prepare_annotation_clips.py \
        --manifest data/manifests/train.csv \
        --output data/annotation_ready/ \
        --num-events 50 \
        --keyframe-interval 15
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import csv

import cv2
import numpy as np
from tqdm import tqdm


def extract_keyframes(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: int = 12,
    keyframe_interval: int = 15,
    output_dir: str = None,
    event_id: str = None,
) -> List[Dict]:
    """
    Extract keyframes from a video event.
    
    Args:
        video_path: Path to source video
        start_sec: Event start time
        end_sec: Event end time
        fps: Target FPS for sampling
        keyframe_interval: Frames between keyframes
        output_dir: Where to save keyframe images
        event_id: Event identifier for naming
    
    Returns:
        List of keyframe metadata dicts
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    end_frame = min(end_frame, total_frames - 1)
    
    duration_frames = end_frame - start_frame
    
    # Handle edge case: very short or zero-duration events
    if duration_frames <= 0:
        print(f"[WARN] Event {event_id}: zero or negative duration, skipping")
        cap.release()
        return []
    
    # Determine keyframe positions
    keyframe_indices = list(range(0, duration_frames, keyframe_interval))
    
    # Handle empty list (shouldn't happen now, but safety check)
    if not keyframe_indices:
        keyframe_indices = [0]
    
    # Always include last frame
    if keyframe_indices[-1] != duration_frames - 1:
        keyframe_indices.append(duration_frames - 1)
    
    keyframes = []
    
    for local_idx in keyframe_indices:
        global_frame = start_frame + local_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        timestamp_sec = local_idx / video_fps
        
        keyframe_info = {
            'event_id': event_id,
            'frame_idx': local_idx,
            'global_frame': global_frame,
            'timestamp_sec': round(timestamp_sec, 3),
            'width': frame.shape[1],
            'height': frame.shape[0],
        }
        
        # Save keyframe image if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            img_filename = f"{event_id}_frame{local_idx:04d}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            cv2.imwrite(img_path, frame)
            keyframe_info['image_path'] = img_path
            keyframe_info['image_filename'] = img_filename
        
        keyframes.append(keyframe_info)
    
    cap.release()
    return keyframes


def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    fps: int = 12,
) -> bool:
    """
    Extract a video clip for an event.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_step = max(1, int(video_fps / fps))
    current_frame = start_frame
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        current_frame += frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    cap.release()
    out.release()
    return True


def create_cvat_task(
    events: List[Dict],
    output_path: str,
    project_name: str = "CrossSucking_BBox"
):
    """
    Create CVAT-compatible task definition.
    """
    task = {
        "name": project_name,
        "labels": [
            {
                "name": "calf",
                "attributes": [
                    {
                        "name": "role",
                        "mutable": False,
                        "input_type": "select",
                        "values": ["initiator", "receiver", "other"]
                    },
                    {
                        "name": "calf_id", 
                        "mutable": False,
                        "input_type": "text",
                        "values": []
                    },
                    {
                        "name": "body_visible",
                        "mutable": True,
                        "input_type": "select",
                        "values": ["full", "head_only", "body_only", "partial", "occluded"]
                    }
                ]
            },
            {
                "name": "target_region",
                "attributes": [
                    {
                        "name": "type",
                        "mutable": False,
                        "input_type": "select",
                        "values": ["ear", "tail", "teat", "other"]
                    }
                ]
            }
        ],
        "events": events
    }
    
    with open(output_path, 'w') as f:
        json.dump(task, f, indent=2)
    
    print(f"[CVAT] Task definition saved to {output_path}")


def create_labelstudio_tasks(
    events: List[Dict],
    keyframes_dir: str,
    output_path: str,
):
    """
    Create Label Studio import file.
    """
    tasks = []
    
    for event in events:
        # Create task for each event's keyframes
        for kf in event.get('keyframes', []):
            task = {
                "data": {
                    "image": f"/data/local-files/?d={kf['image_path']}",
                    "event_id": event['event_id'],
                    "frame_idx": kf['frame_idx'],
                    "timestamp_sec": kf['timestamp_sec'],
                    "behavior": event['behavior'],
                    "initiator_id": event['initiator_id'],
                    "receiver_id": event['receiver_id'],
                },
                "predictions": []  # Pre-fill with model predictions later
            }
            tasks.append(task)
    
    with open(output_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"[LabelStudio] {len(tasks)} tasks saved to {output_path}")


def select_diverse_events(
    manifest_path: str,
    num_events: int,
    stratify_by: List[str] = ['behavior', 'group', 'camera_id'],
    exclude_event_ids: List[str] = None
) -> List[Dict]:
    """
    Select diverse events for annotation.
    Stratifies by behavior type, group, and camera to ensure coverage.
    
    Priority:
    1. Include ALL tail/teat events (rare classes)
    2. Then sample ear events proportionally across groups/cameras
    
    Args:
        manifest_path: Path to CSV manifest
        num_events: Number of events to select
        stratify_by: Columns to stratify by
        exclude_event_ids: List of event IDs to exclude (already annotated)
    """
    events = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        events = list(reader)
    
    # Exclude already-annotated events
    if exclude_event_ids:
        original_count = len(events)
        events = [e for e in events if f"evt_{e.get('event_idx', '')}" not in exclude_event_ids]
        print(f"  Excluded {original_count - len(events)} already-annotated events")
    
    selected = []
    
    # PRIORITY 1: Include all rare class events (tail, teat, other)
    rare_behaviors = ['tail', 'teat', 'other']
    rare_events = [e for e in events if e.get('behavior', '').lower() in rare_behaviors]
    
    # Sample from rare events (up to 30% of total)
    max_rare = min(len(rare_events), int(num_events * 0.35))
    if len(rare_events) <= max_rare:
        selected.extend(rare_events)
        print(f"  Rare classes (tail/teat/other): selected ALL {len(rare_events)} events")
    else:
        # Stratify rare events by group/camera
        rare_sampled = random.sample(rare_events, max_rare)
        selected.extend(rare_sampled)
        print(f"  Rare classes (tail/teat/other): selected {max_rare}/{len(rare_events)} events")
    
    # PRIORITY 2: Fill remaining with ear events, stratified by group/camera
    remaining_slots = num_events - len(selected)
    ear_events = [e for e in events if e.get('behavior', '').lower() == 'ear' and e not in selected]
    
    # Stratify ear events
    ear_groups = {}
    for event in ear_events:
        key = (event.get('group', 'unk'), event.get('camera_id', 'unk'))
        if key not in ear_groups:
            ear_groups[key] = []
        ear_groups[key].append(event)
    
    # Sample proportionally from each group/camera combo
    ear_per_group = max(1, remaining_slots // max(1, len(ear_groups)))
    
    for key, group_events in ear_groups.items():
        n_sample = min(ear_per_group, len(group_events))
        sampled = random.sample(group_events, n_sample)
        selected.extend(sampled)
        print(f"  Ear {key}: selected {n_sample}/{len(group_events)}")
    
    # If still need more, sample randomly
    if len(selected) < num_events:
        remaining = [e for e in events if e not in selected]
        if remaining:
            extra = random.sample(remaining, min(num_events - len(selected), len(remaining)))
            selected.extend(extra)
            print(f"  Additional random: {len(extra)} events")
    
    random.shuffle(selected)  # Shuffle final selection
    return selected[:num_events]


def main():
    parser = argparse.ArgumentParser(description='Prepare clips for annotation')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to event manifest CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for annotation data')
    parser.add_argument('--num-events', type=int, default=50,
                       help='Number of events to prepare')
    parser.add_argument('--keyframe-interval', type=int, default=15,
                       help='Frames between keyframes')
    parser.add_argument('--extract-clips', action='store_true',
                       help='Also extract video clips')
    parser.add_argument('--fps', type=int, default=12,
                       help='Target FPS for clips')
    parser.add_argument('--tool', type=str, default='cvat',
                       choices=['cvat', 'labelstudio', 'both'],
                       help='Annotation tool format')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for selection')
    parser.add_argument('--exclude-events', type=str, default=None,
                       help='Path to annotation_summary.json to exclude already-annotated events')
    parser.add_argument('--prioritize-rare', action='store_true', default=True,
                       help='Prioritize tail/teat/other events in selection')
    parser.add_argument('--batch-name', type=str, default=None,
                       help='Batch name for organizing outputs (e.g., batch1, batch2)')
    
    args = parser.parse_args()
    
    # Handle batch naming
    if args.batch_name:
        args.output = str(Path(args.output) / args.batch_name)
    random.seed(args.seed)
    
    # Create output directories
    output_dir = Path(args.output)
    keyframes_dir = output_dir / 'keyframes'
    clips_dir = output_dir / 'clips'
    tasks_dir = output_dir / 'tasks'
    
    for d in [keyframes_dir, clips_dir, tasks_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load excluded events if provided
    exclude_event_ids = []
    if args.exclude_events and os.path.exists(args.exclude_events):
        with open(args.exclude_events, 'r') as f:
            prev_summary = json.load(f)
            exclude_event_ids = [e['event_id'] for e in prev_summary.get('events', [])]
        print(f"  Loaded {len(exclude_event_ids)} events to exclude")
    
    # Select diverse events
    print(f"\n[1/4] Selecting {args.num_events} diverse events...")
    selected_events = select_diverse_events(
        args.manifest, 
        args.num_events,
        exclude_event_ids=exclude_event_ids
    )
    print(f"  Selected {len(selected_events)} events")
    
    # Process each event
    print(f"\n[2/4] Extracting keyframes...")
    processed_events = []
    
    for event in tqdm(selected_events):
        event_id = f"evt_{event['event_idx']}"
        video_path = event['video_path']
        
        # Parse times (handle both seconds and time strings)
        try:
            if ':' in str(event.get('start_time', '')):
                # Time string format HH:MM:SS
                start_sec = float(event.get('event_offset_sec', 0))
                end_sec = start_sec + float(event.get('duration_sec', 5))
            else:
                start_sec = float(event.get('event_offset_sec', 0))
                end_sec = start_sec + float(event.get('duration_sec', 5))
        except:
            start_sec = 0
            end_sec = 5
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"  [SKIP] Video not found: {video_path}")
            continue
        
        # Extract keyframes
        event_keyframes_dir = keyframes_dir / event_id
        keyframes = extract_keyframes(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=args.fps,
            keyframe_interval=args.keyframe_interval,
            output_dir=str(event_keyframes_dir),
            event_id=event_id,
        )
        
        # Extract clip if requested
        clip_path = None
        if args.extract_clips:
            clip_path = str(clips_dir / f"{event_id}.mp4")
            extract_clip(video_path, start_sec, end_sec, clip_path, args.fps)
        
        processed_event = {
            'event_id': event_id,
            'event_idx': event['event_idx'],
            'video_path': video_path,
            'clip_path': clip_path,
            'behavior': event['behavior'],
            'initiator_id': event['initiator_id'],
            'receiver_id': event['receiver_id'],
            'duration_sec': event['duration_sec'],
            'group': event.get('group', ''),
            'keyframes': keyframes,
            'num_keyframes': len(keyframes),
        }
        processed_events.append(processed_event)
    
    print(f"  Processed {len(processed_events)} events")
    
    # Create task files
    print(f"\n[3/4] Creating annotation task files...")
    
    if args.tool in ['cvat', 'both']:
        create_cvat_task(
            processed_events,
            str(tasks_dir / 'cvat_task.json'),
        )
    
    if args.tool in ['labelstudio', 'both']:
        create_labelstudio_tasks(
            processed_events,
            str(keyframes_dir),
            str(tasks_dir / 'labelstudio_tasks.json'),
        )
    
    # Save event summary
    print(f"\n[4/4] Saving summary...")
    summary = {
        'total_events': len(processed_events),
        'total_keyframes': sum(e['num_keyframes'] for e in processed_events),
        'behaviors': {},
        'events': processed_events,
    }
    
    for event in processed_events:
        b = event['behavior']
        summary['behaviors'][b] = summary['behaviors'].get(b, 0) + 1
    
    with open(tasks_dir / 'annotation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ANNOTATION PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Events prepared:     {len(processed_events)}")
    print(f"Total keyframes:     {summary['total_keyframes']}")
    print(f"Behavior breakdown:")
    for b, count in summary['behaviors'].items():
        print(f"  - {b}: {count}")
    print(f"\nOutput directory:    {output_dir}")
    print(f"Keyframes:           {keyframes_dir}")
    if args.extract_clips:
        print(f"Clips:               {clips_dir}")
    print(f"Task files:          {tasks_dir}")
    print(f"\nEstimated annotation time: {len(processed_events) * 3}-{len(processed_events) * 5} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()