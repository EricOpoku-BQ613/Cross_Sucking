#!/usr/bin/env python3
"""
Video-Level Diagnostic
======================

Check if the model is overfitting to video-level features by analyzing:
1. Events from same video file getting identical predictions
2. Duplicate events in the manifest
3. Video distribution across train/val
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_video_distribution(manifest_path: str, name: str = "dataset"):
    """Check how events distribute across video files."""
    
    df = pd.read_csv(manifest_path)
    
    print(f"\n{'='*70}")
    print(f"VIDEO ANALYSIS: {name}")
    print(f"{'='*70}")
    
    # Find video path column
    video_col = None
    for col in ['video_path', 'clip_path', 'path', 'video']:
        if col in df.columns:
            video_col = col
            break
    
    if video_col is None:
        print("⚠️  No video path column found!")
        print(f"   Available columns: {list(df.columns)}")
        return df
    
    # Extract video filename (without full path)
    df['video_file'] = df[video_col].apply(lambda x: Path(x).name if pd.notna(x) else 'unknown')
    
    print(f"\nTotal events: {len(df)}")
    print(f"Unique videos: {df['video_file'].nunique()}")
    print(f"Events per video: {len(df) / df['video_file'].nunique():.1f} avg")
    
    # Events per video distribution
    events_per_video = df.groupby('video_file').size()
    print(f"\nEvents per video distribution:")
    print(f"  Min: {events_per_video.min()}")
    print(f"  Max: {events_per_video.max()}")
    print(f"  Mean: {events_per_video.mean():.1f}")
    print(f"  Median: {events_per_video.median():.1f}")
    
    # Videos with many events (potential overfitting targets)
    high_event_videos = events_per_video[events_per_video > 5]
    if len(high_event_videos) > 0:
        print(f"\n⚠️  Videos with >5 events ({len(high_event_videos)} videos):")
        for video, count in high_event_videos.nlargest(10).items():
            subset = df[df['video_file'] == video]
            behaviors = subset['behavior'].value_counts().to_dict()
            print(f"    {video}: {count} events - {behaviors}")
    
    # Check class distribution per video
    print(f"\n{'-'*70}")
    print("CLASS DISTRIBUTION BY VIDEO")
    print(f"{'-'*70}")
    
    video_stats = []
    for video in df['video_file'].unique():
        subset = df[df['video_file'] == video]
        n_total = len(subset)
        n_ear = (subset['behavior'] == 'ear').sum()
        n_tail = (subset['behavior'] == 'tail').sum()
        
        video_stats.append({
            'video': video[:40],  # Truncate for display
            'n_events': n_total,
            'n_ear': n_ear,
            'n_tail': n_tail,
            'tail_ratio': n_tail / n_total if n_total > 0 else 0
        })
    
    stats_df = pd.DataFrame(video_stats)
    
    # Show videos with mixed classes (important for learning)
    mixed_videos = stats_df[(stats_df['n_ear'] > 0) & (stats_df['n_tail'] > 0)]
    print(f"\nVideos with BOTH ear and tail events: {len(mixed_videos)}")
    if len(mixed_videos) > 0:
        print(mixed_videos.sort_values('n_tail', ascending=False).head(10).to_string(index=False))
    
    # Show pure-class videos
    ear_only = stats_df[(stats_df['n_ear'] > 0) & (stats_df['n_tail'] == 0)]
    tail_only = stats_df[(stats_df['n_ear'] == 0) & (stats_df['n_tail'] > 0)]
    print(f"\nVideos with ONLY ear events: {len(ear_only)} ({ear_only['n_events'].sum()} total events)")
    print(f"Videos with ONLY tail events: {len(tail_only)} ({tail_only['n_events'].sum()} total events)")
    
    return df


def check_duplicate_events(manifest_path: str):
    """Check for duplicate or near-duplicate events."""
    
    df = pd.read_csv(manifest_path)
    
    print(f"\n{'='*70}")
    print("DUPLICATE EVENT CHECK")
    print(f"{'='*70}")
    
    # Check for exact duplicates
    dup_cols = [c for c in ['video_path', 'clip_path', 'start_frame', 'end_frame', 'start_sec', 'end_sec'] 
                if c in df.columns]
    
    if dup_cols:
        duplicates = df.duplicated(subset=dup_cols, keep=False)
        n_dups = duplicates.sum()
        
        if n_dups > 0:
            print(f"⚠️  Found {n_dups} duplicate events based on {dup_cols}")
            print("\nDuplicate events:")
            print(df[duplicates].head(20).to_string())
        else:
            print(f"✓ No exact duplicates found based on {dup_cols}")
    
    # Check for events with same video + overlapping time
    if 'video_path' in df.columns or 'clip_path' in df.columns:
        video_col = 'video_path' if 'video_path' in df.columns else 'clip_path'
        
        # Group by video and check for time overlaps
        overlaps = []
        for video in df[video_col].unique():
            subset = df[df[video_col] == video].copy()
            if len(subset) > 1:
                # Check if events overlap in time
                if 'start_sec' in subset.columns and 'end_sec' in subset.columns:
                    subset = subset.sort_values('start_sec')
                    for i in range(len(subset) - 1):
                        curr_end = subset.iloc[i]['end_sec']
                        next_start = subset.iloc[i+1]['start_sec']
                        if curr_end > next_start:
                            overlaps.append({
                                'video': video,
                                'event1_idx': subset.index[i],
                                'event2_idx': subset.index[i+1],
                                'overlap_sec': curr_end - next_start
                            })
        
        if overlaps:
            print(f"\n⚠️  Found {len(overlaps)} overlapping event pairs!")
            overlap_df = pd.DataFrame(overlaps)
            print(overlap_df.head(10).to_string(index=False))


def check_train_val_video_overlap(train_path: str, val_path: str):
    """Check if same videos appear in both train and val."""
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"\n{'='*70}")
    print("TRAIN vs VAL VIDEO OVERLAP")
    print(f"{'='*70}")
    
    # Find video column
    video_col = None
    for col in ['video_path', 'clip_path', 'path']:
        if col in train_df.columns:
            video_col = col
            break
    
    if video_col is None:
        print("⚠️  No video path column found!")
        return
    
    # Extract video filenames
    train_videos = set(train_df[video_col].apply(lambda x: Path(x).name if pd.notna(x) else 'unknown'))
    val_videos = set(val_df[video_col].apply(lambda x: Path(x).name if pd.notna(x) else 'unknown'))
    
    overlap = train_videos & val_videos
    train_only = train_videos - val_videos
    val_only = val_videos - train_videos
    
    print(f"\nTrain videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")
    print(f"Videos in BOTH: {len(overlap)}")
    print(f"Videos ONLY in train: {len(train_only)}")
    print(f"Videos ONLY in val: {len(val_only)}")
    
    if len(overlap) > 0:
        print(f"\n⚠️  POTENTIAL DATA LEAKAGE: {len(overlap)} videos appear in both train and val!")
        print("\nOverlapping videos:")
        for v in list(overlap)[:10]:
            train_subset = train_df[train_df[video_col].str.contains(v, na=False)]
            val_subset = val_df[val_df[video_col].str.contains(v, na=False)]
            print(f"  {v}: train={len(train_subset)} events, val={len(val_subset)} events")
    
    if len(val_only) > 0:
        print(f"\n⚠️  Val-only videos ({len(val_only)}):")
        # Check class distribution in val-only videos
        val_df['video_file'] = val_df[video_col].apply(lambda x: Path(x).name)
        val_only_events = val_df[val_df['video_file'].isin(val_only)]
        print(f"   Total events in val-only videos: {len(val_only_events)}")
        print(f"   Class distribution: {val_only_events['behavior'].value_counts().to_dict()}")


def analyze_predictions_by_video(predictions_path: str, manifest_path: str):
    """Match predictions back to videos and check for video-level patterns."""
    
    pred_df = pd.read_csv(predictions_path)
    manifest_df = pd.read_csv(manifest_path)
    
    print(f"\n{'='*70}")
    print("PREDICTIONS BY VIDEO")
    print(f"{'='*70}")
    
    # This requires matching predictions to manifest
    # Predictions use event_0, event_1, etc.
    # Need to know how these map to the manifest
    
    print("\nTo analyze predictions by video, we need to match event indices.")
    print("Prediction indices: 0 to", len(pred_df) - 1)
    print("Manifest indices:", manifest_df.index.min(), "to", manifest_df.index.max())
    
    # Check for probability clusters
    print("\n" + "-"*70)
    print("PROBABILITY CLUSTERING (sign of video-level overfitting)")
    print("-"*70)
    
    prob_counts = pred_df['prob_tail'].round(4).value_counts()
    clusters = prob_counts[prob_counts >= 3]
    
    if len(clusters) > 0:
        print(f"\n⚠️  Found {len(clusters)} probability values appearing 3+ times:")
        for prob, count in clusters.head(15).items():
            events = pred_df[pred_df['prob_tail'].round(4) == prob]
            true_labels = events['true_label'].value_counts().to_dict()
            print(f"  P(tail)={prob:.4f}: {count}x - true labels: {true_labels}")
            
        print("\n   This suggests the model outputs IDENTICAL predictions for different events,")
        print("   likely because they come from the same video or have very similar features.")


def main():
    parser = argparse.ArgumentParser(description='Video-level diagnostic')
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--predictions', type=str, default=None)
    
    args = parser.parse_args()
    
    # Analyze video distribution
    analyze_video_distribution(args.train, "TRAINING SET")
    analyze_video_distribution(args.val, "VALIDATION SET")
    
    # Check duplicates
    check_duplicate_events(args.train)
    check_duplicate_events(args.val)
    
    # Check train/val overlap
    check_train_val_video_overlap(args.train, args.val)
    
    # If predictions provided, analyze by video
    if args.predictions:
        analyze_predictions_by_video(args.predictions, args.val)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
If you see:
- Same video in both train and val → DATA LEAKAGE, re-split by video
- Val-only videos with tail events → Model hasn't seen these videos
- Probability clusters → Video-level overfitting

Key insight: The model may be learning VIDEO features (lighting, camera angle, 
background) rather than BEHAVIOR features. Events from the same video will 
get identical predictions regardless of actual behavior.

Solutions:
1. Split by VIDEO FILE, not by event (no video appears in both train and val)
2. Split by DAY or GROUP (different recording sessions)
3. Use heavier augmentation to break video-specific patterns
4. Add video-level dropout or batch composition constraints
""")


if __name__ == '__main__':
    main()