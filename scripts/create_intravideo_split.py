#!/usr/bin/env python3
"""
Intra-Video Data Splitting
==========================

Creates train/val splits where BOTH sets contain events from the same videos.
This ensures the model learns BEHAVIOR features, not VIDEO features.

Strategy:
- For each video with multiple events, split events ~80/20 into train/val
- Stratify by behavior class within each video
- Ensures model sees similar video conditions in both train and val
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import argparse
from datetime import datetime


def intra_video_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    min_events_for_split: int = 3,
    seed: int = 42,
    video_col: str = 'video_path'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data so both train and val contain events from the same videos.
    
    Args:
        df: Full manifest DataFrame
        val_ratio: Fraction of events per video to put in val
        min_events_for_split: Videos with fewer events go entirely to train
        seed: Random seed
        video_col: Column containing video path
        
    Returns:
        train_df, val_df
    """
    np.random.seed(seed)
    
    train_indices = []
    val_indices = []
    
    # Extract video filename for grouping
    df = df.copy()
    df['_video_file'] = df[video_col].apply(lambda x: Path(x).name if pd.notna(x) else 'unknown')
    
    # Process each video
    for video in df['_video_file'].unique():
        video_mask = df['_video_file'] == video
        video_indices = df[video_mask].index.tolist()
        video_df = df.loc[video_indices]
        
        n_events = len(video_indices)
        
        # Small videos go entirely to train
        if n_events < min_events_for_split:
            train_indices.extend(video_indices)
            continue
        
        # Try stratified split by behavior
        behaviors = video_df['behavior'].values
        unique_behaviors = np.unique(behaviors)
        
        if len(unique_behaviors) == 1:
            # All same class - random split
            n_val = max(1, int(n_events * val_ratio))
            val_idx = np.random.choice(video_indices, size=n_val, replace=False).tolist()
            train_idx = [i for i in video_indices if i not in val_idx]
        else:
            # Stratified split
            try:
                train_idx, val_idx = train_test_split(
                    video_indices,
                    test_size=val_ratio,
                    stratify=behaviors,
                    random_state=seed
                )
            except ValueError:
                # If stratification fails (too few samples), do random split
                n_val = max(1, int(n_events * val_ratio))
                val_idx = np.random.choice(video_indices, size=n_val, replace=False).tolist()
                train_idx = [i for i in video_indices if i not in val_idx]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    
    # Create splits
    train_df = df.loc[train_indices].drop(columns=['_video_file']).copy()
    val_df = df.loc[val_indices].drop(columns=['_video_file']).copy()
    
    return train_df, val_df


def analyze_split(train_df: pd.DataFrame, val_df: pd.DataFrame, video_col: str = 'video_path'):
    """Print analysis of the split."""
    
    print("\n" + "="*70)
    print("SPLIT ANALYSIS")
    print("="*70)
    
    # Extract video filenames
    train_videos = set(train_df[video_col].apply(lambda x: Path(x).name))
    val_videos = set(val_df[video_col].apply(lambda x: Path(x).name))
    
    overlap = train_videos & val_videos
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} events from {len(train_videos)} videos")
    print(f"  Val:   {len(val_df)} events from {len(val_videos)} videos")
    print(f"  Video overlap: {len(overlap)} videos appear in BOTH")
    
    # Class distribution
    print(f"\nClass distribution:")
    print(f"  Train: {train_df['behavior'].value_counts().to_dict()}")
    print(f"  Val:   {val_df['behavior'].value_counts().to_dict()}")
    
    # Tail ratio
    train_tail_ratio = (train_df['behavior'] == 'tail').mean()
    val_tail_ratio = (val_df['behavior'] == 'tail').mean()
    print(f"\nTail ratio:")
    print(f"  Train: {train_tail_ratio:.1%}")
    print(f"  Val:   {val_tail_ratio:.1%}")
    
    # Check videos with tail in both splits
    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    train_df_copy['_video'] = train_df_copy[video_col].apply(lambda x: Path(x).name)
    val_df_copy['_video'] = val_df_copy[video_col].apply(lambda x: Path(x).name)
    
    train_tail_videos = set(train_df_copy[train_df_copy['behavior'] == 'tail']['_video'])
    val_tail_videos = set(val_df_copy[val_df_copy['behavior'] == 'tail']['_video'])
    tail_overlap = train_tail_videos & val_tail_videos
    
    print(f"\nTail event video distribution:")
    print(f"  Videos with tail in train: {len(train_tail_videos)}")
    print(f"  Videos with tail in val: {len(val_tail_videos)}")
    print(f"  Videos with tail in BOTH: {len(tail_overlap)}")
    
    if len(tail_overlap) == 0:
        print("\nWARNING: No videos have tail events in both train and val!")
        print("   Consider using a different split strategy.")
    else:
        print(f"\nGood: {len(tail_overlap)} videos have tail events in both splits")


def boost_minority_class(df: pd.DataFrame, target_ratio: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """
    Boost minority class (tail) by oversampling to reach target ratio.
    
    Args:
        df: DataFrame to boost
        target_ratio: Target ratio of minority class
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Filter to binary (ear/tail only)
    df = df[df['behavior'].isin(['ear', 'tail'])].copy()
    
    n_ear = (df['behavior'] == 'ear').sum()
    n_tail = (df['behavior'] == 'tail').sum()
    current_ratio = n_tail / len(df)
    
    print(f"\nBoosting minority class:")
    print(f"  Before: {n_ear} ear, {n_tail} tail ({current_ratio:.1%} tail)")
    
    if current_ratio >= target_ratio:
        print(f"  Already at or above target ratio ({target_ratio:.1%})")
        return df
    
    # Calculate how many tail samples we need
    # target_ratio = (n_tail + n_extra) / (n_ear + n_tail + n_extra)
    # Solving: n_extra = (target_ratio * n_ear - n_tail * (1 - target_ratio)) / (1 - target_ratio)
    n_extra = int((target_ratio * n_ear - n_tail * (1 - target_ratio)) / (1 - target_ratio))
    n_extra = max(0, n_extra)
    
    # Sample with replacement from tail events
    tail_df = df[df['behavior'] == 'tail']
    extra_samples = tail_df.sample(n=n_extra, replace=True, random_state=seed)
    
    # Combine
    boosted_df = pd.concat([df, extra_samples], ignore_index=True)
    
    n_ear_new = (boosted_df['behavior'] == 'ear').sum()
    n_tail_new = (boosted_df['behavior'] == 'tail').sum()
    new_ratio = n_tail_new / len(boosted_df)
    
    print(f"  After:  {n_ear_new} ear, {n_tail_new} tail ({new_ratio:.1%} tail)")
    print(f"  Added {n_extra} oversampled tail events")
    
    return boosted_df


def main():
    parser = argparse.ArgumentParser(description='Create intra-video train/val split')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to full manifest CSV')
    parser.add_argument('--output-dir', type=str, default='data/manifests',
                       help='Output directory for split files')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Fraction of events per video to put in val')
    parser.add_argument('--boost-ratio', type=float, default=0.0,
                       help='Target tail ratio after boosting (0 = no boosting)')
    parser.add_argument('--filter-binary', action='store_true',
                       help='Filter to ear/tail only')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Total events: {len(df)}")
    print(f"Class distribution: {df['behavior'].value_counts().to_dict()}")
    
    # Filter if requested
    if args.filter_binary:
        df = df[df['behavior'].isin(['ear', 'tail'])].copy()
        print(f"\nFiltered to ear/tail: {len(df)} events")
    
    # Find video column
    video_col = None
    for col in ['video_path', 'clip_path', 'path']:
        if col in df.columns:
            video_col = col
            break
    
    if video_col is None:
        print("ERROR: No video path column found!")
        return
    
    print(f"Using video column: {video_col}")
    
    # Create split
    print("\n" + "-"*70)
    print("Creating intra-video split...")
    print("-"*70)
    
    train_df, val_df = intra_video_split(
        df, 
        val_ratio=args.val_ratio,
        seed=args.seed,
        video_col=video_col
    )
    
    # Analyze split
    analyze_split(train_df, val_df, video_col)
    
    # Boost if requested
    if args.boost_ratio > 0:
        print("\n" + "-"*70)
        print("Boosting minority class...")
        print("-"*70)
        train_df = boost_minority_class(train_df, target_ratio=args.boost_ratio, seed=args.seed)
        val_df = boost_minority_class(val_df, target_ratio=args.boost_ratio, seed=args.seed)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"train_intravideo_{timestamp}.csv"
    val_path = output_dir / f"val_intravideo_{timestamp}.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n" + "="*70)
    print("SAVED")
    print("="*70)
    print(f"Train: {train_path} ({len(train_df)} events)")
    print(f"Val:   {val_path} ({len(val_df)} events)")
    
    # Print config snippet
    print(f"\n" + "-"*70)
    print("Add to your config:")
    print("-"*70)
    print(f"""
data:
  train_csv: "{train_path}"
  val_csv: "{val_path}"
  balanced_sampling: true
  filter_to_id: true
""")


if __name__ == '__main__':
    main()