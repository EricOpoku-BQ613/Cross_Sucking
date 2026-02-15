#!/usr/bin/env python3
"""
Check for Group/Camera Confound
================================

This script analyzes your manifest to identify if:
1. Certain groups/cameras have different class distributions
2. Validation set is missing some groups present in train
3. There's potential data leakage

Run this to diagnose the camera confound issue.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def analyze_group_distribution(manifest_path: str, name: str = "dataset"):
    """Analyze class distribution across groups/cameras."""
    
    df = pd.read_csv(manifest_path)
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*70}")
    print(f"Total events: {len(df)}")
    
    # Check available columns
    group_cols = []
    if 'group' in df.columns:
        group_cols.append('group')
    if 'camera_id' in df.columns:
        group_cols.append('camera_id')
    if 'pen_location' in df.columns:
        group_cols.append('pen_location')
    if 'day' in df.columns:
        group_cols.append('day')
    
    if not group_cols:
        print("⚠️  No group/camera columns found!")
        return df
    
    # Overall class distribution
    print(f"\nOverall class distribution:")
    behavior_counts = df['behavior'].value_counts()
    for behavior, count in behavior_counts.items():
        print(f"  {behavior}: {count} ({100*count/len(df):.1f}%)")
    
    # Analyze each grouping variable
    for col in group_cols:
        print(f"\n{'-'*70}")
        print(f"BY {col.upper()}")
        print(f"{'-'*70}")
        
        groups = df[col].unique()
        print(f"Unique values: {len(groups)}")
        
        group_stats = []
        for group in sorted(groups):
            subset = df[df[col] == group]
            n_total = len(subset)
            n_ear = (subset['behavior'] == 'ear').sum()
            n_tail = (subset['behavior'] == 'tail').sum()
            tail_ratio = n_tail / n_total if n_total > 0 else 0
            
            group_stats.append({
                col: group,
                'n_total': n_total,
                'n_ear': n_ear,
                'n_tail': n_tail,
                'tail_ratio': tail_ratio,
            })
        
        stats_df = pd.DataFrame(group_stats)
        print(stats_df.to_string(index=False))
        
        # Check for imbalance
        tail_ratios = stats_df['tail_ratio'].values
        if tail_ratios.std() > 0.1:
            print(f"\n⚠️  WARNING: High variance in tail ratio across {col}!")
            print(f"   Std of tail ratio: {tail_ratios.std():.3f}")
            print(f"   Some groups may have very different class distributions.")
    
    return df


def check_train_val_overlap(train_path: str, val_path: str):
    """Check if train and val have overlapping or missing groups."""
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"\n{'='*70}")
    print("TRAIN vs VALIDATION GROUP OVERLAP")
    print(f"{'='*70}")
    
    group_cols = []
    for col in ['group', 'camera_id', 'pen_location', 'day']:
        if col in train_df.columns and col in val_df.columns:
            group_cols.append(col)
    
    for col in group_cols:
        train_groups = set(train_df[col].unique())
        val_groups = set(val_df[col].unique())
        
        only_train = train_groups - val_groups
        only_val = val_groups - train_groups
        both = train_groups & val_groups
        
        print(f"\n{col}:")
        print(f"  Train has: {len(train_groups)} unique values")
        print(f"  Val has: {len(val_groups)} unique values")
        print(f"  In both: {len(both)}")
        
        if only_train:
            print(f"  ⚠️  ONLY in train (not in val): {only_train}")
        if only_val:
            print(f"  ⚠️  ONLY in val (not in train): {only_val}")
            
            # Check tail distribution in these groups
            val_only_subset = val_df[val_df[col].isin(only_val)]
            n_tail = (val_only_subset['behavior'] == 'tail').sum()
            print(f"     These val-only groups contain {n_tail} tail events!")
            print(f"     This could explain poor generalization!")


def check_event_index_pattern(val_manifest: str, predictions_path: str = None):
    """Check if errors cluster in certain event index ranges."""
    
    val_df = pd.read_csv(val_manifest)
    
    print(f"\n{'='*70}")
    print("EVENT INDEX ANALYSIS")
    print(f"{'='*70}")
    
    # Check event indices for tail events
    tail_events = val_df[val_df['behavior'] == 'tail']
    
    if 'event_idx' in val_df.columns:
        tail_indices = tail_events['event_idx'].values
        print(f"\nTail event indices: {sorted(tail_indices)}")
        
        # Check if they cluster
        if len(tail_indices) > 0:
            min_idx = min(tail_indices)
            max_idx = max(tail_indices)
            print(f"Range: {min_idx} to {max_idx}")
            
            # Check for gaps
            all_indices = set(range(min_idx, max_idx + 1))
            missing = all_indices - set(tail_indices)
            if len(missing) < len(all_indices) * 0.3:
                print(f"⚠️  Tail events are CLUSTERED in indices {min_idx}-{max_idx}")
                print(f"   This suggests they come from a specific recording/group!")


def main():
    parser = argparse.ArgumentParser(description='Check for group/camera confound')
    parser.add_argument('--train', type=str, required=True,
                       help='Path to training manifest')
    parser.add_argument('--val', type=str, required=True,
                       help='Path to validation manifest')
    parser.add_argument('--predictions', type=str, default=None,
                       help='Path to predictions CSV (optional)')
    
    args = parser.parse_args()
    
    # Analyze train
    analyze_group_distribution(args.train, "TRAINING SET")
    
    # Analyze val
    analyze_group_distribution(args.val, "VALIDATION SET")
    
    # Check overlap
    check_train_val_overlap(args.train, args.val)
    
    # Check event patterns
    check_event_index_pattern(args.val, args.predictions)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
If you see:
- Val-only groups with many tail events → Model hasn't seen this camera/group
- High variance in tail ratio across groups → Class distribution differs by camera
- Clustered tail event indices → Tail events come from specific recordings

Solutions:
1. Re-split data to ensure all groups appear in train AND val
2. Use leave-one-group-out cross-validation
3. Apply heavy augmentation to break camera-specific patterns
4. Use the train_binary_v3_anticonfound.yaml config
""")


if __name__ == '__main__':
    main()