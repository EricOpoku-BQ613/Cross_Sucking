#!/usr/bin/env python3
"""
Convert Windows paths in manifests to ISAAC Linux paths.

Usage:
    python scripts/fix_manifest_paths.py                    # Fix all manifests
    python scripts/fix_manifest_paths.py input.csv          # Fix single file (overwrite)
    python scripts/fix_manifest_paths.py input.csv out.csv  # Fix to new file
"""

import pandas as pd
import sys
from pathlib import Path

# =============================================================================
# EDIT THIS: Map your Windows paths to ISAAC paths
# =============================================================================
PATH_MAPPING = {
    # Windows path -> ISAAC path
    "E:\\Downey cross-sucking videos 2024": "/lustre/isaac24/scratch/eopoku2/downey_cross_sucking_videos_2024",
    "E:/Downey cross-sucking videos 2024": "/lustre/isaac24/scratch/eopoku2/downey_cross_sucking_videos_2024",
    "D:\\cross_sucking": "/home/eopoku2/projects/cross_sucking",
    "D:/cross_sucking": "/home/eopoku2/projects/cross_sucking",
}


def fix_paths(input_csv: str, output_csv: str = None):
    """Fix Windows paths to ISAAC paths in a manifest CSV."""
    df = pd.read_csv(input_csv)
    
    if output_csv is None:
        output_csv = input_csv  # Overwrite in place
    
    # Find the column with video paths
    path_col = None
    for col in ['video_path', 'path', 'file_path', 'clip_path']:
        if col in df.columns:
            path_col = col
            break
    
    if path_col is None:
        print(f"[WARN] No path column found in {input_csv}")
        print(f"       Columns: {df.columns.tolist()}")
        return False
    
    print(f"Processing: {input_csv}")
    print(f"  Path column: {path_col}")
    print(f"  Sample before: {df[path_col].iloc[0][:80]}...")
    
    # Apply path replacements
    for win_path, isaac_path in PATH_MAPPING.items():
        df[path_col] = df[path_col].str.replace(win_path, isaac_path, regex=False)
    
    # Convert any remaining backslashes to forward slashes
    df[path_col] = df[path_col].str.replace("\\", "/", regex=False)
    
    print(f"  Sample after:  {df[path_col].iloc[0][:80]}...")
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"  Saved to: {output_csv}")
    return True


def fix_all_manifests(manifest_dir: str = "data/manifests"):
    """Fix all CSV files in the manifest directory."""
    manifest_path = Path(manifest_dir)
    
    if not manifest_path.exists():
        print(f"[ERROR] Manifest directory not found: {manifest_dir}")
        return
    
    csv_files = list(manifest_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {manifest_dir}\n")
    
    success = 0
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        if fix_paths(str(csv_file)):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Fixed {success}/{len(csv_files)} files")


def verify_paths(manifest_csv: str, num_samples: int = 5):
    """Check if paths in manifest actually exist."""
    df = pd.read_csv(manifest_csv)
    
    path_col = None
    for col in ['video_path', 'path', 'file_path']:
        if col in df.columns:
            path_col = col
            break
    
    if path_col is None:
        print(f"No path column found")
        return
    
    print(f"\nVerifying {num_samples} random paths from {manifest_csv}:")
    sample = df.sample(min(num_samples, len(df)))
    
    found = 0
    for idx, row in sample.iterrows():
        path = row[path_col]
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {path[:70]}...")
        if exists:
            found += 1
    
    print(f"\nResult: {found}/{len(sample)} paths exist")
    return found == len(sample)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No args: fix all manifests
        fix_all_manifests()
        
        # Then verify
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        verify_paths("data/manifests/train_intravideo_20260105_162028.csv")
        
    elif sys.argv[1] == "--verify":
        # Verify mode
        csv_file = sys.argv[2] if len(sys.argv) > 2 else "data/manifests/train_intravideo_20260105_162028.csv"
        verify_paths(csv_file)
        
    else:
        # Fix specific file
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else None
        fix_paths(input_csv, output_csv)