#!/usr/bin/env python3
"""
Project Cleanup Script
======================
Removes unnecessary files before cluster deployment while keeping all essential files.

What gets REMOVED:
- Old training runs (runs/)
- Python cache files (__pycache__, *.pyc)
- Old reports and debug files
- Temporary manifests (old splits)
- Virtual environment (will be recreated on cluster)
- Git files (if any)
- Jupyter checkpoints
- Windows batch files (not needed on Linux)

What gets KEPT:
- All source code (src/, scripts/)
- New intra-video split manifests
- Test manifests
- Configs (all 3 experiments)
- Documentation (README, TRAINING_PLAN, etc.)
- Requirements files
- Raw data annotations
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def get_size_mb(path):
    """Get size of file or directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total / (1024 * 1024)


def cleanup_project(dry_run=True):
    """Clean up project directory."""

    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("="*80)
    print("PROJECT CLEANUP SCRIPT")
    print("="*80)
    print(f"Project root: {project_root}")
    print(f"Mode: {'DRY RUN (no files deleted)' if dry_run else 'LIVE (files will be deleted)'}")
    print("="*80)

    # Track what we're removing
    to_remove = []
    total_size = 0

    # 1. Old training runs
    print("\n[1] Checking old training runs...")
    runs_dir = Path('runs')
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                size = get_size_mb(run_dir)
                to_remove.append(('run', run_dir, size))
                total_size += size
                print(f"  - {run_dir.name}: {size:.1f} MB")

    # 2. Python cache files
    print("\n[2] Checking Python cache files...")
    for cache_dir in project_root.rglob('__pycache__'):
        size = get_size_mb(cache_dir)
        to_remove.append(('cache', cache_dir, size))
        total_size += size
        print(f"  - {cache_dir.relative_to(project_root)}: {size:.2f} MB")

    for pyc_file in project_root.rglob('*.pyc'):
        size = get_size_mb(pyc_file)
        to_remove.append(('pyc', pyc_file, size))
        total_size += size

    # 3. Old reports and debug files
    print("\n[3] Checking old reports and debug files...")
    reports_to_remove = [
        'data/reports',
        'analysis',
        'breakdown_splits.py',
        'check_corrupted.py',
        'check_label_distribution.py',
        'patch_labels.py',
    ]

    for item in reports_to_remove:
        path = Path(item)
        if path.exists():
            size = get_size_mb(path)
            to_remove.append(('report', path, size))
            total_size += size
            print(f"  - {item}: {size:.1f} MB")

    # 4. Old/temporary manifests (keep only new ones)
    print("\n[4] Checking old manifest files...")
    manifests_to_keep = {
        'linked_events.csv',
        'test.csv',
        'test_collapsed.csv',
        'video_hash_index.csv',
        # New intra-video splits
        'train_intravideo_20260105_162028.csv',
        'val_intravideo_20260105_162028.csv',
        'train_intravideo_20260105_162114.csv',
        'val_intravideo_20260105_162114.csv',
        # OLD splits (keep for baseline comparison)
        'train.csv',
        'val.csv',
    }

    manifests_dir = Path('data/manifests')
    if manifests_dir.exists():
        for manifest in manifests_dir.glob('*.csv'):
            if manifest.name not in manifests_to_keep:
                size = get_size_mb(manifest)
                to_remove.append(('manifest', manifest, size))
                total_size += size
                print(f"  - {manifest.name}: {size:.2f} MB")

    # 5. Virtual environment (will be recreated on cluster)
    print("\n[5] Checking virtual environment...")
    for venv_dir in ['venv', '.venv']:
        venv_path = Path(venv_dir)
        if venv_path.exists():
            size = get_size_mb(venv_path)
            to_remove.append(('venv', venv_path, size))
            total_size += size
            print(f"  - {venv_dir}/: {size:.1f} MB")

    # 6. Git files (if any)
    print("\n[6] Checking Git files...")
    git_items = ['.git', '.gitignore', '.gitattributes']
    for item in git_items:
        path = Path(item)
        if path.exists():
            size = get_size_mb(path)
            to_remove.append(('git', path, size))
            total_size += size
            print(f"  - {item}: {size:.2f} MB")

    # 7. Jupyter checkpoints
    print("\n[7] Checking Jupyter checkpoints...")
    for checkpoint_dir in project_root.rglob('.ipynb_checkpoints'):
        size = get_size_mb(checkpoint_dir)
        to_remove.append(('jupyter', checkpoint_dir, size))
        total_size += size
        print(f"  - {checkpoint_dir.relative_to(project_root)}: {size:.2f} MB")

    # 8. Windows-specific files (not needed on Linux cluster)
    print("\n[8] Checking Windows-specific files...")
    windows_files = ['run_training.bat']
    for item in windows_files:
        path = Path(item)
        if path.exists():
            size = get_size_mb(path)
            to_remove.append(('windows', path, size))
            total_size += size
            print(f"  - {item}: {size:.2f} MB")

    # 9. Notebooks (optional - keep for now)
    # print("\n[9] Checking notebooks...")
    # notebooks_dir = Path('notebooks')
    # if notebooks_dir.exists():
    #     size = get_size_mb(notebooks_dir)
    #     print(f"  - notebooks/: {size:.1f} MB (KEEPING - may be useful)")

    # Summary
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)

    print(f"\nTotal items to remove: {len(to_remove)}")
    print(f"Total space to free: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    # Group by type
    by_type = {}
    for item_type, path, size in to_remove:
        if item_type not in by_type:
            by_type[item_type] = {'count': 0, 'size': 0}
        by_type[item_type]['count'] += 1
        by_type[item_type]['size'] += size

    print("\nBreakdown by type:")
    for item_type, stats in sorted(by_type.items()):
        print(f"  {item_type:10s}: {stats['count']:3d} items, {stats['size']:8.1f} MB")

    # Files to KEEP
    print("\n" + "="*80)
    print("ESSENTIAL FILES KEPT")
    print("="*80)

    essential_dirs = [
        'src/',
        'scripts/',
        'configs/',
        'data/manifests/ (cleaned)',
        'data/annotations/',
    ]

    essential_files = [
        'TRAINING_PLAN.md',
        'READY_FOR_CLUSTER.md',
        'FIXED_AND_READY.md',
        'run_training.sh',
        'pyproject.toml',
    ]

    print("\nEssential directories:")
    for d in essential_dirs:
        print(f"  ✓ {d}")

    print("\nEssential files:")
    for f in essential_files:
        if Path(f).exists():
            print(f"  ✓ {f}")

    print("\nEssential manifests kept:")
    for m in sorted(manifests_to_keep):
        manifest_path = Path('data/manifests') / m
        if manifest_path.exists():
            size = get_size_mb(manifest_path)
            print(f"  ✓ {m:50s} ({size:.2f} MB)")

    # Execute cleanup
    if not dry_run:
        print("\n" + "="*80)
        print("EXECUTING CLEANUP")
        print("="*80)

        for item_type, path, size in to_remove:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"  Removed file: {path}")
            except Exception as e:
                print(f"  ERROR removing {path}: {e}")

        print("\n✓ Cleanup complete!")
        print(f"✓ Freed {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    else:
        print("\n" + "="*80)
        print("DRY RUN COMPLETE - No files were deleted")
        print("="*80)
        print("\nTo actually delete files, run:")
        print("  python cleanup_project.py --execute")

    return len(to_remove), total_size


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean up project before cluster deployment')
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete files (default is dry run)')
    args = parser.parse_args()

    # Confirm if executing
    if args.execute:
        print("\n⚠️  WARNING: This will permanently delete files!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cleanup cancelled.")
            return

    cleanup_project(dry_run=not args.execute)

    if not args.execute:
        print("\n💡 Run with --execute to actually delete files")


if __name__ == '__main__':
    main()
