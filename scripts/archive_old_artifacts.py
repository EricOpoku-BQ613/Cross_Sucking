#!/usr/bin/env python3
"""
Archive Old Artifacts
=====================

Moves obsolete files from the old 968-event pipeline to _archive_v1/.
Uses --dry-run by default; pass --execute to actually move files.

Usage:
    python scripts/archive_old_artifacts.py              # dry-run
    python scripts/archive_old_artifacts.py --execute    # actually move
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


# Project root (where this script lives is scripts/, go up one level)
ROOT = Path(__file__).resolve().parent.parent


# ── Files/dirs to archive, keyed by destination subfolder ──────────────

ARCHIVE_MAP = {
    # Root-level files → _archive_v1/root/
    "root": [
        "breakdown_splits.py",
        "check_corrupted.py",
        "check_label_distribution.py",
        "patch_labels.py",
        "cleanup_project.py",
        "activate.bat",
        "setup_venv.bat",
        "_tmp_read.py",
        "run_training.bat",
        "run_training.sh",
        "audit_results.csv",
        "audit_results_FAILURES.csv",
        "CLUSTER_DEPLOYMENT_CHECKLIST.md",
        "FIXED_AND_READY.md",
        "READY_FOR_CLUSTER.md",
        "TRAINING_PLAN.md",
    ],

    # data/manifests/ → _archive_v1/manifests/
    "manifests": [
        "data/manifests/MASTER_FINAL_CLEAN.csv",
        "data/manifests/MASTER_FINAL_CLEAN_BOUNDARYFIX.csv",
        "data/manifests/MASTER_FINAL_CLEAN_v2.csv",
        "data/manifests/linked_events.csv",
        "data/manifests/train.csv",
        "data/manifests/val.csv",
        "data/manifests/test.csv",
        "data/manifests/train_CLEAN.csv",
        "data/manifests/val_CLEAN.csv",
        "data/manifests/test_CLEAN.csv",
        "data/manifests/train_CLEAN_SEGMENTFIX.csv",
        "data/manifests/train_CLEAN_SEGMENTFIX_NO_DOTUNDERSCORE.csv",
        "data/manifests/train_CLEAN_SEGMENTFIX_report.csv",
        "data/manifests/train_FINAL_CLEAN.csv",
        "data/manifests/train_intravideo_20260105_162028.csv",
        "data/manifests/train_intravideo_20260105_162114.csv",
        "data/manifests/val_intravideo_20260105_162028.csv",
        "data/manifests/val_intravideo_20260105_162114.csv",
        # Directories
        "data/manifests/_archive",
        "data/manifests/_verification",
        "data/manifests/splits",
        "data/manifests/intravideo",
        "data/manifests/analysis",
    ],

    # Other data directories → _archive_v1/<subdir>/
    "raw_exports": [
        "data/raw_exports",
    ],
    "reports": [
        "data/reports",
    ],
    "analysis": [
        "analysis",
    ],
    "test_clips": [
        "test_clips_letterbox",
    ],
    "annotations": [
        "data/annotations/combined_group1_days1_4_clean_with_flags.csv",
        "data/annotations/qa_flags.csv",
    ],

    # Configs → _archive_v1/configs/
    "configs": [
        "configs/train_binary_baseline_old_split.yaml",
        "configs/train_binary_v3_intravideo.yaml",
        "configs/train_binary_v3_intravideo_boosted.yaml",
        "configs/E02_simple_augmentation_final.yaml",
        "configs/split_protocol.yaml",
    ],

    # Scripts → _archive_v1/scripts/
    "scripts": [
        "scripts/confirm_smoke_overlap.py",
        "scripts/verify_smoke_final.py",
        "scripts/create_test_events.py",
        "scripts/collapse_labels.py",
        "scripts/boost_val_tail.py",
        "scripts/check_confound.py",
        "scripts/check_video_confound.py",
        "scripts/fix_manifest_paths.py",
        "scripts/fix_manifest_segment_mapping.py",
        "scripts/fix_offset_exceeds_segment.py",
        "scripts/extract_feral_segmentfix_opencv.py",
        "scripts/diagnose_results",
        "scripts/aggregate_ablation_results.py",
        "scripts/test_training_pipeline.py",
        "scripts/__pycache__",
    ],
}

# Files to delete outright (junk)
DELETE_LIST = [
    "nul",
]


def archive(execute: bool = False):
    archive_root = ROOT / "_archive_v1"
    log_entries = []
    stats = {"moved": 0, "deleted": 0, "missing": 0}

    print(f"{'EXECUTING' if execute else 'DRY RUN'}: archiving old artifacts")
    print(f"  Project root: {ROOT}")
    print(f"  Archive dir:  {archive_root}")
    print()

    # ── Delete junk files ──
    for rel in DELETE_LIST:
        src = ROOT / rel
        if src.exists():
            print(f"  DELETE  {rel}")
            if execute:
                try:
                    src.unlink()
                except PermissionError:
                    # 'nul' is a reserved device name on Windows
                    print(f"    SKIP  (reserved name on Windows, cannot delete)")
                    stats["missing"] += 1
                    continue
            log_entries.append({"action": "delete", "path": rel})
            stats["deleted"] += 1
        else:
            stats["missing"] += 1

    # ── Move files to archive ──
    for dest_subdir, paths in ARCHIVE_MAP.items():
        dest_dir = archive_root / dest_subdir

        for rel in paths:
            src = ROOT / rel
            if not src.exists():
                print(f"  SKIP    {rel}  (not found)")
                stats["missing"] += 1
                continue

            dst = dest_dir / src.name
            print(f"  MOVE    {rel}  ->  _archive_v1/{dest_subdir}/{src.name}")

            if execute:
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Handle case where destination already exists
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(src), str(dst))

            log_entries.append({
                "action": "move",
                "from": rel,
                "to": f"_archive_v1/{dest_subdir}/{src.name}",
            })
            stats["moved"] += 1

    # ── Save log ──
    if execute:
        archive_root.mkdir(parents=True, exist_ok=True)
        log_path = archive_root / "archive_log.json"
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "entries": log_entries,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"\n  Log saved to {log_path}")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"  Moved:   {stats['moved']}")
    print(f"  Deleted: {stats['deleted']}")
    print(f"  Missing: {stats['missing']} (already gone or never existed)")
    print(f"{'=' * 50}")

    if not execute:
        print("\n  This was a DRY RUN. Use --execute to actually move files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive old project artifacts")
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (default is dry-run)")
    args = parser.parse_args()
    archive(execute=args.execute)
