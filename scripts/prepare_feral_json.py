#!/usr/bin/env python3
"""
prepare_feral_json.py
=====================
Convert our CSV split manifests to FERAL's JSON annotation format.

FERAL expects:
  {
    "splits": {
      "train": ["evt_0001_ear.mp4", ...],
      "val":   ["evt_0003_tail.mp4", ...],
      "test":  ["evt_0005_ear.mp4", ...]
    },
    "labels": {
      "evt_0001_ear.mp4": [0, 0, 0, ..., 0],   # frame-level: ear=0
      "evt_0003_tail.mp4": [1, 1, 1, ..., 1],  # frame-level: tail=1
    }
  }

Usage (run from repo root):
    python scripts/prepare_feral_json.py
    python scripts/prepare_feral_json.py --clip-dir /lustre/isaac24/scratch/eopoku2/clips_feral
    python scripts/prepare_feral_json.py --dry-run
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_MAP = {"ear": 0, "tail": 1}
CLASS_NAMES = ["ear", "tail"]


def get_frame_count(clip_path: Path) -> int:
    """Get frame count via ffprobe. Falls back to duration * fps estimate."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(clip_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    except Exception:
        pass

    # Fallback: estimate from duration and fps
    cmd2 = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(clip_path),
    ]
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    fps_str = s.get("r_frame_rate", "15/1")
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                    dur = float(s.get("duration", 0))
                    if fps > 0 and dur > 0:
                        return max(1, int(dur * fps))
    except Exception:
        pass

    return 90  # fallback: assume ~6s @ 15fps


def load_split(csv_path: Path, clip_dir: Path, label_map: dict) -> tuple[list, dict, list]:
    """Returns (filenames, labels_dict, missing_list)."""
    df = pd.read_csv(csv_path)
    filenames = []
    labels = {}
    missing = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_path.stem):
        event_idx = int(row["event_idx"])
        behavior = str(row["behavior"]).strip().lower()
        fname = f"evt_{event_idx:04d}_{behavior}.mp4"
        clip_path = clip_dir / fname

        if not clip_path.exists():
            missing.append(fname)
            continue

        label_int = label_map.get(behavior)
        if label_int is None:
            print(f"  WARNING: unknown behavior '{behavior}' in {fname}, skipping")
            continue

        n_frames = get_frame_count(clip_path)
        filenames.append(fname)
        labels[fname] = [label_int] * n_frames

    return filenames, labels, missing


def main():
    ap = argparse.ArgumentParser(description="Prepare FERAL annotation JSON from CSV splits")
    ap.add_argument("--train-csv",   default="data/manifests/train_v4.csv")
    ap.add_argument("--val-csv",     default="data/manifests/val_v4.csv")
    ap.add_argument("--test-csv",    default="data/manifests/test_ood_v4.csv")
    ap.add_argument("--clip-dir",    default="data/processed/clips_feral",
                    help="Directory with 256x256 resized clips (output of resize_clips_feral.sh)")
    ap.add_argument("--output",      default="data/manifests/feral_annotations.json")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    clip_dir    = Path(args.clip_dir) if Path(args.clip_dir).is_absolute() else PROJECT_ROOT / args.clip_dir
    output_path = Path(args.output)   if Path(args.output).is_absolute()   else PROJECT_ROOT / args.output

    print(f"Clip dir:  {clip_dir}")
    print(f"Output:    {output_path}")
    print(f"Labels:    {LABEL_MAP}")
    print()

    if not clip_dir.exists():
        print(f"ERROR: clip_dir does not exist: {clip_dir}")
        print("Run scripts/resize_clips_feral.sh first.")
        raise SystemExit(1)

    train_csv = PROJECT_ROOT / args.train_csv
    val_csv   = PROJECT_ROOT / args.val_csv
    test_csv  = PROJECT_ROOT / args.test_csv

    print("=== Processing splits ===")
    train_files, train_labels, train_miss = load_split(train_csv, clip_dir, LABEL_MAP)
    val_files,   val_labels,   val_miss   = load_split(val_csv,   clip_dir, LABEL_MAP)
    test_files,  test_labels,  test_miss  = load_split(test_csv,  clip_dir, LABEL_MAP)

    print()
    print("=== Summary ===")
    print(f"  Train: {len(train_files)} clips ({len(train_miss)} missing)")
    print(f"  Val:   {len(val_files)} clips ({len(val_miss)} missing)")
    print(f"  Test:  {len(test_files)} clips ({len(test_miss)} missing)")

    # Count label distribution
    for split_name, files, labels in [("Train", train_files, train_labels),
                                       ("Val",   val_files,   val_labels),
                                       ("Test",  test_files,  test_labels)]:
        n_ear  = sum(1 for f in files if labels[f][0] == 0)
        n_tail = sum(1 for f in files if labels[f][0] == 1)
        print(f"  {split_name}: ear={n_ear}, tail={n_tail}")

    if train_miss or val_miss or test_miss:
        print()
        print("Missing clips (first 5 per split):")
        for name, miss in [("train", train_miss), ("val", val_miss), ("test", test_miss)]:
            for m in miss[:5]:
                print(f"  [{name}] {m}")

    # Merge all labels (clips appear in exactly one split, but labels dict covers all)
    all_labels = {**train_labels, **val_labels, **test_labels}

    annotation = {
        "splits": {
            "train": train_files,
            "val":   val_files,
            "test":  test_files,
        },
        "labels": all_labels,
        "meta": {
            "class_names": CLASS_NAMES,
            "label_map":   {v: k for k, v in LABEL_MAP.items()},
            "n_classes":   len(CLASS_NAMES),
        }
    }

    if args.dry_run:
        print("\nDRY RUN — no file written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotation, f)

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nWritten: {output_path} ({size_mb:.1f} MB)")
    print(f"\nNext step:")
    print(f"  python run.py {clip_dir} {output_path}")


if __name__ == "__main__":
    main()
