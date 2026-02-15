#!/usr/bin/env python3
"""
Bout-Grouped Intra-Video Split with OOD Test Set (v4)
=====================================================

Creates train/val/test splits that:
  1. Group temporally adjacent events into "bouts" (gap < 30s)
  2. Keep all events in a bout together (prevents temporal leakage)
  3. Split bouts within each video ~85/15 for train/val (intra-video)
  4. Hold out Groups 4+5 as OOD test set (unseen cameras)
  5. Filter to binary (ear/tail only)

Usage:
    python scripts/create_bout_splits_v4.py
    python scripts/create_bout_splits_v4.py --val-ratio 0.15 --bout-gap 30
    python scripts/create_bout_splits_v4.py --boost-ratio 0.3  # oversample tail
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Bout grouping ────────────────────────────────────────────────────────────


def assign_bouts(df: pd.DataFrame, gap_sec: float = 30.0) -> pd.DataFrame:
    """
    Group temporally adjacent events into bouts within each video.

    Events within `gap_sec` of each other in the same video are grouped
    into the same bout. Returns df with added 'bout_id' column.
    """
    df = df.copy()
    df["bout_id"] = ""

    for video, vdf in df.groupby("video_filename"):
        vdf_sorted = vdf.sort_values("event_offset_sec")
        bout_num = 0
        prev_end = -np.inf

        for idx in vdf_sorted.index:
            event_start = df.loc[idx, "event_offset_sec"]
            event_dur = df.loc[idx, "duration_sec"]

            if event_start - prev_end > gap_sec:
                bout_num += 1

            df.loc[idx, "bout_id"] = f"{video}__bout_{bout_num}"
            prev_end = event_start + event_dur

    return df


# ── Bout-level intra-video split ─────────────────────────────────────────────


def bout_intra_video_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split data at the bout level, keeping intra-video property.

    For each video:
      - List all bouts
      - Label each bout as "has_tail" if any event is tail
      - Videos with < 2 bouts -> all to train
      - Otherwise assign ~val_ratio bouts to val, rest to train
      - Stratify by has_tail when possible

    Returns (train_df, val_df)
    """
    np.random.seed(seed)

    # Build bout summary
    bout_info = {}
    for bout_id, bdf in df.groupby("bout_id"):
        bout_info[bout_id] = {
            "video": bdf["video_filename"].iloc[0],
            "has_tail": (bdf["behavior"] == "tail").any(),
            "n_events": len(bdf),
            "n_tail": (bdf["behavior"] == "tail").sum(),
        }

    # Group bouts by video
    video_bouts = defaultdict(list)
    for bout_id, info in bout_info.items():
        video_bouts[info["video"]].append(bout_id)

    train_bouts = set()
    val_bouts = set()

    for video, bouts in video_bouts.items():
        if len(bouts) < 2:
            # Too few bouts to split — all to train
            train_bouts.update(bouts)
            continue

        # Label bouts for stratification
        has_tail = [bout_info[b]["has_tail"] for b in bouts]

        # Try stratified split
        try:
            if sum(has_tail) >= 2 and sum(not h for h in has_tail) >= 2:
                # Enough of each class for stratification
                train_b, val_b = train_test_split(
                    bouts, test_size=val_ratio, stratify=has_tail, random_state=seed
                )
            else:
                raise ValueError("Not enough for stratification")
        except ValueError:
            # Fallback: random split
            n_val = max(1, int(len(bouts) * val_ratio))
            val_b = list(np.random.choice(bouts, size=n_val, replace=False))
            train_b = [b for b in bouts if b not in val_b]

        train_bouts.update(train_b)
        val_bouts.update(val_b)

    train_df = df[df["bout_id"].isin(train_bouts)].copy()
    val_df = df[df["bout_id"].isin(val_bouts)].copy()

    return train_df, val_df


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_split(train_df, val_df, test_df=None):
    """Print detailed split analysis."""
    print("\n" + "=" * 70)
    print("SPLIT ANALYSIS")
    print("=" * 70)

    splits = [("Train", train_df), ("Val", val_df)]
    if test_df is not None:
        splits.append(("Test OOD", test_df))

    # Basic counts
    print("\nDataset sizes:")
    for name, sdf in splits:
        n_videos = sdf["video_filename"].nunique()
        n_groups = sorted(sdf["group"].unique())
        cameras = sorted(sdf["camera_id"].unique())
        print(f"  {name:10s}: {len(sdf):5d} events, {n_videos:3d} videos, "
              f"groups={n_groups}, cameras={[int(c) for c in cameras]}")

    # Class distribution
    print("\nClass distribution:")
    for name, sdf in splits:
        dist = sdf["behavior"].value_counts().to_dict()
        tail_pct = (sdf["behavior"] == "tail").mean()
        print(f"  {name:10s}: {dist}  (tail: {tail_pct:.1%})")

    # Intra-video check (train vs val)
    train_videos = set(train_df["video_filename"])
    val_videos = set(val_df["video_filename"])
    overlap = train_videos & val_videos
    print(f"\nIntra-video property:")
    print(f"  Videos in train: {len(train_videos)}")
    print(f"  Videos in val:   {len(val_videos)}")
    print(f"  Videos in BOTH:  {len(overlap)} ({len(overlap)/max(len(train_videos|val_videos),1):.0%})")

    # Bout integrity
    train_bouts = set(train_df["bout_id"])
    val_bouts = set(val_df["bout_id"])
    bout_leak = train_bouts & val_bouts
    print(f"\nBout integrity:")
    print(f"  Train bouts: {len(train_bouts)}")
    print(f"  Val bouts:   {len(val_bouts)}")
    print(f"  Bout leakage: {len(bout_leak)} (should be 0)")

    # Temporal leakage check
    print("\nTemporal leakage check (same video, different splits):")
    leaks = 0
    checked = 0
    for video in overlap:
        train_offsets = sorted(train_df[train_df["video_filename"] == video]["event_offset_sec"])
        val_offsets = sorted(val_df[val_df["video_filename"] == video]["event_offset_sec"])
        for t_off in train_offsets:
            for v_off in val_offsets:
                gap = abs(t_off - v_off)
                checked += 1
                if gap < 30:
                    leaks += 1
    print(f"  Pairs checked: {checked}")
    print(f"  Pairs with gap < 30s: {leaks} (should be 0)")

    # Tail distribution by group
    print("\nTail events by group:")
    for name, sdf in splits:
        tail_by_group = sdf[sdf["behavior"] == "tail"].groupby("group").size()
        print(f"  {name:10s}: {tail_by_group.to_dict()}")

    # OOD isolation
    if test_df is not None:
        test_groups = set(test_df["group"])
        train_groups = set(train_df["group"])
        val_groups = set(val_df["group"])
        leak = test_groups & (train_groups | val_groups)
        print(f"\nOOD isolation:")
        print(f"  Test groups: {sorted(test_groups)}")
        print(f"  Train groups: {sorted(train_groups)}")
        print(f"  Val groups: {sorted(val_groups)}")
        print(f"  Group leakage: {sorted(leak)} (should be empty)")


def boost_minority_class(df: pd.DataFrame, target_ratio: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Boost tail class by oversampling to reach target ratio."""
    np.random.seed(seed)

    n_ear = (df["behavior"] == "ear").sum()
    n_tail = (df["behavior"] == "tail").sum()
    current_ratio = n_tail / len(df)

    if current_ratio >= target_ratio:
        return df

    n_extra = int((target_ratio * n_ear - n_tail * (1 - target_ratio)) / (1 - target_ratio))
    n_extra = max(0, n_extra)

    tail_df = df[df["behavior"] == "tail"]
    extra = tail_df.sample(n=n_extra, replace=True, random_state=seed)

    boosted = pd.concat([df, extra], ignore_index=True)
    print(f"  Boosted: {n_tail} -> {(boosted['behavior']=='tail').sum()} tail "
          f"({current_ratio:.1%} -> {(boosted['behavior']=='tail').mean():.1%})")

    return boosted


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Create bout-grouped splits v4")
    parser.add_argument("--input", default="data/manifests/MASTER_LINKED_v4.csv")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--bout-gap", type=float, default=30.0,
                        help="Max gap (seconds) between events in the same bout")
    parser.add_argument("--ood-groups", type=str, default="4,5",
                        help="Groups to hold out as OOD test set")
    parser.add_argument("--boost-ratio", type=float, default=0.0,
                        help="Target tail ratio after boosting (0=no boosting)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output_dir

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total events: {len(df)}")
    print(f"  Behaviors: {df['behavior'].value_counts().to_dict()}")

    # Step 1: Filter to binary (ear/tail only)
    print("\n--- Step 1: Filter to binary ---")
    binary_df = df[df["behavior"].isin(["ear", "tail"])].copy()
    dropped = len(df) - len(binary_df)
    print(f"  Kept: {len(binary_df)} (dropped {dropped} non-binary)")

    # Step 2: Separate OOD test set
    print("\n--- Step 2: Separate OOD test set ---")
    ood_groups = [int(g) for g in args.ood_groups.split(",")]
    test_df = binary_df[binary_df["group"].isin(ood_groups)].copy()
    dev_df = binary_df[~binary_df["group"].isin(ood_groups)].copy()
    print(f"  OOD test (groups {ood_groups}): {len(test_df)} events")
    print(f"  Dev set: {len(dev_df)} events")

    # Step 3: Bout grouping
    print(f"\n--- Step 3: Bout grouping (gap={args.bout_gap}s) ---")
    dev_df = assign_bouts(dev_df, gap_sec=args.bout_gap)
    test_df = assign_bouts(test_df, gap_sec=args.bout_gap)

    n_dev_bouts = dev_df["bout_id"].nunique()
    n_test_bouts = test_df["bout_id"].nunique()

    # Bout stats
    bout_sizes = dev_df.groupby("bout_id").size()
    print(f"  Dev bouts: {n_dev_bouts} (mean {bout_sizes.mean():.1f} events/bout, max {bout_sizes.max()})")
    print(f"  Test bouts: {n_test_bouts}")

    # Bouts with tail
    tail_bouts = dev_df[dev_df["behavior"] == "tail"]["bout_id"].nunique()
    print(f"  Dev bouts with tail: {tail_bouts}")

    # Step 4: Intra-video bout-level split
    print(f"\n--- Step 4: Intra-video bout-level split (val_ratio={args.val_ratio}) ---")
    train_df, val_df = bout_intra_video_split(dev_df, val_ratio=args.val_ratio, seed=args.seed)

    # Step 5: Optional boosting
    if args.boost_ratio > 0:
        print(f"\n--- Step 5: Boosting to {args.boost_ratio:.0%} tail ---")
        print("  Train:")
        train_df = boost_minority_class(train_df, target_ratio=args.boost_ratio, seed=args.seed)
        print("  Val:")
        val_df = boost_minority_class(val_df, target_ratio=args.boost_ratio, seed=args.seed)

    # Analysis
    analyze_split(train_df, val_df, test_df)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop bout_id from output (internal only)
    train_out = train_df.drop(columns=["bout_id"])
    val_out = val_df.drop(columns=["bout_id"])
    test_out = test_df.drop(columns=["bout_id"])

    train_path = output_dir / "train_v4.csv"
    val_path = output_dir / "val_v4.csv"
    test_path = output_dir / "test_ood_v4.csv"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    # Save protocol JSON
    protocol = {
        "created_at": datetime.now().isoformat(),
        "input": str(input_path),
        "seed": args.seed,
        "bout_gap_sec": args.bout_gap,
        "val_ratio": args.val_ratio,
        "ood_groups": ood_groups,
        "boost_ratio": args.boost_ratio,
        "outdoor_cameras_excluded": {"cam_8": [3, 4, 5, 6], "cam_10": [1, 2]},
        "camera_mapping": {
            "group_1": "cam_8", "group_2": "cam_16", "group_3": "cam_12",
            "group_4": "cam_3", "group_5": "cam_1", "group_6": "cam_12",
        },
        "splits": {
            "train": {
                "path": str(train_path),
                "events": len(train_out),
                "tail": int((train_out["behavior"] == "tail").sum()),
                "groups": sorted(train_out["group"].unique().tolist()),
            },
            "val": {
                "path": str(val_path),
                "events": len(val_out),
                "tail": int((val_out["behavior"] == "tail").sum()),
                "groups": sorted(val_out["group"].unique().tolist()),
            },
            "test_ood": {
                "path": str(test_path),
                "events": len(test_out),
                "tail": int((test_out["behavior"] == "tail").sum()),
                "groups": sorted(test_out["group"].unique().tolist()),
            },
        },
        "bout_stats": {
            "dev_bouts": n_dev_bouts,
            "test_bouts": n_test_bouts,
            "dev_bouts_with_tail": tail_bouts,
            "mean_events_per_bout": round(bout_sizes.mean(), 1),
            "max_events_per_bout": int(bout_sizes.max()),
        },
    }

    protocol_path = output_dir / "split_protocol_v4.json"
    with open(protocol_path, "w") as f:
        json.dump(protocol, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("SAVED")
    print(f"{'=' * 70}")
    print(f"  Train:     {train_path} ({len(train_out)} events)")
    print(f"  Val:       {val_path} ({len(val_out)} events)")
    print(f"  Test OOD:  {test_path} ({len(test_out)} events)")
    print(f"  Protocol:  {protocol_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
