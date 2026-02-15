#!/usr/bin/env python3
"""
Clean Full Manifest
===================

Reads final_manifest.xlsx (1,897 events, 6 groups) and produces a clean
master CSV ready for video linking.

Usage:
    python scripts/clean_full_manifest.py
    python scripts/clean_full_manifest.py --input data/manifests/final_manifest.xlsx
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.clean_annotations import (
    load_mapping_rules,
    normalize_behavior,
    normalize_ended_by,
    normalize_pen_location,
    check_duration,
)

# Column rename map: xlsx column → pipeline column
COLUMN_MAP = {
    "Cohort": "group",            # Cohort in xlsx = Group in folder names
    "Date": "date_raw",
    "Day": "day",
    "ID.initiator": "initiator_id",
    "ID.receiver": "receiver_id",
    "Start.time": "start_time",
    "End.time": "end_time",
    "Duration": "duration_sec",
    "Behavior": "behavior_raw",
    "Ended.by.initiator.or.receiver.": "ended_by_raw",
    "Pen.location": "pen_location_raw",
    "Notes_AT": "notes_at",
    "Notes_BD": "notes_bd",
    "Notes_checkedbyBD": "notes_checked_bd",
}

# The xlsx "Group" column is always 1 — we drop it
DROP_COLUMNS = ["Group"]


def parse_date(val) -> str:
    """Parse mixed date formats to YYYY-MM-DD."""
    if pd.isna(val):
        return ""

    # Already a datetime object (from Excel)
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")

    val_str = str(val).strip()

    # Try common formats
    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%y", "%m/%d/%Y",
                "%d/%m/%y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(val_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # If nothing worked, return as-is and flag
    return val_str


def parse_time(val) -> str:
    """Ensure time values are HH:MM:SS format."""
    if pd.isna(val):
        return ""

    if hasattr(val, "strftime"):
        return val.strftime("%H:%M:%S")

    val_str = str(val).strip()

    # Already in HH:MM:SS
    if len(val_str.split(":")) == 3:
        return val_str
    # HH:MM only
    if len(val_str.split(":")) == 2:
        return val_str + ":00"

    return val_str


def clean_manifest(input_path: str, rules_path: str, output_dir: str):
    """Main cleaning pipeline."""
    print(f"Loading manifest: {input_path}")
    df = pd.read_excel(input_path)
    print(f"  Loaded {len(df)} events")
    print(f"  Columns: {list(df.columns)}")

    # ── Drop unused columns ──
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ── Rename columns ──
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    print(f"  Renamed {len(rename_map)} columns")

    # ── Add event_idx ──
    df.insert(0, "event_idx", range(len(df)))

    # ── Load mapping rules ──
    print(f"Loading rules: {rules_path}")
    rules = load_mapping_rules(rules_path)

    # ── Normalize dates ──
    print("Normalizing dates...")
    df["date"] = df["date_raw"].apply(parse_date)
    date_formats_found = df["date_raw"].apply(
        lambda x: "datetime" if hasattr(x, "strftime") else type(x).__name__
    ).value_counts()
    print(f"  Date format types: {dict(date_formats_found)}")

    # ── Normalize times ──
    df["start_time"] = df["start_time"].apply(parse_time)
    df["end_time"] = df["end_time"].apply(parse_time)

    # ── Normalize behavior ──
    print("Normalizing behavior labels...")
    before_behavior = df["behavior_raw"].value_counts().to_dict()
    df["behavior"] = df["behavior_raw"].apply(lambda x: normalize_behavior(x, rules))
    after_behavior = df["behavior"].value_counts().to_dict()

    # ── Normalize ended_by ──
    print("Normalizing ended_by labels...")
    df["ended_by"] = df["ended_by_raw"].apply(lambda x: normalize_ended_by(x, rules))
    after_ended_by = df["ended_by"].value_counts().to_dict()

    # ── Normalize pen_location ──
    print("Normalizing pen_location labels...")
    df["pen_location"] = df["pen_location_raw"].apply(
        lambda x: normalize_pen_location(x, rules)
    )
    after_pen_location = df["pen_location"].value_counts().to_dict()

    # ── Duration validation ──
    print("Validating durations...")
    df["duration_flag"] = df["duration_sec"].apply(lambda x: check_duration(x, rules))
    flagged_count = df["duration_flag"].notna().sum()

    # ── Select and order output columns ──
    output_columns = [
        "event_idx",
        "group",
        "day",
        "date",
        "start_time",
        "end_time",
        "duration_sec",
        "behavior",
        "initiator_id",
        "receiver_id",
        "ended_by",
        "pen_location",
        # Raw columns kept for audit trail
        "behavior_raw",
        "ended_by_raw",
        "pen_location_raw",
        "date_raw",
        # Notes
        "notes_at",
        "notes_bd",
        "notes_checked_bd",
        # Flags
        "duration_flag",
    ]
    # Only include columns that exist
    output_columns = [c for c in output_columns if c in df.columns]
    df_out = df[output_columns]

    # ── Save outputs ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "MASTER_CLEAN_v3.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df_out)} rows)")

    # ── Save cleaning report ──
    reports_dir = Path(output_dir).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "total_events": len(df_out),
        "behavior": {
            "before": before_behavior,
            "after": after_behavior,
        },
        "ended_by": after_ended_by,
        "pen_location": after_pen_location,
        "duration_flagged": flagged_count,
        "nulls": df_out.isnull().sum().to_dict(),
    }

    report_path = reports_dir / "cleaning_report_v3.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {report_path}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"\nTotal events: {len(df_out)}")

    print(f"\nGroup distribution:")
    for g, count in sorted(df_out["group"].value_counts().items()):
        print(f"  Group {g}: {count}")

    print(f"\nBehavior (before -> after):")
    for b in sorted(set(list(before_behavior.keys()) + list(after_behavior.keys()))):
        bef = before_behavior.get(b, 0)
        aft = after_behavior.get(b, 0)
        if bef > 0 or aft > 0:
            print(f"  {b:12s}  {bef:4d} -> {aft:4d}")

    print(f"\nEnded_by distribution:")
    for role, count in sorted(after_ended_by.items(), key=lambda x: -x[1]):
        pct = count / len(df_out) * 100
        print(f"  {role:15s}  {count:4d}  ({pct:.1f}%)")

    print(f"\nPen location distribution:")
    for loc, count in sorted(after_pen_location.items(), key=lambda x: -x[1]):
        pct = count / len(df_out) * 100
        print(f"  {loc:10s}  {count:4d}  ({pct:.1f}%)")

    print(f"\nDuration stats:")
    print(f"  Mean:   {df_out['duration_sec'].mean():.1f}s")
    print(f"  Median: {df_out['duration_sec'].median():.1f}s")
    print(f"  Max:    {df_out['duration_sec'].max():.0f}s")
    print(f"  Flagged: {flagged_count} events")
    if flagged_count > 0:
        flags = df_out[df_out["duration_flag"].notna()]["duration_flag"].value_counts()
        for flag, count in flags.items():
            print(f"    {flag}: {count}")

    print(f"\nDate range: {df_out['date'].min()} to {df_out['date'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean full manifest")
    parser.add_argument(
        "--input",
        default="data/manifests/final_manifest.xlsx",
        help="Input xlsx path",
    )
    parser.add_argument(
        "--rules",
        default="data/annotations/mapping_rules.yaml",
        help="Mapping rules YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="data/manifests",
        help="Output directory for clean CSV",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    input_path = PROJECT_ROOT / args.input
    rules_path = PROJECT_ROOT / args.rules
    output_dir = PROJECT_ROOT / args.output_dir

    clean_manifest(str(input_path), str(rules_path), str(output_dir))
