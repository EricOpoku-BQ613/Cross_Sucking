#!/usr/bin/env python3
"""
Verify manifest alignment AND write verification artifacts next to the manifest.

Outputs are saved under:
  <manifest_dir>/_verification/<manifest_stem>/

Also writes a duration-fixed manifest to support "road to verified data".
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

_TS_RE = re.compile(r"(\d{14})_(\d{14})")

def parse_video_timestamps(filename: str):
    m = _TS_RE.search(str(filename))
    if not m:
        return None, None, None
    try:
        s = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        e = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
        dur = (e - s).total_seconds()
        return s, e, dur
    except Exception:
        return None, None, None

def parse_hms(time_str):
    if time_str is None or (isinstance(time_str, float) and pd.isna(time_str)):
        return None
    s = str(time_str).strip()
    m = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", s)
    if not m:
        return None
    hh, mm, ss = map(int, m.groups())
    return hh, mm, ss

def event_datetime(video_start_dt: datetime, hms_str):
    hms = parse_hms(hms_str)
    if not (video_start_dt and hms):
        return None
    ev = datetime(video_start_dt.year, video_start_dt.month, video_start_dt.day, hms[0], hms[1], hms[2])
    # if event looks way "before" segment start, assume midnight boundary
    if ev < video_start_dt - timedelta(hours=6):
        ev += timedelta(days=1)
    return ev

def calc_offset_sec(video_start_dt: datetime, start_time_str):
    ev = event_datetime(video_start_dt, start_time_str)
    if ev is None:
        return None
    off = (ev - video_start_dt).total_seconds()
    if off < 0:
        off += 24 * 3600
    return off

def calc_duration_sec(video_start_dt: datetime, start_time_str, end_time_str):
    st = event_datetime(video_start_dt, start_time_str)
    en = event_datetime(video_start_dt, end_time_str)
    if st is None or en is None:
        return None
    if en < st:
        en += timedelta(days=1)
    return max(0.0, (en - st).total_seconds())

def detect_schema(df: pd.DataFrame):
    cols = set(df.columns)
    if "video_filename" in cols and "video_path" in cols and "start_time" in cols:
        return "cleaned"
    # You can expand mapping here for original sheet if needed
    return "unknown"

def verify_manifest(manifest_path: str, outdir: Path, max_offset_diff: float, min_dur: float):
    df = pd.read_csv(manifest_path)
    schema = detect_schema(df)

    outdir.mkdir(parents=True, exist_ok=True)

    issues = defaultdict(list)
    stats = defaultdict(int)
    stats["total"] = len(df)

    # Prepare new columns for “fixed” manifest
    df["calc_offset_sec"] = np.nan
    df["duration_sec_fixed"] = np.nan
    df["duration_was_fixed"] = False
    df["verify_ok"] = False
    df["verify_reason"] = ""

    for i, row in df.iterrows():
        evt = row.get("event_idx", i)
        vfn = row.get("video_filename", "")
        vpath = row.get("video_path", "")
        start_time = row.get("start_time", None)
        end_time = row.get("end_time", None)
        behavior = row.get("behavior", row.get("_primary_label", "unknown"))

        vstart, vend, vdur = parse_video_timestamps(vfn)
        if vstart is None or vdur is None or vdur <= 0:
            stats["cannot_parse_segment"] += 1
            df.at[i, "verify_reason"] = "cannot_parse_segment"
            issues["parse_errors"].append({"event_idx": evt, "behavior": behavior, "video_filename": vfn})
            continue

        # Check file existence (we don’t open video here; just path existence)
        if vpath and not Path(str(vpath)).exists():
            stats["missing_video"] += 1
            df.at[i, "verify_reason"] = "missing_video"
            issues["missing_video"].append({"event_idx": evt, "behavior": behavior, "video_path": vpath})
            continue

        off = calc_offset_sec(vstart, start_time)
        if off is None:
            stats["cannot_parse_event_time"] += 1
            df.at[i, "verify_reason"] = "cannot_parse_event_time"
            issues["parse_errors"].append({"event_idx": evt, "behavior": behavior, "start_time": start_time})
            continue

        df.at[i, "calc_offset_sec"] = float(off)

        # Offset vs segment duration
        if off > vdur:
            stats["offset_exceeds_segment"] += 1
            df.at[i, "verify_reason"] = "offset_exceeds_segment"
            issues["offset_exceeds_segment"].append({
                "event_idx": evt, "behavior": behavior, "start_time": start_time,
                "calc_offset_sec": off, "segment_duration_sec": vdur, "excess_sec": off - vdur,
                "video_filename": vfn
            })
            continue

        # Compare stored offset if exists
        stored_off = row.get("event_offset_sec", None)
        if stored_off is not None and str(stored_off).strip() != "":
            try:
                stored_off = float(stored_off)
                if abs(off - stored_off) > max_offset_diff:
                    stats["offset_mismatch"] += 1
                    issues["offset_mismatch"].append({
                        "event_idx": evt, "behavior": behavior,
                        "calc_offset_sec": off, "stored_offset_sec": stored_off,
                        "diff_sec": abs(off - stored_off),
                        "video_filename": vfn
                    })
            except Exception:
                pass

        # Duration handling
        dur = row.get("duration_sec", None)
        dur_val = pd.to_numeric(dur, errors="coerce")
        if not np.isfinite(dur_val) or dur_val <= 0:
            dur2 = calc_duration_sec(vstart, start_time, end_time)
            if dur2 is None or dur2 <= 0:
                dur2 = float(min_dur)
            df.at[i, "duration_sec_fixed"] = float(max(min_dur, dur2))
            df.at[i, "duration_was_fixed"] = True
            stats["bad_duration"] += 1
            issues["bad_duration"].append({
                "event_idx": evt, "behavior": behavior,
                "duration_sec_orig": dur, "duration_sec_fixed": df.at[i, "duration_sec_fixed"],
                "start_time": start_time, "end_time": end_time, "video_filename": vfn
            })
        else:
            df.at[i, "duration_sec_fixed"] = float(max(min_dur, dur_val))

        # If we got here, alignment is OK
        df.at[i, "verify_ok"] = True
        df.at[i, "verify_reason"] = "ok"
        stats["ok"] += 1

    # Write report files
    report_txt = outdir / "report.txt"
    summary_json = outdir / "summary.json"

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MANIFEST VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Manifest: {manifest_path}\n")
        f.write(f"Detected schema: {schema}\n")
        f.write(f"Rows: {len(df)}\n\n")
        f.write("-" * 70 + "\nSUMMARY\n" + "-" * 70 + "\n")
        for k in ["total","ok","missing_video","cannot_parse_segment","cannot_parse_event_time","offset_exceeds_segment","offset_mismatch","bad_duration"]:
            f.write(f"{k:>24}: {stats.get(k,0)}\n")
        ok_pct = (100.0 * stats.get("ok",0) / max(1, stats.get("total",1)))
        f.write(f"{'ok_%':>24}: {ok_pct:.1f}%\n")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({k:int(v) for k,v in stats.items()}, f, indent=2)

    # Save issue tables
    def dump_issue(name: str):
        rows = issues.get(name, [])
        if rows:
            pd.DataFrame(rows).to_csv(outdir / f"{name}.csv", index=False)

    for name in ["missing_video","parse_errors","offset_exceeds_segment","offset_mismatch","bad_duration"]:
        dump_issue(name)

    # Save fixed manifest (same columns + helper columns)
    fixed_path = outdir / "manifest_duration_fixed.csv"
    df.to_csv(fixed_path, index=False)

    # Save strict “verified only” subset (alignment ok)
    verified_only_path = outdir / "manifest_verified_only.csv"
    df[df["verify_ok"] == True].to_csv(verified_only_path, index=False)

    return outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default=None, help="Output folder. Default: data/manifests/_verification/<manifest_stem>/")
    ap.add_argument("--max_offset_diff", type=float, default=1.0)
    ap.add_argument("--min_dur", type=float, default=1.0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if args.outdir is None:
        outdir = manifest_path.parent / "_verification" / manifest_path.stem
    else:
        outdir = Path(args.outdir)

    outdir = verify_manifest(
        str(manifest_path),
        outdir=outdir,
        max_offset_diff=args.max_offset_diff,
        min_dur=args.min_dur
    )

    print(f"\nSaved verification outputs to:\n  {outdir.resolve()}\n")
    print(f"Key files:")
    print(f"  - { (outdir/'report.txt').resolve() }")
    print(f"  - { (outdir/'manifest_duration_fixed.csv').resolve() }")
    print(f"  - { (outdir/'manifest_verified_only.csv').resolve() }")

if __name__ == "__main__":
    main()
