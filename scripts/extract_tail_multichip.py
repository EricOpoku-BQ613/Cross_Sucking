"""
scripts/extract_tail_multichip.py
----------------------------------
Re-extract TAIL events using sliding-window temporal sampling + optional horizontal flip
to multiply effective tail training data without any new annotations.

Strategy:
  - Slide a 1.33s clip window across each tail event with configurable overlap
  - Optionally save a horizontally-flipped version of each window
  - Ear events are untouched (1 clip each, pointing to existing clip_dir)

Scale estimates (default: overlap=0.5, flip=True):
  - 1.33s event  -> 1 window  x2 flip = 2  clips
  - 4.0s event   -> 5 windows x2 flip = 10 clips
  - 6.9s event   -> 9 windows x2 flip = 18 clips  (average)
  - 30s  event   -> 43 windows x2 flip = 86 clips
  Expected total: ~167 events x ~10 avg = ~1,700 tail training clips (was 167)

Usage:
    python scripts/extract_tail_multichip.py \
        --manifest data/manifests/MASTER_LINKED_v4.csv \
        --clip-dir  /lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4 \
        --output-dir /lustre/isaac24/scratch/eopoku2/clips_v5/clips_v5 \
        --output-manifest data/manifests/MASTER_LINKED_v5.csv \
        --overlap 0.5 \
        --flip \
        --max-clips-per-event 20

    # Dry run (count clips without extracting):
    python scripts/extract_tail_multichip.py ... --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


CLIP_LEN_SEC = 16 / 12.0   # 1.333 s at 12 fps — must match model config


def compute_windows(event_start: float, event_end: float,
                    clip_len: float, overlap: float,
                    max_clips: int) -> list[tuple[float, float]]:
    """Slide a clip_len window across [event_start, event_end] with given overlap fraction."""
    stride = clip_len * (1.0 - overlap)
    stride = max(stride, 0.1)  # never shorter than 0.1s

    event_dur = event_end - event_start

    # For very short events, just center one clip
    if event_dur <= clip_len:
        center = (event_start + event_end) / 2
        s = max(0.0, center - clip_len / 2)
        return [(round(s, 3), round(s + clip_len, 3))]

    windows = []
    t = event_start
    while t + clip_len <= event_end + stride * 0.5:  # allow slight overshoot
        clip_s = t
        clip_e = t + clip_len
        # Clamp end to event_end + small buffer
        if clip_e > event_end:
            clip_s = event_end - clip_len
            clip_e = event_end
        windows.append((round(max(0.0, clip_s), 3), round(clip_e, 3)))
        t += stride
        if len(windows) >= max_clips:
            break

    return windows


def extract_ffmpeg(video_path: str, clip_start: float, clip_end: float,
                   output_path: Path, flip: bool = False, crf: int = 20) -> bool:
    """Extract (and optionally horizontally flip) a clip with ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return True  # skip existing

    duration = round(clip_end - clip_start, 3)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(clip_start),
        "-i", video_path,
        "-t", str(duration),
    ]
    if flip:
        cmd += ["-vf", "hflip"]
    cmd += [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  default="data/manifests/MASTER_LINKED_v4.csv")
    ap.add_argument("--clip-dir",  required=True,
                    help="Existing clips_v4 directory (ear clips already here)")
    ap.add_argument("--output-dir", required=True,
                    help="Where to save new tail clips")
    ap.add_argument("--output-manifest",
                    default="data/manifests/MASTER_LINKED_v5.csv")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--splits", nargs="+",
                    default=["train_v4.csv", "val_v4.csv", "test_ood_v4.csv"])
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Fraction of clip_len to overlap between windows (0=no overlap, 0.75=dense)")
    ap.add_argument("--flip", action="store_true", default=True,
                    help="Also save a horizontally-flipped version of each window")
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.add_argument("--max-clips-per-event", type=int, default=20,
                    help="Cap clips per tail event to avoid very long events dominating")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count clips without extracting anything")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    out_dir = Path(args.output_dir)

    ear_rows  = df[df["behavior"].str.lower() == "ear"]
    tail_rows = df[df["behavior"].str.lower() == "tail"]

    # --- Count expected clips ---
    total_tail_clips = 0
    for _, row in tail_rows.iterrows():
        evt_start = float(row["event_offset_sec"])
        evt_end = evt_start + float(row["duration_sec"])
        wins = compute_windows(
            evt_start, evt_end,
            CLIP_LEN_SEC, args.overlap, args.max_clips_per_event,
        )
        total_tail_clips += len(wins) * (2 if args.flip else 1)

    print(f"Ear events      : {len(ear_rows):,} (unchanged, point to existing clip_dir)")
    print(f"Tail events     : {len(tail_rows):,}")
    print(f"Overlap         : {args.overlap} -> stride={CLIP_LEN_SEC*(1-args.overlap):.2f}s")
    print(f"Flip augment    : {args.flip}")
    print(f"Max clips/event : {args.max_clips_per_event}")
    print(f"Expected tail clips: ~{total_tail_clips:,}")
    print(f"Total manifest rows: ~{len(ear_rows) + total_tail_clips:,}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    new_rows = []

    # --- EAR: keep as-is ---
    for _, row in ear_rows.iterrows():
        r = row.to_dict()
        r["clip_id"] = f"evt_{int(row['event_idx'])}_{row['behavior'].lower()}"
        r["clip_dir"] = args.clip_dir
        r["_base_event_idx"] = int(row["event_idx"])
        new_rows.append(r)

    # --- TAIL: sliding window + flip ---
    tail_list = list(tail_rows.iterrows())
    extracted_ok = 0
    extracted_fail = 0

    for i, (_, row) in enumerate(tail_list):
        video_path  = str(row["video_path"])
        base_id     = f"evt_{int(row['event_idx'])}_{row['behavior'].lower()}"
        event_start = float(row["event_offset_sec"])
        event_end   = event_start + float(row["duration_sec"])

        windows = compute_windows(event_start, event_end,
                                  CLIP_LEN_SEC, args.overlap,
                                  args.max_clips_per_event)

        for w_idx, (clip_s, clip_e) in enumerate(windows):
            for flip in ([False, True] if args.flip else [False]):
                suffix = f"_w{w_idx}" + ("_f" if flip else "")
                clip_id  = f"{base_id}{suffix}"
                out_path = out_dir / f"{clip_id}.mp4"

                ok = extract_ffmpeg(video_path, clip_s, clip_e, out_path, flip=flip)
                if ok:
                    extracted_ok += 1
                else:
                    extracted_fail += 1
                    print(f"  [FAIL] {out_path.name}")

                r = row.to_dict()
                r["clip_id"]  = clip_id
                r["clip_dir"] = str(out_dir)
                r["_base_event_idx"] = int(row["event_idx"])
                new_rows.append(r)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tail_list)}] tail events done | "
                  f"clips extracted: {extracted_ok} ok, {extracted_fail} fail")

    new_df = pd.DataFrame(new_rows)
    # Save master manifest without internal tracking column
    new_df.drop(columns=["_base_event_idx"]).to_csv(args.output_manifest, index=False)

    tail_total = (new_df["behavior"].str.lower() == "tail").sum()
    ear_total  = (new_df["behavior"].str.lower() == "ear").sum()
    print(f"\nMaster manifest saved -> {args.output_manifest}")
    print(f"  Ear  : {ear_total:,}")
    print(f"  Tail : {tail_total:,}  (was {len(tail_rows):,}  x{tail_total/len(tail_rows):.1f})")

    # --- Rebuild split CSVs ---
    manifest_dir = Path(args.manifest_dir)
    for split_file in args.splits:
        split_path = manifest_dir / split_file
        if not split_path.exists():
            continue
        split_df  = pd.read_csv(split_path)
        split_event_ids = set(split_df["event_idx"])

        # A new row belongs to a split if its base event_idx is in the original split
        split_new = new_df[new_df["_base_event_idx"].isin(split_event_ids)].copy()
        split_new = split_new.drop(columns=["_base_event_idx"])

        out_split = manifest_dir / split_file.replace("_v4", "_v5")
        split_new.to_csv(out_split, index=False)
        t = (split_new["behavior"].str.lower() == "tail").sum()
        e = (split_new["behavior"].str.lower() == "ear").sum()
        print(f"  {out_split.name}: {len(split_new):,} rows  (ear={e}, tail={t})")


if __name__ == "__main__":
    main()
