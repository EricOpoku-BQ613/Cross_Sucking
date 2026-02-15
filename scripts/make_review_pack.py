#!/usr/bin/env python3
import argparse, random, subprocess, math
from pathlib import Path
import pandas as pd

def run(cmd):
    subprocess.run(cmd, check=True)

def ffmpeg_extract(src, out, start_sec, dur_sec=8, reencode=True):
    out.parent.mkdir(parents=True, exist_ok=True)
    start_sec = max(0, float(start_sec))
    dur_sec = float(dur_sec)

    if reencode:
        # robust: re-encode to avoid keyframe/seek surprises
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(src),
            "-t", f"{dur_sec:.3f}",
            "-an",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            str(out)
        ]
    else:
        # fast but less reliable depending on GOP/keyframes
        cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", str(src), "-t", f"{dur_sec:.3f}", "-c", "copy", str(out)]
    run(cmd)

def pick_no_event_times(video_events, video_len, dur=8, buffer=6, k=1):
    """
    video_events: list of (start,end) event windows in seconds
    buffer: extra seconds around each event to avoid sampling near it
    """
    forbidden = []
    for s,e in video_events:
        forbidden.append((max(0, s-buffer), min(video_len, e+buffer)))
    forbidden.sort()

    def ok(t0):
        t1 = t0 + dur
        if t1 > video_len:
            return False
        for a,b in forbidden:
            if not (t1 <= a or t0 >= b):
                return False
        return True

    times = []
    tries = 0
    while len(times) < k and tries < 5000:
        tries += 1
        t0 = random.uniform(0, max(0.01, video_len-dur))
        if ok(t0):
            times.append(t0)
    return times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per_class", type=int, default=12)
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--pre", type=float, default=2.0)   # seconds before event
    ap.add_argument("--buffer", type=float, default=6.0) # avoid +/- buffer around events for no-event
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--binary", action="store_true", help="only ear/tail")
    args = ap.parse_args()

    random.seed(args.seed)
    outdir = Path(args.outdir)
    events_dir = outdir/"events"
    neg_dir = outdir/"no_event"
    events_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    if args.binary:
        df = df[df["behavior"].isin(["ear","tail"])].copy()

    # sanity: require event_offset_sec
    assert "event_offset_sec" in df.columns, "manifest must contain event_offset_sec"
    assert "video_path" in df.columns, "manifest must contain video_path"

    # sample events per class
    review_rows = []
    for cls in sorted(df["behavior"].unique()):
        pool = df[df["behavior"] == cls]
        n = min(args.per_class, len(pool))
        sample = pool.sample(n=n, random_state=args.seed)
        for _, r in sample.iterrows():
            src = Path(r["video_path"])
            offset = float(r["event_offset_sec"])
            start = max(0.0, offset - args.pre)
            out = events_dir / f"event_{int(r['event_idx'])}_{cls}_t{offset:.2f}.mp4"
            ffmpeg_extract(src, out, start_sec=start, dur_sec=args.dur, reencode=True)
            review_rows.append({
                "kind":"event",
                "event_idx": r["event_idx"],
                "behavior": r["behavior"],
                "video_filename": r.get("video_filename", src.name),
                "video_path": str(src),
                "event_offset_sec": offset,
                "clip_path": str(out),
                "note":""
            })

    # build per-video event windows for no-event sampling
    # window = [offset-pre, offset-pre+dur]
    df2 = df.copy()
    df2["_evt_start"] = (df2["event_offset_sec"].astype(float) - args.pre).clip(lower=0)
    df2["_evt_end"] = df2["_evt_start"] + args.dur

    # estimate video length from segment duration if present; else assume 1800s (30 min)
    # (you can improve later using ffprobe if needed)
    default_len = 1800.0
    video_groups = df2.groupby("video_path")

    neg_count = 0
    for vp, g in video_groups:
        video_len = default_len
        if "segment_duration_sec" in g.columns and pd.notna(g["segment_duration_sec"]).any():
            video_len = float(g["segment_duration_sec"].dropna().iloc[0])

        windows = list(zip(g["_evt_start"].tolist(), g["_evt_end"].tolist()))
        times = pick_no_event_times(windows, video_len, dur=args.dur, buffer=args.buffer, k=1)
        if not times:
            continue

        src = Path(vp)
        t0 = times[0]
        out = neg_dir / f"noevent_{src.stem}_t{t0:.2f}.mp4"
        ffmpeg_extract(src, out, start_sec=t0, dur_sec=args.dur, reencode=True)
        review_rows.append({
            "kind":"no_event",
            "event_idx": "",
            "behavior": "none",
            "video_filename": src.name,
            "video_path": str(src),
            "event_offset_sec": "",
            "clip_path": str(out),
            "note":""
        })
        neg_count += 1

    review_csv = outdir/"review_index.csv"
    pd.DataFrame(review_rows).to_csv(review_csv, index=False)
    print(f"Wrote {len(review_rows)} clips + index: {review_csv}")
    print(f"No-event clips: {neg_count}")

if __name__ == "__main__":
    main()
