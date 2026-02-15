import re
import argparse
from pathlib import Path
from datetime import datetime, date, time
import pandas as pd

TS_RE = re.compile(r"_(\d{14})_(\d{14})\.mp4$", re.IGNORECASE)

def parse_seg_times(filename: str):
    m = TS_RE.search(filename.replace("\\", "/"))
    if not m:
        return None, None
    s = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    e = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    return s, e

def parse_hms(hms: str):
    # accepts "HH:MM:SS"
    hh, mm, ss = hms.strip().split(":")
    return time(int(hh), int(mm), int(ss))

def build_index(camera_dir: Path):
    vids = []
    for p in sorted(camera_dir.glob("*.mp4")):
        s, e = parse_seg_times(p.name)
        if s and e:
            vids.append((s, e, p))
    return vids

def find_segment(vids, event_dt: datetime):
    # linear scan is fine for small folders; can be optimized if needed
    for s, e, p in vids:
        if s <= event_dt < e:
            return s, e, p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--report_csv", default=None)
    ap.add_argument("--max_print", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    needed = ["video_path", "video_filename", "start_time"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing required columns: {miss}")

    # Build per-camera-folder indexes (fast)
    cache = {}
    moved = 0
    unmatched = 0
    bad_ts = 0
    rows_report = []

    for i, r in df.iterrows():
        vp = Path(str(r["video_path"]))
        cam_dir = vp.parent
        fn = str(r["video_filename"])

        seg_s, seg_e = parse_seg_times(fn)
        if seg_s is None:
            bad_ts += 1
            rows_report.append({"event_idx": r.get("event_idx", i), "reason": "cannot_parse_filename_timestamps", "video_filename": fn})
            continue

        # event_dt = (same date as seg_s) + start_time
        try:
            st = parse_hms(str(r["start_time"]))
            event_dt = datetime.combine(seg_s.date(), st)
        except Exception:
            rows_report.append({"event_idx": r.get("event_idx", i), "reason": "bad_start_time", "start_time": r.get("start_time")})
            continue

        # load index for this camera folder
        if cam_dir not in cache:
            cache[cam_dir] = build_index(cam_dir)
        vids = cache[cam_dir]

        hit = find_segment(vids, event_dt)
        if hit is None:
            unmatched += 1
            rows_report.append({
                "event_idx": r.get("event_idx", i),
                "reason": "no_segment_contains_event_time",
                "camera_dir": str(cam_dir),
                "event_dt": str(event_dt),
                "orig_video_filename": fn
            })
            continue

        true_s, true_e, true_p = hit
        new_offset = (event_dt - true_s).total_seconds()

        # Update if different file
        if true_p.name != fn:
            moved += 1

        df.at[i, "video_path"] = str(true_p)
        df.at[i, "video_filename"] = true_p.name
        df.at[i, "event_offset_sec"] = float(new_offset)

        # (optional) also update video_id if you want it to match filename
        if "video_id" in df.columns:
            df.at[i, "video_id"] = true_p.name.lower()

    df.to_csv(args.out_csv, index=False)

    if args.report_csv:
        pd.DataFrame(rows_report).to_csv(args.report_csv, index=False)

    print("DONE")
    print(f"Saved fixed manifest: {args.out_csv}")
    if args.report_csv:
        print(f"Saved report:         {args.report_csv}")
    print(f"Moved to new segment: {moved}")
    print(f"Unmatched:            {unmatched}")
    print(f"Bad filename ts:      {bad_ts}")

if __name__ == "__main__":
    main()
