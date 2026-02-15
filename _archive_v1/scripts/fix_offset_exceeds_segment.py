#!/usr/bin/env python3
import argparse, re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

TS_RE = re.compile(r"_(\d{14})_(\d{14})")

def parse_seg(fn):
    m = TS_RE.search(str(fn))
    if not m: return None
    s = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    e = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    return s, e

def replace_seg(fn, ns, ne):
    base = str(fn)
    m = TS_RE.search(base)
    if not m: return base
    new_ts = f"_{ns.strftime('%Y%m%d%H%M%S')}_{ne.strftime('%Y%m%d%H%M%S')}"
    return base[:m.start()] + new_ts + base[m.end():]

def swap_path_filename(video_path, old_fn, new_fn):
    p = Path(str(video_path))
    if p.name == str(old_fn):
        return str(p.with_name(new_fn))
    return str(p.parent / new_fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_excess", type=float, default=120.0)  # seconds
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    fixed = 0
    for i, r in df.iterrows():
        fn = str(r.get("video_filename", ""))
        seg = parse_seg(fn)
        if not seg:
            continue
        seg_start, seg_end = seg
        seg_len = (seg_end - seg_start).total_seconds()
        off = r.get("event_offset_sec", None)

        # only fix if event_offset_sec exists and exceeds segment by a small amount
        try:
            off = float(off)
        except:
            continue

        excess = off - seg_len
        if excess > 0 and excess <= args.max_excess:
            # shift to next segment
            new_start = seg_start + timedelta(seconds=seg_len)
            new_end   = seg_end   + timedelta(seconds=seg_len)

            new_fn = replace_seg(fn, new_start, new_end)
            df.at[i, "video_filename"] = new_fn
            df.at[i, "event_offset_sec"] = off - seg_len

            # update video_path filename
            vp = str(r.get("video_path", ""))
            df.at[i, "video_path"] = swap_path_filename(vp, fn, new_fn)

            fixed += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Fixed {fixed} boundary events. Wrote: {args.out}")

if __name__ == "__main__":
    main()
