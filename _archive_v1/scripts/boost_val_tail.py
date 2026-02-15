from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse


def norm_path(s: str) -> str:
    return str(s).strip().lower().replace("\\", "/")

def add_video_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust video key:
    - If video_path already includes filename, keep it.
    - Else, join video_path + video_filename.
    """
    df = df.copy()
    if "video_path" not in df.columns:
        raise ValueError("Expected 'video_path' column.")

    vp = df["video_path"].astype(str).map(norm_path)

    if "video_filename" in df.columns:
        vf = df["video_filename"].astype(str).map(norm_path)
        ends = [p.endswith(f) for p, f in zip(vp.tolist(), vf.tolist())]
        key = [
            p if e else (p.rstrip("/") + "/" + f)
            for p, f, e in zip(vp.tolist(), vf.tolist(), ends)
        ]
        df["video_key"] = key
    else:
        df["video_key"] = vp

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--val", type=str, required=True)
    ap.add_argument("--test", type=str, default="data/manifests/test.csv")
    ap.add_argument("--add_tail_events", type=int, default=21)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    train_in = Path(args.train)
    val_in   = Path(args.val)
    test_in  = Path(args.test)

    # outputs (timestamped to avoid Windows locks)
    base = train_in.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_out = base / f"train_boosted_{ts}.csv"
    val_out   = base / f"val_boosted_{ts}.csv"
    audit_out = base / f"splits_boosted_{ts}.json"

    train = add_video_key(pd.read_csv(train_in))
    val   = add_video_key(pd.read_csv(val_in))
    test  = add_video_key(pd.read_csv(test_in))  # only for leakage check

    # normalize behavior
    for df in (train, val):
        if "behavior" not in df.columns:
            raise ValueError("Expected 'behavior' column in train/val event manifests.")
        df["behavior"] = df["behavior"].astype(str).str.lower().str.strip()

    # --- leakage sanity (video-level)
    tr_v = set(train["video_key"].unique())
    va_v = set(val["video_key"].unique())
    te_v = set(test["video_key"].unique()) if "video_key" in test.columns else set()

    if len(tr_v & va_v) > 0:
        raise AssertionError("Leakage: train/val share videos already.")
    if len(tr_v & te_v) > 0:
        raise AssertionError("Leakage: train/test share videos already.")
    if len(va_v & te_v) > 0:
        raise AssertionError("Leakage: val/test share videos already.")

    # current tail count in val
    val_tail_now = int((val["behavior"] == "tail").sum())
    target_tail = val_tail_now + int(args.add_tail_events)

    # candidates: tail videos in TRAIN (video-level move)
    tail_train = train[train["behavior"] == "tail"].copy()
    if len(tail_train) == 0:
        raise RuntimeError("No tail events in train to move.")

    per_vid = (
        tail_train.groupby("video_key").size()
        .reset_index(name="tail_events")
        .sort_values("tail_events", ascending=False)
    )

    chosen = []
    running = val_tail_now

    # shuffle within equal-count buckets for deterministic randomness
    for cnt, group in per_vid.groupby("tail_events", sort=False):
        vids = group["video_key"].tolist()
        rng.shuffle(vids)
        for vk in vids:
            if running >= target_tail:
                break
            chosen.append(vk)
            running += int(cnt)
        if running >= target_tail:
            break

    if running < target_tail:
        raise RuntimeError(
            f"Not enough tail in train to add {args.add_tail_events}. "
            f"val_tail_now={val_tail_now}, max_possible_add={running - val_tail_now}"
        )

    chosen_set = set(chosen)

    # move ALL events for chosen videos train -> val
    move_rows = train[train["video_key"].isin(chosen_set)]
    train_new = train[~train["video_key"].isin(chosen_set)]
    val_new   = pd.concat([val, move_rows], ignore_index=True)

    # final leakage sanity
    tr_v2 = set(train_new["video_key"].unique())
    va_v2 = set(val_new["video_key"].unique())
    if len(tr_v2 & va_v2) > 0:
        raise AssertionError("Leakage after move: train/val overlap.")
    if len(tr_v2 & te_v) > 0:
        raise AssertionError("Leakage after move: train/test overlap.")
    if len(va_v2 & te_v) > 0:
        raise AssertionError("Leakage after move: val/test overlap.")

    def counts(df):
        return df["behavior"].value_counts().to_dict()

    print("\n[Before]")
    print("train:", counts(train))
    print("val:  ", counts(val))

    print("\n[After]")
    print("train:", counts(train_new))
    print("val:  ", counts(val_new))

    added = int((val_new["behavior"] == "tail").sum()) - val_tail_now
    print("\nMoved videos:", len(chosen_set))
    print("Added tail events:", added)
    print("Val tail now:", int((val_new["behavior"] == "tail").sum()))

    # write (drop helper col)
    train_new.drop(columns=["video_key"]).to_csv(train_out, index=False)
    val_new.drop(columns=["video_key"]).to_csv(val_out, index=False)

    audit = {
        "seed": args.seed,
        "action": "boost_val_tail_by_moving_train_videos",
        "requested_add_tail_events": int(args.add_tail_events),
        "actual_added_tail_events": int(added),
        "val_tail_before": int(val_tail_now),
        "val_tail_after": int((val_new["behavior"] == "tail").sum()),
        "moved_video_keys": sorted(list(chosen_set)),
        "inputs": {"train": str(train_in), "val": str(val_in), "test": str(test_in)},
        "outputs": {"train": str(train_out), "val": str(val_out)},
    }
    audit_out.write_text(json.dumps(audit, indent=2))

    print(f"\n[OK] wrote {train_out}")
    print(f"[OK] wrote {val_out}")
    print(f"[OK] wrote {audit_out}")


if __name__ == "__main__":
    main()
