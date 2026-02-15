# scripts/create_test_events.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE = Path("data/manifests")
TRAIN = BASE / "train.csv"
VAL = BASE / "val.csv"
LINKED = BASE / "linked_events.csv"

OUT_TEST = BASE / "test_events.csv"
OUT_TEST_COLLAPSED = BASE / "test_events_collapsed.csv"

# collapse mapping (keep 3-class for now)
COLLAPSE = {
    "ear": "ear",
    "tail": "tail",
    "teat": "other",   # collapse rare classes
    "other": "other",
}

REQUIRED = [
    "duration_sec", "behavior", "video_path", "event_offset_sec",
    "video_filename",  # used if needed
]

def norm_path(s: str) -> str:
    # normalize windows paths for stable grouping/compare
    return str(s).strip().lower().replace("\\", "/")

def get_video_key(df: pd.DataFrame) -> pd.Series:
    # prefer video_path if present; else fallback to filename
    if "video_path" in df.columns:
        return df["video_path"].astype(str).map(norm_path)
    return df["video_filename"].astype(str).str.strip().str.lower()

def main():
    train = pd.read_csv(TRAIN)
    val = pd.read_csv(VAL)
    linked = pd.read_csv(LINKED)

    # sanity: required columns exist in linked
    missing = [c for c in REQUIRED if c not in linked.columns]
    if missing:
        raise ValueError(f"{LINKED} missing columns: {missing}.")
    # also ensure train/val have key col
    if "video_path" not in train.columns or "video_path" not in val.columns:
        raise ValueError("train/val must include video_path to build no-leak test split.")

    train_key = set(get_video_key(train).unique())
    val_key = set(get_video_key(val).unique())
    used = train_key | val_key

    linked_key = get_video_key(linked)
    test = linked[~linked_key.isin(used)].copy()

    # If this ends up empty, it means train/val consumed all event videos.
    if len(test) == 0:
        raise RuntimeError(
            "No unused event videos left for test_events.csv. "
            "You must re-split from linked_events.csv into train/val/test together."
        )

    # Optional: basic cleaning
    test["behavior"] = test["behavior"].astype(str).str.strip().str.lower()
    test["video_path"] = test["video_path"].astype(str)
    test["video_filename"] = test["video_filename"].astype(str)

    # write event-format test
    test.to_csv(OUT_TEST, index=False)

    # collapsed version (3-class)
    test_c = test.copy()
    test_c["behavior"] = test_c["behavior"].map(lambda x: COLLAPSE.get(x, "other"))
    test_c.to_csv(OUT_TEST_COLLAPSED, index=False)

    # report
    print(f"[OK] wrote {OUT_TEST} | n={len(test)}")
    print(test["behavior"].value_counts(dropna=False))
    print()
    print(f"[OK] wrote {OUT_TEST_COLLAPSED} | n={len(test_c)} (collapsed)")
    print(test_c["behavior"].value_counts(dropna=False))

    # leakage check (video-level)
    test_key = set(get_video_key(test).unique())
    print()
    print("[Leakage checks]")
    print("train ∩ test:", len(train_key & test_key))
    print("val   ∩ test:", len(val_key & test_key))
    assert len(train_key & test_key) == 0
    assert len(val_key & test_key) == 0
    print("[OK] No video leakage detected.")

if __name__ == "__main__":
    main()
