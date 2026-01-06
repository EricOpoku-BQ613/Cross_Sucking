# scripts/create_final_splits.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

BASE = Path("data/manifests")
IN_EVENTS = BASE / "linked_events.csv"

OUT_TRAIN = BASE / "train.csv"
OUT_VAL = BASE / "val.csv"
OUT_TEST = BASE / "test.csv"           # event-format test
OUT_SPLITS = BASE / "splits.json"

# Optional: keep collapsed versions too (recommended)
OUT_TRAIN_C = BASE / "train_collapsed.csv"
OUT_VAL_C = BASE / "val_collapsed.csv"
OUT_TEST_C = BASE / "test_collapsed.csv"

# Collapse mapping (for splitting + optional collapsed manifests)
COLLAPSE = {
    "ear": "ear",
    "tail": "tail",
    "teat": "other",
    "other": "other",
}

REQUIRED = [
    "behavior",
    "video_path",
    "video_filename",
    "event_offset_sec",
    "duration_sec",
]

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def norm_path(s: str) -> str:
    return str(s).strip().lower().replace("\\", "/")

def get_video_key(df: pd.DataFrame) -> pd.Series:
    # primary grouping key: video_path
    return df["video_path"].astype(str).map(norm_path)

def mode_label(labels: pd.Series) -> str:
    # stable “mode” (most frequent); if tie, alphabetical
    vc = labels.value_counts()
    top = vc[vc == vc.max()].index.tolist()
    return sorted(top)[0]

def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Write CSV; if target is locked, write a timestamped sibling instead.
    Returns the actual path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new_{_ts()}{path.suffix}")
        df.to_csv(alt, index=False)
        print(f"[WARN] Could not overwrite {path.name} (locked). Wrote {alt.name} instead.")
        return alt

def safe_write_text(path: Path, text: str) -> Path:
    """
    Write text; if target is locked, write a timestamped sibling instead.
    Returns the actual path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new_{_ts()}{path.suffix}")
        alt.write_text(text, encoding="utf-8")
        print(f"[WARN] Could not overwrite {path.name} (locked). Wrote {alt.name} instead.")
        return alt

def stratified_split_video_keys(
    video_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    train_keys, val_keys, test_keys = [], [], []

    for lbl, sub in video_df.groupby("primary_label_split"):
        keys = sub["video_key"].tolist()
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        # ensure at least 1 in train for any class that exists
        if n >= 1 and n_train == 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1

        train_keys.extend(keys[:n_train])
        val_keys.extend(keys[n_train:n_train + n_val])
        test_keys.extend(keys[n_train + n_val:])

    return train_keys, val_keys, test_keys

def apply_split(df: pd.DataFrame, train_keys: set, val_keys: set, test_keys: set) -> pd.DataFrame:
    vk = get_video_key(df)
    split = np.where(vk.isin(train_keys), "train",
             np.where(vk.isin(val_keys), "val",
             np.where(vk.isin(test_keys), "test", "UNASSIGNED")))
    out = df.copy()
    out["_split"] = split
    return out

def collapse_manifest(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y["behavior"] = y["behavior"].map(lambda z: COLLAPSE.get(str(z).strip().lower(), "other"))
    return y

def main(
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 1337,
):
    df = pd.read_csv(IN_EVENTS)

    # sanity
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{IN_EVENTS} missing columns: {missing}. Found={list(df.columns)}")

    # normalize labels
    df["behavior"] = df["behavior"].astype(str).str.strip().str.lower()

    # video keys
    df["video_key"] = get_video_key(df)

    # label used for stratified splitting (collapsed for stability)
    df["behavior_split"] = df["behavior"].map(lambda x: COLLAPSE.get(x, "other"))

    # per-video primary label (for stratification)
    video_primary = (
        df.groupby("video_key")["behavior_split"]
          .apply(mode_label)
          .reset_index()
          .rename(columns={"behavior_split": "primary_label_split"})
    )

    # do split at VIDEO level
    tr_keys, va_keys, te_keys = stratified_split_video_keys(
        video_primary,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    tr_keys, va_keys, te_keys = set(tr_keys), set(va_keys), set(te_keys)

    # leakage checks
    assert len(tr_keys & va_keys) == 0
    assert len(tr_keys & te_keys) == 0
    assert len(va_keys & te_keys) == 0

    # apply to events
    df2 = apply_split(df, tr_keys, va_keys, te_keys)
    if (df2["_split"] == "UNASSIGNED").any():
        bad = df2[df2["_split"] == "UNASSIGNED"]["video_key"].unique()[:5]
        raise RuntimeError(f"Some rows unassigned to split. Examples video_key={bad}")

    train = df2[df2["_split"] == "train"].drop(columns=["_split"])
    val = df2[df2["_split"] == "val"].drop(columns=["_split"])
    test = df2[df2["_split"] == "test"].drop(columns=["_split"])

    # collapsed versions
    train_c = collapse_manifest(train)
    val_c = collapse_manifest(val)
    test_c = collapse_manifest(test)

    # write files (lock-safe)
    p_train = safe_to_csv(train, OUT_TRAIN)
    p_val = safe_to_csv(val, OUT_VAL)
    p_test = safe_to_csv(test, OUT_TEST)

    p_train_c = safe_to_csv(train_c, OUT_TRAIN_C)
    p_val_c = safe_to_csv(val_c, OUT_VAL_C)
    p_test_c = safe_to_csv(test_c, OUT_TEST_C)

    # report counts
    print("\n[Event counts]")
    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

    print("\n[Behavior counts - original]")
    print("train\n", train["behavior"].value_counts())
    print("val\n", val["behavior"].value_counts())
    print("test\n", test["behavior"].value_counts())

    print("\n[Behavior counts - collapsed]")
    print("train\n", train_c["behavior"].value_counts())
    print("val\n", val_c["behavior"].value_counts())
    print("test\n", test_c["behavior"].value_counts())

    # save splits audit (lock-safe)
    payload = {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "group_key": "video_path (normalized to video_key)",
        "stratify_on": "primary_label_split (behavior collapsed: teat->other)",
        "num_videos": {"train": len(tr_keys), "val": len(va_keys), "test": len(te_keys)},
        "num_events": {"train": len(train), "val": len(val), "test": len(test)},
        "written": {
            "train": str(p_train),
            "val": str(p_val),
            "test": str(p_test),
            "train_collapsed": str(p_train_c),
            "val_collapsed": str(p_val_c),
            "test_collapsed": str(p_test_c),
        },
        "videos_preview": {
            "train": sorted(list(tr_keys))[:50],
            "val": sorted(list(va_keys))[:50],
            "test": sorted(list(te_keys))[:50],
        },
    }
    p_splits = safe_write_text(OUT_SPLITS, json.dumps(payload, indent=2))

    print(f"\n[OK] wrote:")
    print(f"  {p_train.name}, {p_val.name}, {p_test.name}")
    print(f"  {p_train_c.name}, {p_val_c.name}, {p_test_c.name}")
    print(f"  {p_splits.name}")

if __name__ == "__main__":
    main()
