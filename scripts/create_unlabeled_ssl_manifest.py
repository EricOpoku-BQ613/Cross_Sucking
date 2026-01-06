# scripts/create_unlabeled_ssl_manifest.py
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime, timezone

def norm_base(p: str) -> str:
    return Path(str(p)).name.lower().strip()

def read_manifest(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash_index", default="data/manifests/video_hash_index.csv")
    ap.add_argument("--train", default="data/manifests/train.csv")
    ap.add_argument("--val", default="data/manifests/val.csv")
    ap.add_argument("--test", default="data/manifests/test.csv")
    ap.add_argument("--smoke_val", default="data/manifests/smoke_to_val.csv")
    ap.add_argument("--smoke_test", default="data/manifests/smoke_to_test.csv")
    ap.add_argument("--out_csv", default="data/manifests/unlabeled_ssl.csv")
    ap.add_argument("--out_yaml", default="configs/ssl_unlabeled_protocol.yaml")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    df = pd.read_csv(args.hash_index)

    # --- Make this robust to differing column names ---
    # Expect at least a path column; try common names.
    path_col = None
    for c in ["video_path", "path", "filepath", "file_path"]:
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        raise ValueError(f"Could not find a path column in {args.hash_index}. Columns={list(df.columns)}")

    # Build basename key (matches your split_protocol grouping_rule: basename_lower)
    df["basename_lower"] = df[path_col].astype(str).apply(norm_base)

    # Determine validity:
    # - If a 'valid' column exists, trust it.
    # - Otherwise: require non-null hash, non-null size, and no obvious error column contents.
    if "valid" in df.columns:
        df_ok = df[df["valid"] == True].copy()
    else:
        # identify likely hash/size/error cols
        hash_col = "phash" if "phash" in df.columns else ("hash" if "hash" in df.columns else None)
        size_col = "size" if "size" in df.columns else ("bytes" if "bytes" in df.columns else None)
        err_col = "error" if "error" in df.columns else ("err" if "err" in df.columns else None)

        df_ok = df.copy()
        if hash_col is not None:
            df_ok = df_ok[df_ok[hash_col].notna()]
        if size_col is not None:
            df_ok = df_ok[df_ok[size_col].notna()]
        if err_col is not None:
            # keep rows with empty/NaN error
            df_ok = df_ok[df_ok[err_col].isna() | (df_ok[err_col].astype(str).str.strip() == "")]
        df_ok = df_ok.copy()

    # Also ensure the files exist on disk (cheap and catches stale paths)
    def exists(p: str) -> bool:
        try:
            return Path(p).exists()
        except Exception:
            return False
    df_ok = df_ok[df_ok[path_col].astype(str).apply(exists)].copy()

    # Build exclusion basenames from labeled & smoke sets
    exclude = set()
    for m in [args.train, args.val, args.test, args.smoke_val, args.smoke_test]:
        dm = read_manifest(m)
        if len(dm) == 0:
            continue
        # prefer video_path if present; otherwise try video_file_name
        m_path_col = "video_path" if "video_path" in dm.columns else ("video_file_name" if "video_file_name" in dm.columns else None)
        if m_path_col is None:
            # fallback: if they have a basename column already
            if "basename_lower" in dm.columns:
                exclude |= set(dm["basename_lower"].astype(str).str.lower().str.strip())
                continue
            continue
        exclude |= set(dm[m_path_col].astype(str).apply(norm_base))

    before = len(df_ok)
    df_unl = df_ok[~df_ok["basename_lower"].isin(exclude)].copy()
    after = len(df_unl)

    # Output columns (keep it simple + stable)
    out = pd.DataFrame({
        "basename_lower": df_unl["basename_lower"].values,
        "video_path": df_unl[path_col].astype(str).values,
    })
    # Add optional helpful columns if present
    for c in ["video_id", "size", "bytes", "duration_sec", "fps", "nframes"]:
        if c in df_unl.columns:
            out[c] = df_unl[c].values

    out = out.drop_duplicates(subset=["basename_lower"]).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    meta = {
        "version": 1,
        "frozen": True,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "source": args.hash_index,
        "exclusions": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "smoke_val": args.smoke_val,
            "smoke_test": args.smoke_test,
            "rule": "exclude if basename_lower in any labeled/smoke manifest",
        },
        "counts": {
            "valid_candidates_before_exclusion": int(before),
            "excluded_by_labeled_or_smoke": int(before - after),
            "unlabeled_ssl_unique_videos": int(len(out)),
        },
        "notes": [
            "Unlabeled SSL pool excludes any video basename found in labeled train/val/test and smoke manifests.",
            "Corrupted/unreadable files are removed via video_hash_index validity + existence checks.",
        ],
        "outputs": {
            "unlabeled_ssl_csv": args.out_csv,
        }
    }
    Path(args.out_yaml).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print("[OK] Wrote:")
    print(f" - {args.out_csv} ({len(out)} unique unlabeled videos)")
    print(f" - {args.out_yaml}")
    print(f"[INFO] Excluded basenames from labeled/smoke: {len(exclude)}")

if __name__ == "__main__":
    main()
