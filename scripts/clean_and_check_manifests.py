import argparse
from pathlib import Path
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def load_csv(path: Path, name: str):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["_split"] = name
    return df

def best_video_key(df: pd.DataFrame):
    # prefer stable hash if present
    for c in ["video_hash", "hash", "video_md5", "video_sha1"]:
        if c in df.columns:
            return c

    if "video_path" in df.columns:
        return "video_path"
    if "video_filename" in df.columns:
        return "video_filename"

    # fallback
    candidates = ["group", "day", "camera_id", "video_filename"]
    if all(c in df.columns for c in candidates):
        df["video_key_fallback"] = (
            df["group"].astype(str) + "|" +
            df["day"].astype(str) + "|" +
            df["camera_id"].astype(str) + "|" +
            df["video_filename"].astype(str)
        )
        return "video_key_fallback"

    raise ValueError("No usable video identifier found (need video_hash/video_path/video_filename/etc.)")

def event_key(df: pd.DataFrame):
    for c in ["event_idx", "event_id", "event_uuid"]:
        if c in df.columns:
            return c
    return None

def overlap(a: pd.DataFrame, b: pd.DataFrame, col: str):
    sa = set(a[col].dropna().astype(str).unique())
    sb = set(b[col].dropna().astype(str).unique())
    inter = sa & sb
    return len(inter)

def ensure_video_key(df: pd.DataFrame, vkey: str):
    # If vkey is fallback, it might not exist in some dfs yet.
    if vkey in df.columns:
        return
    # try to create fallback if needed
    _ = best_video_key(df)

def clean_train_val_test(train, val, test, key: str):
    test_vids = set(test[key].dropna().astype(str).unique())
    train2 = train[~train[key].astype(str).isin(test_vids)].copy()
    val2   = val[~val[key].astype(str).isin(test_vids)].copy()

    # enforce train/val disjoint
    val2_vids = set(val2[key].dropna().astype(str).unique())
    train2 = train2[~train2[key].astype(str).isin(val2_vids)].copy()
    return train2, val2, test.copy()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/manifests",
                    help="Folder containing manifest CSVs")
    ap.add_argument("--write_clean", action="store_true",
                    help="Write train_CLEAN.csv / val_CLEAN.csv / test_CLEAN.csv")
    ap.add_argument("--clean_prefix", type=str, default="",
                    help="Optional prefix for output clean files, e.g. 'v1_' -> v1_train_CLEAN.csv")
    ap.add_argument("--include_intravideo", action="store_true",
                    help="Also clean intravideo train/val pairs if present (keeps same rules vs test + disjoint)")
    args = ap.parse_args()

    root = Path(args.root)

    FILES = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",

        # optional intravideo variants (if present)
        "train_intravideo_162028": "train_intravideo_20260105_162028.csv",
        "val_intravideo_162028": "val_intravideo_20260105_162028.csv",
        "train_intravideo_162114": "train_intravideo_20260105_162114.csv",
        "val_intravideo_162114": "val_intravideo_20260105_162114.csv",
    }

    dfs = {}
    for name, fn in FILES.items():
        df = load_csv(root / fn, name)
        if df is not None:
            dfs[name] = df

    if not dfs:
        raise SystemExit(f"No CSVs found under: {root}")

    print("Loaded:")
    for k, d in dfs.items():
        print(f"  {k:>24}: {d.shape}")

    # Pick vkey/ekey from train if possible
    any_df = dfs["train"] if "train" in dfs else next(iter(dfs.values()))
    vkey = best_video_key(any_df)
    ekey = event_key(any_df)

    # Ensure key exists everywhere if fallback
    for d in dfs.values():
        ensure_video_key(d, vkey)

    print(f"\nUsing VIDEO KEY = {vkey}")
    print(f"Using EVENT KEY = {ekey}\n")

    # ----------------------------
    # Pairwise leakage checks
    # ----------------------------
    names = list(dfs.keys())
    print("================ VIDEO OVERLAPS ================")
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = dfs[names[i]], dfs[names[j]]
            n = overlap(a, b, vkey)
            if n > 0:
                any_overlap = True
                print(f"{names[i]} vs {names[j]}: {n} overlapping videos")
    if not any_overlap:
        print("No overlapping videos detected across loaded files.")

    if ekey:
        print("\n================ EVENT OVERLAPS ================")
        any_e = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = dfs[names[i]], dfs[names[j]]
                if ekey in a.columns and ekey in b.columns:
                    n = overlap(a, b, ekey)
                    if n > 0:
                        any_e = True
                        print(f"{names[i]} vs {names[j]}: {n} overlapping events")
        if not any_e:
            print("No overlapping events detected across loaded files.")

    # ----------------------------
    # Duplicate checks (within file)
    # ----------------------------
    print("\n================ DUPLICATES (WITHIN FILE) ================")
    for n, df in dfs.items():
        dup_v = df[vkey].dropna().astype(str).duplicated().sum() if vkey in df.columns else 0
        print(f"{n:>24}: duplicated video rows = {dup_v}")
        if ekey and ekey in df.columns:
            dup_e = df[ekey].dropna().astype(str).duplicated().sum()
            print(f"{'':>24}  duplicated event rows = {dup_e}")

    # ----------------------------
    # CLEAN train/val/test
    # ----------------------------
    if args.write_clean:
        if not all(k in dfs for k in ["train", "val", "test"]):
            raise SystemExit("Need train.csv, val.csv, test.csv loaded to write CLEAN outputs.")

        train2, val2, test2 = clean_train_val_test(dfs["train"], dfs["val"], dfs["test"], vkey)

        prefix = args.clean_prefix
        out_train = root / f"{prefix}train_CLEAN.csv"
        out_val   = root / f"{prefix}val_CLEAN.csv"
        out_test  = root / f"{prefix}test_CLEAN.csv"

        train2.drop(columns=["_split"], errors="ignore").to_csv(out_train, index=False)
        val2.drop(columns=["_split"], errors="ignore").to_csv(out_val, index=False)
        test2.drop(columns=["_split"], errors="ignore").to_csv(out_test, index=False)

        print("\n================ CLEAN OUTPUTS ================")
        print(f"Saved: {out_train.name}, {out_val.name}, {out_test.name}")
        print(f"Counts: train={len(train2)}, val={len(val2)}, test={len(test2)}")

        # quick post-check
        print("\nPost-check overlaps (should be 0):")
        print("  train vs test:", overlap(train2, test2, vkey))
        print("  val   vs test:", overlap(val2, test2, vkey))
        print("  train vs val :", overlap(train2, val2, vkey))

        # Optional: clean intravideo pairs too
        if args.include_intravideo:
            pairs = [
                ("train_intravideo_162028", "val_intravideo_162028"),
                ("train_intravideo_162114", "val_intravideo_162114"),
            ]
            for tr_name, va_name in pairs:
                if tr_name in dfs and va_name in dfs:
                    tr2, va2, _ = clean_train_val_test(dfs[tr_name], dfs[va_name], test2, vkey)
                    out_tr = root / f"{prefix}{tr_name}_CLEAN.csv"
                    out_va = root / f"{prefix}{va_name}_CLEAN.csv"
                    tr2.drop(columns=["_split"], errors="ignore").to_csv(out_tr, index=False)
                    va2.drop(columns=["_split"], errors="ignore").to_csv(out_va, index=False)
                    print(f"\nSaved intravideo CLEAN: {out_tr.name}, {out_va.name}")
                    print(f"Counts: {tr_name}={len(tr2)}, {va_name}={len(va2)}")
                else:
                    print(f"\nSkip intravideo pair (missing): {tr_name}, {va_name}")

    print("\nDone.")

if __name__ == "__main__":
    main()
