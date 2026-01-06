#!/usr/bin/env python3
"""
Confirm whether Smoke test clips overlap with our corpus.

Key guarantees:
- Corpus scan EXCLUDES the smoke_videos folder (prevents self-matching).
- 'Our Video?' is parsed strictly as label-based membership.
- Label columns are selected strictly (None/Ear/Tail/Teats/Other).
- Hash matching is best-effort; if coverage is low, we warn.
- Conservative split rule: if uncertain -> OVERLAP (goes to Val), not OOD (Test).

Outputs (in data/manifests/):
- video_hash_index.csv
- video_hash_errors.csv
- smoke_hash_errors.csv
- smoke_decisions.csv
- smoke_overlap_confirmed.csv
- smoke_ood_confirmed.csv
- disagreements.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None


# # -----------------------------
# # Logging helpers
# # -----------------------------
# def now() -> str:
#     return time.strftime("%H:%M:%S")


# def log(msg: str, level: str = "INFO") -> None:
#     icon = {"INFO": "[I]", "OK": "[OK]", "WARN": "[W]", "ERR": "[E]"}.get(level, "[I]")
#     try:
#         print(f"[{now()}] {icon} {msg}")
#     except UnicodeEncodeError:
#         # Fallback for systems that can't encode emoji
#         print(f"[{now()}] {icon} {msg}".encode('ascii', 'replace').decode('ascii'))


# # -----------------------------
# # Video hashing utilities
# # -----------------------------
# def hamming(a: int, b: int) -> int:
#     return (a ^ b).bit_count()


# def dhash_int(gray: np.ndarray, hash_size: int = 16) -> int:
#     """
#     Difference hash -> int.
#     Produces hash_size*hash_size bits.
#     """
#     # resize to (hash_size+1, hash_size)
#     img = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
#     diff = img[:, 1:] > img[:, :-1]
#     # pack bits into int
#     bits = diff.flatten().astype(np.uint8)
#     out = 0
#     for bit in bits:
#         out = (out << 1) | int(bit)
#     return out


# def safe_video_meta(path: Path) -> Tuple[int, float]:
#     """
#     Returns (nframes, fps) if possible else (-1, -1).
#     """
#     cap = cv2.VideoCapture(str(path))
#     if not cap.isOpened():
#         return -1, -1.0
#     nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
#     cap.release()
#     if nframes <= 0:
#         nframes = -1
#     if fps <= 0:
#         fps = -1.0
#     return nframes, fps


# def sample_frames(path: Path, n_samples: int = 3) -> Tuple[List[np.ndarray], Optional[str]]:
#     """
#     Sample frames at ~10%, 50%, 90% positions.
#     Returns (frames_gray, error_message)
#     """
#     if cv2 is None:
#         return [], "cv2 not installed"

#     cap = cv2.VideoCapture(str(path))
#     if not cap.isOpened():
#         return [], "cannot open video"

#     nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     if nframes <= 0:
#         cap.release()
#         return [], "frame_count_unknown"

#     idxs = [max(0, int(nframes * p)) for p in (0.10, 0.50, 0.90)]
#     frames = []
#     for idx in idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ok, frame = cap.read()
#         if not ok or frame is None:
#             cap.release()
#             return [], f"failed_read_at_frame_{idx}"
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frames.append(gray)

#     cap.release()
#     return frames, None


# def video_signature(path: Path, hash_size: int = 16) -> Tuple[int, int, int, int, float, int]:
#     """
#     Returns (h1,h2,h3,nframes,fps,size_bytes).
#     If failure: h1=h2=h3=-1 and nframes=-1,fps=-1.
#     """
#     try:
#         size = path.stat().st_size
#     except Exception:
#         size = -1

#     nframes, fps = safe_video_meta(path)
#     frames, err = sample_frames(path, n_samples=3)
#     if err is not None or len(frames) != 3:
#         return -1, -1, -1, nframes, fps, size

#     h = [dhash_int(f, hash_size=hash_size) for f in frames]
#     return h[0], h[1], h[2], nframes, fps, size


# # -----------------------------
# # CSV parsing
# # -----------------------------
# LABEL_COL_PATTERN = re.compile(r"^(none|ear|tail|teats|other)\s*\(1-0\)\s*$", re.IGNORECASE)


# def normalize_cols(cols: List[str]) -> Dict[str, str]:
#     """
#     Map normalized->original column name.
#     """
#     out = {}
#     for c in cols:
#         key = re.sub(r"\s+", " ", str(c).strip().lower())
#         out[key] = c
#     return out


# def parse_yes_no(x) -> Optional[bool]:
#     if pd.isna(x):
#         return None
#     s = str(x).strip().lower()
#     if s in {"y", "yes", "true", "1"}:
#         return True
#     if s in {"n", "no", "false", "0"}:
#         return False
#     return None


# def coerce_01(x) -> Tuple[int, bool]:
#     """
#     Returns (value, was_invalid)
#     """
#     if pd.isna(x):
#         return 0, False
#     s = str(x).strip().lower()
#     if s in {"0", "0.0"}:
#         return 0, False
#     if s in {"1", "1.0"}:
#         return 1, False
#     # allow blank -> 0
#     if s == "":
#         return 0, False
#     return 0, True


# # -----------------------------
# # Corpus collection
# # -----------------------------
# VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".m4v"}


# def iter_videos(root: Path) -> List[Path]:
#     vids = []
#     for p in root.rglob("*"):
#         if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
#             vids.append(p)
#     return vids


# def collect_corpus_from_config(cfg: dict, labeled_only: bool, smoke_videos: Path, exclude_paths: Optional[List[Path]] = None) -> List[Path]:
#     corpus = []
#     # labeled roots
#     for _, node in cfg.get("labeled", {}).items():
#         p = Path(node["path"])
#         corpus.extend(iter_videos(p))

#     if not labeled_only:
#         # unlabeled roots
#         unl = cfg.get("unlabeled", {}).get("root")
#         if unl:
#             corpus.extend(iter_videos(Path(unl)))

#     # Build exclusion list (smoke_videos + any other exclude_paths)
#     if exclude_paths is None:
#         exclude_paths = []
#     exclude_paths = [smoke_videos] + exclude_paths
#     exclude_resolved = [p.resolve() for p in exclude_paths if p]

#     # EXCLUDE paths
#     filtered = []
#     for p in corpus:
#         try:
#             rp = p.resolve()
#             skip = False
#             for excl in exclude_resolved:
#                 if excl in rp.parents or rp == excl:
#                     skip = True
#                     break
#             if skip:
#                 continue
#         except Exception:
#             pass
#         filtered.append(p)
    
#     # unique by resolved path
#     uniq = []
#     seen = set()
#     for p in filtered:
#         k = str(p).lower()
#         if k not in seen:
#             uniq.append(p)
#             seen.add(k)
#     return uniq


# # -----------------------------
# # Hash index cache
# # -----------------------------
# HASH_INDEX_PATH = Path("data/manifests/video_hash_index.csv")
# HASH_ERROR_PATH = Path("data/manifests/video_hash_errors.csv")


# def ensure_parent(p: Path) -> None:
#     p.parent.mkdir(parents=True, exist_ok=True)


# def load_hash_index() -> pd.DataFrame:
#     if HASH_INDEX_PATH.exists():
#         df = pd.read_csv(HASH_INDEX_PATH)
#         # normalize valid column if present
#         if "valid" in df.columns:
#             df["valid"] = df["valid"].astype(str).str.lower().map(
#                 {"true": True, "false": False, "1": True, "0": False}
#             ).fillna(False)
#         return df
#     return pd.DataFrame(columns=["path", "h1", "h2", "h3", "nframes", "fps", "size", "valid"])


# def build_hash_index(
#     corpus_paths: List[Path],
#     workers: int = 4,
#     force_rebuild: bool = False,
#     hash_size: int = 16,
# ) -> pd.DataFrame:
#     ensure_parent(HASH_INDEX_PATH)
#     ensure_parent(HASH_ERROR_PATH)

#     if cv2 is None:
#         log("cv2 missing; cannot build hash index.", "ERR")
#         return load_hash_index()

#     df_cache = pd.DataFrame()
#     if (not force_rebuild) and HASH_INDEX_PATH.exists():
#         df_cache = load_hash_index()

#     cache_map = {}
#     if len(df_cache) > 0:
#         for _, r in df_cache.iterrows():
#             cache_map[str(r["path"]).lower()] = r.to_dict()

#     todo = []
#     for p in corpus_paths:
#         key = str(p).lower()
#         if (not force_rebuild) and key in cache_map:
#             continue
#         todo.append(p)

#     log(f"Cache has {len(cache_map)} videos, computing {len(todo)} new ones (workers={workers})")
#     log(f"Hash metric: dhash | frames=3 | bits={hash_size*hash_size} | hashes=3 (sum per video)")

#     rows = []
#     errors = []
#     processed = 0
#     for p in todo:
#         h1, h2, h3, nframes, fps, size = video_signature(p, hash_size=hash_size)
#         valid = (h1 != -1 and h2 != -1 and h3 != -1)
#         rows.append(
#             {
#                 "path": str(p),
#                 "h1": h1,
#                 "h2": h2,
#                 "h3": h3,
#                 "nframes": nframes,
#                 "fps": fps,
#                 "size": size,
#                 "hash_bits": hash_size * hash_size,
#                 "n_hashes": 3,
#                 "valid": bool(valid),
#             }
#         )
#         if not valid:
#             errors.append({"path": str(p), "reason": "decode_or_hash_failed"})
#         processed += 1
#         if processed % 100 == 0 or processed == len(todo):
#             log(f"Progress: {processed}/{len(todo)}  | Errors: {len(errors)}")

#     # merge cache + new
#     all_rows = list(cache_map.values()) + rows
#     df = pd.DataFrame(all_rows)
#     # enforce types with safe numeric conversion
#     for c in ["h1", "h2", "h3", "nframes", "size", "hash_bits", "n_hashes"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
#     if "fps" in df.columns:
#         df["fps"] = pd.to_numeric(df["fps"], errors="coerce").fillna(-1.0)
#     if "valid" in df.columns:
#         df["valid"] = df["valid"].astype(bool)

#     df.to_csv(HASH_INDEX_PATH, index=False)

#     if errors:
#         pd.DataFrame(errors).to_csv(HASH_ERROR_PATH, index=False)

#     ok = int(df["valid"].sum()) if "valid" in df.columns else 0
#     log(f"Saved hash index to {HASH_INDEX_PATH} ({ok} valid / {len(df)} total)", "OK")
#     if errors:
#         log(f"Saved hash errors to {HASH_ERROR_PATH} ({len(errors)} new this run)", "WARN")
#     return df


# # -----------------------------
# # Matching logic
# # -----------------------------
# def best_hash_match(
#     smoke_sig: Tuple[int, int, int],
#     corpus_df_ok: pd.DataFrame,
#     hash_threshold: int = 12,
#     smoke_meta: Optional[dict] = None,
# ) -> Tuple[Optional[str], Optional[int], bool, bool]:
#     """
#     Returns (best_path, best_dist_raw, is_match, hash_ok).
#     hash_ok=False means the smoke video failed to hash (don't use hash decision).
#     """
#     if smoke_sig[0] == -1 or len(corpus_df_ok) == 0:
#         return None, None, False, False  # smoke failed to hash

#     # Drop rows with NaN hashes
#     df_ok = corpus_df_ok.dropna(subset=["h1", "h2", "h3"])

#     if len(df_ok) == 0:
#         return None, None, False, False  # no valid corpus hashes

#     # vectorize with safe conversion
#     h1 = pd.to_numeric(df_ok["h1"], errors="coerce").fillna(-1).astype("int64").to_numpy()
#     h2 = pd.to_numeric(df_ok["h2"], errors="coerce").fillna(-1).astype("int64").to_numpy()
#     h3 = pd.to_numeric(df_ok["h3"], errors="coerce").fillna(-1).astype("int64").to_numpy()
#     s1, s2, s3 = smoke_sig

#     best_i, best_d = None, 10**9
#     for i in range(len(df_ok)):
#         d = hamming(int(h1[i]), s1) + hamming(int(h2[i]), s2) + hamming(int(h3[i]), s3)
#         if d < best_d:
#             best_d = d
#             best_i = i

#     if best_i is None:
#         return None, None, False, False

#     best_path = str(df_ok.iloc[best_i]["path"])
#     is_match = best_d <= hash_threshold
#     return best_path, int(best_d), bool(is_match), True  # hash_ok=True


# # -----------------------------
# # Main
# # -----------------------------
# def main() -> int:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", default="configs/paths.yaml")
#     ap.add_argument("--smoke_videos", default=None, help="Override smoke video folder (defaults to config).")
#     ap.add_argument("--exclude_path", action="append", default=[], help="Additional paths to exclude from corpus scan (repeatable).")
#     ap.add_argument("--mode", choices=["filename", "hash", "both"], default="both")
#     ap.add_argument("--hash_threshold_strict", type=int, default=8, help="Strict overlap threshold (conservative match).")
#     ap.add_argument("--hash_threshold_loose", type=int, default=15, help="Loose threshold; between strict and loose goes to disagreements.")
#     ap.add_argument("--labeled_only", action="store_true")
#     ap.add_argument("--force_rebuild", action="store_true")
#     ap.add_argument("--workers", type=int, default=4)
#     ap.add_argument("--split_policy", choices=["label", "hash", "label_or_hash", "label_and_hash"], default="label_or_hash")
#     args = ap.parse_args()

#     log("SMOKE TEST OVERLAP CONFIRMATION (ROBUST)")
#     if yaml is None:
#         log("pyyaml missing; install pyyaml or pass explicit paths another way.", "ERR")
#         return 2

#     cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
#     smoke_csv = Path(cfg["annotations"]["smoke_test"])
#     smoke_videos = Path(args.smoke_videos) if args.smoke_videos else Path(cfg.get("smoke_videos", ""))  # optional
#     if not smoke_videos or str(smoke_videos).strip() == "." or str(smoke_videos).strip() == "":
#         # fallback to your known path
#         smoke_videos = Path(r"E:\Downey cross-sucking videos 2024\smoke_test\videos")

#     log(f"Loading smoke CSV: {smoke_csv}")
#     df_smoke = pd.read_csv(smoke_csv)

#     # Normalize columns
#     colmap = normalize_cols(list(df_smoke.columns))
#     # find "video" col
#     video_col = colmap.get("video", None)
#     if video_col is None:
#         log(f"Smoke CSV missing 'Video' column. Found columns: {list(df_smoke.columns)[:10]}", "ERR")
#         return 2

#     # find Our Video? column
#     our_col = None
#     for k, orig in colmap.items():
#         if k in {"our video?", "our video", "our video ?"}:
#             our_col = orig
#             break

#     # strict label columns
#     label_cols = []
#     for c in df_smoke.columns:
#         if LABEL_COL_PATTERN.match(str(c).strip()):
#             label_cols.append(c)

#     # coerce labels safely
#     invalid_cells = 0
#     invalid_rows = []
#     for c in label_cols:
#         new_vals = []
#         for i, x in enumerate(df_smoke[c].tolist()):
#             v, bad = coerce_01(x)
#             if bad:
#                 invalid_cells += 1
#                 invalid_rows.append({"row": i, "col": c, "value": x})
#             new_vals.append(v)
#         df_smoke[c] = new_vals

#     if invalid_cells > 0:
#         ensure_parent(Path("data/manifests/smoke_invalid_cells.csv"))
#         pd.DataFrame(invalid_rows).to_csv("data/manifests/smoke_invalid_cells.csv", index=False)
#         log(f"Smoke CSV has invalid label values in {invalid_cells} cell(s). Coerced to 0 (logged).", "WARN")

#     # parse our video labels
#     if our_col is None:
#         log("Smoke CSV has no 'Our Video?' column -> label policy unavailable.", "WARN")
#         df_smoke["our_video_bool"] = False
#     else:
#         parsed = []
#         bad = 0
#         for x in df_smoke[our_col].tolist():
#             b = parse_yes_no(x)
#             if b is None:
#                 bad += 1
#                 b = False
#             parsed.append(bool(b))
#         df_smoke["our_video_bool"] = parsed
#         if bad > 0:
#             log(f"Could not parse {bad} entries in '{our_col}'. Coerced to False.", "WARN")

#     # basic counts
#     log(f"Smoke clips (rows): {len(df_smoke)}", "OK")

#     # Collect corpus videos (EXCLUDES smoke folder + any --exclude_path)
#     log("Collecting corpus videos...")
#     exclude_paths = [Path(p) for p in args.exclude_path] if args.exclude_path else []
#     corpus_paths = collect_corpus_from_config(cfg, labeled_only=args.labeled_only, smoke_videos=smoke_videos, exclude_paths=exclude_paths)
#     log(f"Total corpus videos: {len(corpus_paths)}", "OK")

#     # Filename matching (only meaningful if corpus filenames resemble smoke)
#     fn_overlaps = {}
#     if args.mode in {"filename", "both"}:
#         log("FILENAME MATCHING")
#         name_to_paths: Dict[str, List[str]] = {}
#         for p in corpus_paths:
#             name_to_paths.setdefault(p.name.lower(), []).append(str(p))
#         for _, r in df_smoke.iterrows():
#             vid = str(r[video_col]).strip().lower()
#             if vid in name_to_paths:
#                 fn_overlaps[str(r[video_col])] = name_to_paths[vid]
#         log(f"Filename matches: {len(fn_overlaps)}", "OK" if len(fn_overlaps) == 0 else "WARN")
#         if len(fn_overlaps) == len(df_smoke):
#             log("All smoke filenames match the corpus. If your corpus scan included the smoke folder, this is a red flag.", "WARN")

#     # Hash index
#     hash_overlaps = {}
#     hash_disagreements = {}
#     smoke_hash_errors = []
#     best_match_paths = {}
#     best_dists = {}
#     hash_ok_map = {}

#     if args.mode in {"hash", "both"}:
#         log("PERCEPTUAL HASH MATCHING")
#         df_idx = build_hash_index(corpus_paths, workers=args.workers, force_rebuild=args.force_rebuild)

#         # robust valid filtering
#         if "valid" in df_idx.columns:
#             df_ok = df_idx[df_idx["valid"] == True].copy()
#         else:
#             df_ok = df_idx.copy()

#         coverage = (len(df_ok) / max(1, len(corpus_paths))) * 100.0
#         log(f"Hash coverage: {coverage:.1f}% ({len(df_ok)}/{len(corpus_paths)})")
#         if coverage < 85:
#             log("WARNING: Low hash coverage -> hash overlap may MISS matches. Treat uncertain as overlap (Val), not OOD (Test).", "WARN")

#         # match smoke videos with two-tier thresholds
#         log("MATCHING SMOKE VIDEOS")
#         log(f"Hash thresholds: strict={args.hash_threshold_strict}, loose={args.hash_threshold_loose}")
#         for i, r in df_smoke.iterrows():
#             vid = str(r[video_col]).strip()
#             vpath = smoke_videos / vid
#             if not vpath.exists():
#                 smoke_hash_errors.append({"video": vid, "reason": "missing_file", "path": str(vpath)})
#                 hash_ok_map[vid] = False
#                 continue

#             h1, h2, h3, nframes, fps, size = video_signature(vpath)
#             if h1 == -1:
#                 smoke_hash_errors.append({"video": vid, "reason": "decode_or_hash_failed", "path": str(vpath)})
#                 hash_ok_map[vid] = False
#                 continue

#             best_path, dist, is_match_strict, hash_ok = best_hash_match((h1, h2, h3), df_ok, hash_threshold=args.hash_threshold_strict)
#             hash_ok_map[vid] = hash_ok
#             best_match_paths[vid] = best_path
#             best_dists[vid] = dist
            
#             if is_match_strict:
#                 hash_overlaps[vid] = best_path
#             elif dist is not None and dist <= args.hash_threshold_loose:
#                 hash_disagreements[vid] = (best_path, dist)

#             if (i + 1) % 10 == 0 or (i + 1) == len(df_smoke):
#                 log(f"Matched {i+1}/{len(df_smoke)}")

#         ensure_parent(Path("data/manifests/smoke_hash_errors.csv"))
#         pd.DataFrame(smoke_hash_errors).to_csv("data/manifests/smoke_hash_errors.csv", index=False)
#         log(f"Hash overlaps (strict, dist <= {args.hash_threshold_strict}): {len(hash_overlaps)}/{len(df_smoke)}", "OK")
#         log(f"Hash disagreements (loose, {args.hash_threshold_strict} < dist <= {args.hash_threshold_loose}): {len(hash_disagreements)}", "WARN" if len(hash_disagreements) > 0 else "OK")

#     # Split decision
#     decisions = []
#     disagreements = []
#     for _, r in df_smoke.iterrows():
#         vid = str(r[video_col]).strip()
#         label_overlap = bool(r.get("our_video_bool", False))

#         fn_overlap = (vid in fn_overlaps)
#         h_overlap = (vid in hash_overlaps)
#         hash_uncertain = (vid in hash_disagreements)
#         h_ok = hash_ok_map.get(vid, False)

#         if args.split_policy == "label":
#             final_overlap = label_overlap
#             reason = "label"
#         elif args.split_policy == "hash":
#             final_overlap = h_overlap
#             reason = "hash"
#         elif args.split_policy == "label_and_hash":
#             final_overlap = label_overlap and h_overlap
#             reason = "label_and_hash"
#         else:  # label_or_hash (conservative against leakage)
#             final_overlap = label_overlap or h_overlap or fn_overlap
#             reason = "label_or_hash"

#         # Flag disagreements for manual review
#         if hash_uncertain and not final_overlap:
#             best_p, best_d = hash_disagreements[vid]
#             disagreements.append(
#                 {
#                     "Video": vid,
#                     "label_overlap": int(label_overlap),
#                     "best_hash_dist": int(best_d),
#                     "best_hash_match": best_p,
#                     "reason": f"hash_dist_in_uncertain_range ({args.hash_threshold_strict}-{args.hash_threshold_loose})",
#                 }
#             )

#         decisions.append(
#             {
#                 "Video": vid,
#                 "label_overlap(OurVideo)": int(label_overlap),
#                 "filename_overlap": int(fn_overlap),
#                 "hash_overlap": int(h_overlap),
#                 "hash_ok": int(h_ok),
#                 "best_hash_match_path": best_match_paths.get(vid, None),
#                 "best_hash_dist": best_dists.get(vid, None),
#                 "final_overlap": int(final_overlap),
#                 "reason": reason,
#             }
#         )

#     df_dec = pd.DataFrame(decisions)
#     ensure_parent(Path("data/manifests/smoke_decisions.csv"))
#     df_dec.to_csv("data/manifests/smoke_decisions.csv", index=False)

#     # Save disagreements for manual review
#     if disagreements:
#         ensure_parent(Path("data/manifests/smoke_hash_disagreements.csv"))
#         pd.DataFrame(disagreements).to_csv("data/manifests/smoke_hash_disagreements.csv", index=False)
#         log(f"Saved {len(disagreements)} uncertain cases to smoke_hash_disagreements.csv (manual review)", "WARN")

#     # Final exports
#     overlap = df_dec[df_dec["final_overlap"] == 1].copy()
#     ood = df_dec[df_dec["final_overlap"] == 0].copy()

#     overlap_path = Path("data/manifests/smoke_overlap_confirmed.csv")
#     ood_path = Path("data/manifests/smoke_ood_confirmed.csv")
#     overlap.to_csv(overlap_path, index=False)
#     ood.to_csv(ood_path, index=False)

#     # Teat counts (only if Teats col exists)
#     teat_col = None
#     for c in df_smoke.columns:
#         if str(c).strip().lower().startswith("teats"):
#             teat_col = c
#             break

#     teat_overlap = teat_ood = 0
#     if teat_col is not None:
#         teat_map = dict(zip(df_smoke[video_col].astype(str).str.strip(), df_smoke[teat_col].astype(int)))
#         teat_overlap = int(sum(teat_map.get(v, 0) for v in overlap["Video"].tolist()))
#         teat_ood = int(sum(teat_map.get(v, 0) for v in ood["Video"].tolist()))

#     log("SUMMARY")
#     log(f"Policy: {args.split_policy} | Mode: {args.mode} | Strict threshold: {args.hash_threshold_strict}, Loose: {args.hash_threshold_loose}")
#     log(f"Overlap: {len(overlap)} clips | Teat: {teat_overlap}", "OK")
#     log(f"OOD: {len(ood)} clips | Teat: {teat_ood}", "OK")
#     if disagreements:
#         log(f"Disagreements (manual review): {len(disagreements)}", "WARN")
#     log(f"Saved: {overlap_path}", "OK")
#     log(f"Saved: {ood_path}", "OK")
#     log("Done!", "OK")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


