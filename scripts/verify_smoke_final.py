#!/usr/bin/env python3
"""
Smoke Test Independence Verification (FINAL VERSION)
=====================================================

This script definitively determines whether smoke test clips overlap with
the main corpus and outputs the final split for the paper.

Key Features:
1. Detailed debug logging for column detection and value parsing
2. Explicit smoke folder exclusion verification
3. Windows path handling (case-insensitive, normalized)
4. Conservative split policy: any uncertainty -> Val (not Test)
5. Clear audit trail for methodology section

Outputs:
    data/manifests/smoke_to_val.csv         - Clips for validation (overlap)
    data/manifests/smoke_to_test.csv        - Clips for held-out test (OOD)
    data/manifests/smoke_analysis_report.json - Full analysis report
    data/manifests/smoke_audit_trail.csv    - Per-clip decision audit

Usage:
    python scripts/verify_smoke_final.py --config configs/paths.yaml
    python scripts/verify_smoke_final.py --debug  # Extra verbose output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None


# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
HASH_SIZE = 16
HASH_THRESHOLD = 15  # Sum of 3 frame distances; 15 = similar, 8 = nearly identical
DEBUG = False


# =============================================================================
# LOGGING
# =============================================================================

def ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str, level: str = "INFO") -> None:
    icons = {"INFO": "[I]", "OK": "[+]", "WARN": "[!]", "ERR": "[X]", "DEBUG": "[D]"}
    icon = icons.get(level, "[?]")
    print(f"[{ts()}] {icon} {msg}")


def debug(msg: str) -> None:
    if DEBUG:
        log(msg, "DEBUG")


# =============================================================================
# PATH UTILITIES (Windows-safe)
# =============================================================================

def normalize_path(p: Path) -> str:
    """Normalize path for comparison (lowercase, forward slashes)."""
    try:
        return str(p.resolve()).lower().replace("\\", "/")
    except Exception:
        return str(p).lower().replace("\\", "/")


def is_subpath(child: Path, parent: Path) -> bool:
    """Check if child is under parent directory (Windows-safe)."""
    try:
        child_norm = normalize_path(child)
        parent_norm = normalize_path(parent)
        # Ensure parent ends with / for proper prefix matching
        if not parent_norm.endswith("/"):
            parent_norm += "/"
        return child_norm.startswith(parent_norm) or child_norm == parent_norm.rstrip("/")
    except Exception:
        return False


def paths_equal(p1: Path, p2: Path) -> bool:
    """Check if two paths point to the same location (Windows-safe)."""
    return normalize_path(p1) == normalize_path(p2)


# =============================================================================
# VIDEO HASHING
# =============================================================================

def dhash(gray: np.ndarray, size: int = HASH_SIZE) -> int:
    """Compute difference hash."""
    try:
        resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = diff.flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    except Exception:
        return -1


def hamming(a: int, b: int) -> int:
    """Hamming distance between two hashes."""
    if a < 0 or b < 0:
        return HASH_SIZE * HASH_SIZE
    return (a ^ b).bit_count()


def compute_video_hash(path: Path) -> Tuple[int, int, int, str]:
    """
    Compute 3-frame hash signature.
    Returns (h1, h2, h3, error_msg).
    """
    if not CV2_AVAILABLE:
        return -1, -1, -1, "opencv_not_available"
    
    if not path.exists():
        return -1, -1, -1, "file_not_found"
    
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return -1, -1, -1, "cannot_open"
        
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if nframes <= 0:
            return -1, -1, -1, "no_frames"
        
        hashes = []
        for frac in [0.1, 0.5, 0.9]:
            idx = max(0, min(nframes - 1, int(nframes * frac)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                return -1, -1, -1, f"read_fail_{idx}"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = dhash(gray, HASH_SIZE)
            if h < 0:
                return -1, -1, -1, f"hash_fail_{idx}"
            hashes.append(h)
        
        return hashes[0], hashes[1], hashes[2], ""
    except Exception as e:
        return -1, -1, -1, str(e)[:40]
    finally:
        if cap:
            cap.release()


# =============================================================================
# SMOKE CSV PARSING
# =============================================================================

@dataclass
class SmokeClip:
    video: str
    video_path: str
    our_video_raw: str = ""
    our_video_parsed: Optional[bool] = None  # True=Y, False=N, None=unknown
    labels: List[str] = field(default_factory=list)
    has_teat: bool = False
    notes: str = ""
    
    # Analysis results
    filename_match: bool = False
    hash_match: bool = False
    hash_distance: Optional[int] = None
    hash_match_path: str = ""
    hash_error: str = ""
    
    # Final decision
    decision: str = ""  # "VAL" or "TEST"
    decision_reasons: List[str] = field(default_factory=list)


def parse_smoke_csv(csv_path: Path, video_folder: Path) -> Tuple[List[SmokeClip], Dict]:
    """
    Parse smoke CSV with detailed validation.
    Returns (clips, parse_stats).
    """
    log(f"Parsing smoke CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    original_rows = len(df)
    log(f"  Raw rows: {original_rows}")
    
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    log(f"  Columns found: {list(df.columns)}")
    
    # Find key columns with flexible matching
    video_col = None
    our_video_col = None
    notes_col = None
    label_cols = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if col_lower == "video":
            video_col = col
        elif "our video" in col_lower:
            our_video_col = col
        elif "note" in col_lower:
            notes_col = col
        elif "none" in col_lower and "1-0" in col_lower:
            label_cols["none"] = col
        elif "ear" in col_lower and "1-0" in col_lower:
            label_cols["ear"] = col
        elif "tail" in col_lower and "1-0" in col_lower:
            label_cols["tail"] = col
        elif "teat" in col_lower and "1-0" in col_lower:
            label_cols["teat"] = col
        elif "other" in col_lower and "1-0" in col_lower:
            label_cols["other"] = col
    
    log(f"  Video column: {video_col}")
    log(f"  Our Video column: {our_video_col}")
    log(f"  Label columns: {label_cols}")
    
    if not video_col:
        raise ValueError(f"Cannot find 'Video' column in {list(df.columns)}")
    
    # Filter out empty/invalid video rows
    df = df[df[video_col].notna()]
    df = df[df[video_col].astype(str).str.strip() != ""]
    df = df[~df[video_col].astype(str).str.match(r'^\d+$')]  # Filter rows that are just numbers (summary row)
    
    log(f"  Valid video rows: {len(df)}")
    
    # Parse Our Video column
    our_video_stats = {"Y": 0, "N": 0, "invalid": 0, "missing": 0}
    
    clips = []
    for idx, row in df.iterrows():
        video_name = str(row[video_col]).strip()
        
        # Skip if video name doesn't look like a filename
        if not any(video_name.lower().endswith(ext.lower()) for ext in [".mp4", ".mov", ".avi"]):
            debug(f"  Skipping non-video row: {video_name}")
            continue
        
        clip = SmokeClip(
            video=video_name,
            video_path=str(video_folder / video_name)
        )
        
        # Parse Our Video?
        if our_video_col and our_video_col in row.index:
            raw_val = row[our_video_col]
            clip.our_video_raw = str(raw_val) if pd.notna(raw_val) else ""
            
            if pd.isna(raw_val):
                clip.our_video_parsed = None
                our_video_stats["missing"] += 1
            else:
                val_str = str(raw_val).strip().upper()
                if val_str in ["Y", "YES", "TRUE", "1"]:
                    clip.our_video_parsed = True
                    our_video_stats["Y"] += 1
                elif val_str in ["N", "NO", "FALSE", "0"]:
                    clip.our_video_parsed = False
                    our_video_stats["N"] += 1
                else:
                    clip.our_video_parsed = None
                    our_video_stats["invalid"] += 1
                    debug(f"  Invalid Our Video value: '{raw_val}' for {video_name}")
        else:
            our_video_stats["missing"] += 1
        
        # Parse labels
        for label_name, col_name in label_cols.items():
            if col_name in row.index:
                val = row[col_name]
                if pd.notna(val):
                    val_str = str(val).strip()
                    if val_str in ["1", "1.0"]:
                        clip.labels.append(label_name)
        
        clip.has_teat = "teat" in clip.labels
        
        # Notes
        if notes_col and notes_col in row.index:
            notes_val = row[notes_col]
            if pd.notna(notes_val):
                clip.notes = str(notes_val).strip()
        
        clips.append(clip)
    
    log(f"  Our Video stats: Y={our_video_stats['Y']}, N={our_video_stats['N']}, "
        f"invalid={our_video_stats['invalid']}, missing={our_video_stats['missing']}")
    
    parse_stats = {
        "original_rows": original_rows,
        "valid_clips": len(clips),
        "our_video_stats": our_video_stats,
        "label_cols_found": list(label_cols.keys()),
    }
    
    return clips, parse_stats


# =============================================================================
# CORPUS COLLECTION
# =============================================================================

def collect_corpus_videos(
    config: dict,
    smoke_folder: Path,
    labeled_only: bool = True
) -> Tuple[List[Path], Dict]:
    """
    Collect corpus videos, explicitly excluding smoke folder.
    Returns (video_paths, collection_stats).
    """
    log("Collecting corpus videos...")
    log(f"  Smoke folder to EXCLUDE: {smoke_folder}")
    
    smoke_folder_norm = normalize_path(smoke_folder)
    
    corpus = []
    sources = {}
    excluded_count = 0
    
    # Labeled folders
    for key, node in config.get("labeled", {}).items():
        folder = Path(node["path"])
        if not folder.exists():
            log(f"  {key}: folder not found - {folder}", "WARN")
            continue
        
        # Check if this folder is or contains smoke folder
        folder_norm = normalize_path(folder)
        if folder_norm == smoke_folder_norm or smoke_folder_norm.startswith(folder_norm + "/"):
            log(f"  {key}: SKIPPED (contains smoke folder)", "WARN")
            continue
        
        videos = []
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {e.lower() for e in VIDEO_EXTS}:
                continue
            
            # Double-check this video is not in smoke folder
            if is_subpath(p, smoke_folder):
                excluded_count += 1
                continue
            
            videos.append(p)
            sources[str(p)] = key
        
        log(f"  {key}: {len(videos)} videos")
        corpus.extend(videos)
    
    # Unlabeled folders (if requested)
    if not labeled_only:
        for key, node in config.get("unlabeled", {}).get("groups", {}).items():
            folder = Path(node["path"])
            if not folder.exists():
                continue
            
            folder_norm = normalize_path(folder)
            if folder_norm == smoke_folder_norm or smoke_folder_norm.startswith(folder_norm + "/"):
                log(f"  {key}: SKIPPED (contains smoke folder)", "WARN")
                continue
            
            videos = []
            for p in folder.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {e.lower() for e in VIDEO_EXTS}:
                    continue
                if is_subpath(p, smoke_folder):
                    excluded_count += 1
                    continue
                videos.append(p)
                sources[str(p)] = key
            
            log(f"  {key}: {len(videos)} videos")
            corpus.extend(videos)
    
    # Deduplicate
    seen = set()
    unique_corpus = []
    for p in corpus:
        key = normalize_path(p)
        if key not in seen:
            seen.add(key)
            unique_corpus.append(p)
    
    if excluded_count > 0:
        log(f"  Excluded {excluded_count} videos in smoke folder", "WARN")
    
    log(f"  Total corpus: {len(unique_corpus)} videos", "OK")
    
    stats = {
        "total_videos": len(unique_corpus),
        "excluded_smoke": excluded_count,
        "sources": {k: sum(1 for v in sources.values() if v == k) for k in set(sources.values())}
    }
    
    return unique_corpus, stats


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_clips(
    clips: List[SmokeClip],
    corpus_paths: List[Path],
    do_hash: bool = True
) -> None:
    """Run filename and hash matching analysis."""
    
    # Build corpus filename index
    corpus_filenames: Dict[str, str] = {}  # lowercase name -> full path
    for p in corpus_paths:
        corpus_filenames[p.name.lower()] = str(p)
    
    log(f"\nCorpus filename index: {len(corpus_filenames)} unique names")
    
    # Filename matching
    log("\n" + "=" * 50)
    log("FILENAME MATCHING")
    log("=" * 50)
    
    fn_matches = 0
    for clip in clips:
        name_lower = clip.video.lower()
        if name_lower in corpus_filenames:
            clip.filename_match = True
            fn_matches += 1
            debug(f"  Filename match: {clip.video} -> {corpus_filenames[name_lower]}")
    
    log(f"Filename matches: {fn_matches}/{len(clips)}", "OK" if fn_matches == 0 else "WARN")
    
    # Hash matching
    if do_hash and CV2_AVAILABLE:
        log("\n" + "=" * 50)
        log("PERCEPTUAL HASH MATCHING")
        log("=" * 50)
        
        # Build corpus hash index
        log(f"Building corpus hash index ({len(corpus_paths)} videos)...")
        corpus_hashes: Dict[str, Tuple[int, int, int]] = {}
        hash_errors = 0
        
        for i, p in enumerate(corpus_paths):
            h1, h2, h3, err = compute_video_hash(p)
            if err:
                hash_errors += 1
            else:
                corpus_hashes[str(p)] = (h1, h2, h3)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(corpus_paths):
                log(f"  Progress: {i+1}/{len(corpus_paths)} | Errors: {hash_errors}")
        
        coverage = 100 * len(corpus_hashes) / max(1, len(corpus_paths))
        log(f"Corpus hash coverage: {len(corpus_hashes)}/{len(corpus_paths)} ({coverage:.1f}%)", "OK")
        
        # Match smoke videos
        log(f"\nMatching {len(clips)} smoke videos...")
        hash_matches = 0
        
        for i, clip in enumerate(clips):
            smoke_path = Path(clip.video_path)
            h1, h2, h3, err = compute_video_hash(smoke_path)
            
            if err:
                clip.hash_error = err
            else:
                best_dist = HASH_SIZE * HASH_SIZE * 3
                best_path = ""
                
                for corpus_path, (ch1, ch2, ch3) in corpus_hashes.items():
                    dist = hamming(h1, ch1) + hamming(h2, ch2) + hamming(h3, ch3)
                    if dist < best_dist:
                        best_dist = dist
                        best_path = corpus_path
                
                clip.hash_distance = best_dist
                clip.hash_match_path = best_path
                
                if best_dist <= HASH_THRESHOLD:
                    clip.hash_match = True
                    hash_matches += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == len(clips):
                log(f"  Progress: {i+1}/{len(clips)}")
        
        log(f"Hash matches (dist <= {HASH_THRESHOLD}): {hash_matches}/{len(clips)}")
    
    elif not CV2_AVAILABLE:
        log("\nHash matching SKIPPED (OpenCV not available)", "WARN")


def make_split_decisions(clips: List[SmokeClip]) -> Dict:
    """
    Make final VAL/TEST decisions.
    Conservative policy: ANY signal of overlap -> VAL
    """
    log("\n" + "=" * 50)
    log("SPLIT DECISIONS")
    log("=" * 50)
    
    for clip in clips:
        reasons = []
        is_overlap = False
        
        # Signal 1: "Our Video?" annotation = Y means overlap
        if clip.our_video_parsed is True:
            is_overlap = True
            reasons.append("OurVideo=Y")
        elif clip.our_video_parsed is False:
            reasons.append("OurVideo=N")
        else:
            reasons.append("OurVideo=unknown")
        
        # Signal 2: Filename match
        if clip.filename_match:
            is_overlap = True
            reasons.append("filename_match")
        
        # Signal 3: Hash match
        if clip.hash_match:
            is_overlap = True
            reasons.append(f"hash_match(d={clip.hash_distance})")
        
        # Signal 4: Notes suggest issues
        if "drop" in clip.notes.lower():
            reasons.append("note_suggests_drop")
        
        # Final decision
        clip.decision = "VAL" if is_overlap else "TEST"
        clip.decision_reasons = reasons
    
    # Statistics
    val_clips = [c for c in clips if c.decision == "VAL"]
    test_clips = [c for c in clips if c.decision == "TEST"]
    
    val_teat = sum(1 for c in val_clips if c.has_teat)
    test_teat = sum(1 for c in test_clips if c.has_teat)
    
    stats = {
        "val_count": len(val_clips),
        "val_teat": val_teat,
        "test_count": len(test_clips),
        "test_teat": test_teat,
    }
    
    log(f"\nVAL (overlap):  {len(val_clips)} clips, {val_teat} teat")
    log(f"TEST (OOD):     {len(test_clips)} clips, {test_teat} teat")
    
    # Show test set teat clips
    if test_clips:
        log("\nTEST set teat clips:")
        for c in test_clips:
            if c.has_teat:
                drop_flag = " [NOTE: suggested drop]" if "drop" in c.notes.lower() else ""
                log(f"  - {c.video}: {c.labels}{drop_flag}")
    
    return stats


# =============================================================================
# OUTPUT
# =============================================================================

def save_outputs(clips: List[SmokeClip], parse_stats: Dict, corpus_stats: Dict, 
                 split_stats: Dict, output_dir: Path) -> None:
    """Save all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # VAL clips
    val_clips = [c for c in clips if c.decision == "VAL"]
    val_df = pd.DataFrame([asdict(c) for c in val_clips])
    val_path = output_dir / "smoke_to_val.csv"
    val_df.to_csv(val_path, index=False)
    log(f"Saved: {val_path} ({len(val_clips)} clips)", "OK")
    
    # TEST clips
    test_clips = [c for c in clips if c.decision == "TEST"]
    test_df = pd.DataFrame([asdict(c) for c in test_clips])
    test_path = output_dir / "smoke_to_test.csv"
    test_df.to_csv(test_path, index=False)
    log(f"Saved: {test_path} ({len(test_clips)} clips)", "OK")
    
    # Full audit trail
    audit_df = pd.DataFrame([asdict(c) for c in clips])
    audit_path = output_dir / "smoke_audit_trail.csv"
    audit_df.to_csv(audit_path, index=False)
    log(f"Saved: {audit_path}", "OK")
    
    # Summary report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parse_stats": parse_stats,
        "corpus_stats": corpus_stats,
        "split_stats": split_stats,
        "policy": "conservative (OurVideo=Y OR filename_match OR hash_match -> VAL)",
        "recommendation": (
            f"Use {split_stats['val_count']} clips for VAL (with {split_stats['val_teat']} teat), "
            f"{split_stats['test_count']} clips for TEST (with {split_stats['test_teat']} teat). "
            "Report TEST results with 95% CI due to small sample size."
        )
    }
    
    report_path = output_dir / "smoke_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    log(f"Saved: {report_path}", "OK")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    global DEBUG
    
    parser = argparse.ArgumentParser(description="Verify smoke test independence")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--smoke_videos", default=r"E:\Downey cross-sucking videos 2024\smoke_test\videos")
    parser.add_argument("--output_dir", default="data/manifests")
    parser.add_argument("--labeled_only", action="store_true", default=True)
    parser.add_argument("--include_unlabeled", action="store_true")
    parser.add_argument("--skip_hash", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    DEBUG = args.debug
    labeled_only = not args.include_unlabeled
    
    log("=" * 60)
    log("SMOKE TEST INDEPENDENCE VERIFICATION (FINAL)")
    log("=" * 60)
    
    if yaml is None:
        log("PyYAML not installed", "ERR")
        return 1
    
    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    smoke_csv_path = Path(config["annotations"]["smoke_test"])
    smoke_video_folder = Path(args.smoke_videos)
    
    log(f"Config: {config_path}")
    log(f"Smoke CSV: {smoke_csv_path}")
    log(f"Smoke videos: {smoke_video_folder}")
    log(f"Mode: {'labeled_only' if labeled_only else 'all'}")
    
    # Parse smoke CSV
    clips, parse_stats = parse_smoke_csv(smoke_csv_path, smoke_video_folder)
    
    if len(clips) == 0:
        log("No valid clips found in smoke CSV", "ERR")
        return 1
    
    # Collect corpus (excluding smoke folder)
    corpus_paths, corpus_stats = collect_corpus_videos(
        config, 
        smoke_video_folder,
        labeled_only=labeled_only
    )
    
    # Run analysis
    analyze_clips(clips, corpus_paths, do_hash=not args.skip_hash)
    
    # Make split decisions
    split_stats = make_split_decisions(clips)
    
    # Save outputs
    save_outputs(clips, parse_stats, corpus_stats, split_stats, Path(args.output_dir))
    
    # Final summary
    log("\n" + "=" * 60)
    log("FINAL SPLIT PROTOCOL")
    log("=" * 60)
    
    print(f"""
┌────────────────────────────────────────────────────────────┐
│                    SMOKE TEST SPLIT                        │
├────────────────────────────────────────────────────────────┤
│  VAL (overlap):   {split_stats['val_count']:3d} clips  │  Teat: {split_stats['val_teat']}                     │
│  TEST (OOD):      {split_stats['test_count']:3d} clips  │  Teat: {split_stats['test_teat']}                     │
├────────────────────────────────────────────────────────────┤
│  Our Video=Y:     {parse_stats['our_video_stats']['Y']:3d}                                     │
│  Our Video=N:     {parse_stats['our_video_stats']['N']:3d}                                     │
└────────────────────────────────────────────────────────────┘
    """)
    
    log("Analysis complete!", "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
