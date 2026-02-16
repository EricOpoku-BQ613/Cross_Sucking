#!/bin/bash
# resize_clips_feral.sh
# Resize existing clips_v4 (4K) to 256x256 for FERAL training.
# Runs as a SLURM CPU job on ISAAC — no GPU needed.
#
# What it does:
#   - Center-crops + resizes each clip from 4K to 256x256
#   - Outputs to /lustre/isaac24/scratch/eopoku2/clips_feral/
#   - Skips already-processed files (safe to rerun)
#   - Estimates ~30-60 min for 1897 clips with 16 CPUs
#
# Submit: sbatch scripts/resize_clips_feral.sh
# Or run interactively: bash scripts/resize_clips_feral.sh

#SBATCH --job-name=resize_feral
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/isaac24/scratch/eopoku2/runs/resize_feral_%j.log
#SBATCH --error=/lustre/isaac24/scratch/eopoku2/runs/resize_feral_%j.log

set -e

SRC="/lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4"
DST="/lustre/isaac24/scratch/eopoku2/clips_feral"
NCPUS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$DST"
mkdir -p /lustre/isaac24/scratch/eopoku2/runs

echo "Source:   $SRC"
echo "Dest:     $DST"
echo "CPUs:     $NCPUS"
echo "Start:    $(date)"
echo ""

# Count files
total=$(ls "$SRC"/*.mp4 2>/dev/null | wc -l)
echo "Total clips to resize: $total"
echo ""

# Use GNU parallel if available, otherwise sequential
if command -v parallel &>/dev/null; then
    echo "Using GNU parallel ($NCPUS jobs)"
    ls "$SRC"/*.mp4 | parallel -j "$NCPUS" '
        SRC_FILE={}
        FNAME=$(basename "$SRC_FILE")
        DST_FILE="'"$DST"'/$FNAME"
        if [ -f "$DST_FILE" ] && [ -s "$DST_FILE" ]; then
            echo "SKIP: $FNAME"
        else
            ffmpeg -y -i "$SRC_FILE" \
                -vf "scale=256:256:force_original_aspect_ratio=increase,crop=256:256" \
                -c:v libx264 -crf 23 -preset fast -an \
                "$DST_FILE" 2>/dev/null \
            && echo "OK: $FNAME" \
            || echo "FAIL: $FNAME"
        fi
    '
else
    echo "GNU parallel not found — using Python multiprocessing"
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate cross_sucking

    python - <<'PYEOF'
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

src = Path("/lustre/isaac24/scratch/eopoku2/clips_v4/clips_v4")
dst = Path("/lustre/isaac24/scratch/eopoku2/clips_feral")
dst.mkdir(parents=True, exist_ok=True)

clips = sorted(src.glob("*.mp4"))
print(f"Processing {len(clips)} clips...")

def resize_clip(clip_path):
    out = dst / clip_path.name
    if out.exists() and out.stat().st_size > 0:
        return f"SKIP: {clip_path.name}"
    cmd = [
        "ffmpeg", "-y", "-i", str(clip_path),
        "-vf", "scale=256:256:force_original_aspect_ratio=increase,crop=256:256",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an",
        str(out)
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode == 0:
        return f"OK:   {clip_path.name}"
    else:
        return f"FAIL: {clip_path.name}: {result.stderr[-200:].decode(errors='ignore')}"

ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
ok, skip, fail = 0, 0, 0
with ThreadPoolExecutor(max_workers=ncpus) as ex:
    futures = {ex.submit(resize_clip, c): c for c in clips}
    for i, f in enumerate(as_completed(futures), 1):
        msg = f.result()
        if msg.startswith("OK"):   ok += 1
        elif msg.startswith("SK"): skip += 1
        else:                      fail += 1; print(msg)
        if i % 100 == 0 or i == len(clips):
            print(f"  Progress: {i}/{len(clips)} | ok={ok} skip={skip} fail={fail}")

print(f"\nDone. OK={ok}, Skipped={skip}, Failed={fail}")
PYEOF
fi

echo ""
echo "Output clips: $(ls $DST/*.mp4 2>/dev/null | wc -l)"
echo "Done: $(date)"
