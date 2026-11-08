#!/usr/bin/env bash
# Train SCRFD-500M-bnkps four times, on 400 / 800 / 1600 / 1988 labelled eye pairs, to find
# out how many the landmark head actually needs. Every run sees every image and every box;
# only the number of faces whose eye points are supervised changes (see subsample_labels.zig).
#
# Natively, with train_scrfd.zig's curve recipe (MMDetection's scrfd_fcs_500m.py settings) on
# the host's NVIDIA GPU: about 3.8 hours per run on an RTX 4090, so an afternoon and an evening.
# Run from the repository root with Zig 0.17 on PATH (Git Bash on Windows, or Linux):
#
#   ./tools/dataset/run_curve.sh /c/Users/grego/Downloads/face-data/openimages work_dirs
#
# Each size writes its own directory and log, so checkpoints never collide.
set -u

DATA=${1:?usage: run_curve.sh <openimages_dir> [out_root]}
OUT=${2:-work_dirs}
mkdir -p "$OUT"

for N in 400 800 1600 1988; do
    echo "=== size ${N}: starting $(date -Is)"
    zig run -O ReleaseFast -lc tools/dataset/train_scrfd.zig tools/dataset/vendor/stb.c -- \
        "$DATA" "${DATA}/labelv2_train_${N}.txt" --recipe curve --out "${OUT}/curve_${N}" \
        > "${OUT}/curve_${N}.log" 2>&1
    status=$?
    last=$(grep '^epoch' "${OUT}/curve_${N}.log" | tail -1)
    echo "=== size ${N}: finished $(date -Is), exit ${status}"
    echo "    last iteration: ${last:-none logged}"
done

echo "=== all sizes done $(date -Is)"
