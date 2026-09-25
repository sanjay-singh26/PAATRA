#!/usr/bin/env bash
# Dense vocabulary grid (L=8) and depth-matched K20K control (L=16).
# NOTE: the reported runs used a vocabulary build with 14 (not 2) non-frequency-ranked
# tokens (e.g. K=20,015 instead of 20,002); a fresh build may differ by a few tokens.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
GRID_ART="$ROOT/artifacts_grid"; DEPTH_ART="$ROOT/artifacts_depth16"
"$PY" "$ROOT/src/setup.py" --output-dir "$GRID_ART" --vocab-sizes 5001 15001 30001 75001
out="$ROOT/results/grid"; mkdir -p "$out"
for name in K5K K15K K30K K75K; do
  "$PY" "$ROOT/src/train.py" --name "$name" --artifacts-dir "$GRID_ART" --output-dir "$out" --seed 42
done
"$PY" "$ROOT/src/evaluate.py" --artifacts-dir "$GRID_ART" --results-dir "$out" --names K5K K15K K30K K75K
"$PY" "$ROOT/src/setup.py" --output-dir "$DEPTH_ART" --n-layer 16 --vocab-sizes 20001
out="$ROOT/results/depth_control"; mkdir -p "$out"
"$PY" "$ROOT/src/train.py" --name K20K --artifacts-dir "$DEPTH_ART" --output-dir "$out" --seed 42
"$PY" "$ROOT/src/evaluate.py" --artifacts-dir "$DEPTH_ART" --results-dir "$out" --names K20K
