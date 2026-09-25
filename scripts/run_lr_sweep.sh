#!/usr/bin/env bash
# Peak-learning-rate sweep (seed 42) for Full, K20K and K10K (learning-rate table).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT/artifacts}"
for name in Full K20K K10K; do
  for pair in 1em4:1e-4 3em4:3e-4 6em4:6e-4 1em3:1e-3; do
    tag="${pair%%:*}"; lr="${pair##*:}"
    out="$ROOT/results/lr_sweep/$name/lr_$tag"
    mkdir -p "$out"
    "${PYTHON:-python}" "$ROOT/src/train.py" --name "$name" --lr "$lr" \
      --artifacts-dir "$ARTIFACTS_DIR" --output-dir "$out" --seed 42
    "${PYTHON:-python}" "$ROOT/src/evaluate.py" --artifacts-dir "$ARTIFACTS_DIR" \
      --results-dir "$out" --names "$name"
  done
done
