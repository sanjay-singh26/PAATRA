#!/usr/bin/env bash
# Seeds 1 and 2 for the four parameter-matched configurations (Table 5, per-seed table).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT/artifacts}"
for seed in 1 2; do
  out="$ROOT/results/replications/seed$seed"
  mkdir -p "$out"
  for name in Full K50K K20K K10K; do
    "${PYTHON:-python}" "$ROOT/src/train.py" --name "$name" \
      --artifacts-dir "$ARTIFACTS_DIR" --output-dir "$out" --seed "$seed"
  done
  "${PYTHON:-python}" "$ROOT/src/evaluate.py" --artifacts-dir "$ARTIFACTS_DIR" --results-dir "$out"
done
