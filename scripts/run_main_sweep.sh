#!/usr/bin/env bash
# Primary sweep (seed 42): Full, K50K, K20K, K10K and the pruning-only Ablation_20K,
# followed by BPB evaluation on the full WikiText-103 test split.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT/artifacts}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/main}"
SEED="${SEED:-42}"
if [[ ! -f "$ARTIFACTS_DIR/meta.json" ]]; then
  "$PYTHON" "$ROOT/src/setup.py" --output-dir "$ARTIFACTS_DIR"
fi
mkdir -p "$RESULTS_DIR"
# All configs share one output directory: evaluate.py reads $RESULTS_DIR/students/*.pt.
for name in Full K50K K20K K10K Ablation_20K; do
  "$PYTHON" "$ROOT/src/train.py" --name "$name" \
    --artifacts-dir "$ARTIFACTS_DIR" --output-dir "$RESULTS_DIR" --seed "$SEED"
done
"$PYTHON" "$ROOT/src/evaluate.py" --artifacts-dir "$ARTIFACTS_DIR" --results-dir "$RESULTS_DIR"
