#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/artifacts/tokenizers}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
for family in bpe unigram; do
  for vocab in 10000 20000 50000; do
    name="${family}_${vocab}"
    "$PYTHON" "$ROOT/src/evaluate_genuine_tokenizer.py" \
      --checkpoint "$RESULTS_DIR/genuine/$name/students/$name.pt" \
      --tokenizer "$TOKENIZER_DIR/$name.json" \
      --output "$RESULTS_DIR/genuine/$name/evaluation/test.json"
  done
done
"$ROOT/scripts/aggregate_genuine.sh"
