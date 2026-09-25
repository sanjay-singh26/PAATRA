#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/artifacts/tokenizers}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/genuine}"
mkdir -p "$RESULTS_DIR"
for family in bpe unigram; do
  for vocab in 10000 20000 50000; do
    name="${family}_${vocab}"
    "$PYTHON" "$ROOT/src/train_genuine_tokenizer.py" \
      --tokenizer "$TOKENIZER_DIR/$name.json" \
      --output-dir "$RESULTS_DIR/$name" --seed 42
  done
done
