#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/src/aggregate_genuine_results.py" --results-root "${RESULTS_ROOT:-$ROOT/results}"
