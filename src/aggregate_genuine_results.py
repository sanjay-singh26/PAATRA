#!/usr/bin/env python3
"""Aggregate validated genuine-tokenizer training/evaluation artifacts."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--results-root", type=Path, required=True,
                    help="Root containing genuine/<name>/ and tokenizers/")
parser.add_argument("--output", type=Path, default=None,
                    help="Output CSV (default: <results-root>/reproduced/genuine_tokenizer_results.csv; "
                         "the bundled reference CSV is never overwritten)")
args = parser.parse_args()
BASE = args.results_root
OUT = args.output or BASE / "reproduced" / "genuine_tokenizer_results.csv"
rows = []
for kind in ("bpe", "unigram"):
    for vocab in (10000, 20000, 50000):
        name = f"{kind}_{vocab}"
        ckpt = BASE / "genuine" / name / "students" / f"{name}.pt"
        evaluation = BASE / "genuine" / name / "evaluation" / "test.json"
        if not ckpt.exists() or not evaluation.exists():
            continue
        import torch
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        row = json.loads(evaluation.read_text())
        row.update({
            "name": name,
            "tokenizer_family": kind,
            "requested_vocab": vocab,
            "steps": bundle["num_steps"],
            "train_loss": bundle["losses"][-1]["loss"],
            "checkpoint": str(ckpt.relative_to(BASE.parent)) if ckpt.is_relative_to(BASE.parent) else str(ckpt),
        })
        tokenizer_path = Path(row.get("tokenizer", ""))
        if tokenizer_path.is_absolute() and tokenizer_path.is_relative_to(BASE.parent):
            row["tokenizer"] = str(tokenizer_path.relative_to(BASE.parent))
        rows.append(row)

fields = ["name", "tokenizer_family", "requested_vocab", "vocab_size", "params",
          "steps", "train_loss", "test_tokens", "test_bytes", "token_nll", "bpb",
          "checkpoint", "tokenizer"]
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {OUT}")
if len(rows) != 6:
    raise SystemExit("expected six completed genuine tokenizer rows")
