"""
PAATRA ICLR — Setup Script
Run once before any training. Builds vocabularies, tokenizes corpus, and
computes parameter-matched architectures for every config in the sweep.

Usage:
    python setup.py --output-dir /path/to/artifacts

    # Swap teacher or dataset:
    python setup.py --output-dir /artifacts \
        --teacher-model Qwen/Qwen2.5-1.5B \
        --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \
        --target-params 200000000

Runtime: ~5–10 min on any machine (CPU only, no GPU needed).
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PAATRA ICLR Setup")

    # Paths
    p.add_argument("--output-dir",    required=True,
                   help="Directory to save artifacts (vocabs, chunks, meta).")

    # Teacher
    p.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B",
                   help="HuggingFace model ID for the frozen teacher.")

    # Dataset
    p.add_argument("--dataset-name",   default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    p.add_argument("--dataset-split",  default="train")
    p.add_argument("--num-docs",       type=int, default=80_000,
                   help="Number of training documents to use.")
    p.add_argument("--text-field",     default="text",
                   help="Column name that holds the document text.")
    p.add_argument("--min-text-len",   type=int, default=20,
                   help="Skip documents shorter than this (characters).")

    # Student architecture (fixed across configs in the main sweep)
    p.add_argument("--seq-len",        type=int, default=512)
    p.add_argument("--n-layer",        type=int, default=8,
                   help="Transformer depth — kept constant across all configs.")
    p.add_argument("--n-head",         type=int, default=8,
                   help="Number of attention heads — kept constant.")
    p.add_argument("--target-params",  type=int, default=64_000_000,
                   help="Shared parameter budget for all parameter-matched configs.")
    p.add_argument("--param-tolerance",type=float, default=0.01,
                   help="Max allowed fractional deviation from target (default 1%%).")

    # Vocabulary sweep
    p.add_argument("--vocab-sizes", nargs="+", type=int,
                   default=[50_001, 20_001, 10_001],
                   help="Subset vocabulary sizes (K) to build. Full vocab is always added.")

    # Training hyperparams (stored in meta for training scripts to read)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--num-steps",     type=int,   default=15_000)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup-steps",  type=int,   default=500)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--kd-temp",       type=float, default=2.0)
    p.add_argument("--kd-alpha",      type=float, default=0.7)
    p.add_argument("--log-interval",  type=int,   default=1_000)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Parameter matching ────────────────────────────────────────────────────────

def count_params_exact(K, d, L, n_pos):
    """
    Exact GPT2LMHeadModel parameter count with tied lm_head (no double-count).
    Per layer: 12d² (weights) + 13d (biases + layernorms)
    Plus: K*d (wte=lm_head, tied), n_pos*d (wpe), 2d (final layernorm).
    """
    return K * d + n_pos * d + L * (12 * d**2 + 13 * d) + 2 * d

def find_matched_d(target_N, K, L, n_pos, n_head, tolerance):
    """
    Solve the quadratic 12L·d² + (K + n_pos + 13L + 2)·d − N = 0 for d,
    then snap to the nearest multiple of n_head and verify tolerance.
    """
    a = 12 * L
    b = K + n_pos + 13 * L + 2
    c = -target_N
    d_float = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)

    candidates = [
        max(n_head, int(d_float // n_head) * n_head),
        (int(d_float // n_head) + 1) * n_head,
    ]
    best_d, best_N, best_err = None, None, float("inf")
    for d in candidates:
        N   = count_params_exact(K, d, L, n_pos)
        err = abs(N - target_N) / target_N
        if err < best_err:
            best_d, best_N, best_err = d, N, err

    if best_err > tolerance:
        raise ValueError(
            f"K={K:,}: best d={best_d} gives {best_err*100:.2f}% deviation "
            f"(tolerance={tolerance*100:.1f}%). Try adjusting --target-params."
        )
    return best_d, best_N


# ── Vocabulary construction ───────────────────────────────────────────────────

def build_subset_vocab(K, token_counts, special_ids, teacher_vocab_size):
    top_k = {tid for tid, _ in token_counts.most_common(K)}
    keep  = sorted(top_k | special_ids)
    assert all(0 <= t < teacher_vocab_size for t in keep)
    s2t = keep                                          # list: student_id → teacher_id
    t2s = {tid: sid for sid, tid in enumerate(keep)}   # dict: teacher_id → student_id
    return s2t, t2s

def build_full_vocab(teacher_vocab_size):
    s2t = list(range(teacher_vocab_size))
    t2s = {i: i for i in range(teacher_vocab_size)}
    return s2t, t2s


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "students").mkdir(exist_ok=True)

    print("=" * 70)
    print("  PAATRA ICLR — Setup")
    print("=" * 70)
    print(f"  Teacher       : {args.teacher_model}")
    print(f"  Dataset       : {args.dataset_name} / {args.dataset_config}")
    print(f"  Docs          : {args.num_docs:,}")
    print(f"  Target budget : {args.target_params/1e6:.0f}M params")
    print(f"  Architecture  : L={args.n_layer}, H={args.n_head}, seq={args.seq_len}")
    print(f"  Output        : {out}")
    print()

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("[1/4] Loading teacher tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    TEACHER_VOCAB_SIZE = len(tokenizer)
    print(f"      Teacher vocab size: {TEACHER_VOCAB_SIZE:,}")

    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(int(tid))
    if getattr(tokenizer, "additional_special_tokens_ids", None):
        special_ids.update(int(t) for t in tokenizer.additional_special_tokens_ids)
    print(f"      Special token IDs  : {sorted(special_ids)}")

    # ── 2. Corpus ─────────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading {args.dataset_name} ({args.dataset_config}, {args.dataset_split})...")
    from datasets import load_dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    texts   = [r[args.text_field] for r in dataset
               if len(r[args.text_field].strip()) >= args.min_text_len]
    texts   = texts[:args.num_docs]
    print(f"      Documents after filter : {len(texts):,}")

    print(f"      Counting token frequencies on {len(texts):,} documents...")
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenizer.encode(text, add_special_tokens=False))
    total_tokens = sum(token_counts.values())
    print(f"      Total tokens  : {total_tokens:,}")
    print(f"      Unique tokens : {len(token_counts):,}")

    # ── 3. Vocabularies & parameter-matched architectures ────────────────────
    # Vocabs are built *before* architectures are solved: a requested subset
    # size K can exceed the corpus's unique-token count, in which case
    # Counter.most_common(K) silently returns fewer than K tokens. Solving the
    # architecture against the requested K in that case understates the real
    # embedding size and produces a model that misses the parameter-tolerance
    # check at train time. Solving against the actual built vocab length
    # avoids that class of bug entirely.
    print(f"\n[3/4] Building vocabularies and matching architectures "
          f"(target={args.target_params/1e6:.0f}M)...")

    # Name → requested K. Full always included; user can pass --vocab-sizes.
    requested = {"Full": TEACHER_VOCAB_SIZE}
    for K in args.vocab_sizes:
        requested[f"K{K//1000}K"] = K
    requested["Ablation_20K"] = 20_001   # 20K vocab + Full's d (intentionally smaller total budget)

    vocabs, arch_table = {}, {}
    print(f"\n  {'Config':<22} {'K (req)':>9}  {'K (actual)':>10}  {'d':>5}  "
          f"{'Total params':>14}  {'Error':>10}  {'Emb%':>6}  {'Trans%':>7}")
    print("  " + "─" * 90)

    for cname, K in requested.items():
        if cname == "Full":
            s2t, t2s = build_full_vocab(TEACHER_VOCAB_SIZE)
            cov = 100.0
        else:
            s2t, t2s = build_subset_vocab(K, token_counts, special_ids, TEACHER_VOCAB_SIZE)
            covered  = sum(c for tid, c in token_counts.items() if tid in t2s)
            cov      = covered / total_tokens * 100
        actual_K = len(s2t)

        if cname == "Ablation_20K":
            d = arch_table["Full"]["d"]   # reuse Full's transformer width, not solved
            N = count_params_exact(actual_K, d, args.n_layer, args.seq_len)
            err_label = "(ablation)"
        else:
            d, N = find_matched_d(
                args.target_params, actual_K, args.n_layer, args.seq_len,
                args.n_head, args.param_tolerance,
            )
            err_label = f"{(N - args.target_params) / args.target_params * 100:+.2f}%"

        emb_N = actual_K * d
        tr_N  = args.n_layer * (12 * d**2 + 13 * d)
        arch_table[cname] = {"K": actual_K, "d": d, "N": N}
        vocabs[cname] = {
            "s2t": s2t, "t2s": t2s,
            "config": {
                "n_embd": d, "n_layer": args.n_layer,
                "n_head": args.n_head, "vocab_K": actual_K,
            },
            "total_params": N,
            "emb_params":   emb_N,
        }
        note = "" if actual_K == K else f"  (capped from {K:,} — corpus has only {len(token_counts):,} unique tokens)"
        print(f"  {cname:<22} {K:>9,}  {actual_K:>10,}  {d:>5}  {N:>14,}  "
              f"{err_label:>10}  {emb_N/N*100:>5.1f}%  {tr_N/N*100:>6.1f}%{note}")
        print(f"  {'':<22} coverage={cov:.2f}%")
    print()

    # ── 5. Tokenize corpus into chunks ────────────────────────────────────────
    print(f"\n[4/4] Tokenizing {len(texts):,} docs into {args.seq_len}-token chunks...")
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    n_chunks = len(all_ids) // args.seq_len
    chunks   = torch.tensor(
        [all_ids[i * args.seq_len : (i + 1) * args.seq_len] for i in range(n_chunks)],
        dtype=torch.long,
    )
    print(f"  Total tokens  : {len(all_ids):,}")
    print(f"  Chunks        : {len(chunks):,} × {args.seq_len} tokens")
    print(f"  Tensor size   : {chunks.element_size() * chunks.nelement() / 1e6:.1f} MB")
    eff_epochs = args.num_steps * args.batch_size / len(chunks)
    print(f"  Effective epochs at {args.num_steps} steps, batch {args.batch_size}: {eff_epochs:.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    meta = {
        "teacher_model":    args.teacher_model,
        "dataset_name":     args.dataset_name,
        "dataset_config":   args.dataset_config,
        "dataset_split":    args.dataset_split,
        "num_docs":         len(texts),
        "text_field":       args.text_field,
        "seq_len":          args.seq_len,
        "n_layer":          args.n_layer,
        "n_head":           args.n_head,
        "target_params":    args.target_params,
        "teacher_vocab_size": TEACHER_VOCAB_SIZE,
        "special_ids":      sorted(special_ids),
        "batch_size":       args.batch_size,
        "num_steps":        args.num_steps,
        "lr":               args.lr,
        "warmup_steps":     args.warmup_steps,
        "weight_decay":     args.weight_decay,
        "grad_clip":        args.grad_clip,
        "kd_temp":          args.kd_temp,
        "kd_alpha":         args.kd_alpha,
        "log_interval":     args.log_interval,
        "seed":             args.seed,
        "configs":          list(arch_table.keys()),
    }

    meta_path   = out / "meta.json"
    chunks_path = out / "chunks.pt"
    vocabs_path = out / "vocabs.pt"

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    torch.save(chunks, chunks_path)
    torch.save(vocabs, vocabs_path)

    print(f"\n✓ Saved:")
    for path in [meta_path, chunks_path, vocabs_path]:
        print(f"  {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    print(f"\n✓ Setup complete. Available configs to train:")
    for cname in vocabs:
        d   = vocabs[cname]["config"]["n_embd"]
        N   = vocabs[cname]["total_params"]
        K   = len(vocabs[cname]["s2t"])
        print(f"  python train.py --name {cname} --artifacts-dir {out} --output-dir {out}")
    print(f"\n  Or submit all at once:")
    print(f"  ARTIFACTS_DIR={out} RESULTS_DIR=<results_dir> ./submit_cluster.sh")


if __name__ == "__main__":
    main()
