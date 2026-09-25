"""
PAATRA ICLR — Evaluation Script
Computes BPB, BPC, token PPL, and tokenization statistics for all trained
checkpoints. Every model is evaluated on exactly the same raw bytes.

Usage:
    python evaluate.py --artifacts-dir /path/to/artifacts --results-dir /path/to/results

    # Evaluate only specific configs:
    python evaluate.py --artifacts-dir /artifacts --results-dir /results \
        --names Full_151K B_20K_paatra

    # Evaluate on full WikiText-103 test split (default) or a subset:
    python evaluate.py ... --max-docs 500
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="PAATRA ICLR Evaluation")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--results-dir",   required=True)
    p.add_argument("--names",         nargs="*", default=None,
                   help="Subset of config names to evaluate. Default: all found.")
    p.add_argument("--eval-dataset",  default=None,
                   help="Override eval dataset (HuggingFace ID). Default: reads from meta.json.")
    p.add_argument("--eval-config",   default=None,
                   help="Override eval dataset config.")
    p.add_argument("--text-field",    default=None,
                   help="Override text field in evaluation dataset.")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow datasets that require remote loading code.")
    p.add_argument("--eval-split",    default="test")
    p.add_argument("--max-docs",      type=int, default=None,
                   help="Limit number of test documents. Default: full test split.")
    p.add_argument("--stride",        type=int, default=512)
    return p.parse_args()


def record_text(row, field):
    value = row[field]
    if isinstance(value, str):
        return value
    if field == "MedlineCitation" and isinstance(value, dict):
        article = value.get("Article", {})
        abstract = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract, str):
            return abstract
        if isinstance(abstract, list):
            return " ".join(str(x) for x in abstract)
    return ""


@torch.inference_mode()
def evaluate_bundle(name, bundle_path, teacher_tokenizer, raw_text,
                    raw_bytes, raw_chars, device, max_length=512, stride=512):
    print(f"\n── {name} {'─'*(50 - len(name))}")
    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")

    cfg          = bundle["config"]
    vocab_size   = bundle["vocab_size"]
    s2t          = bundle["s2t"]
    t2s          = bundle["t2s"]
    total_params = bundle["total_params"]
    emb_params   = bundle["emb_params"]

    # Reconstruct model
    model_cfg = GPT2Config(
        vocab_size          = vocab_size,
        n_embd              = cfg["n_embd"],
        n_layer             = cfg["n_layer"],
        n_head              = cfg["n_head"],
        n_inner             = 4 * cfg["n_embd"],
        activation_function = "gelu_new",
        n_positions         = bundle.get("seq_len", 512),
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=None, eos_token_id=None,
    )
    model = GPT2LMHeadModel(model_cfg)
    model.load_state_dict(bundle["state_dict"])
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    # Encode: teacher tokenizer → remap through t2s
    teacher_ids = teacher_tokenizer.encode(raw_text, add_special_tokens=False)
    student_ids = [t2s.get(tid, 0) for tid in teacher_ids]
    input_ids   = torch.tensor([student_ids], dtype=torch.long)
    seq_len     = input_ids.size(1)

    tokens_per_byte = seq_len / raw_bytes
    tokens_per_char = seq_len / raw_chars

    total_nll          = 0.0
    total_scored_tokens = 0
    previous_end        = 0

    for begin in range(0, seq_len, stride):
        end           = min(begin + max_length, seq_len)
        target_length = end - previous_end

        window = input_ids[:, begin:end].to(device)
        labels = window.clone()
        if target_length < labels.size(1):
            labels[:, :-target_length] = -100

        outputs     = model(input_ids=window, labels=labels)
        valid_count = labels[:, 1:].ne(-100).sum().item()
        if valid_count > 0:
            total_nll           += outputs.loss.float().item() * valid_count
            total_scored_tokens += valid_count

        previous_end = end
        if end == seq_len:
            break

    assert total_scored_tokens > 0, f"{name}: no tokens scored"

    mean_nll         = total_nll / total_scored_tokens
    token_perplexity = math.exp(mean_nll)
    bits_per_byte    = total_nll / (raw_bytes * math.log(2))
    bits_per_char    = total_nll / (raw_chars * math.log(2))

    # Expansion ratio: tokens this model uses vs full-vocab model (set in main)
    print(f"  Total params   : {total_params:,}  (emb {emb_params/total_params*100:.1f}%  "
          f"trans {(total_params-emb_params)/total_params*100:.1f}%)")
    print(f"  Vocab size     : {vocab_size:,}")
    print(f"  Tokens produced: {seq_len:,}  ({tokens_per_byte:.4f} tok/byte)")
    print(f"  Token PPL      : {token_perplexity:.3f}  (NOT comparable across vocabs)")
    print(f"  Bits/byte (BPB): {bits_per_byte:.4f}  ← primary metric")
    print(f"  Bits/char (BPC): {bits_per_char:.4f}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "name":             name,
        "vocab_size":       vocab_size,
        "total_params":     total_params,
        "emb_params":       emb_params,
        "trans_params":     total_params - emb_params,
        "emb_pct":          emb_params / total_params * 100,
        "trans_pct":        (total_params - emb_params) / total_params * 100,
        "n_embd":           cfg["n_embd"],
        "n_layer":          cfg["n_layer"],
        "token_perplexity": token_perplexity,
        "bits_per_byte":    bits_per_byte,
        "bits_per_char":    bits_per_char,
        "tokens_per_byte":  tokens_per_byte,
        "tokens_per_char":  tokens_per_char,
        "scored_tokens":    total_scored_tokens,
        "raw_bytes":        raw_bytes,
        "raw_chars":        raw_chars,
    }


def main():
    args   = parse_args()
    device = get_device()
    arts   = Path(args.artifacts_dir)
    out    = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(arts / "meta.json") as f:
        meta = json.load(f)

    TEACHER_ID    = meta["teacher_model"]
    eval_dataset  = args.eval_dataset  or meta["dataset_name"]
    eval_config   = args.eval_config   or meta["dataset_config"]
    text_field    = args.text_field    or meta["text_field"]

    # ── Load raw evaluation text ───────────────────────────────────────────────
    print(f"Loading eval split: {eval_dataset} / {eval_config} / {args.eval_split}")
    dataset = load_dataset(eval_dataset, eval_config, split=args.eval_split,
                           trust_remote_code=args.trust_remote_code)
    texts   = [record_text(r, text_field) for r in dataset]
    texts   = [t for t in texts if t.strip()]
    if args.max_docs:
        texts = texts[:args.max_docs]

    raw_text  = "\n\n".join(texts)
    raw_bytes = len(raw_text.encode("utf-8"))
    raw_chars = len(raw_text)
    print(f"Documents : {len(texts):,}")
    print(f"Raw bytes : {raw_bytes:,}")
    print(f"Raw chars : {raw_chars:,}")

    # ── Load teacher tokenizer once ────────────────────────────────────────────
    print(f"\nLoading teacher tokenizer: {TEACHER_ID}")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_ID)

    # ── Discover checkpoints ───────────────────────────────────────────────────
    student_dir = out / "students"
    if not student_dir.exists():
        print(f"No student checkpoints found in {student_dir}")
        sys.exit(1)

    available = [p.stem for p in student_dir.glob("*.pt")
                 if not p.stem.endswith("_latest")]
    names_to_eval = args.names if args.names else available
    names_to_eval = [n for n in names_to_eval if (student_dir / f"{n}.pt").exists()]

    if not names_to_eval:
        print(f"No matching checkpoints found. Available: {available}")
        sys.exit(1)

    print(f"\nEvaluating {len(names_to_eval)} configs: {names_to_eval}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    results = []
    for name in names_to_eval:
        result = evaluate_bundle(
            name,
            student_dir / f"{name}.pt",
            teacher_tok, raw_text, raw_bytes, raw_chars,
            device,
            max_length=args.stride,
            stride=args.stride,
        )
        results.append(result)

    df = pd.DataFrame(results).sort_values("bits_per_byte")

    # ── Compute tokenization expansion ratio (vs Full vocab model) ────────────
    full_row = df[df["name"].str.startswith("Full")]
    if not full_row.empty:
        full_tok_per_byte = full_row.iloc[0]["tokens_per_byte"]
        df["expansion_ratio"] = df["tokens_per_byte"] / full_tok_per_byte
    else:
        df["expansion_ratio"] = float("nan")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert df["raw_bytes"].nunique() == 1
    assert df["raw_chars"].nunique() == 1
    assert np.isfinite(df["bits_per_byte"]).all()
    print("\n✓ Sanity checks passed.")

    # ── Save ──────────────────────────────────────────────────────────────────
    csv_path = out / "paatra_iclr_results.csv"
    df.to_csv(csv_path, index=False)

    # ── Print main table ───────────────────────────────────────────────────────
    print("\n" + "═" * 95)
    print("PAATRA ICLR — MAIN RESULTS")
    print("═" * 95)
    print(f"  {'Config':<25} {'Vocab':>8}  {'Params':>10}  {'Emb%':>6}  "
          f"{'BPB↓':>8}  {'BPC↓':>8}  {'Tok/byte':>9}  {'Expansion':>10}")
    print("  " + "─" * 90)
    for _, row in df.iterrows():
        exp = f"{row['expansion_ratio']:.3f}×" if not math.isnan(row.get("expansion_ratio", float("nan"))) else "—"
        print(f"  {row['name']:<25} {row['vocab_size']:>8,}  {row['total_params']:>10,}  "
              f"{row['emb_pct']:>5.1f}%  {row['bits_per_byte']:>8.4f}  "
              f"{row['bits_per_char']:>8.4f}  {row['tokens_per_byte']:>9.4f}  {exp:>10}")
    print("═" * 95)
    print(f"BPB = bits per byte (lower is better, comparable across tokenizers).")
    print(f"Expansion ratio = tokens used / tokens used by Full vocab model (same text).")
    print(f"\nSaved: {csv_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sweep_df = df[~df["name"].str.startswith("Ablation")].sort_values("vocab_size")

        # Figure 1: BPB vs vocab size
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep_df["vocab_size"], sweep_df["bits_per_byte"],
                marker="o", linewidth=2, markersize=8, color="#d62728")
        for _, row in sweep_df.iterrows():
            ax.annotate(
                f"{row['name']}\n{row['bits_per_byte']:.4f}",
                xy=(row["vocab_size"], row["bits_per_byte"]),
                xytext=(8, 4), textcoords="offset points", fontsize=8,
            )
        ax.set_xlabel("Vocabulary size K")
        ax.set_ylabel("Bits per byte (BPB) ↓")
        ax.set_title("PAATRA: Fixed-Budget (~64M) Vocabulary Sweep\nWikiText-103 test")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1_path = out / "fig_bpb_vs_vocab.pdf"
        plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
        plt.savefig(str(fig1_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Figure 2: Training curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i, name in enumerate(names_to_eval):
            bundle_path = out / "students" / f"{name}.pt"
            if not bundle_path.exists():
                continue
            b = torch.load(bundle_path, weights_only=False, map_location="cpu")
            losses = b.get("losses", [])
            if not losses:
                continue
            steps = [e["step"] for e in losses]
            col   = colors[i % len(colors)]
            N     = b["total_params"]
            label = f"{name} ({N/1e6:.1f}M)"
            for ax, key, title in zip(axes, ["loss","kd","ce"], ["Total","KD","CE"]):
                ax.plot(steps, [e[key] for e in losses], label=label,
                        color=col, marker="o", ms=3, linewidth=1.5)
                ax.set_title(f"{title} Loss"); ax.set_xlabel("Step")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        plt.suptitle("PAATRA ICLR — Training Curves (optimization stability)")
        plt.tight_layout()
        fig2_path = out / "fig_training_curves.pdf"
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        plt.savefig(str(fig2_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Figures saved: {out}/fig_*.pdf")
    except Exception as e:
        print(f"Figure generation failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
