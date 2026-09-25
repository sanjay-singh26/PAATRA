#!/usr/bin/env python3
"""Recompute the paper's table values that are derivable from the released files.

Standard library only; no data download, GPU, or checkpoint needed:

    python src/verify_paper_tables.py

Exits with status 1 if any check fails.
"""
import csv
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FAILS = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: computed {got:.6g}, paper {want:.6g}")
    if not ok:
        FAILS.append(label)


def rows(name):
    with (RES / name).open(newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- architecture
def n_params(K, d, L=8, n_pos=512):
    """Exact tied GPT-2 count, as in setup.py::count_params_exact."""
    return K * d + n_pos * d + L * (12 * d * d + 13 * d) + 2 * d


def solve_width(K, L=8, N=64_000_000, n_pos=512, H=8):
    """Eq. (width): snap the quadratic root to the closest multiple of H."""
    a, b = 12 * L, K + n_pos + 13 * L + 2
    d0 = (-b + math.sqrt(b * b + 4 * a * N)) / (2 * a)
    cands = [max(H, int(d0 // H) * H), (int(d0 // H) + 1) * H]
    return min(cands, key=lambda d: abs(n_params(K, d, L, n_pos) - N))


print("== Table 3: configurations")
for name, K, d, P, emb in [("Full", 151665, 344, 63.7, 81.8), ("K50K", 47181, 608, 64.5, 44.4),
                           ("K20K", 20002, 720, 64.6, 22.3), ("K10K", 10002, 760, 63.5, 12.0)]:
    check(f"{name} width d", solve_width(K), d, 0)
    check(f"{name} params (M)", n_params(K, d) / 1e6, P, 0.05)
    check(f"{name} embedding share (%)", 100 * K * d / n_params(K, d), emb, 0.05)
    check(f"{name} within 1% of 64M", abs(n_params(K, d) / 64e6 - 1) * 100, 0, 1.0)
check("Ablation_20K params (M)", n_params(20002, 344) / 1e6, 18.5, 0.05)
check("Ablation_20K embedding share (%)", 100 * 20002 * 344 / n_params(20002, 344), 37.3, 0.05)

# ---------------------------------------------------------------- Table 5 / per-seed
print("== Table 5 and per-seed table")
rep = {(r["experiment"], r["config"]): r for r in rows("paper_reported_values.csv")}
seeds = {c: [float(rep[("main_sweep", c)]["bpb"])] for c in ("Full", "K50K", "K20K", "K10K")}
for f in ("seed1_results.csv", "seed2_results.csv"):
    for r in rows(f):
        seeds[r["name"]].append(float(r["bits_per_byte"]))
for r in rows("category_c_results.csv"):
    if r["source"].startswith("replications/") and r["name"] in ("K50K", "K10K"):
        seeds[r["name"]].append(float(r["bits_per_byte"]))
paper = {"Full": (1.4147, 0.0025, None), "K50K": (1.3452, 0.0064, -4.9),
         "K20K": (1.3460, 0.0026, -4.9), "K10K": (1.4210, 0.0056, 0.4)}
full_mean = st.mean(seeds["Full"])
for c, (m, s, delta) in paper.items():
    check(f"{c} number of seeds", len(seeds[c]), 3, 0)
    check(f"{c} mean BPB", st.mean(seeds[c]), m, 5e-5)
    check(f"{c} sample sd", st.stdev(seeds[c]), s, 5e-5)
    if delta is not None:
        check(f"{c} delta vs mean Full (%)", 100 * (st.mean(seeds[c]) / full_mean - 1), delta, 0.05)
abl = float(rep[("main_sweep", "Ablation_20K")]["bpb"])
check("Ablation delta vs mean Full (%)", 100 * (abl / full_mean - 1), 1.2, 0.05)
check("Ablation worse than mean K20K (%)", 100 * (abl / st.mean(seeds["K20K"]) - 1), 6.4, 0.05)
for c, lo, hi in [("K50K", 4.3, 5.5), ("K20K", 4.5, 5.1)]:
    gains = [100 * (f - x) / f for f, x in zip(seeds["Full"], seeds[c])]
    check(f"{c} min paired per-seed gain (%)", min(gains), lo, 0.05)
    check(f"{c} max paired per-seed gain (%)", max(gains), hi, 0.05)
k10_worse = sum(x > f for f, x in zip(seeds["Full"], seeds["K10K"]))
check("K10K seeds worse than Full", k10_worse, 2, 0)

# ---------------------------------------------------------------- Table: LR sweep
print("== Learning-rate table")
lr_paper = {"Full": [1.5897, 1.4161, 1.3568, 1.3455], "K20K": [1.4230, 1.3448, 1.3620, 1.4213],
            "K10K": [1.4642, 1.4160, 1.4506, 1.5008]}
tags = ["lr_1em4", "lr_3em4", "lr_6em4", "lr_1em3"]
lr = {(r["name"], r["source"].split("/")[2]): float(r["bits_per_byte"])
      for r in rows("category_c_results.csv") if r["source"].startswith("lr_sweep/")}
for c, vals in lr_paper.items():
    for t, v in zip(tags, vals):
        check(f"LR {c} {t}", lr[(c, t)], v, 5e-5)
for c in ("Full", "K20K", "K10K"):
    diff = abs(lr[(c, "lr_3em4")] - float(rep[("main_sweep", c)]["bpb"]))
    check(f"LR 3e-4 rerun vs seed-42 run, {c} (|diff| <= 0.0011)", min(diff, 0.0011), diff, 5e-5)

# ---------------------------------------------------------------- grid / depth
print("== Dense grid and depth control")
grid = {r["name"] + ("_L16" if r["n_layer"] == "16" else ""): r for r in rows("category_c_results.csv")
        if r["source"].startswith(("grid/", "depth_control/"))}
for name, K, d, L, P, bpb in [("K5K", 5014, 784, 8, 63.42, 1.4940), ("K15K", 15014, 736, 8, 63.51, 1.3844),
                              ("K30K", 30014, 672, 8, 63.94, 1.3516), ("K75K", 47194, 608, 8, 64.56, 1.3427),
                              ("K20K_L16", 20015, 528, 16, 64.48, 1.3561)]:
    r = grid[name]
    check(f"{name} K", int(r["vocab_size"]), K, 0)
    check(f"{name} d", int(r["n_embd"]), d, 0)
    check(f"{name} params (M)", int(r["total_params"]) / 1e6, P, 0.005)
    check(f"{name} params = exact formula", n_params(K, d, L), int(r["total_params"]), 0)
    check(f"{name} BPB", float(r["bits_per_byte"]), bpb, 5e-5)

# ---------------------------------------------------------------- CC-News
print("== CC-News")
cc = {r["name"]: float(r["bits_per_byte"]) for r in rows("cc_news_baseline.csv")}
for c, v in [("Full", 2.0743), ("K50K", 1.9617), ("K20K", 1.8510), ("K10K", 2.0143)]:
    check(f"CC-News {c}", cc[c], v, 5e-5)

# ---------------------------------------------------------------- genuine tokenizers
print("== Genuine-tokenizer baselines")
gen = {r["name"]: r for r in rows("genuine_tokenizer_results.csv")}
for name, P, bpb in [("bpe_10000", 57.4, 1.2258), ("bpe_20000", 51.4, 1.2015), ("bpe_50000", 41.9, 1.1862),
                     ("unigram_10000", 57.4, 1.2705), ("unigram_20000", 51.4, 1.2366),
                     ("unigram_50000", 41.9, 1.2285)]:
    r = gen[name]
    check(f"{name} params (M)", int(r["params"]) / 1e6, P, 0.05)
    check(f"{name} BPB", float(r["bpb"]), bpb, 5e-5)
    check(f"{name} test bytes", int(r["test_bytes"]), 1287656, 0)

# ---------------------------------------------------------------- training losses
print("== Training-loss decomposition (Total = 0.7 KD + 0.3 CE)")
for c in ("Full", "K50K", "K20K", "K10K", "Ablation_20K"):
    r = rep[("main_sweep", c)]
    check(f"{c} total loss", 0.7 * float(r["kd_loss"]) + 0.3 * float(r["ce_loss"]), float(r["total_loss"]), 1e-3)

# ---------------------------------------------------------------- analytical FLOPs
print("== Analytical MFLOPs/token (2 FLOPs per MAC, n_ctx = 512, L = 8)")
for c, K, d, want in [("Full", 151665, 344, 132.7), ("K50K", 47181, 608, 138.3), ("K20K", 20002, 720, 140.1),
                      ("K10K", 10002, 760, 138.6), ("Ablation_20K", 20002, 344, 42.1)]:
    flops = 2 * 12 * 8 * d * d + 2 * 2 * 8 * 512 * d + 2 * K * d
    check(f"{c} MFLOPs/token", flops / 1e6, want, 0.05)

# ---------------------------------------------------------------- teacher screen
print("== Teacher-only screen ratios")
screen = [r for r in rows("paper_reported_values.csv") if r["experiment"] == "teacher_screen"]
base = 20.19  # implied unrestricted teacher PPL; consistency check only, not an independent source
for r, ratio in zip(sorted(screen, key=lambda r: -int(r["vocab_size"])), [1.01, 1.11, 1.44, 2.88, 6.70]):
    check(f"screen {r['config']} ratio", float(r["ppl"]) / base, ratio, 0.006)

# ---------------------------------------------------------------- Table 2 audit
# Architecture values are taken from each model's Hugging Face config.json
# (hidden_size, num_hidden_layers, intermediate_size, attention heads, KV heads, head_dim,
# vocab_size, tie_word_embeddings). Gemma 3 270M uses the officially stated 170M/100M split.
print("== Table 2: public model configurations (tied matrices counted once)")


def llama_like(vocab, h, layers, inter, heads, kv, head_dim, tied, qkv_bias=False, norms_per_layer=2):
    attn = h * heads * head_dim + 2 * h * kv * head_dim + heads * head_dim * h
    if qkv_bias:
        attn += heads * head_dim + 2 * kv * head_dim
    per_layer = attn + 3 * h * inter + norms_per_layer * h
    emb = vocab * h * (1 if tied else 2)
    return emb + layers * per_layer + h, emb


models = {
    "SmolLM2-360M": (llama_like(49152, 960, 32, 2560, 15, 5, 64, True), 362, 13.0),
    "TinyLlama-1.1B": (llama_like(32000, 2048, 22, 5632, 32, 4, 64, False), 1100, 11.9),
    "Llama-3.2-3B": (llama_like(128256, 3072, 28, 8192, 24, 8, 128, True), 3213, 12.3),
    "Gemma-2-2B": (llama_like(256000, 2304, 26, 9216, 8, 4, 256, True, norms_per_layer=4), 2614, 22.6),
    "Qwen2.5-0.5B": (llama_like(151936, 896, 24, 4864, 14, 2, 64, True, qkv_bias=True), 494, 27.6),
    "Llama-3.2-1B": (llama_like(128256, 2048, 16, 8192, 32, 8, 64, True), 1236, 21.3),
}
for m, ((total, emb), P, share) in models.items():
    check(f"{m} total (M)", total / 1e6, P, 0.5)
    check(f"{m} embedding share (%)", 100 * emb / total, share, 0.05)

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED:", ", ".join(FAILS))
    sys.exit(1)
print("All checks passed.")
