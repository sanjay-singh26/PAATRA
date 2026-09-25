#!/usr/bin/env python3
"""Regenerate the BPB-vs-vocabulary and teacher-screen figures from reported values.

Per-seed BPB values come from results/*.csv where available; seed-42 values of the
primary sweep, the teacher-only screen, and the Ablation_20K run are transcribed in
results/paper_reported_values.csv (their raw evaluation files are not bundled).
"""
import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "results" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def sweep_points():
    """Return {config: (vocab, [bpb per seed])} for the four parameter-matched configs."""
    rep = {(r["experiment"], r["config"], r["seed"]): r for r in read("paper_reported_values.csv")}
    seeds = {}
    for cfg in ("Full", "K50K", "K20K", "K10K"):
        seeds[cfg] = [float(rep[("main_sweep", cfg, "42")]["bpb"])]
    for fname in ("seed1_results.csv", "seed2_results.csv"):
        for r in read(fname):
            seeds[r["name"]].append(float(r["bits_per_byte"]))
    for r in read("category_c_results.csv"):
        if r["source"].startswith("replications/") and r["name"] in ("K50K", "K10K"):
            seeds[r["name"]].append(float(r["bits_per_byte"]))
    vocab = {"Full": 151665, "K50K": 47181, "K20K": 20002, "K10K": 10002}
    return {c: (vocab[c], v) for c, v in seeds.items()}


def grid_points():
    return [(int(r["vocab_size"]), float(r["bits_per_byte"]), r["name"])
            for r in read("category_c_results.csv") if r["source"].startswith("grid/")]


def fig_bpb(out):
    pts = sweep_points()
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    xs, ms, sds = [], [], []
    offsets = {"K10K": (6, 4), "K20K": (-14, -24), "K50K": (4, -24), "Full": (-30, 6)}
    for cfg, (v, vals) in sorted(pts.items(), key=lambda kv: kv[1][0]):
        assert len(vals) == 3, (cfg, vals)
        xs.append(v); ms.append(st.mean(vals)); sds.append(st.stdev(vals))
        ax.annotate(f"{cfg}\n{st.mean(vals):.4f}", (v, st.mean(vals)), textcoords="offset points",
                    xytext=offsets[cfg], fontsize=7)
    ax.set_ylim(1.30, 1.51)
    ax.errorbar(xs, ms, yerr=sds, fmt="o-", color="#1f5fa8", capsize=3, lw=1.4,
                label="3-seed mean $\\pm$ sd (primary sweep)")
    g = sorted(grid_points())
    ax.plot([p[0] for p in g], [p[1] for p in g], "s", mfc="none", color="#c0392b", ms=5,
            label="dense grid (single seed)")
    ax.set_xscale("log")
    ax.set_xlabel("Student vocabulary size $K$ (log scale)")
    ax.set_ylabel("BPB (lower is better)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(out)


def fig_screen(out):
    rows = [r for r in read("paper_reported_values.csv") if r["experiment"] == "teacher_screen"]
    rows.sort(key=lambda r: -int(r["vocab_size"]))
    labels = [r["config"] for r in rows]
    ppl = [float(r["ppl"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(range(len(ppl)), ppl, "o-", color="#1f5fa8")
    ax.set_xticks(range(len(ppl)), labels)
    ax.set_xlabel("Retained teacher vocabulary $K$")
    ax.set_ylabel("Teacher perplexity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "figures")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fig_bpb(a.out_dir / "fig_bpb_vs_vocab.pdf")
    fig_screen(a.out_dir / "fig_teacher_screen.pdf")
    print("wrote", a.out_dir / "fig_bpb_vs_vocab.pdf", "and", a.out_dir / "fig_teacher_screen.pdf")
