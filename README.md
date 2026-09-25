# Anonymous Reproducibility Package

This package accompanies the anonymous ICLR 2027 submission *PAATRA: Parameter-Allocation-Aware Training for Small Language Models*. It contains the training and evaluation code, the exact settings, command wrappers for every experiment family, per-run result summaries, the paper source, and a script that recomputes the paper's table values from the released files. No author, institution, account, scheduler, or machine-specific identifiers are included.

## Quick check (no GPU, no downloads)

```bash
python src/verify_paper_tables.py
```

The script uses only the Python standard library. It recomputes the table values that can be derived from the released files and checks each against the manuscript. The CPU latency and throughput columns have no source file and are not checked, and the teacher-screen ratios are only a consistency check against the transcribed perplexities. The checks cover:
- parameter counts, widths and embedding shares;
- 3-seed means, standard deviations and relative changes;
- per-seed gains;
- the learning-rate table and the dense grid and depth control;
- CC-News and the genuine-tokenizer baselines;
- the loss decomposition, analytical FLOPs, teacher-screen ratios;
- the Table 2 audit, recomputed from the public Hugging Face `config.json` of each model.

It prints PASS/FAIL per value and exits non-zero if any check fails.

## Contents

| Path | Purpose |
|---|---|
| `src/setup.py` | Downloads teacher and data, builds the frequency-ranked subset vocabularies (Eq. 3), solves the width for each vocabulary (Eq. 8), and tokenizes the training chunks. |
| `src/train.py` | Cross-tokenizer distillation training (Eq. 5). |
| `src/evaluate.py` | BPB evaluation (Eq. 9) on identical raw test bytes for every student. |
| `src/build_tokenizers.py`, `src/train_genuine_tokenizer.py`, `src/evaluate_genuine_tokenizer.py`, `src/aggregate_genuine_results.py` | Genuine BPE/Unigram CE-only baselines. |
| `src/verify_paper_tables.py` | Recomputes the table values derivable from the released files. |
| `src/make_figures.py` | Regenerates the BPB-vs-vocabulary figure and the teacher-screen figure from the CSVs. |
| `scripts/run_main_sweep.sh` | Primary sweep, seed 42: Full, K50K, K20K, K10K, Ablation_20K. |
| `scripts/run_seed_replications.sh` | Seeds 1 and 2 for Full, K50K, K20K and K10K. |
| `scripts/run_lr_sweep.sh` | Learning-rate sweep for Full, K20K and K10K. |
| `scripts/run_grid_and_depth.sh` | Dense vocabulary grid and the L=16 depth control (reconstructed wrapper; see caveats). |
| `scripts/run_genuine_baselines.sh`, `scripts/evaluate_genuine_baselines.sh`, `scripts/aggregate_genuine.sh` | Genuine-tokenizer baselines. |
| `configs/category_c.json` | All settings, including the differences between the main and genuine-tokenizer protocols. |
| `results/` | Reference result summaries (see the mapping below). **Do not overwrite.** |
| `paper/` | Final anonymized manuscript source, figures and compiled PDF. |

## Environment

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/requirements.txt
```

The experiments use Hugging Face `Qwen/Qwen2.5-0.5B` and `Salesforce/wikitext` / `wikitext-103-raw-v1`. The first run downloads these assets and needs network access. CUDA, Apple MPS, or CPU is detected automatically. `train.py` writes the GPU name to each run's log.

## Reproducing the experiments

```bash
python src/setup.py --output-dir artifacts     # vocabularies, widths, training chunks
./scripts/run_main_sweep.sh                    # Table 5 (seed 42), ablation, training losses
./scripts/run_seed_replications.sh             # Table 5 / per-seed table (seeds 1, 2)
./scripts/run_lr_sweep.sh                      # learning-rate table
./scripts/run_grid_and_depth.sh                # dense grid + depth control
python src/build_tokenizers.py --output-dir artifacts/tokenizers
./scripts/run_genuine_baselines.sh && ./scripts/evaluate_genuine_baselines.sh   # genuine-tokenizer table
```

Every run uses 15,000 steps. Distilled students target about 64M parameters; the genuine-tokenizer baselines do not (see caveats). Reproduced outputs go to `results/main/`, `results/replications/`, `results/lr_sweep/`, `results/grid/`, `results/depth_control/` and `results/reproduced/`, all of which are git-ignored. The bundled reference CSVs are never overwritten. Checkpoints are not bundled, because the sweep produces multi-gigabyte artifacts, but the commands above regenerate them.

## Mapping from paper to files

| Paper item | Source |
|---|---|
| Table 3 (configurations) | Computed by `setup.py`; checked by `verify_paper_tables.py` |
| Table 5 and per-seed table, seed 42 | `results/paper_reported_values.csv` (transcribed; see caveats) |
| Table 5 and per-seed table, seeds 1–2 | Full/K20K: `results/seed1_results.csv`, `results/seed2_results.csv`. K50K/K10K: `results/category_c_results.csv` (`source` = `replications/seed{1,2}/…`) |
| Learning-rate table | `results/category_c_results.csv` (`source` = `lr_sweep/…`) |
| Dense grid and depth control | `results/category_c_results.csv` (`source` = `grid/…`, `depth_control/…`) |
| CC-News diagnostic | `results/cc_news_baseline.csv` |
| Genuine-tokenizer table | `results/genuine_tokenizer_results.csv` |
| Teacher-only screen, training-loss table | `results/paper_reported_values.csv` (transcribed) |

## Protocol notes and known caveats

- **Out-of-vocabulary teacher tokens.** Teacher tokens outside the student vocabulary are replaced by student ID 0 (the retained token with the smallest teacher ID) in the student's input. As training targets they are masked. At evaluation they are scored as ID 0. BPB is therefore comparable across students within this protocol, but it is not an exact raw-text likelihood under a genuine reduced tokenizer.
- **K50K is capped.** The 80,000-row training subset contains only about 47.2K distinct teacher tokens, so the 50K and 75K requests are capped: K50K (47,181 tokens in the main build) and K75K (47,194 in the grid build) keep essentially every observed token type.
- **Special-token counts differ between builds.** The primary sweep's vocabularies add 2 non-frequency-ranked tokens (for example 20,002). The dense-grid and depth-control runs used a build that added 14 (for example 20,015). `run_grid_and_depth.sh` is a reconstructed wrapper around the released CLI; a fresh build may differ from the reported vocabulary sizes by a few tokens.
- **Genuine-tokenizer baselines use a different protocol.** They train on all training rows, not the first 80,000, with batch size 8, weight decay 0.1 and a constant learning rate. Their width solver counts the tied embedding twice, which gives 57.4M/51.4M/41.9M-parameter models. Evaluation covers 1,287,656 test bytes, versus 1,293,436 for the main protocol. The code is kept as run, so the reported numbers are reproducible, and these differences are stated in the paper's appendix.
- **Transcribed values.** The raw evaluation files for the seed-42 primary sweep, the Ablation_20K run and the teacher-only screen are not bundled. Their values appear in `results/paper_reported_values.csv`, labeled as transcriptions of the reported numbers. The learning-rate sweep's 3e-4 column is an independent rerun of the seed-42 recipe and differs from those values by at most 0.0011 BPB.
- **Not bundled:** the teacher-only screen script, the CPU runtime benchmark script, the CC-News sampling script, and the script that produced the training-curve figure. Their measurements are reported as-is in the paper.
- **Evaluation data.** All model selection and evaluation use the WikiText-103 test split; there is no separate validation split.

## Build the paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
