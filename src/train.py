"""
PAATRA ICLR — Training Script
Trains one student configuration. Designed to be launched per-config by
submit_cluster.sh on the Mac Studio cluster.

Usage (single run):
    python train.py --name B_20K_paatra --artifacts-dir /path/to/artifacts --output-dir /path/to/results

    # Override any hyperparameter:
    python train.py --name Full_151K --artifacts-dir /artifacts --output-dir /results \
        --num-steps 30000 --batch-size 8 --lr 1e-4

    # Different teacher (setup.py must have been run with the same teacher):
    python train.py --name K20K --artifacts-dir /artifacts_qwen15b --output-dir /results_qwen15b

    # submit_cluster.sh launches this once per config via cluster-submit.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel


# ── Device detection (CUDA → MPS → CPU) ──────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.bfloat16   # MPS handles bfloat16 more reliably
    return "cpu", torch.float32


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PAATRA ICLR Training")

    p.add_argument("--name",          required=True,
                   help="Config name matching a key in vocabs.pt (e.g. B_20K_paatra).")
    p.add_argument("--artifacts-dir", required=True,
                   help="Directory produced by setup.py (contains meta.json, vocabs.pt, chunks.pt).")
    p.add_argument("--output-dir",    required=True,
                   help="Where to write checkpoints and logs.")

    # All hyperparams below override meta.json defaults if specified.
    p.add_argument("--num-steps",      type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--warmup-steps",   type=int,   default=None)
    p.add_argument("--weight-decay",   type=float, default=None)
    p.add_argument("--grad-clip",      type=float, default=None)
    p.add_argument("--kd-temp",        type=float, default=None)
    p.add_argument("--kd-alpha",       type=float, default=None)
    p.add_argument("--log-interval",   type=int,   default=None)
    p.add_argument("--checkpoint-every", type=int, default=5_000,
                   help="Save a resumable checkpoint every N steps.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--resume",         action="store_true",
                   help="Resume from the latest checkpoint if it exists.")

    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────

def create_student(cfg, vocab_size, seq_len):
    """
    GPT2LMHeadModel with tied embeddings.
    bos/eos explicitly set to None — GPT2Config defaults to 50256 which is
    out of range for any vocab smaller than 50257 (the original AAAI bug).
    """
    model_cfg = GPT2Config(
        vocab_size          = vocab_size,
        n_embd              = cfg["n_embd"],
        n_layer             = cfg["n_layer"],
        n_head              = cfg["n_head"],
        n_inner             = 4 * cfg["n_embd"],
        activation_function = "gelu_new",
        n_positions         = seq_len,
        resid_pdrop         = 0.0,
        embd_pdrop          = 0.0,
        attn_pdrop          = 0.0,
        bos_token_id        = None,   # fix: do not inherit GPT2's default 50256
        eos_token_id        = None,
    )
    return GPT2LMHeadModel(model_cfg)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KDDataset(Dataset):
    def __init__(self, chunks, t2s):
        self.chunks = chunks
        self.t2s    = t2s

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        teacher_ids = self.chunks[idx].tolist()
        student_ids = [self.t2s.get(tid, 0) for tid in teacher_ids]
        loss_mask   = [1.0 if tid in self.t2s else 0.0 for tid in teacher_ids]
        return {
            "teacher_ids": torch.tensor(teacher_ids, dtype=torch.long),
            "student_ids": torch.tensor(student_ids, dtype=torch.long),
            "loss_mask":   torch.tensor(loss_mask,   dtype=torch.float),
        }


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, teacher_dtype = get_device()
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "students").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    log  = setup_logging(out / "logs" / f"{args.name}.log")
    arts = Path(args.artifacts_dir)

    # ── Load artifacts ────────────────────────────────────────────────────────
    with open(arts / "meta.json") as f:
        meta = json.load(f)

    # CLI args override meta defaults where provided
    def get(key, cli_val):
        return cli_val if cli_val is not None else meta[key]

    NUM_STEPS      = get("num_steps",    args.num_steps)
    BATCH_SIZE     = get("batch_size",   args.batch_size)
    LR             = get("lr",           args.lr)
    WARMUP_STEPS   = get("warmup_steps", args.warmup_steps)
    WEIGHT_DECAY   = get("weight_decay", args.weight_decay)
    GRAD_CLIP      = get("grad_clip",    args.grad_clip)
    KD_TEMP        = get("kd_temp",      args.kd_temp)
    KD_ALPHA       = get("kd_alpha",     args.kd_alpha)
    LOG_INTERVAL   = get("log_interval", args.log_interval)
    SEQ_LEN        = meta["seq_len"]
    TEACHER_ID     = meta["teacher_model"]

    chunks = torch.load(arts / "chunks.pt",  weights_only=False)
    vocabs = torch.load(arts / "vocabs.pt",  weights_only=False)

    if args.name not in vocabs:
        log.error(f"'{args.name}' not in vocabs. Available: {list(vocabs.keys())}")
        sys.exit(1)

    vocab_entry  = vocabs[args.name]
    s2t          = vocab_entry["s2t"]
    t2s          = vocab_entry["t2s"]
    cfg          = vocab_entry["config"]
    expected_N   = vocab_entry["total_params"]
    expected_emb = vocab_entry["emb_params"]
    vocab_size   = len(s2t)

    log.info("=" * 65)
    log.info(f"  PAATRA ICLR — Training: {args.name}")
    log.info("=" * 65)
    log.info(f"  Device       : {device}")
    if device == "cuda":
        log.info(f"  GPU          : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    log.info(f"  Vocab size   : {vocab_size:,}")
    log.info(f"  Architecture : d={cfg['n_embd']}  L={cfg['n_layer']}  H={cfg['n_head']}")
    log.info(f"  Expected N   : {expected_N:,}  (emb {expected_emb/expected_N*100:.1f}%)")
    log.info(f"  Teacher      : {TEACHER_ID}")
    log.info(f"  Steps        : {NUM_STEPS:,}  batch={BATCH_SIZE}  seq={SEQ_LEN}")
    log.info(f"  LR           : {LR}  warmup={WARMUP_STEPS}  wd={WEIGHT_DECAY}")
    log.info(f"  KD           : τ={KD_TEMP}  α={KD_ALPHA}")

    # ── Build student ─────────────────────────────────────────────────────────
    student  = create_student(cfg, vocab_size, SEQ_LEN).to(device)
    actual_N = sum(p.numel() for p in student.parameters())
    emb_N    = vocab_size * cfg["n_embd"]
    err_pct  = (actual_N - expected_N) / expected_N * 100

    log.info(f"\n  Actual N     : {actual_N:,}  (error {err_pct:+.2f}%)")
    assert abs(err_pct) <= 1.0, (
        f"Parameter count {actual_N:,} deviates {err_pct:.2f}% from expected "
        f"{expected_N:,}. This should not happen — check architecture config."
    )

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    final_path  = out / "students" / f"{args.name}.pt"
    resume_path = out / "students" / f"{args.name}_latest.pt"

    if final_path.exists():
        log.info(f"\n✓ Final checkpoint already exists: {final_path}")
        log.info("  Delete it to retrain. Exiting.")
        return

    # ── Load teacher ──────────────────────────────────────────────────────────
    log.info(f"\n  Loading teacher: {TEACHER_ID}")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, torch_dtype=teacher_dtype, device_map=device,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log.info(f"  Teacher params : {sum(p.numel() for p in teacher.parameters()):,} (frozen)")

    # ── Dataset & dataloader ──────────────────────────────────────────────────
    dataset    = KDDataset(chunks, t2s)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=(device == "cuda"),
    )
    log.info(f"\n  Dataset : {len(dataset):,} sequences  "
             f"({NUM_STEPS * BATCH_SIZE / len(dataset):.2f} effective epochs)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    s2t_tensor  = torch.tensor(s2t, dtype=torch.long, device=device)

    # ── Resume from latest checkpoint if requested ────────────────────────────
    start_step    = 0
    loss_history  = []
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, weights_only=False, map_location=device)
        student.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step   = ckpt["step"]
        loss_history = ckpt.get("losses", [])
        log.info(f"\n  Resuming from step {start_step:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    student.train()
    step          = start_step
    running_loss  = running_kd = running_ce = 0.0
    t_start       = time.time()

    log.info(f"\n  {'Step':>6}  {'Loss':>7}  {'KD':>7}  {'CE':>7}  "
             f"{'LR':>9}  {'Elapsed':>9}  {'ETA':>9}")
    log.info("  " + "─" * 60)

    while step < NUM_STEPS:
        for batch in dataloader:
            if step >= NUM_STEPS:
                break

            lr = cosine_lr(step, WARMUP_STEPS, NUM_STEPS, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            teacher_ids = batch["teacher_ids"].to(device)
            student_ids = batch["student_ids"].to(device)
            loss_mask   = batch["loss_mask"].to(device)

            with torch.no_grad():
                t_logits = teacher(input_ids=teacher_ids).logits.float()
                t_subset = t_logits[:, :-1, :].index_select(2, s2t_tensor)
            del t_logits

            s_logits = student(input_ids=student_ids).logits[:, :-1, :]
            labels   = student_ids[:, 1:]
            mask     = loss_mask[:, 1:]
            mask_sum = mask.sum().clamp(min=1.0)

            t_probs      = F.softmax(t_subset / KD_TEMP, dim=-1)
            s_log_probs  = F.log_softmax(s_logits / KD_TEMP, dim=-1)
            kd_per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
            kd_loss      = (kd_per_token * mask).sum() / mask_sum * (KD_TEMP ** 2)

            ce_per_token = F.cross_entropy(
                s_logits.reshape(-1, s_logits.size(-1)),
                labels.reshape(-1), reduction="none",
            ).reshape(labels.shape)
            ce_loss = (ce_per_token * mask).sum() / mask_sum

            loss = KD_ALPHA * kd_loss + (1 - KD_ALPHA) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()
            running_kd   += kd_loss.item()
            running_ce   += ce_loss.item()
            step         += 1

            if step % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                avg_kd   = running_kd   / LOG_INTERVAL
                avg_ce   = running_ce   / LOG_INTERVAL
                elapsed  = time.time() - t_start
                eta      = elapsed / max(step - start_step, 1) * (NUM_STEPS - step)
                log.info(f"  {step:>6}  {avg_loss:>7.3f}  {avg_kd:>7.3f}  {avg_ce:>7.3f}  "
                         f"{lr:>9.2e}  {elapsed/60:>7.1f}m  {eta/60:>7.1f}m")
                loss_history.append({"step": step, "loss": avg_loss, "kd": avg_kd, "ce": avg_ce})
                running_loss = running_kd = running_ce = 0.0

            # Periodic checkpoint for resumption
            if step % args.checkpoint_every == 0:
                torch.save({
                    "step":            step,
                    "model_state":     student.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "losses":          loss_history,
                }, resume_path)
                log.info(f"  ↳ Checkpoint saved at step {step}")

    total_time = time.time() - t_start
    log.info(f"\n✓ Training complete in {total_time/60:.1f} min")

    # ── Save final bundle ─────────────────────────────────────────────────────
    student.eval()
    bundle = {
        "name":         args.name,
        "state_dict":   student.state_dict(),
        "config":       cfg,
        "s2t":          s2t,
        "t2s":          t2s,
        "losses":       loss_history,
        "total_params": actual_N,
        "emb_params":   emb_N,
        "vocab_size":   vocab_size,
        "seq_len":      SEQ_LEN,
        "num_steps":    NUM_STEPS,
        "hyperparams": {
            "teacher":      TEACHER_ID,
            "lr":           LR,
            "warmup":       WARMUP_STEPS,
            "kd_temp":      KD_TEMP,
            "kd_alpha":     KD_ALPHA,
            "batch_size":   BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "seed":         args.seed,
        },
    }
    torch.save(bundle, final_path)
    fsize = os.path.getsize(final_path) / 1e6
    log.info(f"✓ Saved: {final_path}  ({fsize:.1f} MB)")
    log.info(f"  Final loss : {loss_history[-1]['loss']:.3f}")
    log.info(f"  KD loss    : {loss_history[-1]['kd']:.3f}")
    log.info(f"  CE loss    : {loss_history[-1]['ce']:.3f}")

    # Clean up the resume checkpoint
    if resume_path.exists():
        resume_path.unlink()


if __name__ == "__main__":
    main()
