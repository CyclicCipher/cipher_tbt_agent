"""
train.py - GrokkingMamba3 training loop for modular arithmetic.

Tests three ideas from RESEARCH_IDEAS.md on the standard grokking benchmark
(modular arithmetic, p=97, 50% train/test split).

Usage examples:
    # Baseline (standard Mamba3, no grokking features)
    python train.py

    # GrowableLinear only
    python train.py --growable

    # Nuclear norm regularization (requires --growable for GrowableLinear,
    # or applies to standard weights via SVD -- we only support growable here)
    python train.py --growable --nuclear 0.001

    # Alignment loss only (uses standard nn.Linear)
    python train.py --alignment 0.1

    # All three combined
    python train.py --growable --nuclear 0.001 --alignment 0.1

    # Custom configuration
    python train.py --p 97 --train_frac 0.5 --steps 50000 --d_model 128
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from task import ModArithTask
from grokking_mamba3 import (
    GrokkingConfig, GrokkingMamba3LM,
    collect_growable, get_new_params, total_nuclear_norm,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: GrokkingMamba3LM,
    task:  ModArithTask,
    split: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Exact-match accuracy and cross-entropy loss on train or test split."""
    model.eval()
    triples = task.train if split == "train" else task.test
    total_correct = 0
    total_loss    = 0.0
    total_n       = 0

    for batch in task.eval_batch_iter(triples, batch_size, device):
        logits, _ = model(batch["tokens"])
        pred_logits = logits[:, task.pred_pos, :]   # (B, vocab)
        targets     = batch["targets"]

        loss = F.cross_entropy(pred_logits, targets, reduction="sum")
        correct = (pred_logits.argmax(-1) == targets).sum().item()

        total_loss    += loss.item()
        total_correct += correct
        total_n       += len(targets)

    model.train()
    return {
        "acc":  total_correct / total_n,
        "loss": total_loss    / total_n,
        "n":    total_n,
    }


# ---------------------------------------------------------------------------
# Rank growth
# ---------------------------------------------------------------------------

def maybe_grow(
    model:        GrokkingMamba3LM,
    optimizer:    torch.optim.Optimizer,
    loss_history: List[float],
    step:         int,
    min_plateau:  int   = 300,
    tol:          float = 1e-4,
    lr:           float = 3e-4,
) -> int:
    """Check all GrowableLinear modules and grow those that are plateauing.

    Growth criterion:
        - At least min_plateau steps of loss history available
        - Std of recent window < tol (loss is not improving)
        - Last loss is still above 0.05 (not already solved)

    Returns the number of modules that grew.
    """
    if len(loss_history) < min_plateau:
        return 0
    recent = np.array(loss_history[-min_plateau:])
    if recent.std() >= tol or recent[-1] < 0.05:
        return 0

    growables = collect_growable(model)
    grew = 0
    for g in growables:
        # Check if this module's singular values have small gradients
        # (current rank is saturated, not just under-trained)
        if g.S.grad is not None and g.S.grad.abs().max().item() > 1e-3:
            continue   # still optimizing existing rank -- don't grow yet
        g.grow(n_new=1, step=step)
        grew += 1

    if grew > 0:
        new_params = get_new_params(model, optimizer)
        if new_params:
            optimizer.add_param_group({"params": new_params, "lr": lr})

    return grew


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Task
    task = ModArithTask(p=args.p, train_frac=args.train_frac, seed=args.seed)
    print(task.stats())

    # Model
    config = GrokkingConfig(
        vocab_size    = task.vocab_size,
        d_model       = args.d_model,
        d_state       = 64,
        expand        = 2,
        headdim       = 64,
        chunk_size    = task.seq_len,
        n_layer       = args.n_layer,
        mlp_expand    = 4,
        stable_ssm    = True,
        use_growable  = args.growable,
        initial_rank  = args.initial_rank,
        use_alignment = (args.alignment > 0),
    )
    model = GrokkingMamba3LM(config).to(device)
    print(f"Parameters: {model.n_params():,}")
    if args.growable:
        print("GrowableLinear modules:")
        print(model.rank_summary())

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1
    )

    rng = np.random.default_rng(args.seed)

    # Logging
    log: Dict[str, List] = {
        "step": [], "train_acc": [], "test_acc": [],
        "train_loss": [], "test_loss": [],
        "task_loss": [], "nuclear_loss": [], "align_loss": [],
        "ranks": [],           # list of {name: rank} per eval step
        "nuclear_norms": [],   # list of {name: norm} per eval step
        "growth_events": [],   # list of {step, n_grew}
        "grok_step": None,
    }
    task_loss_window: List[float] = []

    t0 = time.time()
    print(f"\nTraining: steps={args.steps}  batch={args.batch_size}  "
          f"growable={args.growable}  nuclear={args.nuclear}  "
          f"alignment={args.alignment}")
    print(f"{'step':>7}  {'tr_acc':>7}  {'te_acc':>7}  "
          f"{'task_L':>8}  {'nuc_L':>7}  {'aln_L':>7}  "
          f"{'ranks':>12}  {'elapsed':>8}")

    for step in range(1, args.steps + 1):
        model.train()

        batch = task.make_batch(task.train, args.batch_size, device, rng)

        # Forward pass on original batch
        logits, h_list = model(batch["tokens"])
        pred_logits = logits[:, task.pred_pos, :]
        task_loss   = F.cross_entropy(pred_logits, batch["targets"])

        total_loss   = task_loss
        nuclear_loss = torch.tensor(0.0, device=device)
        align_loss   = torch.tensor(0.0, device=device)

        # Nuclear norm penalty
        if args.nuclear > 0 and args.growable:
            nuclear_loss = total_nuclear_norm(model)
            total_loss   = total_loss + args.nuclear * nuclear_loss

        # Alignment loss
        if args.alignment > 0 and model.align is not None:
            shifted_batch = task.make_shifted_batch(batch, shift=1)
            _, h_list_shifted = model(shifted_batch["tokens"])
            align_loss = model.align(h_list, h_list_shifted)
            total_loss = total_loss + args.alignment * align_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track task loss for growth criterion
        tl = task_loss.item()
        task_loss_window.append(tl)
        if len(task_loss_window) > 500:
            task_loss_window.pop(0)

        # Rank growth check (every 100 steps)
        if args.growable and step % 100 == 0:
            n_grew = maybe_grow(
                model, optimizer, task_loss_window, step,
                min_plateau=args.growth_plateau,
                tol=args.growth_tol,
                lr=args.lr,
            )
            if n_grew > 0:
                log["growth_events"].append({"step": step, "n_grew": n_grew})
                print(f"  [step {step}] Grew {n_grew} module(s).")
                print(model.rank_summary())

        # Evaluate periodically
        if step % args.eval_interval == 0 or step == args.steps:
            tr = evaluate(model, task, "train", args.batch_size * 4, device)
            te = evaluate(model, task, "test",  args.batch_size * 4, device)

            # Detect grokking (first time test acc > 95%)
            if log["grok_step"] is None and te["acc"] > 0.95:
                log["grok_step"] = step
                print(f"  *** GROKKING at step {step}! "
                      f"test_acc={te['acc']*100:.1f}% ***")

            log["step"].append(step)
            log["train_acc"].append(tr["acc"])
            log["test_acc"].append(te["acc"])
            log["train_loss"].append(tr["loss"])
            log["test_loss"].append(te["loss"])
            log["task_loss"].append(tl)
            log["nuclear_loss"].append(nuclear_loss.item())
            log["align_loss"].append(align_loss.item())

            # Record rank / nuclear norm state
            ranks = {}
            norms = {}
            for name, m in model.named_modules():
                if hasattr(m, 'rank') and hasattr(m, 'nuclear_norm'):
                    ranks[name] = m.rank
                    norms[name] = m.nuclear_norm.item()
            log["ranks"].append(ranks)
            log["nuclear_norms"].append(norms)

            rank_str = str(sorted(set(ranks.values()))) if ranks else "-"
            elapsed  = time.time() - t0
            print(
                f"{step:>7d}  {tr['acc']:>7.3f}  {te['acc']:>7.3f}  "
                f"{tl:>8.4f}  {nuclear_loss.item():>7.4f}  {align_loss.item():>7.4f}  "
                f"{rank_str:>12}  {elapsed:>7.0f}s"
            )

    # Save results
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out_path = os.path.join(args.save, "log.json")
        # Tensors not JSON-serializable -- convert
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nLog saved to {out_path}")

        # Save model
        ckpt_path = os.path.join(args.save, "model.pt")
        torch.save({
            "config": {
                "vocab_size":    config.vocab_size,
                "d_model":       config.d_model,
                "d_state":       config.d_state,
                "expand":        config.expand,
                "headdim":       config.headdim,
                "chunk_size":    config.chunk_size,
                "n_layer":       config.n_layer,
                "mlp_expand":    config.mlp_expand,
                "stable_ssm":    config.stable_ssm,
                "use_growable":  config.use_growable,
                "initial_rank":  config.initial_rank,
                "use_alignment": config.use_alignment,
            },
            "model_state_dict": model.state_dict(),
            "task_config": {"p": args.p, "train_frac": args.train_frac},
        }, ckpt_path)
        print(f"Model saved to {ckpt_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if log["step"]:
        final_tr = log["train_acc"][-1]
        final_te = log["test_acc"][-1]
        print(f"  Final train accuracy : {final_tr*100:.1f}%")
        print(f"  Final test  accuracy : {final_te*100:.1f}%")
        print(f"  Generalization gap   : {(final_tr - final_te)*100:.1f}pp")
    if log["grok_step"] is not None:
        print(f"  Grokking step        : {log['grok_step']}")
    else:
        print(f"  Grokking step        : not reached (test acc never > 95%)")
    if args.growable and log["growth_events"]:
        print(f"  Growth events        : {len(log['growth_events'])}")
        for ev in log["growth_events"]:
            print(f"    step {ev['step']}: {ev['n_grew']} module(s) grew")
    print("=" * 60)

    return log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GrokkingMamba3 modular arithmetic training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Task
    parser.add_argument("--p",          type=int,   default=97,
                        help="Prime modulus (default 97)")
    parser.add_argument("--train_frac", type=float, default=0.5,
                        help="Fraction of triples in train split (default 0.5)")
    # Model
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_layer",    type=int,   default=2)
    # Training
    parser.add_argument("--steps",      type=int,   default=30000)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0,
                        help="Weight decay (high decay encourages grokking)")
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--seed",       type=int,   default=42)
    # Grokking ideas
    parser.add_argument("--growable",   action="store_true",
                        help="Replace in_proj/out_proj with GrowableLinear")
    parser.add_argument("--initial_rank", type=int, default=16,
                        help="Starting rank for GrowableLinear (default 16)")
    parser.add_argument("--nuclear",    type=float, default=0.0,
                        help="Nuclear norm penalty weight (requires --growable)")
    parser.add_argument("--alignment",  type=float, default=0.0,
                        help="Alignment loss weight (lambda_align)")
    parser.add_argument("--growth_plateau", type=int, default=300,
                        help="Min steps of plateau before growing rank")
    parser.add_argument("--growth_tol", type=float, default=1e-4,
                        help="Loss std threshold for plateau detection")
    # Output
    parser.add_argument("--save",       type=str,   default=None,
                        help="Directory to save log.json and model.pt")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
