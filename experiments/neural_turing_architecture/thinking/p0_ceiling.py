"""P0 — the no-thought ceiling on a diverse task suite (planning doc §20; pre-registered before this ran).

A model with NO thoughts reads a problem and answers at the last position in one forward pass. For each of the six
families of `tasks.py` and each of its 8 depth levels, how accurate is it? The levels where it fails — while the
shallow levels of the same family are solved — are the problems on which thoughts must be NECESSARY (R6, planning
§1), and so the problems P1(b) will use.

Training mixes every family and every level uniformly (a ladder, planning §19.1); answers are scored at the last
position only. Evaluation uses fresh problems from a separate generator stream (Brainfuck: a separate pool, seed 1),
512 per family and level, at the halfway point and at the end — so a failed level can be told apart as still rising
(training-limited) or flat (capacity-limited).

Score per (family, level): accuracy, the chance rate (the majority class's share of those eval answers), and the
NORMALISED accuracy (acc - chance) / (1 - chance). A level is SOLVED at normalised >= 0.9 and FAILED at <= 0.2.
h_solved = the depth of the deepest level such that it and every shallower level are solved; h_fail = the depth of the
shallowest failed level.

Arms (`--arm`):
    loop  — `h1_lid.LoopedModel`: prelude block, ONE tied core block applied `--loops` times with the boundary operator
            (RMS-norm + re-injected anchor) between passes, coda block — our architecture without thoughts;
    plain — `h1_lid.Model`: `--layers` untied blocks — the Coconut-faithful baseline without thoughts.

Usage:
    python p0_ceiling.py --arm loop --loops 4 --json runs/p0/loop4.json --save runs/p0/loop4.pt
    python p0_ceiling.py --arm plain --layers 6 --json runs/p0/plain6.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "transformers"))
import h1_lid as H1  # noqa: E402
from tasks import V, Brainfuck, make_batch, make_families  # noqa: E402

SOLVED, FAILED = 0.9, 0.2


def build_model(args):
    if args.arm == "loop":
        return H1.LoopedModel(d_model=args.d, n_head=args.heads, max_len=args.max_len, pos="rope", n_vocab=V,
                              loops=args.loops, tied=True, n_prelude=1, n_coda=1)
    return H1.Model(d_model=args.d, n_layer=args.layers, n_head=args.heads, max_len=args.max_len, pos="rope", n_vocab=V)


@torch.no_grad()
def evaluate(model, fams, n, dev, seed=12345, chunk=256):
    """Per family and level: accuracy, chance (majority-class share of these answers), normalised accuracy."""
    model.eval()
    rng = np.random.default_rng(seed)
    out = {}
    for f, fam in enumerate(fams):
        rows = []
        for li in range(len(fam.levels)):
            correct, answers = 0, []
            for start in range(0, n, chunk):
                m = min(chunk, n - start)
                tok, ans, _, _ = make_batch([fam], rng, m, dev, level=li)
                pred = model(tok)[:, -1].argmax(-1)
                correct += int((pred == ans).sum())
                answers += ans.tolist()
            acc = correct / n
            chance = max(np.bincount(answers)) / n
            norm = round((acc - chance) / (1 - chance), 4) if chance < 0.999 else None   # one answer only: undefined
            rows.append(dict(h=fam.levels[li], acc=round(acc, 4), chance=round(float(chance), 4), norm=norm))
        out[fam.name] = rows
    model.train()
    return out


def ceilings(rows):
    """h_solved: the deepest level with it and every shallower level solved (0 if level 1 is not); h_fail: the
    shallowest failed level (None if none)."""
    rows = [r for r in rows if r["norm"] is not None]
    h_solved = 0
    for r in rows:
        if r["norm"] < SOLVED:
            break
        h_solved = r["h"]
    return h_solved, next((r["h"] for r in rows if r["norm"] <= FAILED), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["loop", "plain"], default="loop")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--per_family", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=512)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--bf_train", type=int, default=20000)
    ap.add_argument("--bf_eval", type=int, default=1000)
    ap.add_argument("--cache", default=str(HERE / "runs" / "p0"))
    ap.add_argument("--json", default="")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    fams = make_families(bf_seed=0, bf_per_level=args.bf_train, cache_dir=args.cache)
    eval_fams = list(fams)
    eval_fams[4] = Brainfuck(seed=1, per_level=args.bf_eval, cache_dir=args.cache)
    model = build_model(args).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    blocks = (args.loops + 2) if args.arm == "loop" else args.layers
    print(f"device {dev} | arm {args.arm} " + (f"loops {args.loops}" if args.arm == "loop" else f"layers {args.layers}")
          + f" | block applications {blocks} | d {args.d} | params {n_params:,} | steps {args.steps} x {6 * args.per_family}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16 and dev == "cuda"))
    rng = np.random.default_rng(args.seed)
    log, evals, t0 = [], {}, time.time()
    run_loss = None
    for step in range(args.steps):
        tok, ans, _, _ = make_batch(fams, rng, args.per_family, dev)
        with amp:
            logits = model(tok)[:, -1]
        loss = F.cross_entropy(logits.float(), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if (step + 1) % 250 == 0:
            log.append(dict(step=step + 1, loss=round(run_loss, 4), s=round(time.time() - t0, 1)))
            print(f"  step {step + 1:6d}  loss {run_loss:.4f}  {time.time() - t0:7.0f}s", flush=True)
        if step + 1 in (args.steps // 2, args.steps):
            tag = "mid" if step + 1 < args.steps else "final"
            evals[tag] = evaluate(model, eval_fams, args.eval_n, dev)
            print(f"  [{tag} eval at step {step + 1}]", flush=True)

    print(f"\ntrained in {time.time() - t0:.0f}s\n")
    print(f"normalised accuracy per level (final; mid in brackets). solved >= {SOLVED}, failed <= {FAILED}")
    summary = {}
    for name, rows in evals["final"].items():
        mid = evals["mid"][name]
        h_solved, h_fail = ceilings(rows)
        flat = [r["h"] for r, m in zip(rows, mid) if r["norm"] is not None and r["norm"] <= FAILED
                and r["norm"] - m["norm"] <= 0.05]
        summary[name] = dict(h_solved=h_solved, h_fail=h_fail, failed_and_flat=flat)
        cells = "  ".join(f"{r['h']}:{r['norm'] if r['norm'] is None else format(r['norm'], '+.2f')}"
                          f"({m['norm'] if m['norm'] is None else format(m['norm'], '+.2f')})" for r, m in zip(rows, mid))
        print(f"  {name:5s} h_solved {h_solved!s:>4}  h_fail {h_fail!s:>4}  | {cells}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(args=vars(args), device=dev, params=n_params, block_applications=blocks,
                       train_s=round(time.time() - t0, 1), log=log, evals=evals, summary=summary),
                  open(args.json, "w"), indent=1)
    if args.save:
        torch.save(dict(state=model.state_dict(), args=vars(args)), args.save)


if __name__ == "__main__":
    main()
