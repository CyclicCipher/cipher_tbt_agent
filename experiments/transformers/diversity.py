"""Does it COMPOSE, or MEMORISE? The task-diversity sweep.

WHY THIS AND NOT H1. H1 asked how FAST a novel task is acquired, and how that speed tracks the familiarity of its parts.
It cannot be answered as posed, and not for want of a better metric: the measured acquisition rate on held-out
compositions is ZERO for every one of them (0/8, in every optimiser and positional scheme tried). You cannot correlate a
speed against anything when nothing is acquired. The censoring was the finding, not an obstacle to it.

So the question upstream of H1 is whether the model composes at all. The evidence says maybe not: it solves 14/17 TRAINED
compositions (mean accuracy 0.86) and 0/8 held-out ones built from the SAME primitives (0.00–0.37). That is consistent
with it having learned 17 atomic labels rather than a rule for combining seven operations — and a model that never
decomposes has no "local operations" for LID to be about.

THE HYPOTHESIS HAS A KNOWN SHAPE. Raventós et al. 2023 (NeurIPS, arXiv:2306.15063) found a TASK-DIVERSITY THRESHOLD in
in-context regression: trained on few enough distinct tasks a transformer behaves as though it memorised them, and past a
critical count it switches to implementing the general algorithm and generalises to tasks never seen. That would explain
BOTH of our results at once:
  * `linreg.py` drew a fresh `w` every sequence — effectively INFINITE task diversity — and the model learned the general
    algorithm, landing within 0.005–0.02 of Bayes-optimal;
  * `h1_lid.py` trained on 17 discrete tasks — very low diversity — and the model memorised them.
If that is right, held-out accuracy here should be ~0 for small training sets and rise sharply past a threshold. If it
stays flat all the way to the largest training set we can supply, the memorisation story is wrong too and something more
basic is missing — which is the outcome this is built to be able to report.

DESIGN. 12 primitives (rotations, reverse, swap-pairs, value shifts, negate) give 62 DISTINCT 2-compositions, deduplicated
by signature so no two tasks are the same function under different names. **The held-out set is FIXED at 20 across every
condition**, so the test never changes; only the number of training tasks does, and the training sets are NESTED
(each is a superset of the last) so nothing varies but supply. Sampling over training tasks is UNIFORM — H1's graded
frequencies existed to create a familiarity gradient, which is not this question.

Read the two columns together: training accuracy says whether the model fit what it was given at all (few tasks are easy
to memorise, many are not), and held-out accuracy is the claim. The gap between them is the memorisation.

Usage:  python experiments/transformers/diversity.py --sizes 4,8,16
"""
from __future__ import annotations

import argparse
import itertools
import math
import time

import torch
import torch.nn.functional as F

from h1_lid import L, V, Model, out_mask          # same domain (L=6, V=5) and the same RoPE model

PRIMS = {}
for _k in range(1, L):
    PRIMS[f"rot{_k}"] = (lambda k: (lambda s: torch.roll(s, k, dims=-1)))(_k)
PRIMS["reverse"] = lambda s: s.flip(-1)
PRIMS["swap_pairs"] = lambda s: s.reshape(*s.shape[:-1], L // 2, 2).flip(-1).reshape(*s.shape)
for _k in range(1, V):
    PRIMS[f"add{_k}"] = (lambda k: (lambda s: (s + k) % V))(_k)
PRIMS["negate"] = lambda s: (V - s) % V
NAMES = list(PRIMS)


def apply_pair(s, pair):
    return PRIMS[NAMES[pair[1]]](PRIMS[NAMES[pair[0]]](s))


def build(seed: int, n_test: int = 20):
    """Every distinct 2-composition, split into a FIXED held-out set and a NESTED pool of training tasks."""
    g = torch.Generator().manual_seed(seed)
    probe = torch.randint(0, V, (24, L), generator=g)
    ident = tuple(probe.flatten().tolist())
    sigs = {}
    for a, b in itertools.product(range(len(NAMES)), repeat=2):
        sig = tuple(apply_pair(probe, (a, b)).flatten().tolist())
        if sig != ident:
            sigs.setdefault(sig, (a, b))
    pairs = list(sigs.values())
    order = torch.randperm(len(pairs), generator=g).tolist()
    test = [pairs[i] for i in order[:n_test]]
    pool = [pairs[i] for i in order[n_test:]]
    return pool, test


def batch(B, K, tasks, dev, g, fixed=None):
    idx = (torch.full((B,), fixed, dtype=torch.long, device=dev) if fixed is not None
           else torch.randint(0, len(tasks), (B,), generator=g, device=dev))
    x = torch.randint(0, V, (B, K, L), generator=g, device=dev)
    y = torch.empty_like(x)
    for t in idx.unique():
        m = idx == t
        y[m] = apply_pair(x[m], tasks[int(t)])
    return torch.stack([x, y], dim=2).reshape(B, K * 2 * L)


@torch.no_grad()
def accuracy(model, tasks, dev, K, n=384):
    """Exact-match accuracy of the LAST demonstration's output — the most in-context evidence the model ever gets."""
    hits = []
    for t in range(len(tasks)):
        tok = batch(n, K, tasks, dev, None, fixed=t)
        pred = model(tok)[:, :-1].argmax(-1)
        msk = out_mask(K, dev)[1:]
        ok = (pred[:, msk] == tok[:, 1:][:, msk]).reshape(n, K, L).all(-1)
        hits.append(ok[:, -1].float().mean().item())
    return sum(hits) / len(hits), sum(h >= 0.8 for h in hits)


def run(n_train, pool, test, args, dev):
    torch.manual_seed(args.seed)
    tasks = pool[:n_train]
    model = Model(max_len=args.k * 2 * L + 2, pos="rope").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98), fused=(dev == "cuda"))
    warm = args.steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
    trainer = torch.compile(model) if dev == "cuda" else model
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for _ in range(args.steps):
        tok = batch(args.batch, args.k, tasks, dev, g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = trainer(tok)[:, :-1]
        m = out_mask(args.k, dev)[1:]
        loss = F.cross_entropy(logits[:, m].reshape(-1, V).float(), tok[:, 1:][:, m].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    tr_acc, tr_solved = accuracy(model, tasks, dev, args.k)
    te_acc, te_solved = accuracy(model, test, dev, args.k)
    return dict(n=n_train, secs=time.time() - t0, tr_acc=tr_acc, tr_solved=tr_solved,
                te_acc=te_acc, te_solved=te_solved)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="4,8,16,32,42")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool, test = build(args.seed)
    sizes = [int(x) for x in args.sizes.split(",") if int(x) <= len(pool)]
    print(f"device {dev} | {len(pool) + len(test)} distinct 2-compositions | "
          f"{len(test)} held out (FIXED) | training pool {len(pool)} | steps={args.steps}")
    print(f"\n{'|train|':>8}{'train acc':>12}{'train solved':>14}{'HELD-OUT acc':>14}{'held-out solved':>17}{'secs':>7}")
    for n in sizes:
        r = run(n, pool, test, args, dev)
        print(f"{r['n']:>8}{r['tr_acc']:>12.3f}{r['tr_solved']:>9}/{n:<4}{r['te_acc']:>14.3f}"
              f"{r['te_solved']:>12}/{len(test):<4}{r['secs']:>7.0f}")
    print("\nDIVERSITY THRESHOLD predicts held-out accuracy ~0 for small |train|, rising sharply past a critical count.")
    print("FLAT held-out accuracy all the way to the largest training set says memorisation is not the whole story.")


if __name__ == "__main__":
    main()
