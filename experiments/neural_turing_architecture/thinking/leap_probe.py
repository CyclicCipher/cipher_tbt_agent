"""The leap probe (planning doc §21; pre-registered before it ran).

Does the answer to a (family, level, format) correlate with LOW-DEGREE functions of what a frozen model already
computes? If a linear readout of the model's features already predicts it, the level is one step away (leap 1); if only
a quadratic one does, two pieces must be learned together (leap 2) — the plateau that one token per pair removed from
`ptr` (§20.1). The probe costs a regression, not a training run.

Features: the residual stream at each problem's answer position after EVERY block application (prelude, each core
pass, coda), concatenated. Probes: ridge classifiers on one-hot targets, lambda chosen on a validation split, scored on
held-out problems and normalised by the majority-class rate —
    p0  the model's own head (no fitting)
    p1  degree 1: ridge on the standardised features
    p2  degree 2: ridge on the features plus 4,096 random quadratic features (w . x)^2
leap = 0 if p0 >= 0.9, else 1 if p1 >= 0.5, else 2 if p2 >= 0.5, else ">2".

Test 1 (`--test1`): for each of twelve items, the probe on a freshly initialised `loop4`, then the SAME initialisation
trained on that item alone (3,000 steps x 128 problems), scored on 1,024 fresh problems. Writes `runs/leap/test1.json`.

Usage:  python leap_probe.py --test1
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
import tasks  # noqa: E402

LEAP_P0, LEAP_P = 0.9, 0.5


# ------------------------------------------------------------------------------------------------------------ features
@torch.no_grad()
def features(model, fam, li, n, rng, dev="cuda", chunk=500):
    """(features at the answer position after every block application, the head's prediction, the answer)."""
    blocks = list(getattr(model, "prelude", [])) + list(model.blocks) + list(getattr(model, "coda", []))
    store = []
    hooks = [b.register_forward_hook(lambda _m, _i, out: store.append(out)) for b in blocks]
    feats, preds, answers = [], [], []
    try:
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            tok, ans, _, _, last = tasks.make_batch([fam], rng, m, dev, level=li)
            store.clear()
            logits = model(tok)
            idx = torch.arange(m, device=dev)
            feats.append(torch.cat([h[idx, last].float() for h in store], dim=-1))
            preds.append(logits[idx, last].argmax(-1))
            answers.append(ans)
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(feats), torch.cat(preds), torch.cat(answers)


def normalised(acc, y):
    chance = float(torch.bincount(y).max()) / len(y)
    return (acc - chance) / (1 - chance) if chance < 0.999 else float("nan")


def ridge_probe(X, y, n_tr, n_va, lambdas=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Ridge classifier on one-hot targets; lambda (relative to the trace of X'X / D) chosen on the validation split.
    Returns normalised held-out accuracy."""
    X = X.double()
    mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-6
    X = (X - mu) / sd
    X = torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], dim=1)
    classes, yi = torch.unique(y, return_inverse=True)
    Y = F.one_hot(yi, len(classes)).double()
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    G, B = Xtr.T @ Xtr, Xtr.T @ Ytr
    scale = float(torch.diagonal(G).mean())
    best = None
    for lam in lambdas:
        W = torch.linalg.solve(G + lam * scale * torch.eye(len(G), device=X.device, dtype=X.dtype), B)
        va = float(((X[n_tr:n_tr + n_va] @ W).argmax(1) == yi[n_tr:n_tr + n_va]).double().mean())
        if best is None or va > best[0]:
            best = (va, W)
    te = float(((X[n_tr + n_va:] @ best[1]).argmax(1) == yi[n_tr + n_va:]).double().mean())
    return normalised(te, y[n_tr + n_va:])


def quadratic(X, r=4096, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    W = torch.randn(X.shape[1], r, generator=g, device=X.device) / math.sqrt(X.shape[1])
    return torch.cat([Xs, (Xs @ W) ** 2], dim=1)


def leap_scores(model, fam, li, n=10000, seed=1, dev="cuda"):
    rng = np.random.default_rng(seed)
    model.eval()
    X, pred, y = features(model, fam, li, n, rng, dev)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    te = slice(n_tr + n_va, n)
    p0 = normalised(float((pred[te] == y[te]).float().mean()), y[te])
    p1 = ridge_probe(X, y, n_tr, n_va)
    p2 = ridge_probe(quadratic(X), y, n_tr, n_va)
    leap = 0 if p0 >= LEAP_P0 else 1 if p1 >= LEAP_P else 2 if p2 >= LEAP_P else ">2"
    return dict(p0=round(p0, 4), p1=round(p1, 4), p2=round(p2, 4), leap=leap)


# ------------------------------------------------------------------------------------------------------------- test 1
def make_model(seed=0, dev="cuda"):
    torch.manual_seed(seed)
    return H1.LoopedModel(d_model=128, n_head=4, max_len=48, pos="rope", n_vocab=tasks.V, loops=4, tied=True,
                          n_prelude=1, n_coda=1).to(dev)


def train_solo(model, fam, li, steps=3000, batch=128, lr=1e-3, seed=0, dev="cuda"):
    """Train on ONE (family, level) alone; returns (normalised accuracy on 1,024 fresh problems, first step at which the
    running training accuracy reached 0.9 or None)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    rng, ema, hit = np.random.default_rng(seed), None, None
    for step in range(steps):
        tok, ans, _, _, last = tasks.make_batch([fam], rng, batch, dev, level=li)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(tok)[torch.arange(batch, device=dev), last]
        loss = F.cross_entropy(logits.float(), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        acc = float((logits.argmax(-1) == ans).float().mean())
        ema = acc if ema is None else 0.95 * ema + 0.05 * acc
        if hit is None and ema >= 0.9 and step >= 20:
            hit = step + 1
    model.eval()
    rng = np.random.default_rng(12345)
    with torch.no_grad():
        tok, ans, _, _, last = tasks.make_batch([fam], rng, 1024, dev, level=li)
        pred = model(tok)[torch.arange(1024, device=dev), last].argmax(-1)
    return normalised(float((pred == ans).float().mean()), ans), hit


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test1(args):
    bf = tasks.Brainfuck(seed=0, per_level=20000, cache_dir=HERE / "runs" / "p0")
    items = [("ptr2", tasks.PointerChaseTwoToken(), 0), ("ptr", tasks.PointerChase(), 0), ("ptr", tasks.PointerChase(), 1),
             ("aff", tasks.AffineMod(), 0), ("aff", tasks.AffineMod(), 1), ("s5", tasks.S5Word(), 0),
             ("s5", tasks.S5Word(), 3), ("ca-index", tasks.CellularAutomaton("index"), 0),
             ("ca-marked", tasks.CellularAutomaton("marked"), 0),
             ("ca-marked_pairs", tasks.CellularAutomaton("marked_pairs"), 0),
             ("bool", tasks.BoolFormula(), 3), ("bf", bf, 4)]
    rows, t0 = [], time.time()
    for name, fam, li in items:
        model = make_model(seed=0)
        init = leap_scores(model, fam, li)
        final, hit = train_solo(model, fam, li, steps=args.steps)
        trained = leap_scores(model, fam, li)
        rows.append(dict(item=name, h=fam.levels[li], init=init, solo_final=round(final, 4), solo_steps_to_0_9=hit,
                         after=trained))
        print(f"{name:16s} h={fam.levels[li]:<4} init p1 {init['p1']:+.2f} p2 {init['p2']:+.2f} (leap {init['leap']})"
              f"  | solo {args.steps} steps: {final:+.2f}, 0.9 at {hit}  | after: p0 {trained['p0']:+.2f}"
              f"  {time.time() - t0:5.0f}s", flush=True)
    p2s, p1s, fin = [r["init"]["p2"] for r in rows], [r["init"]["p1"] for r in rows], [r["solo_final"] for r in rows]
    out = dict(rows=rows, spearman_p2=spearman(p2s, fin), spearman_p1=spearman(p1s, fin))
    print(f"\nL1: Spearman(p2 at init, solo accuracy) = {out['spearman_p2']:+.2f}   (p1: {out['spearman_p1']:+.2f})")
    (HERE / "runs" / "leap").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(HERE / "runs" / "leap" / "test1.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test1", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    if args.test1:
        test1(args)
