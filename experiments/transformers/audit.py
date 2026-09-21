"""AUDIT THE PRIOR: does the assignment constraint HURT where it does not apply?

WHY. A prior is not bad for existing. It is bad when it is too large, too complicated, or when it helps one task while
hindering another. The Sinkhorn matching layer scores well on the first two — one layer, no task content, it says only
what KIND of object an answer is — and the third has never been tested. This file tests it.

THE TEST HAS TO BE A CASE WHERE THE PRIOR IS FALSE, not merely unnecessary. If output positions read input positions
through a NON-INJECTIVE map (two outputs reading the same input), the truth is not a permutation at all. A row-stochastic
matrix represents that fine; a DOUBLY-stochastic one cannot — every input must be consumed exactly once. So this is the
case where the prior is actively wrong, and if it ever costs anything it costs it here.

Three structures, each fitted both ways:
  PERMUTATION   `y[a] = vmp[x[f[a]]]`, f a permutation           — the prior is TRUE
  MANY-TO-ONE   the same, f sampled WITH replacement             — the prior is FALSE and the alternative can express it
  GLOBAL        `y[a] = vmp[(Σx) mod V]`, no positional reading   — the prior is IRRELEVANT and BOTH are misspecified

Scored by HELD-OUT RECONSTRUCTION: fit `(P, M)` on K demonstrations, then predict `y` for FRESH inputs the fit never
saw. That is the only metric defined across all three structures — `idx` exact-match is meaningless where the truth is
not a permutation — and it is the honest question anyway, since a model that cannot predict new behaviour has not
captured the function whatever its internals look like.

Usage:  python experiments/transformers/audit.py
"""
from __future__ import annotations

import argparse
import time

import torch

from h1_lid import L, V
from sinkhorn import recon_nll, sinkhorn, solve


def make_tasks(kind, n, dev, g):
    """`f` (n, L) output-position -> input-position, and `vmp` (n, V) a value permutation."""
    if kind == "permutation":
        f = torch.rand(n, L, generator=g, device=dev).argsort(-1)
    else:                                                   # with replacement ⇒ generically NOT injective
        f = torch.randint(0, L, (n, L), generator=g, device=dev)
    vmp = torch.rand(n, V, generator=g, device=dev).argsort(-1)
    return f, vmp


def apply_task(kind, f, vmp, x):
    """`x` is (n, K, L) ⇒ `y` is (n, K, L)."""
    n, K, _ = x.shape
    if kind == "global":
        agg = x.sum(-1) % V                                 # depends on ALL of x; no single position can be read
        return vmp[:, None, :].expand(n, K, V).gather(2, agg[:, :, None]).expand(n, K, L)
    xg = x.gather(2, f[:, None, :].expand(n, K, L))
    return vmp[:, None, :].expand(n, K, V).gather(2, xg)


def fit_and_test(kind, f, vmp, K, doubly, args, dev, g):
    """Fit on K demonstrations, then predict `y` for FRESH inputs. Returns (per-slot, whole-sequence) accuracy."""
    n = f.shape[0]
    x = torch.randint(0, V, (n, K, L), generator=g, device=dev)
    xy = torch.cat([x, apply_task(kind, f, vmp, x)], dim=2)
    P, M, _ = solve(xy, args.solver_steps, 0.1, args.sink_iter, args.tau, doubly, args.seed)
    fh, vh = P.argmax(-1), M.argmax(-1)                     # the read map and value map the fit believes in
    xt = torch.randint(0, V, (n, args.test, L), generator=g, device=dev)
    yt = apply_task(kind, f, vmp, xt)
    pred = vh[:, None, :].expand(n, args.test, V).gather(2, xt.gather(2, fh[:, None, :].expand(n, args.test, L)))
    ok = (pred == yt)
    return ok.float().mean().item(), ok.all(-1).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=400)
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--test", type=int, default=16, help="fresh inputs per task, unseen by the fit")
    ap.add_argument("--solver_steps", type=int, default=400)
    ap.add_argument("--sink_iter", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print(f"device {dev} | {args.tasks} tasks per structure | fit on {args.demos} demos, "
          f"tested on {args.test} FRESH inputs")
    print("the question is whether the doubly-stochastic constraint COSTS anything where it does not hold.\n")
    print(f"{'structure':<14}{'prior':<12}{'constraint':<14}{'slot acc':>10}{'exact seq':>11}{'secs':>7}")
    for kind, verdict in (("permutation", "TRUE"), ("many-to-one", "FALSE"), ("global", "IRRELEVANT")):
        g = torch.Generator(device=dev).manual_seed(args.seed)
        f, vmp = make_tasks(kind, args.tasks, dev, g)
        for doubly in (True, False):
            gg = torch.Generator(device=dev).manual_seed(args.seed + 1)
            t0 = time.time()
            slot, ex = fit_and_test(kind, f, vmp, args.demos, doubly, args, dev, gg)
            name = "sinkhorn" if doubly else "row-softmax"
            print(f"{kind:<14}{verdict:<12}{name:<14}{slot:>10.3f}{ex:>11.3f}{time.time() - t0:>7.0f}")

    print(f"\nChance is {1/V:.3f} per slot. A prior that is merely UNNECESSARY should cost nothing; a prior that is")
    print("HARMFUL should show row-softmax winning on the many-to-one row, where a permutation cannot express the truth.")
    print("The global row is the control: no positional reading exists, so both parameterisations are misspecified and")
    print("neither should do well — if one does, the metric is leaking rather than the prior helping.")


if __name__ == "__main__":
    main()
