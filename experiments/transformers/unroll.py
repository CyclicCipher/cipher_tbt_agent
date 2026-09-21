"""FIX AMORTISATION: unroll the solver instead of guessing its answer.

THE GAP. `sinkhorn.py` measured a per-task solver at **1.000** and a one-forward-pass amortised network at **0.756**,
with train ≈ held-out — so the shortfall is OPTIMISATION, not generalisation. One forward pass is simply worse than 400
gradient steps on the same information. And by the no-metric result (`endtoend.py`), 0.756 is not "most of the way
there": a canonical state wrong anywhere names a DIFFERENT function, so this is a 24% total failure rate downstream.

Meanwhile 400 Adam steps per task is fine at this scale and hopeless beyond it. So the target is to get the solver's
accuracy at something much closer to the network's cost.

THE METHOD. The amortised network emits the INITIAL logits, and the reconstruction objective is then descended for T
differentiable steps inside the forward pass, with a learned step size. Training is end-to-end through all T steps, on
the FINAL iterate — so the network is no longer asked to produce a good ANSWER, it is asked to produce a good STARTING
POINT, which is a different and easier thing to learn.

Gradient steps are RMS-NORMALISED per item. Plain gradient descent on this objective is badly scaled — the solver uses
Adam at lr 0.1, and eight raw-gradient steps move almost nowhere — so the update is `lp -= a · g/‖g‖`, which is
scale-free, differentiable, and gives the learned `a` a meaningful range.

The objective stays label-free: reconstruction of executed demonstrations, exactly as in `sinkhorn.py`. `idx` and `vmp`
are never supplied.

MEASURED:
  1. Held-out recovery for T = 0 (the 0.756 baseline, reproduced here) against the unrolled network.
  2. Whether the learned INITIALISATION helps a real solve converge faster — Adam from the network's logits vs from
     noise, at matched step counts. This separates "the network learned a good answer" from "the network learned a good
     basin", which are different claims and only the second survives if T-step refinement is doing the work.

Usage:  python experiments/transformers/unroll.py
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn

from bigroup import cayley, compile_progs, stratify
from canonicaliser import views
from h1_lid import L, V
from sinkhorn import Matcher, recon_nll, score, sinkhorn, split

MAXM = 5


class Unrolled(nn.Module):
    """Amortised initialisation + T differentiable refinement steps on the reconstruction objective."""

    def __init__(self, T, agg="embed", alpha0=0.5):
        super().__init__()
        self.net, self.T = Matcher(agg=agg), T
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha0)))

    def forward(self, xy, n_iter, tau):
        lp, lm = self.net(xy)
        x, y = xy[:, :, :L], xy[:, :, L:]
        a = self.log_alpha.exp()
        for _ in range(self.T):
            loss, _, _ = recon_nll(lp, lm, x, y, n_iter, tau)
            glp, glm = torch.autograd.grad(loss, [lp, lm], create_graph=self.training)
            lp = lp - a * glp / (glp.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8)
            lm = lm - a * glm / (glm.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-8)
        return lp, lm


def adam_refine(lp0, lm0, x, y, steps, n_iter, tau, lr=0.1):
    """A REAL solve started from supplied logits — the test-time question of whether the learned init finds a better
    basin, as opposed to a better answer."""
    lp = lp0.detach().clone().requires_grad_(True)
    lm = lm0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([lp, lm], lr=lr)
    for _ in range(steps):
        loss, _, _ = recon_nll(lp, lm, x, y, n_iter, tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return sinkhorn(lp, n_iter, tau), sinkhorn(lm, n_iter, tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--held", type=int, default=400)
    ap.add_argument("--eval_cap", type=int, default=200)
    ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sink_iter", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--agg", default="embed", choices=["pair", "embed"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    _p, table = cayley(dev, args.seed)
    progs = [w for w in table.values() if 0 < len(w) <= MAXM]
    train, held = split(progs, args.held, args.n_train, args.seed)
    he = stratify(held, args.eval_cap)
    tabs_tr, tabs_he = compile_progs(train, dev), compile_progs(he, dev)
    print(f"device {dev} | train {len(train)} | held-out {len(held)} | eval {len(he)} | {args.demos} demos")
    print("objective: reconstruction only. `idx` and `vmp` are never supplied.\n")

    g_he = torch.Generator(device=dev).manual_seed(args.seed + 777)
    xy_he = views(torch.arange(len(he), device=dev), args.demos, tabs_he, dev, g_he)
    idx_he, vmp_he = tabs_he

    print(f"{'model':<16}{'T':>4}{'IDX exact':>12}{'IDX slot':>11}{'VMP exact':>12}{'secs':>7}")
    trained = {}
    for T in (0, args.T):
        net = Unrolled(T, agg=args.agg).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
        warm = args.steps // 20
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(1, warm) if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / (args.steps - warm))))
        gg = torch.Generator(device=dev).manual_seed(args.seed)
        t0 = time.time()
        for _ in range(args.steps):
            sel = torch.randperm(len(train), generator=gg, device=dev)[:args.batch]
            xy = views(sel, args.demos, tabs_tr, dev, gg)
            lp, lm = net(xy, args.sink_iter, args.tau)
            loss, _, _ = recon_nll(lp, lm, xy[:, :, :L], xy[:, :, L:], args.sink_iter, args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        lp, lm = net(xy_he, args.sink_iter, args.tau)
        with torch.no_grad():
            _l, P, M = recon_nll(lp, lm, xy_he[:, :, :L], xy_he[:, :, L:], args.sink_iter, args.tau)
        ie, isl, ve, _ = score(P, M, idx_he, vmp_he)
        tag = "matcher (T=0)" if T == 0 else "unrolled"
        print(f"{tag:<16}{T:>4}{ie:>12.3f}{isl:>11.3f}{ve:>12.3f}{time.time() - t0:>7.0f}")
        # Store the network's INITIAL emission, not the post-refinement logits. Refining from the latter would only
        # re-measure the already-solved answer and would look like a spectacular init for no reason.
        with torch.no_grad():
            trained[T] = tuple(z.detach() for z in net.net(xy_he))

    # Does the learned init find a better BASIN? Adam from the network's logits vs from noise, matched steps.
    print(f"\n{'init':<16}{'adam steps':>12}{'IDX exact':>12}{'IDX slot':>11}{'VMP exact':>12}")
    x, y = xy_he[:, :, :L], xy_he[:, :, L:]
    gn = torch.Generator(device=dev).manual_seed(args.seed)
    noise = (0.01 * torch.randn(len(he), L, L, generator=gn, device=dev),
             0.01 * torch.randn(len(he), V, V, generator=gn, device=dev))
    for name, (lp0, lm0) in (("noise", noise), ("matcher init", trained[0]), ("unrolled init", trained[args.T])):
        for st in (10, 40, 200):
            P, M = adam_refine(lp0, lm0, x, y, st, args.sink_iter, args.tau)
            ie, isl, ve, _ = score(P, M, idx_he, vmp_he)
            print(f"{name:<16}{st:>12}{ie:>12.3f}{isl:>11.3f}{ve:>12.3f}")

    print("\nReference: the per-task solver from noise reaches 1.000 at 400 Adam steps and 8 demonstrations;")
    print("one-shot amortisation reached 0.756 (`sinkhorn.py`). A learned INIT that converges in far fewer steps is a")
    print("different and more useful result than a learned ANSWER, because it keeps the solver's accuracy at a fraction")
    print("of its cost. By the no-metric result, anything short of 1.000 is a total failure rate, not a partial credit.")


if __name__ == "__main__":
    main()
