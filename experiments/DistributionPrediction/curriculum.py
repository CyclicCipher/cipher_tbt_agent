"""Experiment 2 (EXPERIMENT_2.md): the four ways of weighting the 28 buckets (7 families × 4 horizon groups).

Every rule produces per-bucket multipliers c (28,) for the loss Σ_b c_b·sum_b / Σ_b c_b·count_b, except AF, which keeps
a subset of TERMS. Curricula start after `start` steps (uniform before) and mix 20% uniform: c = 0.8·s/mean(s) + 0.2.
"""
from __future__ import annotations

import torch

N_BUCKETS = 28
HGROUP = torch.tensor([0] + [1] * 3 + [2] * 4 + [3] * 8)          # horizon k = 1..16 -> group 0..3


def mix(s):
    s = s.clamp(min=0)
    return 0.8 * s / s.mean().clamp(min=1e-12) + 0.2


class Uniform:
    name = "U"

    def __init__(self, dev, start=100):
        self.c = torch.ones(N_BUCKETS, device=dev)

    def observe(self, step, L_b):
        pass

    def multipliers(self):
        return self.c


class LearningProgress(Uniform):
    """s_b = |EMA_slow(L_b) − EMA_fast(L_b)| — how fast bucket b's loss is moving (absolute learning progress)."""
    name = "LP"

    def __init__(self, dev, start=100, fast=0.1, slow=0.01):
        super().__init__(dev)
        self.start, self.fast_r, self.slow_r = start, fast, slow
        self.fast = self.slow = None

    def observe(self, step, L_b):
        L_b = L_b.detach()
        if self.fast is None:
            self.fast, self.slow = L_b.clone(), L_b.clone()
        else:
            self.fast += self.fast_r * (L_b - self.fast)
            self.slow += self.slow_r * (L_b - self.slow)
        if step >= self.start:
            self.c = mix((self.slow - self.fast).abs())


class GradientAlignment(Uniform):
    """s_b = |<grad L_b, P ⊙ (θ_past − θ_now)>| (Self-Play Pretraining with Zero Data, eq. 2): θ_past is the snapshot
    nearest half the steps so far; P = lr/(sqrt(v̂) + eps), Adam's per-weight step. Recomputed every `every` steps."""
    name = "GA"

    def __init__(self, dev, start=100, every=100, snap_every=50):
        super().__init__(dev)
        self.start, self.every, self.snap_every = start, every, snap_every
        self.snaps = {}

    def snapshot(self, step, params):
        if step % self.snap_every == 0:
            self.snaps[step] = [q.detach().clone() for q in params]

    def due(self, step):
        return step >= self.start and step % self.every == 0

    def refresh(self, step, bucket_losses, params, opt):
        """bucket_losses: (28,) tensor with graph (buckets with no terms may be 0 without graph)."""
        past_step = min(self.snaps, key=lambda s: abs(s - step // 2))
        past = self.snaps[past_step]
        P, delta = [], []
        for q, qp in zip(params, past):
            st = opt.state.get(q, {})
            if "exp_avg_sq" not in st:
                P.append(None)
                delta.append(None)
                continue
            b2 = opt.param_groups[0]["betas"][1]
            t = float(st["step"])
            vhat = st["exp_avg_sq"] / (1 - b2 ** t)
            P.append(opt.param_groups[0]["lr"] / (vhat.sqrt() + opt.param_groups[0]["eps"]))
            delta.append(qp - q.detach())
        s = torch.zeros(N_BUCKETS, device=bucket_losses.device)
        for b in range(N_BUCKETS):
            if not bucket_losses[b].requires_grad:
                continue
            g = torch.autograd.grad(bucket_losses[b], params, retain_graph=True, allow_unused=True)
            s[b] = sum((gi * Pi * di).sum() for gi, Pi, di in zip(g, P, delta) if gi is not None and Pi is not None).abs()
        self.c = mix(s)
        return past_step


class AdvantageFilter(Uniform):
    """Per term: advantage = loss − its bucket's running mean loss; only the half of the terms with the largest
    |advantage| enter the loss (Ataraxos's advantage filtering)."""
    name = "AF"

    def __init__(self, dev, start=100, rate=0.05):
        super().__init__(dev)
        self.start, self.rate = start, rate
        self.mean = None

    def observe(self, step, L_b):
        L_b = L_b.detach()
        self.mean = L_b.clone() if self.mean is None else self.mean + self.rate * (L_b - self.mean)

    def term_mask(self, step, nll, ids, m):
        if step < self.start or self.mean is None:
            return m
        adv = (nll.detach() - self.mean[ids]).abs()
        thr = adv[m > 0].median()
        return m * (adv >= thr).float()


ARMS = dict(U=Uniform, LP=LearningProgress, GA=GradientAlignment, AF=AdvantageFilter)
