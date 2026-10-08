"""Experiment 1 model (EXPERIMENT_1.md): point features -> the shared h1_lid transformer -> one of three heads.

Every head works at EVERY position t of the causal transformer, i.e. for every context size n = t + 1 at once:
  MixtureHead(horizons=1)   A0  next point                    — the standard next-step objective
  MixtureHead(horizons=16)  A2  the point k = 1..16 ahead      — candidate 3: a density queryable at any (x, y), any k
  ParticleHead              A1  16 whole continuations        — candidate 2: sampled continuations (energy score)
Means and particles are predicted as OFFSETS from the current point (identical in every arm).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "transformers"))
import h1_lid as H1  # noqa: E402

K = 16                       # horizons predicted
FREQS = (1.0, 2.0, 4.0, 8.0)
LOG_S_MIN, LOG_S_MAX = math.log(0.002), math.log(0.5)


def features(p, diff=False):
    """(B, L, 2) points -> coordinates plus sin/cos at frequencies 1, 2, 4, 8 (same for every arm). With `diff`, also
    the STEP since the previous mark, Δ_t = p_t − p_{t−1} (0 at t = 0), and sin/cos(2π f Δ) for f = 1, 2, 4 — the time
    derivative as a first-class input (the notes: time as a first-class quantity); sin/cos(2πΔ) also see through a
    wrap-around modulo 1."""
    ang = 2 * math.pi * p[..., None] * torch.tensor(FREQS, device=p.device)
    out = [p, ang.sin().flatten(-2), ang.cos().flatten(-2)]
    if diff:
        d = torch.cat([torch.zeros_like(p[:, :1]), p[:, 1:] - p[:, :-1]], 1)
        dang = 2 * math.pi * d[..., None] * torch.tensor(FREQS[:3], device=p.device)
        out += [d, dang.sin().flatten(-2), dang.cos().flatten(-2)]
    return torch.cat(out, -1)


def n_features(diff=False):
    return 2 + 4 * len(FREQS) + (2 + 4 * 3 if diff else 0)


class Backbone(nn.Module):
    """`loops` > 0: an h1_lid LoopedModel (prelude block, ONE tied core block applied `loops` times with the boundary
    operator, coda block) in place of `layers` untied blocks — the same depth from fewer weights."""

    def __init__(self, d=64, layers=3, heads=4, max_len=64, diff=False, loops=0):
        super().__init__()
        self.diff = diff
        self.inp = nn.Linear(n_features(diff), d)
        if loops:
            self.tf = H1.LoopedModel(d_model=d, n_head=heads, max_len=max_len, pos="rope", n_vocab=2, loops=loops, tied=True)
        else:
            self.tf = H1.Model(d_model=d, n_layer=layers, n_head=heads, max_len=max_len, pos="rope", n_vocab=2)

    def embed(self, p):
        return self.inp(features(p, self.diff))

    def forward(self, p):
        return self.tf.norm(self.tf.forward_embedded(self.embed(p)))


class MixtureHead(nn.Module):
    """For each of `horizons` steps ahead, a C-component axis-aligned Gaussian mixture over the point."""

    def __init__(self, d, horizons, C=8):
        super().__init__()
        self.H, self.C = horizons, C
        self.out = nn.Linear(d, horizons * C * 5)

    def forward(self, h, p):
        o = self.out(h).view(*h.shape[:2], self.H, self.C, 5)
        logw = o[..., 0].log_softmax(-1)
        mu = p[:, :, None, None, :] + o[..., 1:3]
        logs = (o[..., 3:5] - 2.5).clamp(LOG_S_MIN, LOG_S_MAX)
        return logw, mu, logs


class QueryMixtureHead(nn.Module):
    """Candidate 3 with the horizon as an INPUT: an MLP over (state, code of k) gives the mixture for horizon k, so one
    network answers any k (the density is queryable in time as well as space). Same outputs as MixtureHead(K)."""

    def __init__(self, d, C=8, hidden=128, kfreq=(1, 2, 4, 8), hybrid=False, past=False, unif=False):
        super().__init__()
        self.C, self.unif = C, unif
        self.linear = nn.Linear(d, K * C * 5) if hybrid else None
        # with `past`, the head also answers k = -K..-1 (where the marks WERE), trained on the context itself; the
        # future horizons are always the LAST K entries
        klist = list(range(-K, 0)) + list(range(1, K + 1)) if past else list(range(1, K + 1))
        self.nk = len(klist)
        ks = torch.tensor(klist, dtype=torch.float32)[:, None]
        ang = 2 * math.pi * ks * torch.tensor(kfreq, dtype=torch.float32) / (2 * K)
        self.register_buffer("kcode", torch.cat([ks / K, ang.sin(), ang.cos()], 1), persistent=False)   # (K, 1+2F)
        self.net = nn.Sequential(nn.Linear(d + self.kcode.shape[1], hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, C * 5))
        # with `unif`, one more output per (position, horizon): the log-odds of a uniform component over the square
        self.unif_net = nn.Linear(d + self.kcode.shape[1], 1) if unif else None
        self.last_log_u = None

    def forward(self, h, p):
        B, Lh, _ = h.shape
        x = torch.cat([h[:, :, None].expand(-1, -1, self.nk, -1), self.kcode.to(h.dtype)[None, None].expand(B, Lh, -1, -1)], -1)
        o = self.net(x).view(B, Lh, self.nk, self.C, 5)
        self.last_log_u = F.logsigmoid(self.unif_net(x).squeeze(-1) - 3.0) if self.unif_net is not None else None
        if self.linear is not None:                         # hybrid: plus a separate linear output per horizon
            o = o + self.linear(h).view(B, Lh, K, self.C, 5)
        logw = o[..., 0].log_softmax(-1)
        mu = p[:, :, None, None, :] + o[..., 1:3]
        logs = (o[..., 3:5] - 2.5).clamp(LOG_S_MIN, LOG_S_MAX)
        return logw, mu, logs


def mixture_logpdf(logw, mu, logs, y):
    """log density (nats) of y (..., 2) under the mixture (..., C, ·)."""
    z = (y[..., None, :] - mu) / logs.exp()
    comp = -0.5 * (z ** 2).sum(-1) - logs.sum(-1) - math.log(2 * math.pi)
    return torch.logsumexp(logw + comp, -1)


def with_uniform(lp, log_u):
    """log of (1 - u) * mixture + u * 1 (the uniform density on the unit square is 1)."""
    if log_u is None:
        return lp
    return torch.logaddexp(lp + torch.log1p(-log_u.exp().clamp(max=1 - 1e-6)), log_u)


def mixture_sample(logw, mu, logs, gen=None):
    c = torch.distributions.Categorical(logits=logw).sample()
    m = mu.gather(-2, c[..., None, None].expand(*c.shape, 1, 2)).squeeze(-2)
    s = logs.gather(-2, c[..., None, None].expand(*c.shape, 1, 2)).squeeze(-2).exp()
    return m + s * torch.randn_like(m)


class ParticleHead(nn.Module):
    """From the state and a random code z ~ N(0, I), an MLP emits the next K points (as offsets from the current one)."""

    def __init__(self, d, zdim=16, hidden=128):
        super().__init__()
        self.zdim = zdim
        self.net = nn.Sequential(nn.Linear(d + zdim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, K * 2))

    def forward(self, h, p, P):
        z = torch.randn(*h.shape[:2], P, self.zdim, device=h.device, dtype=h.dtype)
        x = torch.cat([h[:, :, None].expand(-1, -1, P, -1), z], -1)
        return p[:, :, None, None, :] + self.net(x).view(*h.shape[:2], P, K, 2)


def targets(p):
    """Y[b, t, k-1] = p[b, t + k] and its validity mask (t + k < L)."""
    B, L, _ = p.shape
    Y = torch.zeros(B, L, K, 2, device=p.device)
    m = torch.zeros(B, L, K, device=p.device)
    for k in range(1, K + 1):
        Y[:, :L - k, k - 1] = p[:, k:]
        m[:, :L - k, k - 1] = 1
    return Y, m


def past_targets(p):
    """Z[b, t, j] = p[b, t - (K - j)] for j = 0..K-1 (k = -K..-1) and its validity mask (t + k >= 0)."""
    B, L, _ = p.shape
    Z = torch.zeros(B, L, K, 2, device=p.device)
    m = torch.zeros(B, L, K, device=p.device)
    for j in range(K):
        k = K - j
        Z[:, k:, j] = p[:, :L - k]
        m[:, k:, j] = 1
    return Z, m


def energy_score(X, y, m):
    """Energy score per (sequence, position): X (B, L, P, K, 2) particles, y (B, L, K, 2), m (B, L, K) validity."""
    w = m[..., None]
    Xm, ym = (X * w[:, :, None]).flatten(-2), (y * w).flatten(-2)
    d1 = ((Xm - ym[:, :, None]) ** 2).sum(-1).add(1e-8).sqrt().mean(-1)
    P = X.shape[2]
    pair = ((Xm[:, :, :, None] - Xm[:, :, None]) ** 2).sum(-1).add(1e-8).sqrt()
    d2 = (pair.sum((-1, -2)) - P * 1e-4) / (P * (P - 1))     # mean over i != j: the unbiased, strictly proper form
    return d1 - 0.5 * d2
