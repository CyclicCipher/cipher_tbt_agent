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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "transformers"))
import h1_lid as H1  # noqa: E402

K = 16                       # horizons predicted
FREQS = (1.0, 2.0, 4.0, 8.0)
LOG_S_MIN, LOG_S_MAX = math.log(0.002), math.log(0.5)


def features(p):
    """(B, L, 2) points -> coordinates plus sin/cos at frequencies 1, 2, 4, 8 (same for every arm)."""
    ang = 2 * math.pi * p[..., None] * torch.tensor(FREQS, device=p.device)
    return torch.cat([p, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)


class Backbone(nn.Module):
    def __init__(self, d=64, layers=3, heads=4, max_len=64):
        super().__init__()
        self.inp = nn.Linear(2 + 4 * len(FREQS), d)
        self.tf = H1.Model(d_model=d, n_layer=layers, n_head=heads, max_len=max_len, pos="rope", n_vocab=2)

    def forward(self, p):
        return self.tf.norm(self.tf.forward_embedded(self.inp(features(p))))


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


def mixture_logpdf(logw, mu, logs, y):
    """log density (nats) of y (..., 2) under the mixture (..., C, ·)."""
    z = (y[..., None, :] - mu) / logs.exp()
    comp = -0.5 * (z ** 2).sum(-1) - logs.sum(-1) - math.log(2 * math.pi)
    return torch.logsumexp(logw + comp, -1)


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


def energy_score(X, y, m):
    """Energy score per (sequence, position): X (B, L, P, K, 2) particles, y (B, L, K, 2), m (B, L, K) validity."""
    w = m[..., None]
    Xm, ym = (X * w[:, :, None]).flatten(-2), (y * w).flatten(-2)
    d1 = ((Xm - ym[:, :, None]) ** 2).sum(-1).add(1e-8).sqrt().mean(-1)
    P = X.shape[2]
    pair = ((Xm[:, :, :, None] - Xm[:, :, None]) ** 2).sum(-1).add(1e-8).sqrt()
    d2 = (pair.sum((-1, -2)) - P * 1e-4) / (P * (P - 1))     # mean over i != j: the unbiased, strictly proper form
    return d1 - 0.5 * d2
