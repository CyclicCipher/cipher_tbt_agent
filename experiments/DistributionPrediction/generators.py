"""Experiment 1 data: ordered 2-D marks from seven rule families (EXPERIMENT_1.md). One task per sequence.

Each family returns a (n, L, 2) tensor of points in the unit square, IN THE ORDER THEY WERE MADE. Everything except the
pen is vectorised in torch on the device; the pen rule (`make_figures.draw_marks`) is sequential, so pen sequences come
from a pool built once and cached in `runs/e1/pen_pool_L{L}.pt`.

`batch(per_family, L, g, dev, ood=False)` -> points (B, L, 2), family index (B,).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

FAMILIES = ["circle", "spiral", "lissajous", "billiard", "pen", "rotation", "uniform"]
STRUCTURED = ["circle", "spiral", "lissajous", "billiard", "rotation"]
SIGMA = 0.005
HERE = Path(__file__).resolve().parent


def _u(n, lo, hi, g, dev):
    return lo + (hi - lo) * torch.rand(n, generator=g, device=dev)


def _sign(n, g, dev):
    return torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -1.0, 1.0)


def _noise(p, g):
    return p + SIGMA * torch.randn(p.shape, generator=g, device=p.device)


def circle(n, L, g, dev, ood=False):
    r = _u(n, 0.3, 0.4, g, dev) if ood else _u(n, 0.1, 0.3, g, dev)
    c = r[:, None] + 0.02 + torch.rand(n, 2, generator=g, device=dev) * (1 - 2 * r[:, None] - 0.04)
    w = _sign(n, g, dev) * _u(n, 0.15, 0.35, g, dev)
    th = _u(n, 0, 2 * math.pi, g, dev)[:, None] + w[:, None] * torch.arange(L, device=dev)
    return _noise(c[:, None] + r[:, None, None] * torch.stack([th.cos(), th.sin()], -1), g)


def spiral(n, L, g, dev, ood=False):
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        b = _sign(m, g, dev) * (_u(m, 0.12, 0.2, g, dev) if ood else _u(m, 0.02, 0.12, g, dev))
        w = _sign(m, g, dev) * _u(m, 0.1, 0.25, g, dev)
        r0 = _u(m, 0.03, 0.35, g, dev)
        t = torch.arange(L, device=dev)
        r = r0[:, None] * torch.exp(b[:, None] * w.abs()[:, None] * t)
        ok = (r.max(1).values <= 0.45) & (r.min(1).values >= 0.01)
        rmax = r.max(1).values
        c = rmax[:, None] + 0.02 + torch.rand(m, 2, generator=g, device=dev) * (1 - 2 * rmax[:, None] - 0.04).clamp(min=0)
        th = _u(m, 0, 2 * math.pi, g, dev)[:, None] + w[:, None] * t
        p = c[:, None] + r[..., None] * torch.stack([th.cos(), th.sin()], -1)
        out.append(p[ok])
    return _noise(torch.cat(out)[:n], g)


RATIOS = [(1, 2), (2, 1), (1, 3), (3, 2)]


def lissajous(n, L, g, dev, ood=False):
    if ood:
        pq = torch.tensor([[2.0, 3.0]], device=dev).expand(n, 2)
    else:
        pq = torch.tensor(RATIOS, dtype=torch.float32, device=dev)[torch.randint(0, 4, (n,), generator=g, device=dev)]
    w = _u(n, 0.08, 0.2, g, dev)
    A = torch.stack([_u(n, 0.1, 0.3, g, dev), _u(n, 0.1, 0.3, g, dev)], 1)
    c = A + 0.02 + torch.rand(n, 2, generator=g, device=dev) * (1 - 2 * A - 0.04)
    ph = _u(n * 2, 0, 2 * math.pi, g, dev).view(n, 2)
    t = torch.arange(L, device=dev)
    arg = pq[:, None, :] * w[:, None, None] * t[None, :, None] + ph[:, None, :]
    return _noise(c[:, None] + A[:, None] * torch.sin(arg), g)


def billiard(n, L, g, dev, ood=False):
    s = _u(n, 0.05, 0.08, g, dev) if ood else _u(n, 0.02, 0.05, g, dev)
    th = _u(n, 0, 2 * math.pi, g, dev)
    p0 = 0.1 + 0.8 * torch.rand(n, 2, generator=g, device=dev)
    u = p0[:, None] + (s[:, None] * torch.stack([th.cos(), th.sin()], -1))[:, None] * torch.arange(L, device=dev)[None, :, None]
    a, wd = 0.05, 0.9
    y = torch.remainder(u - a, 2 * wd)
    return _noise(torch.where(y > wd, 2 * wd - y, y) + a, g)


def rotation(n, L, g, dev, ood=False):
    step = (_u(2 * n, 0.02, 0.1, g, dev) if ood else _u(2 * n, 0.1, 0.9, g, dev)).view(n, 2)
    x0 = torch.rand(n, 2, generator=g, device=dev)
    p = torch.remainder(x0[:, None] + step[:, None] * torch.arange(L, device=dev)[None, :, None], 1.0)
    return torch.remainder(_noise(p, g), 1.0)


def uniform(n, L, g, dev, ood=False):
    return torch.rand(n, L, 2, generator=g, device=dev)


_PEN = {}


def pen_pool(L, size=2400, seed=0):
    """Pen sequences from the notes' own rule, built once (numpy, sequential) and cached; first 90% train, rest eval."""
    if L in _PEN:
        return _PEN[L]
    path = HERE / "runs" / "e1" / f"pen_pool_L{L}.pt"
    if path.exists():
        _PEN[L] = torch.load(path)
        return _PEN[L]
    from make_figures import draw_marks
    rng = np.random.default_rng(seed)
    seqs = []
    while len(seqs) < size:
        p = draw_marks(L, grid=12, target=2, rng=rng, start=tuple(rng.uniform(0.2, 0.8, 2)),
                       turn=float(rng.uniform(0.3, 0.8)), step=float(rng.uniform(0.03, 0.06)))
        if len(p) == L:
            seqs.append(p)
    pool = torch.tensor(np.stack(seqs), dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pool, path)
    _PEN[L] = pool
    return pool


def pen(n, L, g, dev, ood=False, split="train"):
    pool = pen_pool(L)
    cut = int(0.9 * len(pool))
    part = pool[:cut] if split == "train" else pool[cut:]
    idx = torch.randint(0, len(part), (n,), generator=torch.Generator().manual_seed(int(torch.randint(0, 2**31, (1,), generator=g, device=dev).item())))
    return part[idx].to(dev)


GEN = dict(circle=circle, spiral=spiral, lissajous=lissajous, billiard=billiard, pen=pen, rotation=rotation, uniform=uniform)


def batch(per_family, L, g, dev, ood=False, families=FAMILIES, split="train"):
    pts, fam = [], []
    for i, f in enumerate(families):
        if ood and f in ("pen", "uniform"):
            continue
        kw = dict(split=split) if f == "pen" else {}
        pts.append(GEN[f](per_family, L, g, dev, ood=ood, **kw))
        fam.append(torch.full((per_family,), FAMILIES.index(f), device=dev))
    return torch.cat(pts), torch.cat(fam)
