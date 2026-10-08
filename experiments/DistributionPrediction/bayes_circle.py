"""A Bayesian reference for in-context extrapolation on the circle family (exploratory, EXPERIMENT_1.md follow-ups).

The ceiling "knows the true rule and parameters" (~11.2 bits) is unfair at small context sizes: from n noisy points the
circle's centre, radius, angular speed and phase are only partly determined, and the uncertainty grows with the horizon.
This computes what an inference engine that KNOWS the family is a circle can achieve: least-squares fit of
(cx, cy, r, omega, phi) to the n points (several starts), a Laplace approximation of the parameter posterior
(cov = (J^T J / sigma^2)^-1), and the linearised predictive density of the point k steps ahead
(N(mu_k, J_k cov J_k^T + sigma^2 I)). Bits gained = log2 of that density at the true point, 1% uniform mixed in.

Usage:  python bayes_circle.py        (prints the in-context curve by horizon bucket, n = 4..32)
"""
from __future__ import annotations

import math

import torch

import generators as G

L, K = 48, 16
SIG = G.SIGMA
BUCKETS = {"k1": (1, 1), "k2-4": (2, 4), "k5-8": (5, 8), "k9-16": (9, 16)}


def model_xy(theta, t):
    cx, cy, r, w, ph = theta.unbind(-1)
    a = ph[..., None] + w[..., None] * t
    return torch.stack([cx[..., None] + r[..., None] * a.cos(), cy[..., None] + r[..., None] * a.sin()], -1)


def fit(x, starts=6, iters=300):
    """x (N, n, 2) -> theta (N, 5), least squares with several starts for omega; batched Adam then refinement."""
    N, n, _ = x.shape
    t = torch.arange(n, dtype=torch.float64, device=x.device)
    c0 = x.mean(1)
    r0 = (x - c0[:, None]).norm(dim=-1).mean(1).clamp(min=0.05)
    ang = torch.atan2(x[..., 1] - c0[:, None, 1], x[..., 0] - c0[:, None, 0])
    dang = torch.remainder(ang[:, 1:] - ang[:, :-1] + math.pi, 2 * math.pi) - math.pi if n > 1 else torch.zeros(N, 1, device=x.device)
    w0 = dang.mean(1) if n > 1 else torch.zeros(N, device=x.device)
    best, best_loss = None, None
    for s in range(starts):
        w_init = w0 if s == 0 else (torch.rand(N, device=x.device, dtype=x.dtype) * 0.7 - 0.35)
        th = torch.stack([c0[:, 0], c0[:, 1], r0, w_init, ang[:, 0]], 1).clone().requires_grad_(True)
        opt = torch.optim.Adam([th], lr=0.02)
        for i in range(iters):
            loss = ((model_xy(th, t) - x) ** 2).sum((1, 2))
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            if i == iters // 2:
                for gp in opt.param_groups:
                    gp["lr"] = 0.003
        with torch.no_grad():
            loss = ((model_xy(th, t) - x) ** 2).sum((1, 2))
            if best is None:
                best, best_loss = th.detach().clone(), loss
            else:
                better = loss < best_loss
                best[better], best_loss[better] = th.detach()[better], loss[better]
    return best


def predictive_bits(theta, n, y):
    """theta (N, 5) fitted on points 0..n-1; y (N, K, 2) true points n..n+K-1 -> bits (N, K)."""
    N = theta.shape[0]
    t_fit = torch.arange(n, dtype=torch.float64, device=theta.device)
    t_new = torch.arange(n, n + K, dtype=torch.float64, device=theta.device)
    J = torch.func.vmap(torch.func.jacrev(lambda th: model_xy(th, t_fit).reshape(-1)))(theta)     # (N, 2n, 5)
    H = J.transpose(1, 2) @ J / SIG ** 2 + 1e-6 * torch.eye(5, dtype=theta.dtype, device=theta.device)
    cov = torch.linalg.inv(H)
    Jk = torch.func.vmap(torch.func.jacrev(lambda th: model_xy(th, t_new)))(theta)               # (N, K, 2, 5)
    mu = model_xy(theta, t_new)
    S = Jk @ cov[:, None] @ Jk.transpose(-1, -2) + SIG ** 2 * torch.eye(2, dtype=theta.dtype, device=theta.device)
    d = (y - mu)[..., None]
    logp = -0.5 * (d.transpose(-1, -2) @ torch.linalg.solve(S, d)).squeeze(-1).squeeze(-1) - 0.5 * torch.logdet(S) - math.log(2 * math.pi)
    return torch.logaddexp(logp + math.log(0.99), torch.full_like(logp, math.log(0.01))) / math.log(2)


if __name__ == "__main__":
    dev = "cuda"
    ge = torch.Generator(device=dev).manual_seed(10_000)          # the same seed as run_e1's held-out test set
    x = G.circle(64, L, ge, dev).double()
    print("Bayesian reference (knows it is a circle), bits gained, mean of 64 held-out circles")
    print("   n  " + "  ".join(f"{b:>6s}" for b in BUCKETS))
    for n in (4, 8, 16, 32):
        th = fit(x[:, :n])
        b = predictive_bits(th, n, x[:, n:n + K])
        print(f"  {n:2d}  " + "  ".join(f"{float(b[:, lo - 1:hi].mean()):+6.2f}" for lo, hi in BUCKETS.values()))
