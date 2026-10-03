"""How much of the available improvement does each way of choosing a thought step recover, as the thought's
dimension d grows? A toy, numpy only (CURRICULUM_LESS_COCONUT_AND_SEARCH.md §10.1).

Every candidate step has the same length r (a trust region: a thought may move only so far from the policy's mean).
  linear objective   Q(δ) = gᵀδ, g a random unit vector (the local picture of any smooth value). The best possible
                     step is r·g, worth r, so each method is reported as the fraction of r it recovers.
                     g lies in a random m-dimensional subspace (m = 8), to model a value that depends on only a
                     few directions of the thought.
  needle objective   success iff the step lands within ρ = 0.5·r of a target δ* at distance r: the sparse case.
Methods, each with K evaluations of Q unless stated:
  isotropic best-of-K     K random directions in all d dimensions (what sampling-based search does);
  subspace best-of-K      K random directions inside the m-dim subspace g lives in (an oracle subspace);
  gradient step           one step along the true gradient (one backward pass, no evaluations);
  noisy gradient (cos c)  one step along a gradient estimate whose cosine with the truth is c — a learned value's
                          gradient that is mostly wrong (c = 0.3) still points somewhere useful.
    python experiments/neural_turing_architecture/research/dim_search_toy.py      (seconds, CPU)
"""
import json
import sys
from pathlib import Path

import numpy as np

R, M, TRIALS = 1.0, 8, 4000
rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def linear_cell(d, K):
    basis = np.linalg.qr(rng.standard_normal((d, min(M, d))))[0]          # the relevant subspace
    out = {"iso": [], "sub": [], "grad_noisy_0.3": []}
    for _ in range(TRIALS):
        g = unit(basis @ rng.standard_normal(basis.shape[1]))
        iso = R * unit(rng.standard_normal((K, d)))
        out["iso"].append((iso @ g).max() / R)
        sub = R * unit(rng.standard_normal((K, basis.shape[1])) @ basis.T)
        out["sub"].append((sub @ g).max() / R)
        n = rng.standard_normal(d); n -= (n @ g) * g; n = unit(n)
        ghat = 0.3 * g + np.sqrt(1 - 0.3 ** 2) * n
        out["grad_noisy_0.3"].append(R * ghat @ g / R)
    return {k: float(np.mean(v)) for k, v in out.items()}


def needle_cell(d, K, trials=20000):
    target = R * unit(rng.standard_normal((trials, d)))
    hits = 0
    for t in range(trials):
        cand = R * unit(rng.standard_normal((K, d)))
        hits += bool((np.linalg.norm(cand - target[t], axis=1) < 0.5 * R).any())
    return hits / trials


rows = []
for d in (2, 12, 64, 256, 1024):
    for K in (16, 256):
        lin = linear_cell(d, K)
        row = dict(d=d, K=K, isotropic=lin["iso"], subspace_m8=lin["sub"], gradient=1.0,
                   noisy_gradient_cos03=lin["grad_noisy_0.3"],
                   needle_hit_rate=needle_cell(d, K) if d <= 64 else None)
        rows.append(row)
        print(f"d={d:5d} K={K:4d} | isotropic best-of-K {row['isotropic']:.3f} | subspace(m=8) best-of-K "
              f"{row['subspace_m8']:.3f} | gradient 1.000 | noisy gradient (cos .3) {row['noisy_gradient_cos03']:.3f} | "
              f"needle hit rate {row['needle_hit_rate'] if row['needle_hit_rate'] is not None else 'n/a (d>64)'}",
              flush=True)
out = Path(__file__).with_suffix(".json")
json.dump(dict(R=R, m=M, trials=TRIALS, rows=rows), open(out, "w"), indent=1)
print(f"-> {out}")
