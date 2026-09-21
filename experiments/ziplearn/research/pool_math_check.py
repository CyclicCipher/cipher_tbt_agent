"""pool_math_check -- a feel-experiment for refs/opinion_pools_and_products_of_experts.md (no gradient anywhere).

Synthetic binary prediction with M = 3 sources under Satopaa's Gaussian partial-information model: a latent
z = u_1 + ... + u_K (independent Gaussians), the outcome y = 1[z > 0], and source i sees a SUBSET S_i of the u_k and
reports the calibrated P(y = 1 | u_{S_i}).  Three information structures:
  nested   : S_1 c S_2 c S_3   (E34's order-1 c order-2 c order-3 contexts -- the shorter context is a function of the longer)
  disjoint : S_i pairwise disjoint (conditionally independent given nothing but z -- the naive-Bayes case)
  overlap  : each pair shares one component (partly redundant -- the realistic case)
For each we measure bits/symbol of: the best single source; the linear pool at its best simplex weights; the geometric
pool with exponents on the simplex (sum 1) at its best weights; the geometric pool with FREE exponents (grid, sum up
to M) with the prior divided out (Bordley / Allard eq. 15); the naive-Bayes product (all exponents 1); the exact
posterior P(y | all u seen by anyone); and two CLOSED-FORM weightings computed from counts, not fitted:
  lindley : Genest & Zidek eq. (4.1) / LDA -- w = Xi^{-1}(mu_1 - mu_0) on the probit-stretched forecasts, from the
            class-conditional means and pooled covariance of the forecasts (needs the outcome);
  partial : Satopaa & Ungar 2015 revealed aggregator w = diag(Sigma)' Sigma^{-1} on the LATENT scale (each source's
            information delta_i recovered from the variance of its probit-stretched forecast), from the covariance
            of the forecasts ALONE (no outcome).
NB the disjoint case: the sources are marginally independent but NOT conditionally independent given y (y is the sign
of a shared latent), so the exponent-1 naive-Bayes product is not exact there; the right exponents are larger.
Weights are searched on a grid (simplex step 0.05, free exponents step 0.5 up to 3; no gradient).  N = 50k, runs in ~20 s.  Output: research/pool_math_check.json
    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/pool_math_check.py
"""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)
N = 50_000
V = np.array([0.35, 0.30, 0.25, 0.10])          # variances of u_1..u_4, sum 1 -> z ~ N(0, 1)
STRUCT = {
    "nested":   [[0], [0, 1], [0, 1, 2]],
    "disjoint": [[0], [1], [2]],
    "overlap":  [[0, 1], [1, 2], [2, 0]],
}

def bits(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(np.where(y == 1, np.log2(p), np.log2(1 - p))))

def calibrated(u, S):
    """P(y=1 | u_S) = Phi(sum_S u / sqrt(1 - sum_S v))  (the remaining components are N(0, 1 - sum_S v))."""
    s = u[:, S].sum(1); rest = 1.0 - V[S].sum()
    return norm.cdf(s / np.sqrt(rest)) if rest > 1e-12 else (s > 0).astype(float)

def geo_pool(L, w, prior, divide_prior):
    """P_G ∝ prior^(1 - sum w) * prod P_i^w_i  (Allard 2012 eq. 15); divide_prior=False is E34's `pr_`/`g` form.
    L = (log P, log(1 - P)) precomputed once."""
    lo, lo1 = L
    a = lo @ w; b = lo1 @ w
    if divide_prior:
        a += (1 - w.sum()) * np.log(prior); b += (1 - w.sum()) * np.log(1 - prior)
    return 1 / (1 + np.exp(b - a))

def simplex_grid(M, step=0.05):
    ks = np.arange(0, 1 + 1e-9, step)
    for c in itertools.product(ks, repeat=M - 1):
        if sum(c) <= 1 + 1e-9:
            yield np.array(list(c) + [1 - sum(c)])

def free_grid(M, step=0.5, top=3.0):
    ks = np.arange(0, top + 1e-9, step)
    for c in itertools.product(ks, repeat=M):
        yield np.array(c)

out = {}
for name, subsets in STRUCT.items():
    u = rng.normal(size=(N, len(V))) * np.sqrt(V)
    z = u.sum(1); y = (z > 0).astype(int)
    P = np.stack([calibrated(u, S) for S in subsets], 1)            # (N, M) calibrated forecasts
    M = P.shape[1]; prior = y.mean()
    seen = sorted(set(k for S in subsets for k in S))
    exact = calibrated(u, seen)
    L = (np.log(np.clip(P, 1e-9, 1 - 1e-9)), np.log(np.clip(1 - P, 1e-9, 1 - 1e-9)))
    r = {"best_single": min(bits(P[:, i], y) for i in range(M)), "exact_posterior": bits(exact, y)}
    r["linear_best_simplex"] = min(bits(P @ w, y) for w in simplex_grid(M))
    r["geometric_best_simplex_sum1"] = min(bits(geo_pool(L, w, prior, True), y) for w in simplex_grid(M))
    best = min(((bits(geo_pool(L, w, prior, True), y), w) for w in free_grid(M)), key=lambda t: t[0])
    r["geometric_best_free_exponents"] = best[0]; r["free_exponents"] = best[1].round(2).tolist()
    r["free_exponent_sum"] = float(best[1].sum())
    r["naive_bayes_product_exp1"] = bits(geo_pool(L, np.ones(M), prior, True), y)
    r["e34_product_no_prior_division"] = bits(geo_pool(L, np.ones(M), prior, False), y)
    # closed-form weightings on the probit-stretched forecasts s_i = Phi^{-1}(P_i)
    S_ = norm.ppf(np.clip(P, 1e-6, 1 - 1e-6))
    mu1, mu0 = S_[y == 1].mean(0), S_[y == 0].mean(0)
    Xi = 0.5 * (np.cov(S_[y == 1].T) + np.cov(S_[y == 0].T))
    w_l = np.linalg.solve(Xi, mu1 - mu0)
    logit = np.log(prior / (1 - prior)) + (S_ - (mu1 + mu0) / 2) @ w_l
    r["lindley_lda_closed_form"] = bits(1 / (1 + np.exp(-logit)), y); r["lindley_weights"] = w_l.round(2).tolist()
    # Satopaa's model is for forecasts X_i = E[z | info_i] of the LATENT z (reliable: Cov(X_i, z) = Var(X_i) = delta_i).
    # A calibrated probability is P_i = Phi(X_i / sqrt(1 - delta_i)), so s_i = X_i / sqrt(1 - delta_i) and
    # Var(s_i) = delta_i / (1 - delta_i)  ->  delta_i = Var(s_i) / (1 + Var(s_i)); X_i = s_i * sqrt(1 - delta_i).
    # Everything below uses the forecasts alone (no outcome): Sigma = Cov(X), E[z|X] = diag(Sigma)' Sigma^{-1} X,
    # Var(z|X) = 1 - diag(Sigma)' Sigma^{-1} diag(Sigma), P(y=1|X) = Phi(E[z|X] / sqrt(Var(z|X))).
    vs = S_.var(0); delta = vs / (1 + vs); X = S_ * np.sqrt(1 - delta)
    Sig = np.cov(X.T); w_p = np.linalg.solve(Sig, np.diag(Sig))     # diag(Sigma)' Sigma^{-1}
    cvar = max(1.0 - np.diag(Sig) @ w_p, 1e-6)
    r["partial_info_closed_form"] = bits(norm.cdf((X @ w_p) / np.sqrt(cvar)), y)
    r["partial_info_weights_latent"] = w_p.round(2).tolist(); r["partial_info_delta"] = delta.round(3).tolist()
    out[name] = r
    print(f"[{name}] " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)

here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, "pool_math_check.json"), "w"), indent=1)
print("wrote", os.path.join(here, "pool_math_check.json"))
