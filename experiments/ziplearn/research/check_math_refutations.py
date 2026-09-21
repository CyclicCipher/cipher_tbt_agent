"""check_math_refutations -- numerical checks of the mathematical claims in
notes/gradient_free_mixing_and_features.md (the adversarial-checker pass, lens = mathematics). No gradient anywhere.

Checks:
  1. Mattern 2013 Example 4.8 at V = 70: a geometric pool with SIMPLEX exponents (1/2, 1/2) gives the true symbol MORE
     probability than either component and has LOWER entropy than either -- refutes the synthesis I.0 sentence "with
     exponents on the simplex it re-shapes but cannot sharpen beyond the components".
  2. Binary alphabet (Mattern 2012's experimental setting): with simplex exponents the pooled logit is a convex combination
     of the component logits, so geo <= p_max per bit -- the normaliser effect needs V > 2.
  3. S_w is scale-dependent, not "the effective number of independent sources": on pool_math_check's `disjoint`
     structure (3 sources with disjoint information, latent z = u1+u2+u3+u4, y = 1[z>0]) the EXACT posterior is a
     geometric pool of the calibrated sources on the PROBIT scale with exponents sqrt(1 - v_i)/sqrt(v_4) (sum ~ 8), while
     the same three sources combine with weights (1, 1, 1) (sum 3) on Satopaa's conditional-expectation scale. Same
     sources, same information structure, S_w = 8 or 3 depending on the parametrisation. The free-exponent grid of
     pool_math_check.py stops at 3.0 per exponent, so its "(3, 3, 3), S_w = 9" is a grid-boundary value.
  4. Remark 4.7 is a PER-SYMBOL ceiling: two experts that alternate in quality; the fixed-share Bayesian linear mixture
     codes the whole stream far below either expert's total -- "cannot gain" does not follow from the remark.
  5. Arithmetic in the note: -log2(0.98); the KT 0.5/(1+35); the dominance-inequality price of the per-context grid.
Runs in < 5 s.  Output: research/check_math_refutations.json
    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/check_math_refutations.py
"""
from __future__ import annotations
import json, math, os
import numpy as np
from scipy.stats import norm

out = {}

# ---- 1. Example 4.8 at V = 70 -------------------------------------------------------------------------------------
V, q, eps = 70, 0.6, 1e-3
p1 = np.full(V, (1 - q) * eps / (V - 2)); p1[0] = q; p1[1] = (1 - q) * (1 - eps)
p2 = np.full(V, (1 - q) * eps / (V - 2)); p2[0] = q; p2[2] = (1 - q) * (1 - eps)
geo = np.sqrt(p1 * p2); geo /= geo.sum()
H = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
out["ex48_V70"] = {"p_max(symbol 1)": q, "geo_simplex(symbol 1)": float(geo[0]),
                   "entropy_components": [H(p1), H(p2)], "entropy_geo_simplex": H(geo),
                   "bits_if_symbol1_true": {"best_expert": -math.log2(q), "geo": -math.log2(geo[0])}}

# ---- 2. Binary alphabet: simplex geo cannot exceed p_max ----------------------------------------------------------
rng = np.random.default_rng(0)
P = rng.uniform(0.01, 0.99, size=(10000, 3))
w = rng.dirichlet(np.ones(3), size=10000)
logit = np.log(P / (1 - P)); g = 1 / (1 + np.exp(-(w * logit).sum(1)))
out["binary_simplex_geo_exceeds_pmax_count"] = int((g > P.max(1) + 1e-12).sum())

# ---- 3. S_w on two scales, disjoint structure of pool_math_check.py ---------------------------------------------
Vv = np.array([0.35, 0.30, 0.25, 0.10]); N = 200_000
u = rng.normal(size=(N, 4)) * np.sqrt(Vv); z = u.sum(1); y = (z > 0).astype(int)
def bits(p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(np.where(y == 1, np.log2(p), np.log2(1 - p))))
src = [norm.cdf(u[:, i] / np.sqrt(1 - Vv[i])) for i in range(3)]         # calibrated P(y=1 | u_i)
exact = norm.cdf(u[:, :3].sum(1) / np.sqrt(Vv[3]))
# probit-scale geometric pool: Phi(sum_i a_i Phi^-1(p_i)) with a_i = sqrt(1 - v_i)/sqrt(v_4) reproduces `exact` exactly
a = np.sqrt(1 - Vv[:3]) / np.sqrt(Vv[3])
probit_pool = norm.cdf(sum(a[i] * norm.ppf(src[i]) for i in range(3)))
# logit-scale geometric pool (the PAQ / Mattern form) with the SAME exponents, and a 1-D search of a common scale
lg = [np.log(s / (1 - s)) for s in src]
def logit_pool(c): return 1 / (1 + np.exp(-sum(c * a[i] * lg[i] for i in range(3))))
scales = np.arange(0.4, 1.61, 0.05)
best_c = min(scales, key=lambda c: bits(logit_pool(c)))
# Satopaa scale: X_i = E[y | u_i] = src[i]; centre; Cov(X_i, y - prior) = Var(X_i) under the model; w = diag(S)' S^-1
X = np.stack(src, 1) - 0.5
S = np.cov(X.T); w_sat = np.diag(S) @ np.linalg.inv(S)
out["disjoint_Sw_two_scales"] = {
    "bits_exact": bits(exact), "bits_probit_pool": bits(probit_pool), "probit_exponents": a.round(3).tolist(),
    "probit_Sw": float(a.sum()), "logit_pool_same_exponents_bits": bits(logit_pool(1.0)),
    "logit_pool_best_common_scale": float(best_c), "logit_Sw_at_best_scale": float(best_c * a.sum()),
    "logit_pool_best_bits": bits(logit_pool(best_c)),
    "satopaa_weights_conditional_expectation_scale": w_sat.round(3).tolist(), "satopaa_Sw": float(w_sat.sum()),
    "pool_math_check_free_grid_cap_per_exponent": 3.0}

# ---- 4. Remark 4.7 is per-symbol: alternating experts, Bayesian fixed-share mixture ---------------------------------
T, Vb = 20000, 10
truth = rng.integers(0, Vb, size=T)
A = np.full((T, Vb), 0.02 / (Vb - 1)); B = A.copy()
good = np.arange(T) // 1000 % 2 == 0                                  # expert A knows even blocks, B odd blocks
A[good, truth[good]] = 0.98; A[~good] = 1 / Vb
B[~good, truth[~good]] = 0.98; B[good] = 1 / Vb
wmix = np.array([0.5, 0.5]); tot = 0.0; alpha = 0.02
for t in range(T):
    probs = np.array([A[t, truth[t]], B[t, truth[t]]]); p = float(wmix @ probs); tot -= math.log2(p)
    post = wmix * probs / p; wmix = (1 - alpha) * post + alpha / 2
out["remark47_per_symbol_only"] = {"bits_per_symbol": {"expert_A": float(-np.log2(A[np.arange(T), truth]).mean()),
                                   "expert_B": float(-np.log2(B[np.arange(T), truth]).mean()), "fixed_share_linear": tot / T}}

# ---- 5. arithmetic -----------------------------------------------------------------------------------------------
out["arithmetic"] = {"-log2(0.98)": -math.log2(0.98), "KT_unseen_N1_V70": 0.5 / (1 + 35),
                     "dominance_price_global_63_over_60k": math.log2(63) / 60000,
                     "dominance_price_per_context_36x42_over_60k": 36 * math.log2(42) / 60000,
                     "mattern_book1_gap_rel_GEO": (2.313 - 2.212) / 2.212, "mattern_book1_gap_rel_BETA": (2.313 - 2.212) / 2.313,
                     "mattern_avg_gap_rel_GEO": (2.265 - 2.187) / 2.187, "mattern_avg_gap_rel_BETA": (2.265 - 2.187) / 2.265}

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "check_math_refutations.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
