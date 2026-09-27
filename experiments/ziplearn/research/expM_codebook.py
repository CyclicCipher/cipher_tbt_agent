"""expM (S2, 2026-09-27) -- interpretable superposition: is a designed codebook's interference actually predictable?

DESIGN.md S22.2 claims three things that make superposition a DESIGN object rather than an accident of gradient
descent, and all three are checkable with numpy and no network:
  1. the Welch bound mu >= sqrt((m-d)/(d(m-1))) floors the coherence of m atoms in d dimensions;
  2. k active atoms are recovered exactly when k < (1 + 1/mu)/2  (Donoho-Elad / Tropp);
  3. the interference is therefore a CONSTANT you compute once, so S22.5 can price how superposed a layer should be.

Measured here, for codebooks of m atoms in d dims at m/d in {1, 2, 4, 8, 16}: the empirical coherence against the
Welch bound; the exact-readout rate of a bundle of k atoms against the k < (1 + 1/mu)/2 prediction; the largest k
with >= 99% readout; and the two-part price of S22.5 -- bits saved by packing m features into d dims, minus bits
lost to the misreads -- to see WHICH allocation the price picks.

Three codebook families: gaussian (the generic random frame), sparse (+-1 with k_nz nonzeros -- Kanerva's random
sparse addressing, S21.8a), hadamard (rows of a Hadamard matrix with random signs -- a tight, equiangular-ish frame
standing in for the explicit ETF/Gold constructions).

Prints JSON to runs/research/expM.json. No neural network is built, run or trained.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

HERE = Path(__file__).resolve().parent
TRIALS = 400


def codebook(kind, m, d, rng):
    if kind == "gaussian":
        A = rng.standard_normal((m, d))
    elif kind == "sparse":
        nz = max(2, int(round(math.log2(d))) * 2)
        A = np.zeros((m, d))
        for i in range(m):
            j = rng.choice(d, size=min(nz, d), replace=False)
            A[i, j] = rng.choice([-1.0, 1.0], size=len(j))
    elif kind == "hadamard":
        p = 1 << int(math.ceil(math.log2(max(d, 2))))
        H = hadamard(p)[:, :d].astype(np.float64)
        reps = int(math.ceil(m / p))
        # per-BLOCK column sign flips: row i of block b is H[i] * s_b, so atoms from different blocks have inner
        # product ~ 1/sqrt(d). (Flipping ROW signs instead repeats every atom up to a sign -- coherence 1.0.)
        A = np.vstack([H * rng.choice([-1.0, 1.0], size=(1, d)) for _ in range(reps)])[:m]
        A = A + 1e-9 * rng.standard_normal(A.shape)
    else:
        raise ValueError(kind)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def welch(m, d):
    return math.sqrt((m - d) / (d * (m - 1))) if m > d else 0.0


def readout_rate(A, k, rng, trials=TRIALS):
    """Bundle k random atoms; read out by matched filter (top-k of A @ x). Fraction of bundles recovered EXACTLY."""
    m = A.shape[0]
    ok = 0
    for _ in range(trials):
        idx = rng.choice(m, size=k, replace=False)
        x = A[idx].sum(0)
        s = A @ x
        got = np.argpartition(-s, k)[:k]
        if set(got.tolist()) == set(idx.tolist()):
            ok += 1
    return ok / trials


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    out = {"trials": TRIALS, "families": {}}

    for kind in ("gaussian", "sparse", "hadamard"):
        rows = []
        for d in (64, 128, 256):
            for rho in (1, 2, 4, 8, 16):
                m = rho * d
                A = codebook(kind, m, d, rng)
                G = np.abs(A @ A.T)
                np.fill_diagonal(G, 0.0)
                mu = float(G.max())
                wb = welch(m, d)
                k_pred = 0.5 * (1 + 1 / mu) if mu > 0 else float("inf")
                curve, kstar = {}, 0
                for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
                    if k > m:
                        break
                    r = readout_rate(A, k, rng)
                    curve[k] = r
                    if r >= 0.99:
                        kstar = k
                rows.append({"d": d, "m": m, "m_over_d": rho, "coherence": mu, "welch_bound": wb,
                             "coherence_over_welch": (mu / wb) if wb > 0 else None,
                             "k_predicted_exact": k_pred, "k_star_99pct": kstar,
                             "features_per_dim_at_99pct": (rho if kstar >= 1 else 0.0),
                             "readout_curve": curve})
        out["families"][kind] = rows

    # ---- S22.5's price, stated correctly ----
    # The comparison is: the SAME m features, held in d dimensions, for d from m (disjoint) down to m/16.
    # Storage costs d * PREC bits per stored vector; every misread costs log2(m choose k) bits (say which bundle
    # was really meant). Shrinking d saves storage at a rising error rate: the argmin is the allocation to pick.
    PREC, N_STORED, N_READS = 8.0, 2_000, 10_000
    price = []
    M_FIX = 1024
    for k in (2, 4, 8):
        for rho in (1, 2, 4, 8, 16):
            d = M_FIX // rho
            A = codebook("hadamard", M_FIX, d, rng)
            r = readout_rate(A, k, rng)
            eps = 1.0 - r
            store = d * PREC * N_STORED
            per_err = math.log2(max(2.0, math.comb(M_FIX, k)))
            lost = N_READS * eps * per_err
            # a store only pays for itself against NOT HAVING ONE, which costs the full answer every read
            baseline = N_READS * per_err
            price.append({"k": k, "m": M_FIX, "d": d, "m_over_d": rho, "readout": r,
                          "storage_bits": store, "bits_lost_to_misreads": lost, "total_bits": store + lost,
                          "no_store_baseline_bits": baseline, "beats_no_store": (store + lost) < baseline})
    out["price"] = price
    out["price_pick"] = {str(k): min((p for p in price if p["k"] == k), key=lambda p: p["total_bits"])["m_over_d"]
                         for k in (2, 4, 8)}
    out["largest_packing_at_99pct_readout"] = {
        str(k): max([p["m_over_d"] for p in price if p["k"] == k and p["readout"] >= 0.99], default=0)
        for k in (2, 4, 8)}
    out["price_note"] = ("m features held in d dims, d shrinking; storage d*8 bits per stored vector against "
                         "misreads at log2(m choose k) bits each. The argmin is the allocation S22.5 picks.")
    out["seconds"] = time.time() - t0
    dd = HERE.parent / "runs" / "research"; dd.mkdir(parents=True, exist_ok=True)
    (dd / "expM.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["price_pick"]), f"{out['seconds']:.1f}s")


if __name__ == "__main__":
    main()
