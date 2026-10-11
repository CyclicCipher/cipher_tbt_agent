# Experiment 5 — combining the winners (pre-registered 2026-10-10)

Status: **pre-registered before any arm ran.** The user asked: "Combine the winners first, then test SGD vs Adam".

On O4 they said: "fine as long as it does not require the model to be differentiable". It does not:
- **No derivative is taken.** The local map's transpose is just its own weights, and the activation's slope is MEASURED
  by perturbation.
- **The one requirement:** the layer between a block's writer and its hub or hidden units is linear.

## The winners so far (training at matched or lower cost, all reused, not re-run)

| arm | validation (2 seeds) | cost per run | cost to first reach 2.37 |
|---|---|---|---|
| baseline | 2.3687 | 86k | ~86k |
| O4 (K = 56) | 2.3329 | 83k | 50–52k |
| O4 + orth (K = 56) | 2.2969 | 83k | 44k |
| O4 + local (K = 128) | 2.2815 | 46k | 18–29k |
| cache-free simultaneous perturbation (K = 143) | 2.4065 | 43k | not reached; ≈ the baseline per unit of cost |

## Arms

| arm | settings | question |
|---|---|---|
| **W1: O4 + local + orth**, K = 128 | `hub_T=1 local=1 noise=orth`, aux heads after each block | do all three stack? Compare O4 + local (2.2815, same cost) |
| **W2: O4 + local, K = 256** | about the baseline's cost, if it fits in 2 minutes (otherwise the largest K that fits, reported) | is O4 + local still improving with more draws, or saturated? |
| **W3: cache-free + local** ("local simultaneous") | every site perturbed in every pass (`simul.py`), but each block's sites are scored by the NEXT auxiliary head's per-token loss, computed in the same pass; the last block uses the final loss. No cache, no reruns. K = 143 (half cost), as T3 | does local scoring rescue the cache-free form, whose weakness is that every site hears every other's noise? Compare T3 (2.4065, same cost) |

All arms: 300 steps, two seeds, Adam lr 1e-2, validation every 10 steps.

## Predictions

- **P1:** W1 beats O4 + local by ≥ 0.02 (orth stacks).
- **P2:** W2 beats O4 + local (K = 128) by ≥ 0.02.
- **P3:** W3 beats T3 by ≥ 0.03, at the same half cost. Local scoring limits each site's interference to its own block.
  If it holds, the cache-free form becomes competitive with the rerun form, which matters for looped models.
- **P4 — the headline:** the best of W1–W3 reaches 2.37 with ≤ 1/4 of the baseline's cost (≤ 22k).

**Margin:** 0.02 nats in both seeds (local-scored runs have shown seed spreads up to 0.09, so a single-seed reversal
is reported, not hidden).

## Deviations found while smoke-testing (before any training run)

- **W1, the orthogonal draws.**
  - **Why:** per-token orthogonalisation at K = 128 would take ~475 s per run. With K = 2·D it is two float64 Cholesky
    factorisations per site per token-batch. (An fp32 Cholesky also failed on a full block of 64.)
  - **The replacement, `noise=orth_sign`:** one random orthonormal basis per call, with each token flipping signs
    coordinate by coordinate. That stays exactly orthogonal per token (measured off-diagonal 3e-7), and the run takes
    ~79 s.
  - **Its price:** M1 on O4 + orth at K = 56 gives 0.749 against per-token QR's 0.795; O4 alone gives 0.673.
- **W2:** K = 256 would take ~149 s, so by the stated rule it runs at **K = 192**, the largest that fits (~112 s).
- **W3:** K = 143, ~112 s, as planned.

## Files

`simul.py` (`local` scoring for W3). Results: `runs/e5/`.

## Results — RAN 2026-10-10 (`runs/e5/`)

| arm | validation per seed | mean | vs its comparison | cost | cost to first reach 2.37 | s/run |
|---|---|---|---|---|---|---|
| W1: O4 + local + orth_sign, K = 128 | 2.2776 / 2.3075 | 2.2926 | **+0.011** vs O4 + local (2.2815) | 46k | 21k / 24k | 91 |
| W2: O4 + local, K = 192 | 2.2282 / 2.3145 | 2.2713 | −0.010 vs O4 + local, at 1.5× the cost | 69k | 30k / 41k | **138 (over the 2-min limit)** |
| W3: cache-free + local, K = 143 | 2.3956 / 2.4029 | 2.3992 | −0.007 vs T3 (2.4065) | 43k | not reached | **122 (over)** |

### Against the predictions

- **P1 — FAILED.** Orthogonal draws do not stack on O4 + local. W1 is worse in mean, and the seeds split: −0.042 on
  seed 0, +0.020 on seed 1 against O4 + local.
- **P2 — FAILED.** 1.5× the draws buys 0.010: saturated.
- **P3 — FAILED.** Local scoring barely helps the cache-free form (−0.007).
- **P4 — FAILED, narrowly.** Reaching 2.37 took 21k and 24k (bar ≤ 22k in both seeds). O4 + local alone took 18k and
  29k.

### What it says

**The estimator tweaks are saturated around O4 + local:**
- about 3–4× cheaper than the Dust baseline at reaching 2.37;
- about 50–75× backprop's cost.

Adding orthogonal draws, more draws, or local scoring to the cache-free form changes little. The user's reading fits:
this route improves Dust by a constant factor but does not open orders of magnitude.

**A consistent seed effect in every local-scored arm.** Seed 0 ends at 2.228–2.278, seed 1 at 2.307–2.327: a 0.03–0.09
gap that non-local arms do not show (their seeds differ by ≤ 0.01–0.03). Local scoring is sensitive to the data order
or the draws, so its results deserve more seeds than two.
