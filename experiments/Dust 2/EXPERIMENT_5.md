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
