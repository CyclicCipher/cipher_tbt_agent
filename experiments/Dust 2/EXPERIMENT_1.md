# Experiment 1 — Gemini's five ideas for cutting Dust's cost (pre-registered 2026-10-10)

Status: **pre-registered before any variant's cosine or training result was looked at.** Apparatus numbers (§0) were
measured before this file, and the predictions below take them into account. Asked for by the user: "do experiments to
test Gemini's ideas, and see if they fail or succeed".

## 0. Apparatus (measured before pre-registration)

- **The estimator matches theory.** At the 64-wide sites, the per-token error cosine is 0.56 at K = 32 and 0.89 at
  K = 256, against √(K/(K+D)) = 0.58 and 0.89. At the 256-wide MLP hidden it is 0.32 and 0.70, against 0.33 and 0.71.
- **Precision (the user asked for 16-bit wherever possible).** The perturbation passes run on a native fp16 copy of the
  weights, and the rewards, accumulators and gradients stay fp32.
  - **fp16:** mean weight-gradient cosine 0.841 at K = 256, against fp32's 0.843, at 2× the speed (0.17 against
    0.34 s per step).
  - **bf16:** loses 0.04 at the residual writers (0.805), so it is not used for the estimator.
  - **Backprop:** bf16 autocast gives the same validation loss as fp32 (2.178 against 2.179).
- **Credit decay γ for rerun sites** (the cosine at K = 256): γ = 0 gives 0.841, 0.5 gives 0.824, and 0.9 gives 0.666,
  so γ = 0.
- **Learning rates (Adam, 300 steps).**
  - **Dust:** lr 1e-2 reaches validation 2.371 (2e-2: 2.390).
  - **Backprop:** lr 2e-2 reaches 2.153, at 3 forward-equivalents per step against Dust's 287.
- **The baseline's cosine against draws** (K is the draws per site), at the checkpoints below:

| K | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| task 1 (Latin), mean cosine | 0.279 | 0.390 | 0.509 | 0.637 | 0.748 | 0.834 | 0.892 |
| task 2 (recall), mean cosine | 0.203 | 0.265 | 0.353 | 0.444 | 0.539 | 0.621 | 0.693 |
| cost per step (forward-equivalents) | 72.5 | 144 | 287 | 573 | 1145 | 2289 | 4577 |

- **The TRUE per-token errors have low effective rank** (participation ratio over the 2,048 tokens of a batch, from
  backprop):
  - Latin: attention hubs 1.7–3.2 of 64; residual writers 3.6–6.9 of 64; MLP hidden 10–18 of 256; q/k/v 11–26 of 192.
  - Recall: hubs 5–6; writers 6–14; hidden 4–9.
- **The ceiling for subspace methods.** The oracle subspace (backprop's own top-8 error directions per site, all draws
  inside it) reaches mean cosine 0.736 at K = 32 on Latin. That is the baseline's level at K ≈ 128, about **4×**.
  This is the most a rank-8 subspace could give (diagnostic only; it uses backprop).

## 1. Setup

**Task 1** (`data.py`) is character-level Latin: Cicero + Livy, 38 symbols, T = 64, B = 32.

**Task 2** (`recall.py`) is in-context associative recall: 12 key→value pairs, then 20 queries, with only the answers
scored. Backprop needs about 2,000 steps to leave chance (2.77 nats) on it. So on task 2 only the cosine diagnostic runs,
because no 2-minute Dust run could learn it.

**Model** (`model.py`): d = 64, 4 layers, 4 heads, the MLP 4× wide with ReLU², parameter-free RMSNorm, and the
soft-cap 15.

**Baseline D0** (`estimators.py` defaults):
- K = 32 draws per rerun site: embedding, the attention hub `o`, the projection, the MLP hidden and the MLP output, in
  each of 4 blocks, so 17 sites.
- Chunks of 16 draws per pass.
- σ = 0.2 at rerun sites, and γ = 0.
- q, k, v are scored locally through the hub, with σ = 0.05 and γ_loc = 0.98.
- The head is perturbed on its logits, with 4K draws.
- Cost is 287 forward-equivalents per step.

**Training** runs 300 steps with Adam lr 1e-2, 20 warm-up steps and cosine decay, about 1 minute per run.

**Checkpoints for the cosine diagnostic** are backprop-trained states:
- task 1: 300 steps, validation 2.34, the region Dust training passes through;
- task 2: 4,000 steps, validation 2.15.

They are made by `train.py --method bp --steps 300 --lr 3e-3 --amp '' --save runs/ckpt_bp300.pt` and
`train.py --method bp --task recall --steps 4000 --lr 3e-3 --save runs/ckpt_recall_bp4000.pt`. They are not committed:
`*.pt` is git-ignored.

## 2. Arms (Gemini's ideas, `IDEAS.md` §2), at the baseline's cost unless stated

| arm | what changes (`--set`) | Gemini idea |
|---|---|---|
| **lr_rand** | every draw inside a fixed random 8-dimensional subspace per site, rescaled to the same norm (`noise=lr_rand rank=8`; rank 16 in the diagnostic too) | 1 (fixed random projection) |
| **lr_pca** | every draw inside the top-8 principal directions of the site's clean OUTPUTS in this batch (`noise=lr_pca rank=8`) | 1 (running PCA of activations) |
| **orth** | per token, the 32 draws are mutually orthogonal, each of norm √D (`noise=orth`) | 2 (orthogonal ensembles) |
| **sobol** | scrambled Sobol points mapped to Gaussians; per token a random sign per coordinate and its own order of the points (`noise=sobol`) | 2 (quasi-Monte-Carlo) |
| **local** | an auxiliary next-character head after blocks 0, 1 and 2; each site is scored by the change of the NEXT auxiliary head's loss, so the rerun stops there; the auxiliary heads are trained by jitter on their logits (`--aux_every 1 --set local=1`); cost 73.5 per step at K = 32, so the matched-cost arm uses **K = 128** (cost ≈ 291) | 3 (local losses, truncated credit) |
| **ent50** | only the 50% of tokens with the highest predictive entropy are perturbed (`ent_frac=0.5`; 25% in the diagnostic too) | 4 (entropy gating) |
| **guided** | an EMA (rate 0.1) of each site's estimated error covariance; draws a = √(1−β)·ε + √(βD/k)·U·ε_k, with U its top k = 8 directions and β = 0.5 (`noise=guided`); the diagnostic warms the EMA with 20 estimator calls at the checkpoint | 5 (covariance adaptation; = Guided ES) |
| *anti* (reference) | antithetic pairs ±a (`noise=anti`) | not Gemini's: standard practice |
| *oracle* (reference) | as guided, but U = backprop's top-8 error directions (β = 1 and 0.5); diagnostic only | the ceiling for 1 and 5 |

## 3. Measures

- **M1, gradient quality.** The mean, over the 19 weight matrices, of the cosine between the estimated and the backprop
  gradient, on 4 held-out batches at the checkpoint, on both tasks. It is converted into **equivalent draws K_eq** by
  log-linear interpolation of the baseline curve (§0). **Gain = K_eq / (the baseline's K at the same cost):**
  - for every arm, that K is 32;
  - for local at K = 32, it is 8;
  - for local at K = 128, it is 32.
- **M2, training (task 1).** Validation loss after 300 steps at matched cost, two seeds (0, 1), for D0, lr_rand,
  lr_pca, orth, sobol, anti, local (K = 128), ent50 and guided. Wall-clock is reported too: orth, sobol and guided have
  overhead that the forward-equivalents do not count.
- **M0, depth (Gemini's premise for idea 3).** The baseline at L = 8 (d = 64, a backprop checkpoint after 300 steps),
  K = 32: the per-token error cosine of the hub and the writers, by block.

## 4. Predictions

- **P0 — depth is not the cost.** At L = 8, the writers' and hubs' per-token error cosine differs by less than 0.05
  between block 0 and block 7. Gemini said variance "grows exponentially with depth"; the measured law says the cost is
  the site's width.
- **P1 — idea 1 fails as stated.**
  - **lr_rand:** gain < 1 on both tasks. A random subspace misses most of the error.
  - **lr_pca:** gain < 1.1 on both tasks. The activations' main directions are not the errors' main directions.
- **P2 — idea 2 is a constant factor at most.**
  - **orth:** gain between 1.1 and 2.0 on both tasks, mostly at the 64-wide sites.
  - **sobol:** gain between 0.9 and 1.1.
- **P3 — idea 3 fails at matched cost.** local (K = 128) has a lower M1 than the baseline, because the local loss's
  gradient is not the global one, and its M2 is no better than the baseline by more than 0.02 nats.
- **P4 — idea 4 fails.** ent50 has a gain < 1, and its M2 is not better than the baseline.
- **P5 — idea 5 succeeds, modestly.**
  - **guided:** gain ≥ 1.3 on task 1, because the errors are low-rank (§0) and the subspace persists between batches.
  - **M2:** better than the baseline by ≥ 0.02 in both seeds.
  - **Ceiling:** below the oracle's 4×.
- **P6 — no idea gives orders of magnitude.** No arm reaches a gain of ≥ 10 on either task.

## 5. Verdicts

An idea **SUCCEEDS** if both of these hold:
- its M1 gain is ≥ 1.3 on both tasks;
- its M2 beats the baseline by at least the margin in both seeds.

It **FAILS** if either of these holds:
- its gain is < 1.1 on either task;
- its M2 is worse than the baseline.

Anything else is **PARTIAL**.

**The margin** is 0.02 nats, or twice the spread between the baseline's two seeds if that is larger.

## Files

- Code: `estimators.py`, `measure.py`, `train.py`.
- Results: `runs/e1/curve/` (baseline curve), `runs/e1/m1/` (diagnostics), `runs/e1/m2/` (training).

## Results — RAN 2026-10-10 (`runs/e1/m1/`, `runs/e1/m1_log.txt`, `runs/e1/m2/`)

### M0 — depth (L = 8, K = 32): the per-token error cosine by block, 0 → 7

| site | block 0 … block 7 |
|---|---|
| hub `o` | 0.555 0.558 0.551 0.553 0.557 0.561 0.560 0.558 |
| projection | 0.555 0.559 0.555 0.558 0.558 0.562 0.562 0.564 |
| MLP output | 0.553 0.553 0.556 0.556 0.558 0.559 0.563 0.563 |
| MLP hidden | 0.320 0.323 0.319 0.319 0.317 0.319 0.320 0.318 |

Flat. The cosine is set by the site's width (64 vs 256), not its depth.

### M1 — gradient quality at the checkpoints (gain = equivalent draws ÷ the baseline's draws at the same cost)

| arm | Latin mean cos | Latin gain | recall mean cos | recall gain |
|---|---|---|---|---|
| base (K = 32) | 0.509 | 1.00 | 0.353 | 1.00 |
| lr_rand, rank 8 | 0.281 | **0.25** | 0.221 | 0.30 |
| lr_rand, rank 16 | 0.349 | 0.39 | 0.263 | 0.49 |
| lr_pca, rank 8 | 0.587 | **1.52** | 0.383 | 1.26 |
| orth | 0.584 | **1.50** | 0.399 | 1.42 |
| sobol | 0.481 | 0.85 | 0.329 | 0.83 |
| anti (reference) | 0.413 | 0.57 | 0.279 | 0.56 |
| local, K = 32 (cost 73.5; vs baseline K = 8) | 0.389 | 1.99 | 0.255 | 1.78 |
| local, K = 128 (cost 291; vs baseline K = 32) | 0.551 | **1.26** | 0.375 | 1.18 |
| ent50 | 0.342 | 0.37 | 0.338 | 0.89 |
| ent25 | 0.292 | 0.27 | 0.305 | 0.69 |
| guided (rank 8, β 0.5) | 0.657 | **2.27** | 0.460 | 2.25 |
| oracle, β 1 (reference) | 0.736 | 3.72 | 0.542 | 4.11 |
| oracle, β 0.5 (reference) | 0.708 | 3.12 | 0.508 | 3.19 |

### M2 — training, Latin, 300 steps, two seeds, matched cost (86–87k forward-equivalents)

| arm | validation per seed | mean | vs base | seconds per run |
|---|---|---|---|---|
| base | 2.3670 / 2.3704 | 2.3687 | — | 57 |
| lr_rand | 2.3947 / 2.3824 | 2.3885 | +0.020 | 62 |
| lr_pca | 2.3765 / 2.3822 | 2.3794 | +0.011 | 83 |
| orth | 2.3509 / 2.3599 | 2.3554 | −0.013 | 114 |
| sobol | 2.3705 / 2.3788 | 2.3746 | +0.006 | 109 |
| anti (reference) | 2.3865 / 2.3870 | 2.3868 | +0.018 | 62 |
| **local, K = 128** | **2.2469 / 2.3161** | **2.2815** | **−0.087** | **135 (over the 2-minute limit)** |
| ent50 | 2.5027 / 2.4826 | 2.4927 | +0.124 | 63 |
| guided | 2.3630 / 2.3912 | 2.3771 | +0.008 | 89 |

The baseline's seeds differ by 0.003, so the margin is 0.02. For reference, backprop reaches 2.153 in 300 steps, at 900
forward-equivalents.

### Against the predictions

- **P0 — HELD.** Depth is not the cost: block 0 and block 7 differ by < 0.01. Gemini's "variance grows exponentially with
  depth" is wrong here.
- **P1 — split.**
  - **lr_rand: HELD.** Gain 0.25–0.49; idea 1 as a fixed random subspace FAILS.
  - **lr_pca: REFUTED in gradient quality.** Gain 1.52 / 1.26: the activations' main directions do overlap the errors'.
    But it trains slightly WORSE (+0.011), so by the verdict rule it FAILS.
- **P2 — orth HELD** (gain 1.50 / 1.42: a constant factor). It trains 0.013 better, in both seeds, but under the margin:
  **PARTIAL**. **sobol REFUTED, on the low side** (0.85 / 0.83, slightly worse than plain Gaussian): FAILS.
- **P3 — REFUTED.**
  - **Gradient quality:** local scoring at matched cost is better, even measured against the GLOBAL gradient (1.26 /
    1.18).
  - **Training:** it trains much better (−0.087; −0.122 and −0.053 by seed), the only large training effect in this
    experiment.
  - **Verdict:** PARTIAL by the rule, because the cosine gain is under 1.3.
  - **Caveat:** its runs took 135 s, over the 2-minute limit. Its many short reruns launch more kernels at the same
    forward-equivalent cost.
- **P4 — HELD.** Entropy gating FAILS badly (−0.37 gain; +0.124 in training).
- **P5 — REFUTED in training.** Guided has the best gradient quality of any real arm (2.27 / 2.25, about 60% of the
  oracle's ceiling), but it trains no better (+0.008). By the rule it FAILS.
- **P6 — HELD.** No arm reaches 10×. The best real gain is 2.3×; the oracle subspace is 3–4×.

### What it says

1. **Gemini's ideas, scored.**
   - **Idea 1:** fails as a fixed random subspace. As a PCA of activations it improves the gradient but not training.
   - **Idea 2:** orthogonal draws are a small real gain (1.4–1.5×, slightly better training); Sobol is worse.
   - **Idea 3:** local losses are the one clear training win, not for the reason Gemini gave: depth is not the cost.
   - **Idea 4:** entropy gating fails.
   - **Idea 5:** covariance adaptation (guided) gives the best gradients, but not better training.

   **None gives orders of magnitude.**
2. **Gradient cosine at a checkpoint is a poor predictor of training here.**
   - guided: 2.3× cosine, no training gain;
   - lr_rand: 4× worse cosine, only 0.02 worse;
   - local: 1.2× cosine, the best training.

   The Dust authors wrote that cosine picks candidates and training decides; this is a strong case of it. Momentum (Adam
   β1 = 0.9) already averages roughly 10 steps of noise. So the per-step cosine overstates how noise-limited training
   is, and something else holds Dust at 2.37 against backprop's 2.15 at equal steps. Experiment 2's V test (K = 128 vs
   32 vs backprop) asks whether that something is variance at all.
3. **Local losses change what is optimised.** Each block also gets a direct next-character objective, and its draws are
   4× cheaper. Experiment 2's bp_aux control asks whether the gain is the objective (deep supervision) or the estimator.
