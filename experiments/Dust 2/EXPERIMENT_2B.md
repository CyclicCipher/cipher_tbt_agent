# Experiment 2b — the guide fix, and stacking what trained well (pre-registered 2026-10-10)

Status: **pre-registered before any arm ran**, after Experiment 2's training results (see `EXPERIMENT_2.md` results).

## Why

Experiment 2's training split the ideas in two.

**These trained BETTER than the baseline:**
- O4, the local transposes: −0.036 at matched cost;
- local losses (Experiment 1): −0.087;
- orth (Experiment 1): −0.013.

**The anisotropic-draw ideas trained WORSE, despite the best gradient cosines:**
- the guided subspace (Experiment 1): +0.008;
- O3, the top-error guide: +0.06;
- the combo: +0.15.

**My diagnosis.** With draws a ~ N(0, Σ), the average of r·a estimates **Σg, not g**. That is Guided ES's own
estimator, and it over-weights the guessed directions: about 20× along the top-error guide, and 4.5× inside the subspace.
Wherever the guess is poor (the attention projections, the embedding), the update mostly follows the guess. Worse, the
guided subspace is the EMA of these estimates, which are already stretched along it. So it reinforces itself and stops
tracking the true errors as the model moves.

A cosine measured at a fixed checkpoint, with the subspace warmed there, cannot see either problem.

**The fix ("whitened").**
- Use Σ⁻¹ times the average, which estimates g itself, per token, by the Woodbury identity.
- Build the guided subspace from the whitened estimates.

The guide then only decides WHERE the draws look harder. It no longer decides where the update goes.

## Arms (training: 300 steps, two seeds, val every 10 steps; M1 for the whitened arms)

| arm | settings | compare with (reused, not re-run) |
|---|---|---|
| **guided_w** | guided rank 8, β 0.5, `whiten=1`, K = 32 | Experiment 1 guided (2.377), baseline (2.369) |
| **top_w** | top_guide 0.3, exact head, `whiten=1`, K = 32 | Experiment 2 O3_top (2.432) |
| **combo_w** | combo, K = 56, `whiten=1` | Experiment 2 combo_K56 (2.522) |
| **O4_orth** | hub_T + orth, K = 56 | Experiment 2 O4_hubT_K56 (2.333) |
| **O4_local** | hub_T + local losses, K = 128, at about HALF the baseline's cost (the 2-minute limit forbids matched cost) | baseline at full cost (2.369) |

## Predictions

- **F1 — the fix works.**
  - guided_w trains better than the baseline by ≥ 0.02.
  - top_w trains better than O3_top by ≥ 0.05, and is not worse than the baseline.
  - combo_w trains better than combo_K56 by ≥ 0.10.
- **F2 — whitening costs some checkpoint cosine.** guided_w's M1 gain is below guided's 2.27, because Σg is better aligned
  with g along the guessed directions than the unbiased estimate is at K = 32. This is the measure's blind spot showing.
- **F3 — the good ideas stack.**
  - O4_orth trains better than O4 alone (any margin).
  - O4_local, at half the cost, trains better than the full-cost baseline by ≥ 0.02.

## Verdicts

Success or failure is judged against the comparison column, by the 0.02-nat margin in both seeds.

## Results — RAN 2026-10-10 (`runs/e2b/`, `runs/e2/m1_arms2b_log.txt`; compiled rerun path, identical maths)

### M1 — gradient quality (gain over the baseline at matched cost)

| arm | Latin | recall |
|---|---|---|
| guided_w | **0.24** | 0.27 |
| top_w | 0.30 | 0.26 |
| combo_w_K56 | 0.19 | 0.18 |
| **O4_orth_K56** | **6.10** | **5.46** |

O4_orth's per-site cosines: writers 0.92–0.93. With O4 only the 64-wide writers are rerun, and 56 orthogonal draws in
64 dimensions span almost all of it, so each token's error is nearly exact.

### M2 — training, 300 steps, two seeds

| arm | validation per seed | mean | vs its comparison | cost to first reach 2.37 | s/run |
|---|---|---|---|---|---|
| guided_w | 2.4017 / 2.3997 | 2.4007 | +0.024 vs guided (2.377); +0.032 vs baseline | not reached | 111 |
| top_w | 2.3785 / 2.3737 | 2.3761 | **−0.056 vs O3_top (2.432)**; +0.007 vs baseline | not reached | 56 |
| combo_w_K56 | 2.4088 / 2.4064 | 2.4076 | **−0.114 vs combo_K56 (2.522)**; +0.039 vs baseline | not reached | 96 |
| **O4_orth_K56** | **2.2922 / 2.3017** | **2.2969** | **−0.036 vs O4 (2.333); −0.072 vs baseline** | **44k / 44k** | 116 |
| **O4_local_K128** (≈ half the cost) | **2.2356 / 2.3274** | **2.2815** | **−0.087 vs the FULL-cost baseline** | **18k / 29k** | 94 |

### Against the predictions

- **F1 — split.**
  - **guided_w FAILED:** whitening made the guided subspace worse.
  - **top_w HELD:** whitening repaired the top-error guide (−0.056), back to the baseline's level.
  - **combo_w HELD:** whitening repaired the combo (−0.114), but it is still worse than the baseline.
- **F2 — HELD in direction, not in size.** Whitening costs far more checkpoint cosine than predicted: gain 0.24, not just
  below 2.27. Each draw moves along the guessed directions and the rest at once. The large guessed-direction part then
  dominates every reward, and dividing by Σ amplifies that noise in all the other directions.
- **F3 — HELD.**
  - **O4 + orth:** trains better than O4 alone (−0.036).
  - **O4 + local, at half the cost:** beats the full-cost baseline by 0.087. Its seeds differ, −0.133 and −0.041; local
    scoring's results vary by seed, as in Experiment 1.

### What it says

1. **The anisotropic-draw family is a dead end in this form.**
   - Σg over-weights the guessed directions.
   - The unbiased Σ⁻¹ version is too noisy, because each draw mixes the guessed and unguessed directions.
   - The repair is to make SEPARATE draws inside and outside the guessed subspace, so neither's reward carries the
     other's signal. It has not been built.

   The over-weighting explanation now fits better as a CURVATURE effect than as "bias": stretching the update along the
   top error directions acts like a larger learning rate exactly where the loss is sharpest. `PROXY.md` tests that
   reading.
2. **Stacking the ideas that train well works:**
   - **O4 + orth:** 2.0× cheaper than the baseline to reach 2.37.
   - **O4 + local:** 3–4.7× cheaper.

   Against backprop (390 forward-equivalents to 2.37), the gap goes from about 220× to about **50–75×**.
