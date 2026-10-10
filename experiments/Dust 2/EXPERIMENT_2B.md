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
