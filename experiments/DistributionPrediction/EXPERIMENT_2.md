# Experiment 2 — does a frontier curriculum help in-context extrapolation? (pre-registered 2026-10-08)

Status: **pre-registered before any code for it ran.** Asked for by the user after the exploration (`EXPLORATION.md`):
"pre-register the curriculum comparison and run it". Results are reported against this file.

## The question

Every training sequence already covers every context size and every horizon, so a curriculum here decides how much
WEIGHT each part of the problem gets in the loss. The parts ("buckets") are the 7 families × 4 horizon groups (k = 1,
2–4, 5–8, 9–16) = 28 buckets. Does weighting them by a signal of "learnable now" beat uniform weighting at the same
wall-clock — and in particular, does it rescue the rule the model learns all-or-nothing (rotation)?

## Base model (fixed; the exploration's best at 3 layers)

A2 density with the horizon-as-input head (`--head query`), the step input (`--diff 1`), the learned uniform component
(`--unif 1`), d = 64, 3 layers, lr 1.6e-2 with 100 warmup steps then cosine — the cosine runs over WALL-CLOCK, not steps.
**Equal wall-clock: 70 seconds of training per run**, so a curriculum pays for its own cost in fewer steps. Same data
generator, same evaluation as Experiment 1 (`run_e1.gain_table`). Three seeds per arm.

## Arms

The loss is a weighted mean of the 28 bucket losses: Σ_b count_b · c_b · L_b / Σ_b count_b · c_b, where L_b is bucket b's
mean negative log-likelihood in the batch, count_b its number of terms, and c_b the curriculum's multiplier (c_b = 1 is
exactly Experiment 1's term-mean loss). Curricula start after 100 steps (uniform before), and mix 20% uniform:
c_b = 0.8 · s_b / mean(s) + 0.2, so no bucket is ever starved.

| arm | signal s_b | source |
|---|---|---|
| **U** uniform | — (c_b = 1) | the baseline |
| **LP** learning progress | \|EMA_slow(L_b) − EMA_fast(L_b)\| (rates 0.01 and 0.1), updated every step | Oudeyer; Teacher–Student; ALP-GMM; "epiplexity" in our TBT notes |
| **GA** gradient alignment | \|⟨∇L_b, P ⊙ (θ_past − θ_now)⟩\|, θ_past = the snapshot nearest half the steps so far, P = Adam's per-weight step lr/(√v̂ + ε); recomputed every 100 steps (28 extra backward passes each time — counted in the wall-clock) | Self-Play Pretraining with Zero Data, eq. 2 |
| **AF** advantage filter | per TERM, advantage = its loss − its bucket's running mean loss; only the half of the terms with the largest \|advantage\| in each batch enter the loss | Ataraxos's advantage filtering |

The P0 staircase (`p0_ceiling.Frontier`) is not an arm: it needs a "mastered" threshold, which a continuous log score
does not supply.

## Measures (held-out, as Experiment 1)

- **M1** — far-horizon (k = 9–16) gain at n = 8, 16, 32, mean over circle, spiral, lissajous, billiard ("4-fam"); mean
  and spread over seeds.
- **M2** — rotation: learned if its k = 1 gain at n = 16 is ≥ 3 bits; seeds learned, per arm.
- **M3** — guards: uniform-noise worst gain at n ≥ 2; pen far gain.
- **M4** — steps actually trained in the 70 s, and what each curriculum emphasised (multipliers per family and per
  horizon group at the probes).

## Predictions

- **C1 — on the smooth families, no curriculum beats uniform by more than 0.3 bits** (4-fam far, n = 16, seed means).
  They learn without a plateau (Experiment 1: A2's far gain rose from the first probe), so there is little for a
  curriculum to reallocate, and LP and GA lose steps to their own bookkeeping.
- **C2 — the curriculum's blind spot: LP and GA learn rotation in no more seeds than U.** Both measure PROGRESS, and a
  rule on a plateau shows none — so they will not raise its weight before it starts to move, which is exactly when help
  is needed. (The 20% uniform mixing keeps rotation trained at a reduced rate.) No prediction for AF.
- **C3 — guards hold:** every arm's uniform-noise worst gain at n ≥ 2 stays within 0.3 bit of U's.

**Refuted / what it means.** C1 refuted (some curriculum wins by > 0.3 bits) → reallocating weight helps even without a
plateau; adopt it. C2 refuted (LP or GA learns rotation in more seeds than U) → the signal does see a plateau about to
break, which is the result we would most want. If GA's bookkeeping costs more than ~25% of its steps, that is reported as
its price.

## Files

`curriculum.py` (the four weighting rules), `run_e2.py` (time-budgeted training with a curriculum; reuses `run_e1`'s
model and evaluation), results in `runs/e2/<arm>_s<seed>.{json,log}`.
