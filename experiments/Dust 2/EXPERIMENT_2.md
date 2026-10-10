# Experiment 2 — my own ideas for cutting Dust's cost (pre-registered 2026-10-10)

Status: **pre-registered before any of these arms ran.** It was written after Experiment 1's gradient-quality results
(`EXPERIMENT_1.md`), which shaped it. The user asked: "then come up with your own ideas".

## What Experiment 1 established, and how it shapes this one

1. **The noise at every site is pure "dimension noise".**
   - **The law:** the per-token error cosine follows √(K/(K+D)) exactly.
   - **Antithetic pairs:** they remove the second-order term and only LOST (gain 0.57), because they halve the
     independent directions.
   - **Depth:** irrelevant. Block 0 and block 7 of an 8-layer model differ by < 0.01.

   So the only levers are: fewer effective dimensions per measurement, more information per draw, and better-placed
   draws.
2. **The true per-token errors are LOW-RANK** (effective rank 2–18 out of 64–256). The oracle rank-8 subspace is worth
   3–4×, and the learned (guided) subspace 2.3×.
3. **Rerun sites dominate the cost.** There are 17 of them, each perturbed in its own passes. Half are the attention
   hubs and MLP hidden layers, whose errors are tied to their own block's writer by a single linear map.

## Arms

| arm | what changes (`--set`) | idea (`IDEAS.md` §5) |
|---|---|---|
| **O1_sparse** | at the MLP hidden, perturb only units within 2σ of being active (`sparse_c=2`); an inactive unit's error is exactly 0 for ReLU-type units, so nothing true is lost | O1 (sparsity cuts the dimension) |
| **top10_O1** | the same, on a model whose MLP keeps only its top 10% units (`--topk 0.1`), against that model's own baseline (`top10_base`) | O1 with event-like sparsity |
| **O2_exact_head** | the head's error in closed form (softmax − one-hot), no draws | O2 |
| **O3_top**, **O3_top05** | per token, a share (0.3 or 0.5) of each draw's variance at the residual writers (embedding, attention projection, MLP output) goes along the EXACT error at the top of the residual stream. That error is computed from the head in closed form: no pass through the network, nothing stored. A writer's output reaches the top through the identity path, so this is a per-token guess for its error (like Direct Feedback Alignment, but measured, not trusted) | O3 |
| **O4_hubT**, **O4_hubT_K56** | the attention hub's and the MLP hidden's errors are no longer rerun sites; they come from their own block's writer estimate through the one linear map between them (W_projᵀ, W_outᵀ). The hidden layer's activation slope is measured by per-unit draws (4), so the activation may be non-differentiable. Rerun sites drop from 17 to 9 (cost 287 → ~145 at K = 32), so the matched-cost arm uses K = 56 | O4 (narrow groups with their own score; Dust's hub trick applied to the MLP) |
| **combo**, **combo_K56** | guided subspace (rank 8, β 0.5) + O1 + O3 (0.3) + O4 + exact head, at K = 32 and at matched cost (K = 56) | all together |

**Amendment, before any arm of this experiment ran** (after Experiment 1's training showed local losses training
0.05–0.12 nats better than the baseline, with only a 1.2× gain in gradient cosine):

| arm | what changes | why |
|---|---|---|
| **combo_local** | combo + auxiliary heads after every block, each site scored by the next one (`--aux_every 1 --set local=1` + COMBO), at matched cost | does local scoring stack with the rest? |
| **bp_aux** (control, backprop) | backprop with the same auxiliary losses added (weight 1) vs plain backprop, 300 steps, two seeds | is local's gain the OBJECTIVE (deep supervision helps any learner) or the ESTIMATOR (cheaper, cleaner draws)? |

**What O4 gives up, said plainly.** It multiplies an estimated error by the transpose of ONE linear map inside a block.
That is a local backward step through a linear layer. It is not a pass through the network, and it stores nothing
across blocks. The activation function stays free: no derivative is used, its slope is measured. If this crosses a line
the user wants kept, O4 drops out, and the other arms do not depend on it.

## Measures

- **M1 (as Experiment 1).** The mean cosine to backprop over the weight matrices, at the same checkpoints, on both
  tasks. It is converted into equivalent draws K_eq with the baseline curve of the SAME model; the top-10% model gets its
  own curve. **Gain = K_eq / (the baseline's K at the arm's cost).**
- **W (width scaling), task 1, d = 64, 128, 256** (backprop checkpoints after 300 steps, head width 16). For each width:
  - the effective rank of the true errors per site type;
  - the baseline's cosine at K = 32, 128 and 512;
  - the gain of the oracle rank-8 subspace, the guided subspace and the combo.

  This is the question behind the user's premise that Dust's "cost scales with the width of the network".
- **M2 (training, task 1).** 300 steps, two seeds, at matched cost:
  - arms: O1_sparse, O3_top, O4_hubT_K56 and combo_K56, against Experiment 1's baseline;
  - the top-10% model: its baseline vs O1;
  - plus the headline: **the forward-equivalents each method needs to reach validation 2.37** (the baseline's
    300-step loss), against backprop's (from a backprop run evaluated every 10 steps).

- **V (is Dust's training limited by variance or by bias?).** Added after seeing Experiment 1's first training results,
  before any arm here ran:
  - **What prompted it:** lr_rand trained only ~0.02 nats worse than the baseline with a 4× worse gradient cosine, and
    lr_pca slightly worse with a 1.5× better one.
  - **The runs:** 100 steps (same schedule shape, two seeds) of baseline Dust at K = 32 and at K = 128 (4× the draws),
    and backprop.
  - **The reading:** if K = 128 closes most of the gap to backprop, training is variance-limited, and the estimator
    ideas matter. If not, the gap is BIAS:
    - γ = 0 drops credit through attention to later tokens;
    - σ smoothing;
    - the local q/k/v scoring.

    Then those, not the draw count, are what to fix first.

## Predictions

- **Q1 — O1 helps only where units are really sparse.**
  - **On ReLU²:** about 2/3 of the hidden units are within 2σ of active, so the gain is 1.0–1.15.
  - **On the top-10% model** (~17% of units kept): the MLP-hidden weight cosines rise by ≥ 0.15, and the mean gain
    is ≥ 1.2.
- **Q2 — O2 alone is a small gain (1.0–1.1).** The head is one matrix of 19, and its draws were already cheap.
- **Q3 — O3_top gains ≥ 1.5 on task 1 and ≥ 1.3 on task 2.**
- **Q4 — O4_hubT_K56 gains ≥ 1.5 on both tasks.** At the hidden layer, a 64-wide estimate passed through W_outᵀ beats a
  256-wide direct one.
- **Q5 — combo_K56 gains ≥ 4 on task 1 and ≥ 3 on task 2,** but < 10.
- **Q6 — width.**
  - The writers' effective rank at d = 256 is at most 2× its value at d = 64.
  - The oracle rank-8 gain at d = 256 is at least 2× its gain at d = 64.

  If both hold, subspace-guided perturbation's cost grows much more slowly than width. Then the orders of magnitude
  appear at Dust's widths (512–4096), not at ours.
- **Q7 — training (M2) at matched cost:**
  - combo_K56 beats the baseline by ≥ 0.05 nats in both seeds;
  - O3_top and O4_hubT_K56 beat it by ≥ 0.02;
  - O1_sparse is within ±0.02 on ReLU², and top10_O1 beats top10_base by ≥ 0.02.
- **Q10 — bp_aux.** Backprop with auxiliary losses is within ±0.02 of plain backprop. If so, local's gain under Dust is
  the estimator's (cheaper, lower-noise scoring of early blocks), not deep supervision. **combo_local** beats combo_K56
  by ≥ 0.02.
- **Q9 — V: bias-limited.** At 100 steps, K = 128 closes less than half of the gap between K = 32 and backprop.
- **Q8 — the headline.** combo_K56 reaches validation 2.37 with at most 1/3 of the baseline's forward-equivalents.

## Verdicts

The rules are Experiment 1's:
- **Success:** M1 gain ≥ 1.3 on both tasks, and M2 better by the margin in both seeds.
- **Failure:** a gain below 1.1, or a worse M2.
- **Partial:** anything else.

## Files

- Code: `estimators.py` (`sparse_c`, `top_guide`, `hub_T`, `act_K`; `top_error`, `act_local`), `model.py` (`margin`,
  `xL`), `e2_m1.py`.
- Results: `runs/e2/m1/`, `runs/e2/m2/`.

## Results — RAN 2026-10-10 (`runs/e2/m1/`, `runs/e2/m1_arms_log.txt`, `runs/e2/m1_width_log.txt`, `runs/e2/m2/`, `runs/e2/v/`)

**Reused, not re-run:**
- the Dust baseline: Experiment 1's `runs/e1/m1/*_base.json` and `runs/e1/m2/base_s*`;
- the d = 64 width row: Experiment 1's curve, oracle and guided numbers;
- backprop at seed 0, 300 steps: 2.1526, from Experiment 1 §0.

After the code changes, a one-batch check showed the baseline gradients differ from before only at the GPU's run-to-run
level (~1e-9).

### M1 — gradient quality (gain = equivalent draws ÷ the baseline's draws at the same cost)

| arm | Latin gain | recall gain | note |
|---|---|---|---|
| O1_sparse (ReLU² model) | 1.17 | 1.42 | MLP hidden fc0 cos 0.38 → 0.46 |
| top10_O1 (top-10% model, vs its own curve) | 1.47 | 1.55 | per-matrix numbers of that model's baseline were not saved (my omission) |
| O2_exact_head | 1.02 | 1.06 | |
| O3_top (0.3) / O3_top05 | 1.60 / 1.60 | 1.45 / 1.44 | helps the MLP outputs (out0 0.65 → 0.86), HURTS the attention projections (0.61 → 0.43) and embedding (0.44 → 0.25): the top error is a good guess only for some writers |
| O4_hubT, K = 32 (cost 158) / K = 56 (cost 275) | 2.71 / **2.62** | 2.51 / **2.37** | the MLP hidden's cosine doubles: 0.36 → 0.73 |
| combo, K = 32 / K = 56 | 5.62 / **3.76** | 4.22 / **2.79** | |
| combo_local, K = 56 (cost 67) | 6.49 | 5.59 | |

### W — width (Latin; d = 64 from 4 batches, d = 128 and 256 from 2)

| d | effective rank of true errors: hub / proj / out / hidden / qkv / emb | baseline cos at K = 32 / 128 / 512 | oracle rank-8 gain | guided gain | combo gain (K = 32) |
|---|---|---|---|---|---|
| 64 | 2.5 / 5.7 / 6.6 / 14.1 / 15.9 / 7.3 | 0.509 / 0.748 / 0.892 | 3.7 | 2.4 | 5.7 |
| 128 | 3.9 / 7.8 / 8.1 / 21.2 / 34.0 / 11.7 | 0.404 / 0.642 / 0.834 | 6.7 | 2.9 | 11.4 |
| 256 | 6.2 / 11.5 / 10.7 / 25.1 / 34.1 / 20.9 | 0.285 / 0.496 / 0.728 | 9.9 | 3.3 | **19.7** |

**What the width rows show:**
- **Plain Dust's cosine at fixed draws falls with width:** cost ∝ width, the user's premise, confirmed.
- **The errors' effective rank grows about 2× for 4× the width.**
- **The combo's cosine barely moves** (0.71 → 0.70 → 0.66), so its gradient-quality advantage grows with width.

### V — variance or bias? (100 steps, two seeds)

| | validation |
|---|---|
| Dust, K = 32 | 2.4367 / 2.4377 |
| Dust, K = 128 | 2.4228 / 2.4216 |
| backprop | 2.3954 / 2.4046 |

4× the draws closes 0.015 of the 0.037 gap: 40%.

### M2 — training, Latin, 300 steps, two seeds

| arm | validation per seed | mean | vs baseline | s/run | cost (fwd-eq) | cost to first reach 2.37 |
|---|---|---|---|---|---|---|
| baseline (Experiment 1) | 2.3670 / 2.3704 | 2.3687 | — | 57 | 86,107 | ~86k (seed 0 at step 300; seed 1 not reached) |
| O1_sparse | 2.3629 / 2.3691 | 2.3660 | −0.003 | 61 | 86,107 | 66k / 77k |
| O3_top | 2.4252 / 2.4392 | 2.4322 | **+0.064** | 61 | 86,107 | not reached |
| **O4_hubT_K56** | **2.3216 / 2.3442** | **2.3329** | **−0.036** | 68 | 82,515 | **50k / 52k** |
| combo_K56 | 2.5294 / 2.5141 | 2.5217 | **+0.153** | 83 | 82,515 | not reached |
| combo_local_K128 (about half the cost) | 2.5097 / 2.5110 | 2.5103 | +0.142 | 104 | 45,830 | not reached |
| top10_base | 2.3796 / 2.3877 | 2.3837 | (top-10% model) | **170, over the 2-min limit** | 86,107 | not reached |
| top10_O1 | 2.3673 / 2.3651 | 2.3662 | −0.018 vs top10_base | **169, over** | 86,107 | 75k / 66k |
| backprop | 2.1526 / 2.1646 | 2.1586 | | 5 | 900 | **390** (step 130) |
| backprop + auxiliary losses (bp_aux) | 2.1907 / 2.2154 | 2.2031 | +0.045 vs backprop | 6 | 900 | 390 / 450 |

**Two flags.**
- **Time limit:** the top-10% runs took ~170 s, breaking the 2-minute rule. The global top-k selection, a sort, runs in
  every rerun.
- **Top-k as a model:** it is a poor stand-in for the user's critical-state sparsity: global, fixed activity, no
  avalanches. Experiment 4 replaces it with per-unit thresholds and homeostasis.

### Against the predictions

- **Q1 — O1, mixed.**
  - **ReLU²:** helps a little more than predicted in gradient (1.17 / 1.42 against 1.0–1.15); training is unchanged
    (−0.003), as predicted.
  - **Top-10% model:** gradient gain 1.47 / 1.55 (✓ ≥ 1.2). Training −0.018 against its own baseline (−0.012 and
    −0.023 by seed), just under the 0.02 bar: PARTIAL.
- **Q2 — HELD** (1.02 / 1.06).
- **Q3 — REFUTED in training.** O3 has a gradient gain of 1.60 / 1.45 (✓), but it trains 0.064 WORSE: FAIL.
- **Q4 — HELD: the first SUCCESS of the project.** O4 gains 2.62 / 2.37 in gradient, and trains better by 0.047 and
  0.025. It reaches validation 2.37 with ~51k forward-equivalents against the baseline's ~86k: **1.7× cheaper**.
- **Q5 — REFUTED.** combo's gradient gain is 3.76 / 2.79 (below the predicted 4 / 3), and it trains much WORSE (+0.153):
  FAIL.
- **Q6 — mostly HELD.**
  - **Rank:** grows sub-linearly with width, ×1.6–2.0 at the writers for ×4 width. proj is 2.02×, a hair over the stated
    bound.
  - **Oracle gain:** ×2.65 at d = 256 vs d = 64 (✓ ≥ 2).
  - **Caveat:** this is gradient quality. The guided ideas that carry the width advantage failed in training here (next
    point), so the width result only matters if Experiment 2b's fix works.
- **Q7 — split.** O4 ✓; O1 ±0.02 ✓; top10_O1 narrowly missed (0.018); combo ✗; O3 ✗.
- **Q8 — REFUTED.** combo never reached 2.37. The best real arm, O4, needs 0.59 of the baseline's cost, not ≤ 1/3.
- **Q9 — HELD in the letter (40% < half), but only for the first 100 steps,** where Dust trails backprop by only 0.04.
  The gap opens LATER: backprop goes 2.398 → 2.165 between steps 100 and 300, while Dust goes 2.43 → 2.37. So the
  100-step test does not settle what holds Dust back later on.
- **Q10 — REFUTED, and informative.** Auxiliary losses make BACKPROP worse (+0.045), yet they made Dust better in
  Experiment 1 (−0.087). So local scoring's gain under Dust is the ESTIMATOR's (truncated, cheaper, cleaner draws), not
  a better objective. It wins despite optimising a slightly worse one.

### What it says

1. **Two ideas really help training, both by giving the estimator less to do:**
   - **O4:** hub and hidden-layer errors come through the one local linear map from their own block's writer estimate,
     with the activation's slope measured by per-unit draws. Rerun sites drop from 17 to 9, and the MLP hidden gets a
     64-wide estimate in place of a 256-wide one. **1.7× cheaper to reach the baseline's loss.**
   - **Local scoring** (Experiment 1): truncated draws.
2. **Every anisotropic-draw idea failed in training, however good its gradient cosine:** guided, O3 and combo. My
   diagnosis: Dust's average estimates Σg, not g. Σg over-weights the guessed directions up to 20×, and the guided
   subspace reinforces itself because it is built from Σg. Experiment 2b tests the fix: Σ⁻¹-whitened estimates, and the
   subspace built from them.
3. **The cost gap to backprop:** the baseline needs ~86k forward-equivalents to reach 2.37 against backprop's 390, about
   **220×**. O4 brings that to about **130×**.
