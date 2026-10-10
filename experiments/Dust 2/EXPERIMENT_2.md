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
