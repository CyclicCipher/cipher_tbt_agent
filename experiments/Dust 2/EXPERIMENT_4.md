# Experiment 4 — event-driven units: what would delta reruns save, and what do they cost? (pre-registered 2026-10-10)

Status: **pre-registered before any of it ran.** Approved order (user, 2026-10-10): compile the rerun path (done: 1.2×,
identical maths), then this measurement, then delta reruns only if the avalanches are small. The user's correction
applies: top-k is not the model of their idea. The brain's critical-state dynamics are LOCAL thresholds with fluctuating
activity. So the event-driven variant here uses per-unit thresholds with homeostasis.

## The question

Every Dust draw reruns the network from a perturbed site. In an event-driven network, a rerun would only need to
recompute what CHANGED. How much changes after a perturbation (the "avalanche"), in a dense model and in an event-driven
one? And how good is the gradient estimate from those draws?

**Two facts shape the design:**
1. **Dust's population comes from perturbing every token at once.** So a delta rerun only saves work if the
   perturbation is sparse in units: a few units per token.
2. **A transformer has densifying operations.** Each one spreads a single change to every coordinate of a token, or to
   every later token:
   - RMSNorm: one common scale factor per token;
   - the residual write W_out·Δz: every column is dense;
   - the attention softmax: across tokens.

## Models (both trained by backprop for 300 steps on task 1, then frozen; measurement only, no Dust training)

**DENSE.** Experiment 1's checkpoint (ReLU² MLP).

**EV: the MLP's hidden units are event-driven.**
- **The unit:** z_j = (h_j − θ_j)₊², where each unit has its own threshold θ_j: elementwise, no sort.
- **Homeostasis:** θ_j += 0.05·(rate_j − target) after every training step, where rate_j is unit j's firing fraction in
  the batch. θ starts, block by block, at each unit's (1 − target) quantile of its pre-activation over one batch.
  (Amended before running: the first draft said 0.01 and a zero start, which would need thousands of steps to reach the
  target.)
- **Target activity:** 5% (EV5) and 2% (EV2).

## Measures (on 4 held-out batches)

1. **Activity:** the measured firing fraction, with its spread over tokens and units.
2. **The cascade from a sparse perturbation.** At block 0's MLP-output writer, perturb m = 4 random coordinates per
   token (σ = 0.2). Then, at every later layer, the fraction of entries that change (|Δ| > 1e-3 of that tensor's rms):
   - in the residual stream;
   - at the MLP hidden pre-activation h;
   - at the hidden output z;
   - at the attention output.

   The **branching ratio** is the changed fraction of z in block b+1 ÷ in block b.
3. **The avalanche-size distribution:** per token, the number of changed hidden-output entries summed over blocks. Its
   mean, median and 99th percentile; a heavy tail is the user's concern.
4. **What delta reruns would save:** the fraction of rerun multiply-adds that touch a changed input:
   - a matmul whose input change is sparse costs that fraction;
   - an elementwise op or a norm costs only its changed entries;
   - attention costs in full.
5. **Estimator quality on EV, against DENSE:**
   - the gradient cosine of the baseline estimator;
   - the cosine of O1, perturbing only units within 2σ of their thresholds.

## Predictions

- **G1 — the dense model densifies at once.** After one block, the residual-stream change is dense (≥ 95% of entries)
  and so is the hidden pre-activation. The hidden output change matches the active fraction (ReLU² ~ 30–50%).
- **G2 — EV keeps the hidden outputs sparse but not the stream.**
  - **Hidden output:** the changed fraction ≈ the active fraction plus the threshold crossings, ≤ 2× target.
  - **Residual stream and hidden pre-activation:** stay dense (≥ 95%), because W_out writes densely and RMSNorm rescales
    the whole token.
- **G3 — so delta reruns would save little in this architecture.** The rerun-cost fraction (measure 4) is ≥ 0.6 for EV5:
  only the MLP's second matmul and elementwise ops benefit. Sparse activity ALONE is not enough. Event-driven savings also
  need sparse connectivity (each unit writes to a few coordinates) and no per-token global norms. That is an architecture
  change, not a training trick.
- **G4 — EV's estimator is no worse than DENSE's,** within 0.05 of mean cosine. O1 on EV gains ≥ 1.5 against EV's
  baseline, because the activity is genuinely sparse.

**What would change the plan.** If G3 holds, the event-driven route to cheap reruns goes through the architecture:
sparse writes, local normalisation. That joins the user's DiffusionBlocks / Block-AttnRes design questions, and delta
reruns are not built for the current transformer. If G3 fails (savings ≥ 2× already), build delta reruns now.

## Files

`ev.py` (the event-driven unit, homeostasis, the cascade and cost measurements). Results: `runs/e4/`.
