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

## Results — RAN 2026-10-10 (`runs/e4/measure.json`, `runs/e4_measure_log.txt`, `runs/e4/train_ev*.json`)

**Training (backprop, 300 steps).**
- **EV5:** validation 2.357; EV2: 2.355. DENSE (the same recipe, Experiment 1's checkpoint): 2.344.
- **Event-driven hidden units cost almost nothing in quality.**
- **Homeostasis had not converged in 300 steps.** The measured firing fraction at the end:
  - EV5 ("5%"): 7.6 / 6.7 / 6.1 / 5.8% by block;
  - EV2 ("2%"): 5.5 / 4.0 / 3.3 / 3.6%.

  Still falling.

**"Changed" means |Δ| > 0.001 × the clean tensor's rms.** That is SEND-ON-DELTA: changes below the tolerance are not
propagated, as in event-driven hardware. So these numbers describe an APPROXIMATE delta rerun, not an exact one.

**Perturbation:** 4 random coordinates per token of block 0's MLP output, σ 0.2.

| | DENSE | EV5 | EV2 |
|---|---|---|---|
| active hidden units (block 1–3) | 36 / 27 / 26% | 6.7 / 6.1 / 5.8% | 4.0 / 3.3 / 3.6% |
| changed: residual stream (after block 1 / 2 / 3) | 13 / 15 / 16% | 54 / 62 / 65% | 63 / 71 / 72% |
| changed: MLP hidden pre-activation h | 75 / 68 / 65% | 91 / 89 / 87% | 92 / 90 / 89% |
| changed: MLP hidden output z | 23 / 18 / 18% | 5.7 / 5.1 / 5.0% | 3.6 / 3.0 / 3.2% |
| changed: attention output | 77 / 70 / 67% | 93 / 91 / 90% | 95 / 93 / 93% |
| branching ratio of z (block 2/1, 3/2) | 0.76, 1.04 | 0.89, 0.99 | 0.84, 1.06 |
| avalanche per token, all tokens perturbed (changed z entries; mean / median / p99) | 152 / 146 / 259 | 40 / 39 / 62 | 25 / 24 / 42 |
| avalanche per sequence, ONE token perturbed (mean / median / p99 / max) | 147 / 144 / 250 / 269 | 126 / 98 / 398 / 447 | 131 / 97 / 519 / 602 |
| **rerun multiply-adds needed (fraction)** | **0.25** | **0.43** | **0.48** |
| estimator cos at K = 32 / O1's gain | 0.509 (Experiment 1) / — | 0.474 / **1.54** | 0.467 / **1.70** |

### Against the predictions

- **G1 — REFUTED.** Under send-on-delta the dense model is NOT dense:
  - the residual-stream change stays at 13–16% of entries;
  - h at 65–75%;
  - z at 18–23%, about two-thirds of its active units.
- **G2 — half.**
  - **✓** EV keeps the hidden output change at its activity level (3–6%).
  - **✗** Its stream changes far less than the predicted ≥ 95% (54–72%). But it changes MORE than the dense stream,
    relative to its own scale. A sparser network writes a smaller stream, so the same absolute change is a larger
    relative one.
- **G3 — REFUTED.**
  - **The savings:** delta reruns would need only 25% (DENSE), 43% (EV5) or 48% (EV2) of a rerun's multiply-adds, so
    2–4× savings, more than predicted.
  - **Event-driven units do NOT add to it.** Here they make it worse.
- **G4 — HELD.** EV's estimator is within 0.04 of DENSE's, and O1 gains 1.5–1.7× on it.

### Two findings worth carrying forward

1. **Event-driven units produce the heavy tails the user worried about.** With one token perturbed:
   - **Typical avalanches are SMALLER than DENSE's:** median 97–98 against 144.
   - **The tail is much longer:** p99/median 4–5×, against DENSE's 1.7×.

   This is the regime where the user's rank or sign shaping of the rewards should matter. In the smooth dense network,
   rank shaping was exactly neutral (Experiment 3).
2. **The delta-rerun saving depends on the tolerance,** and it is unknown whether ignoring sub-tolerance changes keeps
   the draw's SIGNAL. That signal is the per-token loss change, itself a small number. **Experiment 4b** measures it
   before any delta rerun is built.

## Experiment 4b — does send-on-delta keep the signal? (pre-registered 2026-10-10, before it ran)

**What is compared.** For each draw, the per-token loss change from:
- an EXACT rerun;
- a SEND-ON-DELTA rerun, where after every operation a change smaller than tol × that tensor's clean rms is dropped:
  the value stays clean and the change is not propagated.

**Set-up.**
- Models: DENSE and EV5.
- Site: block 0's MLP output.
- Probes:
  - sparse: m = 4 coordinates per token, the delta-friendly probe;
  - dense: all 64 coordinates, Dust's own draw.
- tol = 1e-4, 1e-3, 1e-2.
- 32 draws on each of 2 held-out batches.

**Measures.**
- **(a)** Pearson correlation between the exact and send-on-delta per-token loss changes.
- **(b)** The cosine to backprop of that site's weight gradient estimated from the send-on-delta rewards, against the
  same draws' exact-reward estimate.
- **(c)** The rerun multiply-add fraction (as in Experiment 4).

**Predictions.**
- **H1:** at tol 1e-3, the correlation is ≥ 0.95 for DENSE and ≥ 0.9 for EV5, with sparse probes. The signal survives
  the 2–4× saving.
- **H2:** at tol 1e-2, the correlation is < 0.8. The saving grows, but the signal breaks.
- **H3:** dense probes save much less (fraction ≥ 0.6 at tol 1e-3), because the whole site changes.
- **H4:** sparse probes' estimator cosine at K = 32 is within 0.05 of dense probes' (exact rewards). Sparse probes are
  not worse estimators.

**What would change the plan.** If H1 and H4 hold, delta reruns with sparse probes are worth building: 2–4× on top of
everything else. If H1 fails, the event-driven saving does not survive contact with Dust's small signals at this
tolerance.

## Experiment 4b — RESULT (`runs/e4/fidelity.json`, `runs/e4b_fidelity_log.txt`)

| model / probe | estimator cos, exact rewards | tol 1e-4: corr / cos / cost | tol 1e-3: corr / cos / cost | tol 1e-2: corr / cos / cost |
|---|---|---|---|---|
| DENSE, sparse (4 of 64) | 0.577 | 1.000 / 0.577 / 0.68 | **0.993 / 0.574 / 0.42** | 0.511 / 0.226 / 0.16 |
| DENSE, dense (64) | 0.628 | 1.000 / 0.628 / 0.78 | 0.996 / 0.625 / 0.73 | 0.436 / 0.191 / 0.39 |
| EV5, sparse (4 of 64) | 0.460 | 1.000 / 0.460 / 0.64 | 1.000 / 0.460 / 0.58 | **0.897 / 0.408 / 0.25** |
| EV5, dense (64) | 0.553 | 1.000 / 0.553 / 0.71 | 1.000 / 0.553 / 0.70 | 0.935 / 0.518 / 0.57 |

("cost" is the rerun multiply-add fraction; the exact-rerun fractions in Experiment 4 used a looser definition.)

### Against the predictions

- **H1 — HELD.** At tol 1e-3, send-on-delta keeps the signal: correlation 0.993 (DENSE) and 1.000 (EV5), with the
  estimator cosine unchanged.
- **H2 — split.**
  - **DENSE ✓:** breaks at 1e-2 (0.51).
  - **EV5 ✗:** still 0.90 at 1e-2. **Event-driven units make send-on-delta far more robust**, because thresholds absorb
    small changes. So EV can run at a 10× coarser tolerance, at 0.25 of the rerun cost.
- **H3 — HELD.** Dense probes save little (0.70–0.73 at 1e-3).
- **H4 — REFUTED.** Sparse probes ARE worse estimators: −0.05 cos on DENSE, −0.09 on EV5.

### What it adds up to

**The net saving.** Converted with the cosine law cos² = K/(K + c):
- DENSE, sparse probes at tol 1e-3: ~1.33× more draws at 0.42 of the cost each, about **1.8× cheaper** than exact
  dense draws.
- EV5, sparse probes at tol 1e-2: ~2.2× more draws at 0.25 of the cost, also about **1.8×**.

**The verdict.** Delta reruns are a real but modest lever, about 2×, which multiplies with the others. Event-driven units
are what make coarse send-on-delta safe. They do not pay for themselves in this transformer, because its residual
stream, attention and norms stay dense. The architecture changes that would let them pay more are:
- sparse writes to the stream;
- local normalisation;
- sparse attention.
