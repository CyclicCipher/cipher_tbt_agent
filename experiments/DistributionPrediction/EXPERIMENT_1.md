# Experiment 1 — predicting the extrapolated distribution: sampled continuations vs a queryable density

Status: **DESIGNED 2026-10-08; nothing built or run.** Pre-registration below; it is committed before any code for it
runs, and results are reported against it. Context: `IDEA.md` §6 (the objective and its three candidate outputs),
`TRANSCRIPT.md` (the user's notes). The user chose to test candidates 2 and 3 of §6 and to start from a transformer.

## The question

The notes propose replacing "predict the next token" with "predict the best extrapolated distribution of this pattern,
then predict the next token from it". Experiment 1 asks, at the smallest scale that can answer it:

1. **Does the objective help?** Trained to predict the distribution of the CONTINUATION, does a transformer extrapolate
   better than the same transformer trained on next-step prediction and rolled out?
2. **Which output representation?** Candidate 2 — a weighted set of **sampled continuations** (particles) — against
   candidate 3 — a **density you can query at any coordinate and time**.
3. **Is time necessary?** The notes' central claim: without the order of the data the rule cannot be recovered.

## Data: ordered 2-D marks from known rules

A sequence is a list of points in the unit square, in the order they were made — the pencil marks of the notes, with
the order kept. Seven families, chosen to differ in structure (BRAINSTORM §0.1 item 3 of the NTA line: diverse
environments). Observation noise is Gaussian, σ = 0.005, except where stated. Every generator is a short program with
known parameters, so the true continuation distribution is known.

| family | rule | parameters (training range) | what it tests |
|---|---|---|---|
| F1 circle | constant angular speed around a centre | centre ∈ [0.3, 0.7]², radius ∈ [0.1, 0.3], ω ∈ ±[0.15, 0.35] rad/step | the "donut" |
| F2 log spiral | r = r₀·e^{b·θ}, θ advancing at ω | as F1, r₀ ∈ [0.05, 0.15], b ∈ ±[0.02, 0.15] | the horseshoe that is not one (notes Fig 3/4) |
| F3 Lissajous | x, y sinusoids with frequency ratio (1:2, 2:1, 1:3, 3:2) | amplitudes, phases, ω ∈ [0.1, 0.25] | a closed shape with a hidden ratio |
| F4 billiard | constant velocity, reflecting off the walls of [0.05, 0.95]² | speed ∈ [0.02, 0.05]/step | a rule (the bounce) that may not appear in the prefix |
| F5 pen | the notes' own drawing rule (`make_figures.draw_marks`): spiral, jump to the nearest unfilled section | grid, target, turn, step as in Fig 1 | the actual pencil experiment; stochastic |
| F6 rotation | x ← frac(x + α), y ← frac(y + β): a deterministic sequence that FILLS the square evenly | α, β ∈ [0.1, 0.9] | looks like uniform noise as a picture; trivial with order (notes Fig 2) |
| F7 uniform | independent uniform points (no noise added) | — | real noise: the right extrapolation is uniform |

Families are mixed uniformly in training. A training example is a prefix of n points (n uniform in 8–32) and its next
H = 24 points; evaluation uses n = 24. Trajectories are kept inside the square for n + 24 steps by rejection. Held-out
evaluation sets: 512 prefixes per family with in-range parameters, and 256 per family with **out-of-range** parameters
(F1/F2 radius up to 0.4, F2 |b| ∈ [0.15, 0.25], F3 the ratio 2:3 (never trained), F4 speed ∈ [0.05, 0.08], F6 α, β ∈
[0.02, 0.1]).

## Model: one transformer, three heads

Backbone, shared by every arm: `experiments/transformers/h1_lid.py` blocks (`Model.forward_embedded`), d = 128, 4
layers, 4 heads, RoPE over the position in the sequence (= time). A point enters as `Linear(2 → 128)`. The arms differ
ONLY in the head and the loss.

| arm | output | training loss | its "next token" |
|---|---|---|---|
| **A0 next-step** (the standard objective, the reference) | at every position, a mixture of 8 2-D Gaussians for the next point | negative log-likelihood of the next point, all positions (teacher forcing) | itself; extrapolation = 64 autoregressive rollouts |
| **A1 particles** (candidate 2) | from the prefix's last state plus a random code z ~ N(0, I₁₆), an MLP (2 × 256) emits a whole continuation of 24 points; 32 particles per prefix in training, 64 at evaluation | the **energy score** of the 32 particles against the true continuation — a strictly proper scoring rule for samples, minimised only by the true distribution | the particles' first points |
| **A2 density field** (candidate 3) | an MLP (3 × 256) f(prefix state, k, x, y) → log-density of the point k steps ahead being at (x, y); (x, y) enter as Fourier features, k as a sinusoidal code — queryable at ANY coordinate and time | log-likelihood of the true point at each k, the density normalised over a 32 × 32 grid of query points; 4 random k per prefix per step | f at k = 1 |
| **A2-shuffled** (the time test) | as A2 | as A2, but the prefix points are SHUFFLED before they enter (a set: the order, i.e. time, is destroyed) | f at k = 1 |

Budget, equal for every arm: 20,000 steps × 64 sequences, AdamW (lr 3e-4, 5% warmup + cosine), bf16 on the GPU;
checkpoints every 2,000 steps for the learning curves. Expected ~10–15 minutes per arm on the RTX 3050 Ti — about an
hour for the four.

## Measures

All on held-out prefixes (n = 24), per family, at horizons k = 1, 2–4, 5–12 and 13–24.

- **M1 — information gained over uniform (bits per point), the common proper score.** Each arm's predicted distribution
  of the point k steps ahead is put on a 64 × 64 grid of the square: A2 by querying it; A0 and A1 by a Gaussian kernel
  density of their 64 samples (bandwidth chosen on a validation set, per arm and k); a floor of 10⁻⁴ per cell. Score:
  12 bits (= log₂ 4,096, uniform) minus the bits needed to name the true point's cell. Higher is better; 0 = no better
  than uniform. Reference ceiling: the **true-parameter oracle** (knows the family and its parameters; its spread is the
  observation noise alone) — the best any predictor could do.
- **M2 — whole-continuation consistency.** The energy score of complete 24-point continuations. A0 and A1 produce
  continuations directly; A2 only marginals per k, so its continuations are assembled from independent per-k samples —
  the measure shows what predicting continuations JOINTLY is worth, and A2 is expected to lose on it by construction.
- **M3 — ambiguity.** A confusable set: prefixes from F1 circles and from F2 spirals with |b| ∈ [0.02, 0.06]. Reference:
  the posterior weight of "spiral" against "circle" from fitting both families to the prefix (least squares, then
  BIC weights). For each arm, the share of its k = 24 mass lying within 0.03 of the fitted spiral's continuation versus
  the fitted circle's. Error = |predicted share − posterior weight|.
- **M4 — data efficiency.** From the checkpoints: training sequences needed to reach half of A0's FINAL far-horizon
  (k = 13–24) gain, per family; and the plateau length — sequences spent within 0.2 bits of uniform before the gain rises.
- **M5 — out-of-range parameters.** M1 on the out-of-range sets, as a fraction of the in-range gain.

## Predictions (pre-registered)

- **P1 — the objective helps far ahead.** At k = 13–24, averaged over the structured families F1–F4 and F6, A1 and A2
  each gain at least 1 bit per point more than A0. At k = 1 all three are within 0.5 bit of each other.
- **P2 — the representations split the work.** A2 ≥ A1 on M1 at every horizon (it is trained on the log score M1
  measures); A1 < A2 on the M3 error (particles can commit to "spiral" or "circle" as whole continuations; per-k marginals
  only blur them) and A1 > A2 on M2.
- **P3 — time is necessary.** On F6 (rotation) A2 gains at least 3 bits per point at k = 1, while A2-shuffled stays
  within 0.5 bit of uniform. On F1–F3, A2-shuffled loses at least 1 bit per point at k = 13–24 relative to A2 (the shape
  survives shuffling; the direction and phase do not).
- **P4 — data efficiency (the notes' prediction):** A1 or A2 reaches half of A0's final far-horizon gain with at most
  half the training sequences A0 needs, on at least three of F1–F4.
- **P5 — no rule induction yet:** every arm keeps less than half its in-range gain on out-of-range parameters. (A
  transformer with these objectives is not expected to have found the RULES; this sets the bar later variants must
  clear.)
- **P6 — no invented structure:** on F7 every arm is within 0.2 bit of uniform at every horizon. No prediction for F5.

**Refuted / what changes.** If P1 fails — A1 and A2 do not beat next-step rollouts far ahead — the objective does not help
even here, and the next step is to ask why before scaling. If P2's ranking reverses, the choice of representation for
later experiments follows the measurements, not this table. If P3 fails (shuffled order does as well on F6), the notes'
time claim is not supported at this scale.

## Out of scope for Experiment 1

Parameter count and decompilable circuits (the notes' first two predictions) — they need a model-size sweep and an
interpretability pass, and come after the objective shows a gain at all. Rules as the output (candidate 1). CTM-style
internal ticks, DiffusionBlocks training, discovered coordinates — the modifications to the transformer come after the
baseline transformer has been measured with each head.

## Files (to be written when this is approved to run)

`generators.py` (the seven families and the true-parameter oracle), `heads.py` (the A0/A1/A2 heads and their losses on
the shared `h1_lid` backbone), `train_e1.py` (one arm per invocation, checkpoints), `evaluate_e1.py` (M1–M5, tables),
results in `runs/e1/`.
