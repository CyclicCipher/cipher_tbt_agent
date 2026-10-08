# Experiment 1 — predicting the extrapolated distribution in context: sampled continuations vs a queryable density

Status: **v2, revised 2026-10-08 before anything ran** (v1, same day, was a 4-arm, ~1-hour design; the user asked for
test runs of 2 minutes at most, and for the experiment to be built around IN-CONTEXT LEARNING). Pre-registered: this file
is committed before any full run; results are reported against it. Context: `IDEA.md` §6 (the objective and its three
candidate outputs — the user chose candidates 2 and 3), `TRANSCRIPT.md`.

## The question

The notes propose: predict the best extrapolated distribution of the pattern, then the next token from it. The user
expects a successful version to show **in-context learning** — inferring the distribution BEYOND the data seen so far
from as few points as possible, and acquiring that ability in as few training steps as possible. Experiment 1 asks:

1. **Does the objective help?** Trained to predict the distribution of the continuation, does a transformer
   extrapolate better, from fewer in-context points, than the same transformer trained on next-step prediction and
   rolled out?
2. **Which output?** Candidate 2, **sampled continuations** (particles), against candidate 3, **a density you can query
   at any coordinate and time**.
3. **Is time necessary?** The notes' claim: without the order of the data the rule cannot be recovered.

## How the design serves in-context learning

- **Every sequence is a new task:** a rule family and fresh parameters drawn from continuous ranges — unlimited task
  diversity, the condition under which transformers learn general in-context inference rather than memorising tasks.
- **Every prefix length is trained at once:** at EVERY position t of a causal transformer, the head predicts the
  distribution of the next 16 points given the t + 1 points so far. One forward pass trains in-context inference from
  1, 2, …, 32 points; there is no single "context size".
- **The headline measure is the in-context curve:** bits gained against the number of points seen, n = 1, 2, 4, 8, 16,
  32 — how few data the model needs — and how it develops over training steps — how few steps.

## Data: ordered 2-D marks from known rules

A sequence is 48 points in the unit square, in the order they were made. Seven families (diverse by construction);
Gaussian observation noise σ = 0.005 except where stated; generated vectorised on the GPU (the pen family from a cached
pool, since its rule is sequential).

| family | rule | training parameters | out-of-range test | what it tests |
|---|---|---|---|---|
| circle | constant angular speed ω around a centre | radius 0.1–0.3, \|ω\| 0.15–0.35 rad/step | radius 0.3–0.4 | the "donut" |
| spiral | radius ∝ e^{b·\|ω\|·t} | \|b\| 0.02–0.12 (in and out), \|ω\| 0.1–0.25 | \|b\| 0.12–0.2 | the horseshoe that is not one |
| lissajous | sinusoids with frequency ratio 1:2, 2:1, 1:3 or 3:2 | ω 0.08–0.2, amplitudes 0.1–0.3 | ratio 2:3 (never trained) | a hidden ratio |
| billiard | constant velocity, reflecting off the walls | speed 0.02–0.05 /step | speed 0.05–0.08 | a rule (the bounce) that may not appear in the context |
| pen | the notes' drawing rule (`make_figures.draw_marks`) | turn 0.3–0.8, step 0.03–0.06 | — | the pencil experiment itself |
| rotation | x ← frac(x + α), y ← frac(y + β) | α, β 0.1–0.9 | α, β 0.02–0.1 | fills the square evenly: noise as a picture, trivial with order |
| uniform | independent uniform points | — | — | real noise: the right answer is uniform |

## Model: one transformer, three heads

Shared backbone: `experiments/transformers/h1_lid.py` `Model.forward_embedded`, d = 64, 3 layers, 4 heads, RoPE over
the position (= time). A point enters as its coordinates plus sin/cos features at frequencies 1, 2, 4, 8 (identical for
every arm), through one linear layer. Arms differ ONLY in head and loss:

| arm | head, at every position t | loss | extrapolation used for scoring |
|---|---|---|---|
| **A0 next-step** (the reference objective) | an 8-component Gaussian mixture for point t + 1 | log-likelihood of point t + 1 | 32 autoregressive rollouts, 16 steps |
| **A1 particles** (candidate 2) | an MLP takes the state and a random code z ~ N(0, I₁₆) and emits the next 16 points; 16 particles per position | energy score of the particles against the true next 16 points (strictly proper for samples) | 32 particles |
| **A2 density** (candidate 3) | for each horizon k = 1..16, an 8-component Gaussian mixture — a closed-form density queryable at any (x, y) for any k | log-likelihood of the true point at every horizon k | the mixtures directly |
| **A2-shuffled** (the time test) | as A2 | as A2, but every training sequence is randomly PERMUTED (inputs and targets): order — time — is destroyed, so it can only learn where marks of a process land, not when | scored on the true, ordered futures |

*Why a Gaussian mixture for candidate 3:* it is the cheapest exactly-normalised density that can be queried anywhere; a
general implicit field (an MLP over coordinates) needs a numerical normalisation ~100× more expensive per step, which the
2-minute budget does not allow. If candidate 3 wins, the implicit field is its first follow-up.

**Budget per arm: at most 2 minutes wall-clock, training plus evaluation.** 126 sequences per step (18 per family),
AdamW (lr 1e-3, 100 warmup steps, cosine), bf16 on the GPU. The step count S is fixed from a smoke run BEFORE any full
run, as the largest multiple of 500 at which the slowest arm trains in ≤ 75 s, and is the same for every arm; it is
written here when fixed.

**Fixed from the smoke runs (200–300 steps each, numbers not read), before any full run:**
- **S = 1,500 steps for every arm.** The slowest arm, A1, trains at ~40 ms/step (75 s → 1,875 steps → 1,500 is the
  largest multiple of 500). A0 trains fastest (~23 ms/step) but its evaluation runs autoregressive rollouts; with a k/v
  cache (checked equal to a full pass to 10⁻⁶) its whole run is ~1.5 minutes.
- **A1 uses 8 particles per position in training,** not 16 (pairwise cost; 16 did not fit), and 32 at evaluation as
  planned.
- **The energy score is the unbiased form,** averaging the spread term over pairs i ≠ j. The first version averaged over
  all pairs including i = j, which shrinks that term and so rewards particle sets that are too narrow — a scoring rule
  that is not strictly proper. Fixed before any full run.

## Measures

Bits gained per point, **log₂ of the predicted density at the true point** (the uniform density on the square is 1, so
0 bits = no better than uniform). Every predicted density is mixed with 1% uniform before scoring, so one miss costs at
most 6.6 bits. A0 and A1 densities are Gaussian kernel densities of their samples, bandwidth chosen per arm and horizon
from {0.003, 0.01, 0.03, 0.1} on a separate validation batch. Ceiling: a predictor that knows the rule and its
parameters gains ≈ 11.2 bits per point (the observation noise alone) on the deterministic families, 0 on uniform.

- **M1 — the in-context curve:** gain at context size n ∈ {1, 2, 4, 8, 16, 32} and horizon k ∈ {1, 2–4, 5–8, 9–16}, per
  family, 64 held-out sequences per family.
- **M2 — data needed in context:** n½ = the smallest n at which an arm reaches half of its own n = 32 gain at
  k = 9–16, per family.
- **M3 — steps needed:** the far-horizon (k = 9–16) gain at n = 8 measured every 10% of training — when does in-context
  extrapolation appear?
- **M4 — out of range:** M1 at n = 16 on the out-of-range sets, as a fraction of the in-range gain.

## Predictions (pre-registered)

Structured families = circle, spiral, lissajous, billiard, rotation.

- **P1 — the objective helps far ahead:** at k = 9–16 and n = 16, A1 and A2 each gain at least 1 bit per point more than
  A0's rollouts, averaged over the structured families.
- **P2 — density beats samples on this score:** A2 ≥ A1 at every horizon, averaged over the structured families (A2 is
  trained on the log score M1 measures; 32 samples smooth a density only coarsely).
- **P3 — time is necessary:** on rotation at k = 1, n ≥ 8, A2 gains at least 3 bits per point while A2-shuffled stays
  within 0.5 bit of uniform; on circle at k = 1, n ≥ 8, A2 beats A2-shuffled by at least 2 bits (the shuffled model can
  still learn the ring, not where on it the next mark falls).
- **P4 — in-context efficiency (the user's expectation):** on at least three of the structured families, A2's and A1's
  n½ is no larger than A0's, and A2 reaches a far-horizon gain of 2 bits at n = 8 earlier in training than A0 (or A0
  never does).
- **P5 — no rule induction yet:** every arm keeps less than half its in-range gain on the out-of-range sets.
- **P6 — no invented structure:** on uniform, every arm is within 0.2 bit of 0 at every n and k. No prediction for pen.

**Refuted / what changes.** P1 fails → the objective does not beat next-step rollouts even here; find out why before
anything else. P2 reverses → later experiments use particles. P3 fails → the time claim is not supported at this scale.
P4 fails → the objective does not by itself make in-context inference more data-efficient.

## Out of scope for Experiment 1

Parameter count and decompilable circuits (the notes' first two predictions); rules as the output (candidate 1);
transformer modifications (CTM-style ticks, DiffusionBlocks, discovered coordinates) — they come after the plain
transformer has been measured with each head.

## Files

`generators.py` (the seven families), `heads.py` (input features, the shared backbone, the three heads and their
losses), `run_e1.py` (one arm per invocation: train, periodic M3 probes, final M1/M2/M4; writes `runs/e1/<arm>.json`).
