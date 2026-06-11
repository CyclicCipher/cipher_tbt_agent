# Stage 3 — Unified (time × value) Field Predictor vs Separate Heads

> **Status: design, for redline. No code, no GPU run until this is approved.**
> Temporal fork, Stage 3. Belongs to the *fork* (Stages 0–2 = EventStream / temporal-PoPE /
> ShiftSeq), **not** the shelved DEQ "Stage 3" in `implementation_plan.md`.

## The question

Stages 0–2 predicted **"what, given when."** Stage 3 adds **"when, given what"** — but the real
question is whether the two need to be *separate*. The hypothesis: a model that **predicts the
dynamics as one object** answers both by *querying* that object in two directions. "What" and
"when" are not two tasks; they are two reads of one estimated function (Makushkin's
"function estimate"; the project's [[project_prediction_not_transformation]] — predict the
trajectory, don't transform input→answer; the "one operator, many modes" bet).

We test that directly, against the separate-head alternative, under both deterministic and
stochastic dynamics, with controls.

## The unified object

The model predicts a **field over (time bucket × value bucket)** — for every time, a distribution
over value. Two reads:

- **What at τ** = the column slice → `p(value | t=τ)` (vertical read).
- **When does value reach θ** = the threshold-crossing read across time → `p(t | first-passage θ)`
  (horizontal, survival-weighted read).

Deterministic = the field is a **sharp ridge** (the trajectory curve); stochastic = a **diffuse
band** (uncertainty envelope). Same architecture; the data decides peaked vs spread. The field's
**entropy along a slice = predictive surprise** (the HTM-bursting *principle*, extracted — not the
machinery; HTM's hand-tuned columns/SDRs are not ported).

**Honest asymmetry:** "what" is a raw column slice; "when" is first-passage (survival-weighted),
a slightly more processed read. The object is shared; the reads are not perfectly symmetric. We
expect "when" to be the harder/slower direction, and that is itself a measurement.

## Generator — Brownian-motion-with-drift (closed-form everything)

One generator, two regimes via the diffusion coefficient. Per sequence, sample latent dynamics,
emit an **irregularly-timed partial observation path**, then query forward/inverse.

- Latent: `v(t) = v0 + μ·t + σ·W(t)`, with `v0 ~ U[lo,hi]`, drift `μ ~ U[μlo,μhi]` per sequence.
- Observations: `K` samples at irregular times `t_i` (reuse EventStream's irregular grid +
  continuous-PoPE coordinate, the Stage-1 winner), `v_i = v(t_i)`.
- **Deterministic** = `σ = 0`: `what(τ) = v0 + μτ` exact; `when(θ) = (θ − v0)/μ` exact.
- **Stochastic** = `σ > 0`: `what(τ) ~ N(v0+μτ, σ²τ)`; `when(θ) ~ InverseGaussian` (first-passage
  of drifted Brownian motion — **closed-form density and entropy**).

Closed forms matter: they give us the **analytic floor `H(p_true)`** per query, so the stochastic
metric is `KL(true‖model) = CE − H(p_true)` (last turn's result), not raw CE.

**Decorrelation (the confound discipline, cf. `fixed_dist`).** `v0` and `μ` are randomized per
sequence, so neither query is answerable from the time axis alone *or* the value axis alone — the
model must infer the dynamics from the observed path, then answer. This is what makes it a genuine
function-estimation problem and stops the field from collapsing to a 1-D marginal.

**OOD axis = horizon extrapolation.** Train queries with `τ, θ` inside `[0,T]`; test with
`τ, θ` beyond `T`. Learned the *rate law* ⇒ extrapolates; memorized a lookup ⇒ caps at `T`.
(Same shape as the Stage-1 2× gap extrapolation.)

## Primary factor matrix (2 × 2)

Identical transformer trunk (continuous-PoPE, the Stage-1 winner) across all arms; **only the
readout differs**. Each arm evaluated in-dist **and** OOD-horizon, on **both** query directions.

| arm | output | dynamics |
|---|---|---|
| **U-det** | unified field, sliced both ways | deterministic (σ=0) |
| **U-stoch** | unified field, sliced both ways | stochastic (σ>0) |
| **S-det** | separate query-conditioned what-head + when-head | deterministic |
| **S-stoch** | separate query-conditioned what-head + when-head | stochastic |

- **Unified** = query-agnostic: the model emits the whole field once; *we* slice it. It never sees
  the query.
- **Separate** = query-conditioned: a `what`-head (input: queried τ → value dist) and a `when`-head
  (input: queried θ → time dist), trained jointly on their own supervision.

## Controls

- **C1 — Zero-shot inverse (the headline unification test; unified only, both dynamics).**
  Supervise the field with **forward observations only** (observed `(t,v)` points → NLL), and
  query **"when" cold**, with *no* when-supervision. If the field is a faithful function estimate,
  the inverse query works for free. Separate heads *cannot* do this by construction (the when-head
  was never trained) → structural baseline = chance. Also run a "both-directions supervised"
  unified variant to measure how much the free inverse costs vs. explicit supervision.
- **C2 — Parameter-matched.** Trunk identical; report readout param counts. The unified field's
  `(T_bins·V_bins)` readout is larger than `V_bins + T_bins` separate heads — so a unified *win*
  must be shown not to be just capacity. If readout params dominate, add a **factored-field**
  variant (C6).
- **C3 — Entropy calibration (stochastic).** (a) Run the *stochastic* (field-entropy) predictor on
  *deterministic* data → field entropy must collapse to ≈0 (confirms entropy is a real uncertainty
  readout, not noise — the bursting-surprise signal). (b) On stochastic data → **reliability /
  quantile coverage**: do predicted p-quantiles match realized frequencies?
- **C4 — Floors / baselines.** Predict-the-marginal (accuracy floor, deterministic); analytic
  `H(p_true)` (KL floor, stochastic). Every number reported against its floor.
- **C5 — Degenerate-task sanity.** A deliberately collapsible task (one axis ≈ constant) to confirm
  the metric *flags* vacuity and that the main decorrelated task isn't accidentally collapsible.
- **C6 — Factored (rank-1) field (optional).** Field forced to an outer product of a time-factor
  and a value-factor ⇒ time⊥value independence. Should **fail** to capture the trajectory
  correlation, isolating that the (time,value) *coupling* is what the full field buys.

## Metrics

- **Deterministic:** per-query **accuracy** (argmax) in-dist + OOD; plus trajectory-fit MSE
  (recovered μ, v0). Drop accuracy is *not* used here — det. has a single correct answer.
- **Stochastic:** **`KL = CE − H(p_true)`** per query, in-dist + OOD (NOT accuracy — argmax is
  meaningless on a distribution); plus **calibration** (C3b).
- **Both:** **field-entropy** per slice as the surprise readout (feeds the Stage 4 bridge);
  **time-to-threshold** and **curve shape** (the learnability signatures from data points #1/#2).

## Falsifiable predictions (this is data point #3 in waiting)

1. **Unification:** U arms learn both queries at least as well as S arms, and **C1 succeeds** —
   forward-only training yields a usable inverse. *Negative* (needs separate heads / C1 fails) =
   equally valuable; it bounds the "one operator, many modes" bet.
2. **Shared structure ⇒ joint learnability:** U makes both queries short-under-the-prior ⇒ they
   co-emerge; S risks two separately-memorized circuits with worse OOD on at least one direction.
3. **Asymmetry:** "when" (first-passage read) is slower/weaker than "what" in every arm.
4. **Entropy = surprise:** stochastic field entropy tracks true predictive uncertainty (C3
   calibrated), and collapses on deterministic data — giving Stage 4 its allocation signal for free.

## What we do NOT do

- No HTM machinery (columns, SDRs, hand-tuned dendrites) — **principle only** (surprise =
  predictive entropy; high entropy gates allocation). HTM as a system underperformed; we keep the
  idea, not the apparatus. [[feedback_bitter_lesson]]
- No handcoded "what/when" routing — the model predicts the field; querying is arithmetic on the
  field, not a learned domain rule.
- No GPU run until this doc is redlined. Then: tasks + smoke tests first (Mistake #36).
- **No Δ-encoding factor for now (deferred).** The absolute/delta/delta+anchor × absolute/relative
  cross is a strong follow-up — its predicted outcome (the winner flips with the query's invariance;
  delta+anchor dominates) is recorded as **Prediction P1** in
  `Theory/representability_and_learnability.md`. Stage 3 stays absolute-encoded to keep the matrix
  small. The biological grounding (some neurons code absolute level, e.g. pain; some code change)
  argues the eventual answer is a hybrid — a later experiment.

## Build order (once approved)

1. `tasks/driftfield.py` — `DriftField` generator (Brownian-with-drift; det/stoch; irregular path;
   forward/inverse query sampling; closed-form `H(p_true)`; decorrelated v0/μ; OOD-horizon split).
2. Readouts on the existing trunk: unified `(T_bins×V_bins)` field head; separate what/when heads.
3. `tests/test_driftfield_smoke.py` — target-correctness (analytic what/when match samples),
   shift/decorrelation invariants, closed-form-floor sanity, both readouts run, smoke `run_*`.
4. `train_field.py` — arm runner (U/S × det/stoch + C1 forward-only + C5 collapsible), KL-above-
   floor + accuracy + calibration + entropy logging, dense early evals, diagnostics JSON.
5. Record outcome as **data point #3** in `Theory/representability_and_learnability.md` + force-add
   the diagnostics JSON to `Theory/data/`.
