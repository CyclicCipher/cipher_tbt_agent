# Representability & Learnability — STUB

> **Status: stub.** Scaffolding for the theoretical thread, not an answer. Seeded by
> the empirical finding (item-52 SNR run, 2026-06-09) that an optimizer-level
> generalization fix could *not* close our OOD gap — because the gap is
> representational, not memorization. See `Docs/LESSONS.md`.

## The question

**Given an architecture A and hyperparameters θ, can a model *theoretically* learn / compress the general algorithm for a domain D?**

This is the question our experiments keep bottoming out on. It is *not* one question — it stacks two, and conflating them is where confusion lives.

### (i) Representability — is the target program even *in the function class*?
Does there exist a setting of A's weights that computes the domain's generative program (not just fits the training sample)? This is independent of training — it's about the *expressive capacity* of the architecture.

- For us concretely: can a fixed-depth PoPE transformer of width d, depth L *represent* the "compose n affine maps, for n beyond training length" algorithm at all? Length generalization is the sharp case: a fixed-depth feedforward net has a fixed compute budget per token; an algorithm whose required steps grow with input length may simply **not be representable** at fixed depth (this is the bitter-lesson / recurrent-depth argument from the other side).
- Relevant theory to pull on later: circuit-complexity characterizations of transformers (what's in TC⁰ / log-depth / threshold circuits), "what algorithms can transformers express," uniform vs non-uniform expressibility, the role of chain-of-thought / scratchpad as a way to *buy* representable depth.

### (ii) Learnability — will the optimizer *find* it?
Given the program *is* representable, will SGD-from-init actually converge to it (rather than a memorizing solution)? This is the optimization/landscape question.

- For us: the grokking phenomenon, memorization→generalization transitions, implicit bias of SGD/Adam, and the item-52 signal/noise story all live *here*. item-52 is entirely a learnability intervention; it assumes representability is solved.
- Our OOD plateau is a failure of **(i)**, which is why a **(ii)** tool (the SNR gate) couldn't fix it. Diagnosing *which* layer a failure lives in is the first job of this theory.

## The goal — a computable decision procedure (north star)

Not just understanding: the aim is a **theoretically-grounded, analytical answer a
simple program can compute.** Given (a) an architecture A + hyperparameters and
(b) a *sufficiently-determined* domain / general function D, return:

- `Representable(A, D) -> bool / bound` — can A express D's general algorithm? (capacity; **this doc**)
- `Learnable(A, D, optimizer) -> bool / bound` — can the training algorithm *reach* it? (optimization; **separate problem**, §ii)

These are two distinct decision procedures and must not be conflated. *"Sufficiently
determined"* = D's generative program / function class is specified enough to
analyze (e.g. "compose n affine maps mod P"). Success = a calculator, not a vibe.

## The compression framing (the through-line)

Restated in MDL/Solomonoff terms: generalization = compressing the **source** (the generative program), memorization = compressing the **sample** (the training set). Train and test "look like different distributions" only at the *surface* level; at the *program* level they are the same distribution.

- **Representability** ⇔ the source program is in the architecture's hypothesis class (has *finite* description length under A's "code").
- **Learnability** ⇔ the source program is the *MDL-shortest* solution the optimizer's implicit bias is drawn toward, and the path there is navigable.
- The dream "learn anything fast" is bounded by No-Free-Lunch: achievable only on the subset of domains whose shared structure A's prior matches (the bet: that shared structure is compositional/relational).

## Starting clues & first probes (agenda set 2026-06-09)

Two complementary lenses toward `Representable(A, D)`:

**Circuit completeness** — bounds what's expressible (the necessary condition).
Fixed-depth log-precision transformers ⊆ uniform **TC⁰** (Merrill & Sabharwal); a
fixed-depth net can't express depth that grows with input length. *But* our task is
benign: composing affine maps is **associative** → computable by a **parallel prefix
scan in O(log n) depth** → in TC⁰ → a fixed-depth transformer *can* represent it in
principle. So representability likely isn't the blocker for the algorithm; the
question is whether the *learned* solution is the scan (length-generalizes) or a
depth-n unrolled shortcut (capped at trained length). Tools: **RASP-L** (Zhou et al.,
"What Algorithms Can Transformers Learn") — conjecture: length-generalizes iff a short
RASP-L program exists; **CoT/scratchpad buys depth past TC⁰** (Merrill & Sabharwal;
Feng et al.) — which is the recurrent-depth thesis, provable here.

**Categorical deep learning** — whether A's structure matches D's, making the right
solution natural (the inductive-bias side; Gavranović & Veličković, "An Algebraic
Theory of Architectures"). Representable ⇔ a **composition-preserving functor**
(monoid homomorphism) from the affine-map monoid into A's representable functions.
Measurable as **composition fidelity** `M(a∘b) = M(a)·M(b)`; if it holds, length
generalization is free. Head start: the project already has the CTKG categorical
apparatus (Yoneda / functors) to point at this; parametric lenses (item 21) are the
optimization-side companion.

**First experiments (all on the existing PoPE transformer + ModularChain, no new arch):**
1. **Composition-fidelity probe** — does the learned representation compose as a monoid homomorphism? (categorical representability test)
2. **Scratchpad length-generalization** — emit per-step intermediate values; does OOD lift? (circuit/CoT depth test)
3. **Write the affine prefix-scan in RASP-L** — does the short-program prediction hold?

Decision tree → the **X1 diagnosis**: scratchpad fixes OOD ⇒ representability ceiling,
CoT lifts it (recurrent depth justified); composition fidelity holds but OOD fails ⇒
a positional/scaffolding *learnability* issue; neither ⇒ wrong bias, need a different
relational/positional scheme.

## Worked data point #1 — the temporal-PoPE 3-way (a learnability microscope)

Our first *controlled* learnability experiment (2026-06-10, `train_temporal.py`, clean
`fixed_dist=1` EventStream where token-count carries 0 info about elapsed time, r=0).
Three models, **identical in every way except how time is represented**, same AdamW,
same data, same task ("read the elapsed gap, compute a decay"). This is the ideal shape
for theory: hold everything fixed, vary one knob (the representation / inductive bias),
watch how *learnability* moves. The three outcomes separate the two axes cleanly:

| arm | time is… | curve shape | reaches 90%? | final (nz in/ood) | verdict |
|---|---|---|---|---|---|
| **integer** | absent | flat at the floor forever | never | 0.35 / 0.34 | **not representable** (axis i) — solution not in the function class |
| **time_input** | a content feature (log t) | long plateau ~0.33 until ~step 1000, then a **grokking jump** | ~step 1200 | 0.85 / 0.90 (caps <1, oscillates) | **representable but HARD to learn** (axis ii) |
| **continuous** | the PoPE coordinate (relative phase) | monotone, fast, convex climb | ~step 200 | **1.00 / 1.00** (perfect, incl. 2× OOD extrapolation) | **representable AND easy to learn** |

(Floor 0.35 > chance 0.10 because the value k is given as content; the model can predict
the best answer given k while ignoring the unknown gap. integer is pinned there: it
literally cannot read the gap.)

**What it isolates.** `integer` vs the other two = the **representability** axis (i): is the
solution in the function class at all? `time_input` vs `continuous` = the **learnability**
axis (ii): *the same representable solution*, but the inductive bias of the representation
changes how reachable it is by gradient descent — by a factor of ~6× in time-to-90%, and
the difference between perfect and capped-with-grokking. **This is the clean proof that
learnability ≠ representability: two architectures that can both express the solution can be
worlds apart in how easily GD finds it.**

**Candidate measurable learnability signatures** (toward `Learnable(A, D, opt) -> bound`):
- **Time-to-threshold** (steps to X% acc) — continuous ~200, time_input ~1200, integer ∞.
- **Curve shape** — monotone-convex (continuous, easy) vs plateau-then-phase-transition
  (time_input, grokking) vs flat (integer, unlearnable). The *presence of a plateau* is a
  learnability red flag: no gradient signal until a circuit is mostly assembled.
- **Final ceiling** — does GD reach the *perfect* solution (continuous 1.00) or stall below
  it (time_input ~0.9)?
- **Grokking present?** — grokking is the signature of a *representable-but-long* solution:
  the generalizing circuit exists but the optimizer wanders/memorizes first.

**Unifying hypothesis (theory-shaped, falsifiable).** *Learnability tracks the description
length of the solution under the architecture's inductive bias.* continuous-PoPE makes
"relative elapsed time" a **single built-in primitive** (phase difference) → the solution is
SHORT in the architecture's code → gradient points at it immediately → fast, monotone,
perfect. time_input makes the model **compose a differencing circuit** from absolute
log-times → the solution is LONG → no loss signal until the circuit assembles → plateau →
grokking. integer has no primitive for it at all → infinite length → unlearnable. So the
same **Occam/MDL principle that governs generalization also governs learnability**: GD finds
*short-under-the-prior* solutions fast. Grokking = a long-but-finite description.

**To turn this into the algorithm** (what to measure next, on this same 3-way as ground
truth): (a) **eNTK alignment at init** — project the target onto the architecture's empirical
tangent kernel; predict continuous has high alignment (target in the fast/signal subspace),
time_input low (target in a slow direction) — item 52's signal/noise machinery applied to
*learnability* rather than generalization. (b) **Solution description length** in the
architecture's primitives (a RASP-L-style program length, §clues). (c) **Plateau detection**
from the early loss curve (does cumulative-online loss stall?). If any of these *predicts*
the observed time-to-threshold ordering across the 3 arms, it's a candidate computable
`Learnable()` estimator — validated on a datum we already have.

## Worked data point #2 — Δ-encoding vs absolute under distribution shift (a *pure* learnability datum)

Our second controlled experiment (2026-06-10, `train_delta.py`, `ShiftSeq` L=8 D=4, target =
total change Σdₜ, 25 classes, chance 0.04). A value `v0` takes 8 increments `dₜ∈{0,1,2,3}`;
the target is **shift-invariant** (independent of `v0`). Two arms, **identical architecture**
(continuous-input PoPE transformer, same AdamW), differing ONLY in the input representation;
train v0∈[0,100], **OOD shifts v0 to [1000,1100]** — an unseen absolute range. The shift lives
*entirely* in `v0`.

| arm | input is… | in-dist curve | OOD curve | final (in/ood) | verdict |
|---|---|---|---|---|---|
| **absolute** | running values `[v0…vL]` | slow, noisy climb to ~0.74 (never perfect; final loss 1.08) | **pinned at chance** (~0.08, range 0.04–0.14) the *entire* run | 0.738 / 0.105 | representable, but GD **memorizes the training value-range**; OOD ≈ chance |
| **delta** | increments `[0,d0…d₇]` | fast convex climb to 1.00 | **identical to in-dist at every step** | 1.000 / 1.000 | the representation makes the shift **invisible**; OOD ≡ in-dist |

**Why this is the sharpest axis-(ii) datum yet — representability is held EQUAL.** Unlike #1
(where `integer` was genuinely *non*-representable), here **both** encodings can express the
generalizing solution: `vL − v0` is linear, attention can subtract position-L from position-0,
and a linear differencing circuit would *extrapolate* to the shifted range for free. So the
absolute arm's OOD collapse is **not** a capacity failure — the extrapolating solution is in its
function class. GD simply didn't find it: it found a **range-specific lookup** keyed to the
trained absolutes (in-dist climbs, OOD pinned at chance from step 1). This isolates **learnability
alone**, with representability provably equalized — the cleanest separation of the two axes we have.

**The generalization-gap trajectory is the headline signature.** For delta, `|acc_in − acc_ood|`
≈ 0 at *every* step from initialization (e.g. 500: 0.945/0.945; 2600: 1.000/1.000) — there is
*never* a generalization gap, because in the delta representation **train and OOD are literally the
same distribution** (the deltas don't contain `v0`; the shift cannot reach them). For absolute the
gap **opens immediately and never closes**: in-dist rises to 0.74 while OOD stays flat at chance.
This suggests a new measurable: **gap≈0 throughout ⇒ the representation has dissolved the
nuisance variable; gap opens and persists ⇒ representation-specific memorization.** It is
diagnosable from the *first* evals, before in-dist accuracy even moves.

**This is Makushkin's thesis as a learnability statement.** A non-stationary signal (absolute `v`)
with a stationary derivative (delta) becomes stationary the moment you represent the derivative.
"Train and test look like different distributions" was true only in *absolute* space; in *delta*
space they are identical — the distribution shift was an artifact of the representation, not the
data. Restated in the compression framing (§through-line): delta makes the source program
("sum the increments" = a running sum, which attention does natively) **short under the
architecture's code**, so GD finds it fast and the solution is automatically shift-invariant;
absolute makes the generalizing program longer (attend-first/last, difference, *and* be
range-invariant) and the range-invariant clause is exactly the part GD drops in favor of
memorizing the training range.

**The dips (after delta reaches ~1.0).** Occasional drops — 700: 0.703/0.727, 1300: 0.672/0.672,
3000: 0.500/0.500, 3600: 0.773/0.766 — that recover within 100 steps. They hit in-dist and OOD
**identically** (different eval batches, same accuracy) and correlate with **training-loss spikes**
(3000: loss 0.21 vs neighbours ~0.01), so they are the **model weights** transiently knocked off a
sharp minimum, not eval noise or a generalization failure. Cause: flat-LR AdamW + weight decay +
momentum jitter at near-zero loss (the same flat-LR-at-the-optimum instability seen in the
Muon/Aurora investigation). A WSD cooldown would smooth them; left as-is since they don't affect
the conclusion.

**Consistency with the unifying hypothesis.** Same moral as #1 on an orthogonal axis
(generalization-under-shift rather than time-representation): *learnability/generalization tracks
the description length of the solution under the architecture's inductive bias.* #1 varied the
representation and watched time-to-threshold move ~6×; #2 holds representability fixed and watches
the *generalization gap* go from ~0 (short solution) to permanently-open (long solution GD skips).
Together they are two independent confirmations that **the representation, not the raw capacity,
decides whether GD reaches the generalizing solution.** (Data: `Theory/data/delta_shift_seed0.json`.)

## Worked data point #3 — DriftField field-vs-heads (channel determines generalization)

Third controlled experiment (2026-06-11, `train_field.py`, deterministic DriftField, WSD schedule,
censoring-masked `when` metric). One trunk (continuous-PoPE; reads an irregular Brownian-with-drift
path; distils it to a summary ≈ {v0, μ}); **four readouts that differ only in how queries access that
summary**. Two queries: `what(τ)` = value at a time, `when(θ)` = first-passage time. Final accuracy
(chance: what 0.031, when 0.05; in-dist / OOD-horizon):

| arm | readout / loss | what in/ood | when in/ood |
|---|---|---|---|
| **unified** | one field, both queries read it; what+when loss | 0.752 / **0.359** | 0.593 / 0.10 |
| **unified_fwd** | one field; **what-loss only** (when read cold = C1) | 0.758 / **0.045** | **0.411** / 0.16 |
| **separate** | two query-conditioned heads; what+when loss | 0.767 / 0.196 | 0.631 / 0.31 |
| **separate_fwd** | two heads; what-loss only | 0.766 / 0.160 | **0.056** / 0.03 |

**What it isolates — pure *channeling*, capacity and optimisation held fixed.** Same trunk, same
AdamW+WSD, same information in the summary. Read by information-availability:
- **`what_in` ≈ 0.75, all four equal** — answer *directly observed* (interpolate the window); channel
  irrelevant, everyone ties. The control that the trunk is equal across arms.
- **`what_ood` (extrapolate; info absent)** — set entirely by whether μ is *channeled* to the unseen
  future: cross-query consistency (`unified` 0.36) > explicit query-coordinate (`separate` 0.20) ≫
  nothing (`unified_fwd` **0.045 = chance, flat 4000 steps**). `unified` vs `unified_fwd` is the clean
  proof: identical net + info, the inverse-query loss is the only difference and is what carries μ
  forward.
- **`when_in` (invert; needs v0 and μ)** — set by channel to the inverse: dedicated head (`separate`
  0.63) ≳ shared field (`unified` 0.59) > **free from a what-only field** (`unified_fwd` 0.41, C1) ≫
  untrained head (`separate_fwd` 0.056 = chance).
- **`when_ood`** — compounds extrapolation+inversion; low everywhere; explicit dual channel
  (`separate` 0.31) wins; field-derived arms inherit the weak forward extrapolation.

**WSD is the null control for the OOD claim.** The schedule lifted in-dist stability (killed
`separate`'s late jitter) and sharpened the inverse reads during cooldown (every field-reading arm's
`when_in` jumped in the final 1000 steps) — but moved **no** `what_ood` number (`unified_fwd` still
exactly 0.045). Optimisation is not the OOD bottleneck; channeling is. (Same shape as item-52: an
optimiser-level fix cannot close a representational gap.)

**The lesson.** Generalisation here is gated by whether the information a query needs has a *channel*
to the prediction point — not by capacity, not by the optimiser. In-dist (info present) →
channel-invariant; the moment information must travel (forward in time, or backward through inversion)
the number is set entirely by the channel: cross-query consistency, a shared representation, an
explicit coordinate, or none → chance. H1 made mechanical, and it converges with
[[project_prediction_not_transformation]] and the settling-core "one object, many clamp modes" bet.
**Design consequences** (predict one latent object; impose every query as a consistency clamp; read
it through a coordinate-conditioned *functional evaluator*) are the basis for the next architecture
step — see below.

## Worked data point #4 — the OOD ceiling is informational, not architectural

Fourth result (2026-06-11, `train_field.py` + `train_lewm.py`, deterministic DriftField, per-time-bin
accuracy logged, observation-density sweep). The question after Stage 3 + LeWM: why does *no*
architecture beat OOD ≈ 0.27–0.44? Three architecturally unrelated models cluster there:

| model | what in/ood (n_obs=12) | what in/ood (n_obs=48) |
|---|---|---|
| unified field (+consistency) | 0.752 / 0.359 | 0.882 / 0.421 |
| functional coordinate readout (+consistency) | 0.762 / 0.441 | — |
| LeWorldModel autoregressive rollout (sigreg) | 0.588 / 0.336 | 0.830 / 0.268 |

**Two independent confirmations the ceiling is the task's information content** (chance = 0.031):
1. **Per-bin accuracy falls monotonically with horizon τ** (n_obs=48), for *both* the field and LeWM,
   and the decline begins *exactly* at the OOD boundary (τ>10):
   - field: 0.66 (τ=10.5) → 0.55 → 0.45 → 0.36 → 0.20 → 0.16 (τ=19.5); in-dist flat ~0.87.
   - LeWM:  0.85 (τ=10.5) → 0.35 → 0.16 → 0.09 → 0.08 (τ=19.5); in-dist flat ~0.83.
   Extrapolation error grows with horizon — the signature of `value_err(τ) ≈ δμ · τ`, a finite-window μ
   estimate amplified linearly.
2. **Denser observations raise the (clean) field ceiling:** n_obs 12→48 lifts OOD 0.359→0.421 and in-dist
   0.752→0.882 — more samples → better μ → higher wall. The information limit *moves with the
   information*, exactly H1; no architecture beats a fixed window's μ-uncertainty.

**LeWM autoregressive rollout actively *hurts* here (two pathologies the per-bin curve exposes):**
- **Compounding drift / exposure bias:** LeWM's *first* OOD step is in-dist quality (τ=10.5: **0.852**)
  then crashes (0.08 by τ=19.5) — each rolled step feeds its own error back. The direct field readout has
  no autoregression and degrades *gracefully* (0.66→0.16). Net OOD-average: LeWM 0.268 < field 0.421.
- **Action-distribution mismatch:** denser obs *lower* LeWM OOD (0.336→0.268) because at n_obs=48 the
  training `Δt≈0.21` but rollout steps to the OOD grid are `≈1.0` (5× larger, OOD actions). The field,
  having no rollout, only benefits from density.

**Conclusion.** The Stage-3→LeWM arc tried to break the OOD ceiling by swapping the *readout* (coordinate
field, data point #3) then the *substrate* (latent world model + SIGReg/Sub-JEPA). Neither did. The ceiling
is **informational** — μ inferred from a finite window, its error amplified over the extrapolation horizon —
measured two ways (monotone per-bin decay + the density lift). H1 confirmed on a generalization-under-
*extrapolation* axis (data points #1/#2 were channel/representation; #4 is the raw information limit).
Corollary: autoregressive rollout is *not* a free extrapolation win — it converts graceful degradation into
great-then-catastrophic drift and is sensitive to train/rollout action mismatch. The lever to push next is
therefore *information* (denser/richer observation, or a task whose window determines the rule), not another
architecture. (Data: `Theory/data/field_det_n48_seed0.json`, `Theory/data/lewm_det_sigreg_n48_seed0.json`.)

## Prediction P1 — invariance-matching: when Δ-encoding helps vs hurts (not yet run)

A *prediction* (2026-06-10), reasoned from the Stage-2 mechanism *before* Stage 3 runs, so the
outcome is a falsifiable test rather than a post-hoc story. Stage 2 showed Δ-encoding wins
**because the target was level-invariant** (`Σdₜ ⊥ v0`) — the shift lived entirely in the discarded
constant. Stage 2 and Stage 3 share the generator form `v(t) = v0 + μt + σW(t)`, and differencing
annihilates exactly one term:

- **Δ-encoding keeps `{μ, σ}`** (drift + diffusion = the entire *local* dynamics — first and second
  moment of the increment) and **destroys `{v0}`** (the global anchor / integration constant).

So a representation's value is set by the **query's invariance group, not the domain**:

| query | needs v0? | predicted winner |
|---|---|---|
| Σ-change / "rise by Δ from here" (relative) | no | **delta** (Stage-2 mechanism; OOD-shift for free) |
| rate / OOD-horizon *shape* | no | **delta** (μ is the derivative; absolute buries it) |
| predictive spread / surprise (σ) | no | **delta** (`var(Δv)=σ²Δt` lives in the increments) |
| **what(τ) = v0+μτ** (absolute level) | **yes** | **absolute** (delta destroyed the anchor) |
| **when(θ) = (θ−v0)/μ** (absolute threshold) | **yes** | **absolute** |

**The principle.** Δ-encoding trades absolute-level information for level-invariance; it is optimal
*iff the query is also level-invariant*. Stage 3's absolute queries are exactly the case Stage 2
happened not to contain — so Stage 3 is the experiment that *exposes* pure-Δ's **integration-constant
hole** (lose the anchor, lose the DC offset) where Stage 2 hid it.

**The resolution is a hybrid.** Derivative + one boundary condition reconstructs the absolute
trajectory (`v(τ) = v0 + ∫Δ`). The optimum is neither pure-delta nor pure-absolute but **Δ-stream
(robust rate/spread) + one retained anchor** — re-integrate from a reference. This is the
continual-learning posture (track change cheaply, re-anchor periodically), and it is biologically
grounded: not all neurons code change — some code **absolute level** (e.g. nociceptive/pain
intensity), some code **derivatives** (adaptation / transient cells). A real system uses both.

**Status: deferred.** Stage 3 runs *without* a Δ-encoding factor for now (keep the matrix small).
When run, cross `encoding ∈ {absolute, delta, delta+anchor} × query ∈ {absolute, relative}`; the
falsifiable prediction is **the winner flips with the query's invariance, and delta+anchor dominates
both** — a sharper representability statement than either stage alone (the optimum is
query-determined, not fixed).

## Hypothesis H1 — generalization as information sufficiency (the "enrichment" view)

> The user's hypothesis (2026-06-11), formed from the temporal 3-way (data point #1). The claim and
> evidence ladder are theirs; the tie-in to the MDL framing and the falsifiable predictions are mine.

**Claim.** Grokking is *not* intrinsically a tens-of-thousands-of-epochs phenomenon, and networks do
**not** inherently prefer memorization. **Memorization is the rational response to an underdetermined
problem** — the model compresses the *sample* because there is *too much information missing to form
the general solution*. Supply the missing information and the grokking delay collapses; the general
solution is found immediately.

**The evidence ladder (data point #1 *is* this experiment).**
- *No* time information → memorized (`integer`: flat, never generalizes).
- Time as *extra content* (a text/token feature) → long delay before the model relates it to the
  content, then a grokking jump (`time_input`: plateau → grok, caps <1).
- Time as *positional* information on each datum → **no grokking delay, test accuracy reaches exactly
  100%** (`continuous`-PoPE).

**The sharp, tested sub-claim — the *channel* matters, not just the amount.** The *same* information
as a **positional/structural side-channel** beats supplying it as **content tokens** (`continuous` ≫
`time_input`, both representable). Position gave the model "an eye for time" — a built-in sense —
rather than another symbol to be related by a learned circuit. The user ties this to the **grounding
problem**: a mind spawned in a dark void, thrown trillions of meaningless symbols, needs enormous data
to expose their regularities and still fails OOD because nothing anchors them to a referent; a body +
senses *enriches* every datapoint so context and meaning are cheap. Macro-scale instance: **V-JEPA 2**
controls arbitrary robots from only ~62 h of robot data *after* enough video to model the world — the
data was pre-enriched. Human sensory data is heavily enriched, which is why we generalize from far
less than a language model does.

**Connection to this document's framing.** H1 is the *input-side* statement of the same
description-length thesis the data points argue from the *architecture* side:
- "Missing information" = the source program is **not identifiable** from the data, so the
  MDL-shortest fit *is* the memorized sample. Enrichment makes the source identifiable, so the general
  program becomes the short (hence learned) solution (§the-compression-framing).
- The channel sub-claim refines learnability≠representability: at *fixed* information, changing the
  channel (content→positional) changes the **description length of the solution under the
  architecture's prior** — positional time is a built-in primitive (short); content time must be
  composed (long → grokking). So H1 and data point #1 are one fact seen from input vs architecture:
  *generalization speed tracks how short the general solution is, and both enrichment (input) and the
  right channel/prior (architecture) shorten it.*

**Falsifiable predictions H1 implies.**
1. **Information-deficit dial.** For a fixed task, monotonically adding grounding features per datum
   should monotonically shrink the grokking delay and raise OOD — optimizer and capacity unchanged.
   (The 3-way is 3 points on this curve; a denser sweep traces it.)
2. **Channel dominance.** The *same* feature as positional encoding beats it as an appended content
   token, at matched information, across tasks — directly testable on Stage 3 (coordinate-as-PoPE vs
   coordinate-as-input-token).
3. **No free lunch on truly-absent info.** If the information the rule needs is in *no* channel, no
   training closes OOD (`integer` — a representability wall, not a learnability delay). H1 predicts the
   wall is *informational*: it moves the moment the info is supplied in any usable channel.

**Caveat (mine).** H1 is strongest as a claim about *identifiability* (is the general solution
determined by the enriched data?). It does not by itself dissolve the pure-learnability gap: data
point #1 shows that even with the information *present* (`time_input`), the channel can still force a
grokking detour. Honest synthesis is two-factor: **enrichment makes the solution exist-and-be-short in
principle; the channel/prior decides whether GD walks straight to it.** H1 owns the first factor and
points at the second.

## Open sub-questions (to develop)

- **R1.** Can we *certify* (or refute) that a given architecture can represent a given algorithmic family — ideally constructively (exhibit the weights) or via a complexity-class argument?
- **R2.** What is the cheapest architectural change that moves a non-representable target into the function class (recurrent depth? scratchpad/CoT? a different positional code)? — connects directly to why we explored the settling model.
- **L1.** Given representability, what determines whether SGD finds the generalizing vs memorizing solution, and can it be measured *during* training (validation-free)? — item-52 is one attempt; we found its signal noisy.
- **L2.** Is there an architecture-aware notion of "epiplexity ceiling" (max extractable structure) vs "epiplexity rate" (speed) — and can we predict both from A, θ before training?
- **X1.** Diagnosis: given a failure (e.g. our OOD plateau), a *procedure* to attribute it to (i) vs (ii). The single most useful near-term deliverable.

## Connection to the experiments

Every empirical result is a probe of one of these. PoPE fixing OOD = a representability fix (the right positional code made length structure representable). The settling-model plateau = representability ceiling of weight-tied iteration. The SNR gate's failure = a learnability tool on a representability problem. The job of this document, as it grows, is to turn that pattern-matching into something predictive.
