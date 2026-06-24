# Experiment plan — few-shot arithmetic on a PoPE transformer

> **Goal.** Take the existing small PoPE transformer and modify it — primarily through its **training
> objective** — so it learns **addition from 2 examples, multiplication from 3, and never more than
> ~10**. The point is to test the central claim of `PURE_MATH_FOR_ML.md` in a neural substrate:
> *given the right representation (the number line / group-phase basis) and an objective that prefers
> the group circuit, sample complexity drops to the residual degrees of freedom.*
>
> **Why this is iterable fast:** ≤10 examples + a tiny model (dim≈32, 2 layers) + CPU = sub-second to
> seconds per run. Unlike the GPU jobs (Mistake #36), these few-shot runs are cheap enough to sweep
> objectives and example-counts directly and rapidly. That speed is the whole appeal.

## Substrate (already in the repo)

- **Model:** `baselines.FixedDepthTransformer` with `pos_mode="pope"` — the base PoPE transformer
  `train_transformer.py` already trains. PoPE itself is in `core/block.py`
  (`PolarPositionalEmbedding`): magnitude = content, **phase = a coordinate** (`pos·θ`), score
  factorizes (what-match)×(where-match). Crucially, the phase is driven by a **real `coord` passed at
  forward time** — not a fixed position table.
- **Objective today:** plain next-token cross-entropy (`ce_loss` in `train_transformer.py`). That is
  the thing we change.
- **Task scaffolding:** `tasks/` already has `ModularChain` (modular arithmetic). Cyclic modular
  arithmetic ℤ/nℤ is the cleanest test — it *is* the group whose representation is PoPE's phase (the
  Nanda grokking setting). `baselines/sigreg.py` (isotropy/anti-collapse) is available as one arm.

## The two levers, and which is the prior vs the question

1. **Representation (the number line) — CORRECTED (see `PURE_MATH_FOR_ML.md §9`).** An earlier version
   of this plan said: "drive PoPE's `coord` with the token's numeric value" to supply the number line.
   **That was wrong.** A positional encoding carries *where a token arrived in the input* — an
   externally injected coordinate — not where a concept sits on a reference frame the model has
   *learned*. Feeding `coord = value` hands the model the metric through the attention channel and
   short-circuits the question. The number line must be the **learned geometry of the value
   representations**, grown in a **discovery phase** (below), with `coord` left to do its real job
   (sequence position only; `value_coord=False` in both tasks). The two readings of "already knows the
   number line" are then: idea #1 = *hardcode* a fixed `phase(value)` embedding (rejected — smuggles in
   the structure, not general); idea #2 = *discover* it from succ/ordering/comparison experience
   (chosen — forces us to answer how a basis is discovered at all).
2. **Objective (the open question — the actual experiment).** With the number line given, can the
   objective make the model learn the *operation* from few examples instead of memorizing the pairs?
   The principal candidate is a **Thousand-Brains objective**: TBT recast as a *loss*, not an
   architecture (see `PURE_MATH_FOR_ML.md §7`).

## The Thousand-Brains objective (the primary lever) — *proposed, untested*

This repo has always built TBT into the *network* (reference frames, displacement layers,
`ManuallyCodedTBT`). The new move is to recast TBT as the *objective*. What the theory says the system
optimizes: **predict the sensory consequence of a movement, in a reference frame where displacements
compose coherently.** That second clause *is* a **group action on a learned reference frame** — the
exact structure §1 says collapses sample complexity — but **discovered**, domain-agnostically, rather
than the task's algebra hand-typed. For arithmetic the mapping is literal: the **number line is the
reference frame**, a number is a **location**, `+b` is a **movement**, `a+b` is *"start at a, move by
b, predict the landing."* (Status: a proposed objective — not yet run; "highly impressive **if true**.")

## Phase 0 — discover the number line first (`tasks/number_line.py`, `train_numberline.py`)

Because the number line must be *learned representation geometry* (§1, corrected), we teach it before
addition, from succ / pred / **circular-distance comparison** on ℤ/m (`coord` = position only). A
ring-probe measures whether an ordered ring actually emerges in the value reps (`ring_var`, `dist_corr`;
clean ≈ 0.8+) and held-out comparison accuracy checks the metric *generalizes* (a real line, not
memorized pairs).

- **Finding 1 (relational CE is not enough).** succ/pred/compare under CE solves all three and
  comparison generalizes (~0.63 vs ~0.11 chance), but **no clean ring forms** (ring_var/dist_corr
  ~0.15–0.34, decaying in embeddings, weakly stable in activations). CE rewards *separability, not
  geometry*; a ring is sufficient but not necessary, so it doesn't appear. (Negative result — recorded
  in `PURE_MATH_FOR_ML.md §9`.)
- **Finding 2 (the ring-forcing term works).** Require the `+1` movement to act as a **single shared
  transform** `z(a+1) ≈ R z(a)` (one learned `R`), the **orbit closing** (`Rᵐ ≈ I`, supplied by the
  wrap `z(0)=R z(m-1)`), plus SIGReg. At `w_equiv ≈ 1` the internal value geometry climbs to
  `dist_corr ≈ 0.68` (from ≈ 0), cmp held-out 0.93 — a discovered ring. **Over-weighting collapses it**
  (trivial low-rank solution; balance matters). Constrains only the generator + closure, not arbitrary
  `+b`, so phase-1 addition stays few-shot.
- **`R` is a true rotation** (`R = exp(G)`, `G` skew-symmetric) — see `PURE_MATH_FOR_ML.md §10`: this is
  what lets the discrete ℤ/m generator extend to **fractional / continuous** movements (`R^t = exp(t·G)`)
  and makes `R` a genuine Lie generator. The movement magnitude `b` is *given* here (passive slice of
  TBT); the **long-term goal is the active agent setting** (movements decided by a policy, known via
  efference copy = path integration), where these ideas get their real test.

The discovered model (its value reps) is the **init** for the phase-1 addition arms below.

## Arms (objectives), held on identical data/init/representation

- **A0 · CE control.** Plain next-token CE on the N labeled pairs. **Prediction:** memorizes — train
  → 100%, held-out → chance (overparameterized lookup; the §1 DoF is unconstrained).
- **A1 · TBT objective = sensorimotor prediction + reference-frame coherence.** Two terms:
  1. **Prediction** — CE on the ≤10 labeled pairs (the sensorimotor *anchor*: what "move by b" does at
     a place).
  2. **Coherence** — enforce that movements *compose*, checked on **unlabeled** triples via the model's
     own predictions (cycle-consistency): `f(f(a, b₁), b₂) ≈ f(a, b₁⊕b₂)`, plus identity `f(a,0)=a`.
     Holds for *all* `(a, b₁, b₂)` with no answer key, so it **propagates the few anchors to the whole
     domain** — the mechanism that makes few-shot possible (anchor + propagate).
  Commutativity/associativity are **not** typed in — they *emerge* from reference-frame coherence,
  which is why this is general (same loss for vision/navigation), not arithmetic-specific.
  **SIGReg is a required partner, not a separate arm.** Coherence alone has a trivial collapse (map
  everything to one location → composition is vacuously consistent); the isotropy/variance floor
  (`baselines/sigreg.py`) keeps the reference frame spread out. So A1 = prediction + coherence + SIGReg.
  **Prediction:** generalizes from few — approaching 2 (add) / 3 (mult).
- **A2 · Hardcoded-algebra ablation.** The task-specific control: bolt commutativity `f(a,b)=f(b,a)`,
  identity, and successor-consistency in *directly* instead of letting them emerge from coherence.
  **Prediction:** works on arithmetic but is the bitter-lesson-violating, non-general version —
  included to show A1 matches it *without* the hand-coded algebra (the whole point of the TBT framing).

## Task & metric

- Modular addition `c=(a+b) mod m` and multiplication `c=(a·b) mod m`, small `m` (start `m=17`, Nanda's
  setting; `m²` total pairs). Input `a OP b =`, target `c`.
- Train on `N` randomly chosen pairs (`N = 2,3,…,10`); evaluate on **held-out** pairs.
- **Headline metric: held-out accuracy vs N** — the sample-complexity curve, per arm.
- **Success:** the structured arm (A1) reaches high held-out accuracy at **N=2 for addition, N=3 for
  multiplication, ≤10 throughout**, where A0 (CE) sits at chance. Hitting the exact DoF count is the
  stretch goal; the primary result is the *gap* between A1 and A0.

## Predictions (falsifiable)

1. A0 (CE) memorizes: ~100% train, chance held-out, for all N ≤ 10. (If A0 *generalizes* from 2, the
   representation alone is doing it and the objective claim is moot — also informative.)
2. A1 (TBT objective) generalizes from few — addition by ~2, multiplication by ~3 — via anchor +
   propagate (the coherence loss carries the labels to the whole domain).
3. A1 ≈ A2 (the TBT objective matches the hand-coded-algebra control *without* hardcoding the algebra)
   — the headline if it holds: a *general* objective reproducing a task-specific one.
4. A1 **without** SIGReg collapses (held-out → chance, representation degenerate) — confirms coherence
   needs its anti-collapse partner.
5. If A1 ≈ A0 at N ≤ 10, **the TBT-objective / "objective is the lever" hypothesis is wrong for the
   neural case** — a clean negative that sends us back to the representation or to baked-in equivariance.

## Honest caveats

- **2-shot with a neural net is a stretch target,** not a guarantee. The reliable, measurable outcome
  is the *curve* (examples-to-generalize per objective); "exactly 2" requires the objective +
  representation to fully collapse the effective hypothesis class, and a transformer's spare capacity
  may demand a few more.
- **The representation pre-supplies the number line on purpose.** That's the legitimate prior (the
  symbolic case assumed it too); we are testing whether the *operation* is few-shot **given** the
  number line — not claiming the number line itself is free.
- **Per `PURE_MATH_FOR_ML.md §4`, this is not MDL.** The coherence loss is a *differentiable surrogate
  for the group structure* (a soft consistency constraint), not a circuit-compression objective (which
  isn't differentiable) and not a hard group quotient. If it works, it's because it encodes the
  symmetry, not because it "compresses" — and being *soft* is why the effective DoF won't collapse
  perfectly (so "exactly 2" is the stretch, the curve is the result).
- **Collapse is the main failure mode** (hence SIGReg as A1's partner); the interface below must make
  "compose movements" differentiable and stable.
- Modular (cyclic-group) first because it's the cleanest match to PoPE's phase; integer addition
  (carry, unbounded magnitude) is a harder follow-on.

## The `f(location, movement)` interface (what A1 needs)

The coherence term needs the model to expose a composable move operator. Minimal version on the PoPE
transformer, no architecture change:
- **location** = a numeral's value (carried as the PoPE `coord`); **movement** = the second operand.
- `f(a, b)` = the model's prediction for the sequence `a + b =` (a numeral → a numeral, same space).
- **Composition (cycle-consistency):** run `f(a, b₁)=ŝ`, then feed `ŝ + b₂ =` → `ŝ′`, and penalize
  `‖ŝ′ − f(a, b₁⊕b₂)‖` (on the logits/soft-argmax so it's differentiable). `b₁⊕b₂` is computed in the
  *movement* space (here integers mod m), which the task supplies — we are learning how movements *act
  on locations*, not what movement-composition is.
- **Identity:** penalize `f(a, 0) ≠ a`. These two suffice; commutativity/associativity follow.

## Build list (small)

1. A minimal few-shot arithmetic task (or a thin wrapper on `ModularChain`): `(a,b) → (a∘b) mod m`,
   with a train/held-out split by pair and an `N` knob.
2. **Value-as-coordinate wiring**: pass each token's numeric value as PoPE `coord` (or a fixed
   `phase(value)` embedding) — the number-line representation.
3. The objective arms: **A1 = prediction + coherence (cycle-consistency, §interface) + SIGReg**;
   A2 = the hand-coded-algebra ablation; A0 = plain CE. Coherence evaluated on sampled unlabeled triples.
4. A sweep harness: for arm × N ∈ {2..10}, train the tiny model, log held-out accuracy; plot the
   curve. Sub-second runs ⇒ iterate freely (also an ablation: A1 with SIGReg off, to confirm collapse).

## Connections
- `PURE_MATH_FOR_ML.md` — the theory this tests (DoF, the group structure, PoPE = group rep, the
  objective distinction).
- `experiments/ProgramSynthesis/volume/` — the symbolic side where MDL *is* directly the objective;
  this is the neural counterpart, where the objective must be a differentiable group surrogate.
- Ties to the two standing targets from the thread: **the objective** (the lever here) and
  **distribution shift** (few-shot learnability and transport are the same property — held-out
  accuracy *is* a shift test).
