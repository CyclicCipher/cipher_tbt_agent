# Pure Math for Machine Learning

Insights connecting sample complexity, group theory, and the training objective — distilled from the
arithmetic/learnability thread. The throughline: **the number of examples a perfect learner needs to
pin down a function is a readout of how much of the function's *algebraic structure* the
representation already encodes — and the same structure is what makes the learned rule transport.**

## 1 · Sample complexity = degrees of freedom (given a prior)

For a perfect learner (noiseless, realizable, picks the simplest consistent hypothesis), the minimum
number of examples to *uniquely determine* a function equals the **free parameters of the hypothesis
class after the structural prior**, with examples in **general position** (each constrains a new
parameter; full-rank Jacobian `∂f_θ(xᵢ)/∂θ`).

Two non-negotiable caveats:
- **The count is finite only relative to a prior.** For a generic smooth function it is ∞ —
  infinitely many curves pass through any finite point set. The natural prior for an *operator* is its
  **algebraic identity** (its functional equation / group structure), which collapses the count.
- **Structural fixed points come for free and must be avoided as examples.** Each operator's defining
  equation forces values (`a+0=a`, `x×1=x`, `ln 1=0`, `b⁰=1`); those reduce the parameter count, and a
  training example placed on a forced point is wasted (it's "out of general position").

## 2 · The operator table (under the algebraic-identity prior)

| operator | defining structure (functional equation) | forced points | residual DoF | min examples |
|---|---|---|---|---|
| `ln x` | homomorphism (ℝ⁺,×)→(ℝ,+): `f(xy)=f(x)+f(y)`, continuous | `f(1)=0` | scale `c` | **1** (any `x≠1`) |
| `exp / bˣ` | homomorphism (ℝ,+)→(ℝ⁺,×): `f(x+y)=f(x)f(y)`, continuous | `f(0)=1` | base `b=f(1)` | **1** (any `x≠0`) |
| `a+b` | translation; shift affine in the operand | identity `a+0=a` ⟹ intercept 0 | shift slope | **2** ("affine shift"); **1** if the identity is known |
| `a×b` | bilinear / distributive `f(a,b+c)=f(a,b)+f(a,c)` | `a×0=0`, `a×1=a` | one scale | **1–2**; ≈0 (derivable) if both identities assumed |
| geometric/exponential **without** the multiplicative-group prior | "some smooth curve" — must *detect curvature* | — | line + a bend | **3** (2 points always fit a line; the 3rd reveals the bend) |

The last row is the key contrast: **the same exponential is 1 example with its group prior and 3
without it.** The example count *is* a measure of how much structure the representation has captured.

## 3 · Group theory in ML — why the generalizing circuit is the group's representation

`exp`/`ln` are the isomorphism between the additive group (ℝ,+) and the multiplicative group (ℝ⁺,×);
`+` is the group ℤ (translation); modular `+` is the cyclic group ℤ/nℤ. The empirical finding is that
**neural nets, when they generalize on these tasks, discover exactly this structure:**

- **Chughtai, Chan & Nanda 2023 ("A Toy Model of Universality")** — nets trained to compose a finite
  **group** learn it via the group's **irreducible representations**. Universal mechanism.
- **Nanda et al. 2023 ("Progress Measures for Grokking")** — grokked *modular addition* is computed in
  the **Fourier basis** (sines/cosines = the characters of ℤ/nℤ): numbers become rotations, addition
  becomes composing rotations.

This is the same fact as §1–2 in a neural substrate: **the right representation for a group operation
is the group's character/phase basis**, and in it the operation is "compose the phases" — a few-DoF,
few-shot, transportable rule. The number-line that makes `+` 2-shot symbolically *is* ℤ's translation
structure; the Fourier circuit that groks `+` neurally *is* the same structure as phases.

**PoPE is, by construction, a group-representation positional scheme** (`core/block.py`): content lives
in magnitude, position in **phase** (`pos · θ_c`), and the attention score factorizes into
(what-match)×(where-match). Phase = the cyclic group's character — so PoPE already encodes the "where"
the way the grokked circuit encodes numbers. **But (see §9) PoPE encodes the position a token arrived at
in the *input stream* — an externally supplied coordinate — not where a concept sits on a reference
frame the model has *learned*.** An earlier draft proposed driving `coord` with a token's **value** to
"put the number line in" directly; that injects the metric from outside, through the attention channel,
and short-circuits the very question the experiment asks. The number line must instead be the *learned
geometry of the value representations* — §9.

**Irreducible vs regular: grid and place are one object in two bases.** The "generalising circuit = the
group's *irreducible* representations" thesis has a dual that matters for *memory*. A group's **regular
representation** (its action on itself — the localized / one-hot basis) **decomposes into the direct sum
of all irreducibles**. So the **metric, distributed, low-D code** (a few irreps = the *grid* / Fourier
basis, where the group acts as a diagonal rotation-per-frequency — good for *path integration*) and the
**orthogonal, sparse, high-D code** (the regular rep = the *place* basis, where the group acts as a
*permutation* — good for *binding / memory*) are the **same information related by the group Fourier
transform** (DFT for cyclic groups; Peter-Weyl in general). The transform **preserves the group action**,
so converting grid→place orthogonalises the code *without breaking movement*. This is the entorhinal↔
hippocampal split, and it is the analytic backbone of gap C — see `GRID_PLACE_REFERENCE.md`.

## 4 · The objective — where MDL works and where it doesn't

A correction worth pinning down:

- **Symbolic regime — MDL is a usable objective.** Description length is *computable* over discrete
  structures, so "shortest hypothesis consistent with the data" is directly optimizable. This is what
  `experiments/ProgramSynthesis/volume/` does (`fit_box_concept`, `relation.fit` minimize a two-part
  code). MDL there picks the 2-parameter line over the 10⁴-entry table.
- **Neural regime — MDL is *not* a usable training objective.** You cannot measure a *circuit's*
  description length and gradient-descend toward a simpler circuit; circuit/algorithmic complexity
  isn't differentiable. The crude proxies act on *weights*, not function complexity:
  variational/bits-back MDL (Hinton & van Camp 1993) and weight decay. Compression *describes* the
  grokked outcome (deMoss et al. 2024, "The Complexity Dynamics of Grokking": complexity rises then
  falls) but is **not** the optimized quantity — it emerges from weight-decay dynamics.
- **The neural lever is the group structure, not a compression loss:** (a) **equivariance** — bake the
  task's symmetry into the architecture so the group circuit is the default (geometric deep learning;
  CNNs for translation); or (b) **differentiable algebraic constraints** — add an auxiliary loss that
  enforces the functional equation (commutativity, associativity, identity, the homomorphism), which
  *is* a differentiable surrogate for "be the group operation"; or (c) grokking dynamics (weight
  decay + time), which reach it slowly and by luck.

## 5 · The unifying thesis

**Capturing a task's symmetry/group structure simultaneously (i) collapses sample complexity to the
residual DoF and (ii) gives distribution-shift robustness — they are the same property.** Few-shot
learnability ⟺ transport: the 2-parameter line predicts `97+4` you never saw; the lookup table
doesn't. Today's training needs millions of examples *and* breaks under shift for the same reason —
prediction-loss + a metric-free representation lets it settle for the table instead of the line. The
fix is two-sided: a **representation** that encodes the structure (the number line / the group's phase
basis — but §9: this has to be *learned representation geometry*, not a positional coordinate injected
from outside) and an **objective** that prefers the structured circuit (symbolically: MDL; neurally:
equivariance or algebraic-constraint losses).

## 6 · Connections in this repo

- `experiments/ProgramSynthesis/volume/` — MDL *is* the objective (symbolic; computable code length).
- `experiments/ProgramSynthesis/docs/phase2/VOLUME_CONCEPTS.md` — concepts as regions; "MDL decides the
  geometry"; the dimension-tower (a tower of concepts-as-dimensions).
- The lottery-ticket / feature-learning view (Frankle & Carbin 2018; Ramanujan et al. 2020): gradient
  descent reorganizes a random substrate toward reachable circuits — good for crystallized skill, but
  the *general* circuit is the rare needle, hence the fluid-intelligence gap. Symbolic lacks the
  substrate (must search) but is fully legible.
- PoPE (`core/block.py`; Gopalakrishnan et al. 2025, arXiv:2509.10534) — the group-representation
  positional scheme; the substrate for the few-shot arithmetic experiment.

## 7 · The Thousand-Brains objective (proposed) — a *general* group-structure objective

The repo has always put Thousand-Brains Theory into the *architecture* (reference frames, displacement
layers, `experiments/ManuallyCodedTBT`). The unrealized move is to make TBT the **objective**. What the
theory says the cortex optimizes:

> learn objects as **(feature, location-in-a-reference-frame)**, recognized by **predicting the sensory
> consequence of a movement**, where **displacements compose** consistently (grid-cell path integration).

As a loss, two terms:
1. **Sensorimotor prediction** — given a location and a movement, predict the feature at the new location.
2. **Reference-frame coherence** — movements must compose path-independently:
   `f(f(s, m₁), m₂) = f(s, m₁⊕m₂)`, identity `f(s,0)=s`.

Term 2 *is the group law*. A reference frame + a consistent composable displacement action **is a group
acting on a space** — the §1–3 structure that collapses sample complexity — but here it is **discovered**
rather than the task's algebra hand-typed. That generality is the point and the bitter-lesson fix:
the *same* loss applies to arithmetic (number line, `+b`), vision (object pose), navigation; the
specific algebra (commutativity, associativity) **emerges** from coherence instead of being typed in.
It is "learned equivariance": don't assume the symmetry, make the network find a reference frame in
which a learned movement-action composes.

**Why it could give few-shot.** Term 2 holds for *all* `(s, m₁, m₂)` with **no labels** — checkable as
cycle-consistency against the model's own predictions. So a couple of labeled examples only have to
*anchor* what "move by m" does at one place; coherence **propagates that anchor across the whole
domain**. That is the DoF-collapse of §1 made operational — *anchor + propagate* — and it's the first
objective in this line with an actual mechanism for turning a few labels into the general rule.

**Caveats** (consistent with §4): it is a *soft* surrogate for the group, not a hard quotient, so the
effective DoF won't collapse perfectly; and coherence alone has a trivial **collapse** (everything → one
location makes composition vacuously true), so it needs an anti-collapse partner — **SIGReg**
(`baselines/sigreg.py`). Status: proposed, untested — see `FEW_SHOT_ARITHMETIC.md`.

This unifies four threads: TBT, the group-theory/few-shot math (§2–3), the "objective is the lever"
direction, and the repo's own world-model + PoPE + SIGReg pieces.

## 8 · Biological basis & parallels

The group/phase structure §3 and §7 rely on is not merely convenient for ML — it is, on strong
evidence, how the mammalian brain encodes structured spaces. The grokked-Fourier circuit and the
cortical grid code appear to be **convergent instances of the same principle**.

**Grid cells are a group representation of self-motion (established).**
- Entorhinal **grid cells** (Hafting & Moser 2005; Nobel 2014) encode position as **multi-scale
  phases** — within a module, cells share spacing/orientation and differ in phase; scales step up
  geometrically across modules. That is a **Fourier/harmonic basis for space**.
- The population code is **literally toroidal** — Gardner et al. 2022 (Nature) measured a grid module's
  activity lying on a torus. The phases are real geometry, not metaphor.
- Movement updates the code by **path integration** = a **phase shift** = the action of the translation
  group. Formalized explicitly: Gao, Xie, Zhu & Wu (2021) model grid cells as a *vector* representation
  of self-position acted on by a *matrix* representation of self-motion (conformal-isometry / Lie-group
  structure). "Movement = rotate the code" is the brain's mechanism, and it is the translation group's
  representation — the **same object** as Nanda's grokked modular-addition circuit (numbers→phases,
  addition→rotation = ℤ/nℤ characters). The transformer rediscovered, for ℤ/nℤ, the code the
  entorhinal cortex uses for ℝ².

**The objective grounding (the key parallel for §7).** Train a network on the **path-integration
objective** — "predict your position after this sequence of movements" — and grid cells *emerge*:
Banino et al. 2018 (Nature, DeepMind), Cueva & Wei 2018 (ICLR), with the *why* in Sorscher, Mel, Ocko,
Giocomo & Ganguli 2019 (the path-integration objective + mild constraints *forces* the hexagonal/Fourier
code). So the **movement-prediction objective → the group/grid representation, emergently** — exactly
the Thousand-Brains objective of §7, already validated in brains *and* RNNs. There is also a formal
transformer↔brain link: Whittington, Warren & Behrens 2022 (ICLR) show a transformer with the right
recurrent positional structure is equivalent to the Tolman-Eichenbaum Machine (Whittington et al. 2020,
Cell) and develops grid/place cells — a PoPE transformer trained on a coherence objective sits in this
family.

**The cortical-column generalization (Hawkins' TBT — evidenced, not proven).** TBT claims the cortex
reuses grid-cell machinery for *every* reference frame — objects, concepts, math — not just space
(Lewis, Purdy, Ahmad & Hawkins 2019, "Locations in the Neocortex"; Hawkins et al. 2019, Frontiers).
Supporting evidence: Constantinescu, O'Reilly & Behrens 2016 (Science) found a **grid-like fMRI code in
humans navigating an abstract conceptual space** — hexagonal symmetry in a non-spatial task. So the
brain demonstrably reuses grid-like codes beyond physical space, which is the crux of the generalization.

**Why the brain settled on *this* format (and the "6 values").** The toroidal/hexagonal grid is the
simultaneous optimum of three constraints — and each maps onto a design choice for our model:
1. **Metric preservation ⟹ rotation (conformal isometry).** Path integration needs the code to preserve
   distances/angles of space (equal physical step → equal representational step, isotropically). The
   representation satisfying this in 2D *is* the hexagonal lattice — "hexagons all the way down"
   (Schøyen et al. 2025). A module is the superposition of **three plane waves at 60°**; with optimal
   phases, **3 cells represent the unit cell and 6 optimally-phased cells form the torus** (vs ~20 with
   random phases) — the "6 values that determine position" within a module. → our **rotation `R=exp(G)`**.
2. **Periodicity ⟹ torus.** Indefinite integration with finite neurons must wrap; position-mod-lattice
   is intrinsically a torus, realised by a toroidal continuous attractor. → our **closing ring/orbit**.
3. **Unique long range ⟹ multiple scales.** A single module is only *locally* unique (periodicity →
   global ambiguity — the single-frequency failure we hit in §9); the brain stacks **modules at
   geometric scales (~3/2)** for exponential range + error correction. → our **multi-frequency** lesson.
So "why this format" = the convergent solution to metric-preserving + periodic + long-range coding,
self-organising from generic pattern-formation dynamics. The 2D hexagonal version is because physical
space is 2D; a 1D quantity (the number line) is the 1-torus (circle); abstract conceptual spaces reuse
the same machinery (Constantinescu's 6-fold fMRI). **This three-pillar decomposition is the blueprint
for the redesigned model — see `BLUEPRINT.md`.**

**Grid → place → memory (the binding pipeline).** The entorhinal grid (metric, for path integration)
feeds the hippocampus, where **place cells form by summing multi-scale grid cells** (Solstad, Moser &
Einevoll 2006) — the *inverse Fourier* synthesis (§3) — and the **dentate gyrus sparsifies and
orthogonalises** that code (pattern separation, an expansion recode) so distinct locations become
orthogonal keys for CA3 memory. The Tolman-Eichenbaum Machine (Whittington 2020) models the conjunction
**grid ⊗ sensory → place** — exactly our feature⊗location binding. So the brain keeps *two coupled codes*
(grid for movement, place for memory) related by the group Fourier transform — the design rule gap C
adopts. Full synthesis + refs in `GRID_PLACE_REFERENCE.md`.

**Honest caveats.**
- **Convergence, not identity.** A transformer is not a cortical column; both find the group/phase code
  because it is the *efficient, transportable* representation of a group action — convergent solutions,
  which is more telling than a literal equivalence, not less.
- **Modular arithmetic specifically** is not known to be done this way in the brain (the grokking result
  is about transformers); the *general mechanism* (group/phase codes for structured spaces) is grounded.
- **TBT's full reach** (all cognition, including symbolic math, on grid machinery) is Hawkins'
  hypothesis — well-motivated, partially evidenced, unsettled.

**What it means here.** The Thousand-Brains objective (§7) is the same objective that — biologically and
in path-integrating RNNs — *grows* the group/grid representation; PoPE's phase is the same code; ℤ/nℤ is
the cleanest group to test it on. If the few-shot experiment works it echoes a real cortical principle;
if it needs the *architecture* and not just the *objective*, that gap is itself a finding about how much
of the brain's grid machinery is the objective vs. the wiring.

## 9 · Positional encoding is not a reference frame (correction + first finding)

A conflation worth pinning down, because it reshaped the experiment (owner's catch, 2026-06-16).

**Two different "wheres."** (1) A **positional encoding** answers *"where in the input did this token
arrive?"* — a sequence index, metadata about the data stream, supplied **externally**. RoPE,
sinusoidal, PoPE all turn a *given* coordinate into something attention can use. (2) A **learned
reference frame** (the TBT / grid-cell sense) is **internal and learned**: a representational space the
model *built*, in which a concept has a *location* and an operation is a *movement*. PoPE maps a given
coord → phase; it does **not** supply a learned frame. Feeding `coord = value` does #1 while claiming
#2 — it injects the number-line metric from outside, into the attention/"where" channel (not even the
content), and so cannot test whether the model can *build/use* a number line; it hands it over.

**For a plain transformer, the reference frame just *is* the geometry of the learned representations** —
how the model's internal vectors for `VAL(0..m-1)` are arranged. That is what the ring-probe in
`train_numberline.py` measures, and what a discovery phase must *grow*. So the honest setup gives PoPE
its real job (`coord` = sequence position only) and demands the number line emerge in the content.

**First finding (2026-06-16, `train_numberline.py`).** Train succ / pred / circular-distance comparison
under cross-entropy with `coord` = position only. Result: all three solved, and the metric even
**generalizes** to held-out comparison pairs (~0.63 vs ~0.11 chance) — yet **no clean ring forms**.
`ring_var`/`dist_corr` stay ~0.15–0.34 (a clean ring ≈ 0.8+): weak and *decaying* in the input
embeddings, only weakly present and stable in the hidden activations (~0.34). Lesson: **cross-entropy on
relational tasks rewards *separability*, not *geometry*.** A ring is *sufficient* to classify succ/compare
but not *necessary*, so the model settles for "separable enough." **Relational supervision alone
underdetermines the space** — a real (if negative) answer to "how is a basis discovered": not like this.

**Finding 2 — the equivariance term forces the ring (confirmed, 2026-06-16)** (§7's coherence made
concrete; §4(a) equivariance). Require the `+1` movement to act as a **single shared transform** —
`z(a+1) ≈ R z(a)` for *one* learned `R` — with the **orbit closing** (the wrap `z(0)=R z(m-1)` ⟹
`Rᵐ ≈ I`) and SIGReg preventing collapse. At the right weight (`w_equiv ≈ 1`) the internal value
geometry climbs from corr ≈ 0 (relational CE alone) to **dist_corr ≈ 0.68, stable** (`ring_var ≈ 0.48`),
held-out comparison 0.93 — a genuinely *discovered* ring, no injected coordinate. The orbit of one
rotation that closes after `m` steps **is** a ring (the Fourier/grid code of §8). It constrains only the
generator + closure, **not** arbitrary `+b`, so phase-1 addition stays a genuine few-shot test (`+b = Rᵇ`
must still be learned from anchors).

*Two regimes for `R`:*
- **Free matrix `R`.** Over-weighting (`w_equiv ≈ 3`) triggers the §7 collapse from the other side — the
  term is satisfied *trivially* by squashing `z` to low rank where `R` acts cheaply (`ring_var` → 0.02),
  overwhelming SIGReg and destroying ring **and** generalization. Best ≈ 0.68 at `w_equiv ≈ 1`.
- **Rotation `R = exp(G − Gᵀ)` (adopted; §10).** A rotation is norm-preserving, so it *cannot* shrink
  `z` — the only way to satisfy `z(a+1)=R z(a)` hard is to lay the values on an actual circle. This
  **removes the collapse mode**: at `w_equiv ≈ 3` the hidden value space becomes a **near-perfect ring**
  (`dist_corr 0.98`, `ring_var 0.99`). New trade-off, though: that purity *starves the downstream
  readout* — held-out comparison drops to 0.56 (vs 0.85 at `w_equiv ≈ 1`, ring ≈ 0.58). So discovery
  balances **geometry purity vs task transfer**; the right operating point depends on what phase-1 needs.

**Deferred open question (owner).** A model learns *many* separate reference frames; "the token's
position in my frame" is then ill-posed — *which* frame? A genuine learned-coord→phase PoPE (the model
computes a concept's location, and that drives the phase) must first solve frame-selection. Parked.

## 10 · From a discrete generator to a Lie group: continuity, and how the movement is decided

Two questions about `R` (owner, 2026-06-16) that decide whether this scales past a discrete lookup.

**Fractional / continuous values need `R` to be a real rotation.** `R` is the discrete generator of
ℤ/m (`z(a)=Rᵃ z(0)`, integer `a`). A fractional movement (`+0.5`, or a continuous quantity) needs `R^t`
for real `t`, well-defined **only if `R` is orthogonal with eigenvalues on the unit circle** (the m-th
roots of unity). Then `R = exp(G)` for a generator `G`, `R^t = exp(t·G)` interpolates the phase, and the
discrete ring is revealed as a *sampling of a continuous circle* — the jump from a **cyclic group ℤ/m to
a Lie group** (U(1); ℝ for the unbounded line). `G` is a "velocity"; finite movement is its matrix
exponential. An *unconstrained* `R` (eigenvalues off the unit circle, `Rᵐ≈I` only soft) does **not**
support this — `R^t` drifts or blows up. So continuity is a direct argument for parametrising
`R = exp(G)` with `G` skew-symmetric — a guaranteed rotation, closure = angles are multiples of 2π/m.
**[Adopted — `train_numberline.py`.]**

**How is the movement `b` decided? Passive symbol vs active self-motion.**
- *In the arithmetic task `b` is **given*** (the second operand). The model decides nothing; it learns
  the action `label b → Rᵇ`. Our experiment is therefore a **passive slice of TBT**.
- *In TBT proper `b` is the agent's **own movement*** — a motor command, known *because it issued it*
  (**efference copy**). That is why path integration works: you know how far you moved because you
  commanded it. For continuous self-motion `Rᵇ = exp(b·G)` *is* path integration.
- **Long-term goal (owner).** Turn this model into an **agent** like our agent experiment (the
  `WorldModelAgent` in `experiments/ProgramSynthesis`), where movements are **decided by a policy and
  known via the action taken** — the active setting in which
  `b`, continuity, and reference-frame coherence can actually be tested. The continuous-active form
  (a self-chosen real `b` feeding `exp(b·G)`) is the same object as the deferred *learned-coord→phase*
  PoPE (§9) and the agent's action-conditioning; they converge.

**Neuroscience: the movement is always *known* self-motion, and the code is a (Lie) group rep.**
- **Gao, Xie, Zhu & Wu 2021** — grid cells as a vector code `v(x)` acted on by a **matrix representation
  of self-motion** `v(x+Δx)=M(Δx)v(x)`, `M(Δx)=exp(Δx·B)`: `R` generalised to *continuous* `Δx`, their
  conformal-isometry constraint = "make `M` a rotation" (answers the fractional case directly).
- **Sorscher et al. 2019 / Banino et al. 2018** — the path-integration objective feeds the network the
  agent's **velocity** (decided, known self-motion); grid cells emerge.
- **Whittington et al. 2020 / 2022 (TEM)** — actions index transitions; the transformer↔hippocampus
  equivalence puts a PoPE transformer in this family.

Throughline: movement = the agent's *known* self-motion; representation = a (Lie) group rep; composition
= matrix product; continuous motion = matrix exponential. Fractions and "how `b` is decided" dissolve
once `R` is a Lie generator driven by efference copy — which is why **the agent setting is the real
test**, and the modular passive task is the clean warm-up.

## References
- Frankle & Carbin 2019, *The Lottery Ticket Hypothesis* — arXiv:1803.03635.
- Ramanujan et al. 2020, *What's Hidden in a Randomly Weighted Neural Network?* (CVPR).
- Chughtai, Chan & Nanda 2023, *A Toy Model of Universality: Reverse Engineering How Networks Learn
  Group Operations* (ICML).
- Nanda et al. 2023, *Progress Measures for Grokking via Mechanistic Interpretability* — arXiv:2301.05217.
- deMoss et al. 2024, *The Complexity Dynamics of Grokking*.
- Hinton & van Camp 1993, *Keeping Neural Networks Simple by Minimizing the Description Length of the
  Weights* (COLT) — variational/MDL.
- Gopalakrishnan et al. 2025, *PoPE: Polar Coordinate Positional Embeddings* — arXiv:2509.10534.
- (classical) Cauchy functional equations / characterization of continuous group homomorphisms.

Neuroscience / grid cells (§8):
- Hafting, Fyhn, Molden, Moser & Moser 2005, *Microstructure of a spatial map in the entorhinal
  cortex* (Nature) — grid cells.
- Gardner et al. 2022, *Toroidal topology of population activity in grid cells* (Nature).
- Schøyen et al. 2025, *Hexagons all the way down: grid cells as a conformal isometric map of space*
  (PLOS Comp Bio) — three-plane-wave model; conformal isometry ⟹ hexagonal; 6-cell minimal torus.
- Sreenivasan & Fiete 2016, *Connecting multiple spatial scales to decode grid-cell activity* (Sci Adv);
  Stemmler/Fiete, *Robust and efficient coding with grid cells* (PLOS Comp Bio) — multi-scale necessity.
- Solstad, Moser & Einevoll 2006, *From grid cells to place cells: a mathematical model* (Hippocampus) —
  place = weighted sum of multi-scale grid cells (the inverse Fourier synthesis). See `GRID_PLACE_REFERENCE.md`.
- Rodriguez & Caplan 2019, *A hexagonal Fourier model of grid cells*; *Does the entorhinal cortex use the
  Fourier transform?* (2013) — grid as a Fourier basis, place as its inverse transform.
- *Reassessing pattern separation in the dentate gyrus* — sparsification/orthogonalisation (grid→place).
- Whittington et al. 2020, *The Tolman-Eichenbaum Machine* (Cell) — grid ⊗ sensory → place (binding).
- *Capacity Analysis of Vector Symbolic Architectures* (arXiv:2301.10352) — binding/superposition capacity.
- Babadi & Sompolinsky 2014, *Sparseness and Expansion in Sensory Representations* (Neuron) — expansion
  recoding → orthogonalisation, with the linear-expansion noise-amplification caveat (needs a nonlinearity).
- Gao, Xie, Zhu & Wu 2021, *On the Representation of Grid Cells: Group Representation & Isotropic
  Scaling* (NeurIPS) — grid cells as a matrix/group representation of self-motion.
- Banino et al. 2018, *Vector-based navigation using grid-like representations in artificial agents*
  (Nature) — grid cells emerge from a path-integration objective.
- Cueva & Wei 2018, *Emergence of grid-like representations by training RNNs to perform spatial
  localization* (ICLR).
- Sorscher, Mel, Ocko, Giocomo & Ganguli 2019/2023, *A unified theory for the origin of grid cells
  through the lens of pattern formation* (NeurIPS).
- Whittington et al. 2020, *The Tolman-Eichenbaum Machine* (Cell); Whittington, Warren & Behrens 2022,
  *Relating Transformers to Models of the Hippocampal Formation* (ICLR).
- Constantinescu, O'Reilly & Behrens 2016, *Organizing conceptual knowledge in humans with a gridlike
  code* (Science) — grid-like code for an abstract conceptual space.
- Lewis, Purdy, Ahmad & Hawkins 2019, *Locations in the Neocortex* (Frontiers in Neural Circuits);
  Hawkins et al. 2019, *A Framework for Intelligence and Cortical Function Based on Grid Cells*
  (Frontiers) — the Thousand Brains generalization.
