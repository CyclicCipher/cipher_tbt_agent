# ProgramSynthesis — Experiment Goals

> **What this is.** The research charter for the side-project: what we are trying to learn
> about *program-centric intelligence* and how the ARC-AGI-3 replica + the RecurrentWorldModel
> (RWM) results combine into a single testable program. Head-in-the-clouds in §1–§3,
> falsifiable in §4–§6.
>
> **Status legend:** `established` empirically confirmed (ours or literature) · `directional`
> promising, unproven · `theoretical` unconfirmed · `open` active question · `commitment`
> project decision.

---

## 0 · One-sentence aim

Find out whether **program-centric abstraction** (Chollet's exact, recombinable, discrete
structure) can be obtained *inside* a continuous deep model by **binding information to the
right coordinates**, and use the ARC-AGI-3 replica as the arena that forces the question.

---

## 1 · The binding thesis (what the temporal-PoPE experiment actually showed)

`established` (3-way is our data point #1, `RWM/Theory/representability_and_learnability.md`)

We ran one knob — *how is time represented* — three ways, everything else fixed:

| time is… | channel | result |
|---|---|---|
| **absent** (`integer`) | none | memorizes; flat forever; never generalizes |
| **a content token** (`time_input`, log t appended) | value/content | representable but **hard**: long plateau → grokking jump, caps below perfect |
| **a coordinate** (`continuous`-PoPE phase) | positional/structural | **easy & exact**: fast monotone climb to 1.00, extrapolates 2× OOD |

The decisive comparison is the last two: **same information, same capacity, same optimizer —
only the *channel* differs**, and it is the difference between grokking-and-capping versus
immediate-and-perfect. The user's H1 names the principle: *the channel matters, not just the
amount.* Position gave the model "an eye for time" — a built-in sense — instead of one more
symbol to be related by a learned circuit.

**The head-in-the-clouds reading.** Binding is **not an operation a model performs on tokens;
it is a property of the coordinate system the tokens live in.** Two features are bound when
they share a coordinate (same position, same phase, same object-slot, same time). When time is
a PoPE phase, "elapsed time between A and B" is *one primitive* — a phase subtraction the
architecture already owns. When time is a token, the model must first *learn to associate* the
time-token with the event and *then* learn to difference them: binding-by-circuit, which is
long, slow, and fragile. So:

> **A relationship is cheap iff it is a coordinate operation, and expensive iff it must be
> discovered as a correlation among tokens.** Learnability tracks the description length of the
> solution under the architecture's inductive bias (the doc's MDL thesis); the right binding is
> exactly what makes the general solution *short*.

This is **Chollet's two-kinds-of-abstraction, rediscovered as an engineering knob** (see
`chollet_connection.md`). Token-binding = **value-centric** abstraction: a learned continuous
association, interpolative, range-bound (the absolute arm memorizes its training range).
Coordinate-binding = **program-centric** abstraction: the relationship is structural, exact,
and *composes* (phase diffs, deltas — little programs baked into the geometry). The temporal
result is therefore direct evidence for the program-synthesis bet: **the way to get exact,
recombinable structure out of a continuous net is to move the relationship from a learned
value-association into a coordinate of the representation.**

---

## 2 · Consequences for a program-synthesis model

`directional`

**Recombination = coordinate composition.** Chollet's Kaleidoscope says intelligence mines
reusable *atoms* and recombines them. An atom that is a free-floating token can only be
recombined by a learned circuit (long, fragile). An atom that is a structured object in a
coordinate system can be recombined by coordinate algebra (short, exact, extrapolating). This
is *why* RWM's composition-fidelity question — does `M_{r2}∘M_{r1} = M_{r2∘r1}`? (RWM Q1/A4) —
is the crux: it asks whether the binding is structural enough that relations compose like
program primitives rather than blur like interpolations.

**ARC-AGI-3's Core Knowledge priors are binding structures, not features to be learned.** The
benchmark is built on objectness, geometry, basic physics, agentness. The temporal lesson says:
do **not** feed an agent a bag of pixel-tokens and hope it learns objectness as a circuit (the
`time_input` path — it will grok slowly and cap below perfect and fail OOD). Instead, give it
representations where each prior is a *coordinate*:

| Core Knowledge prior | the wrong (token) channel | the right (coordinate) channel |
|---|---|---|
| **objectness** | pixels correlated into a blob by a learned circuit | a bound object-slot (the entity *is* the unit) |
| **geometry** | absolute pixel indices as features | a spatial reference frame (2-D PoPE / grid-cell / learned frame — RWM agenda item 29) |
| **basic physics** | absolute positions over time | **Δ-encoded dynamics** (data point #2: deltas dissolve the shift) |
| **agentness** | "an action happened" as a token | action as a *causal coordinate* binding my-act → resulting-Δ |

**The unifying claim (`theoretical`).** *Program search and binding are the same problem:
finding the coordinate system in which the rule is a one-liner.* When the primitive is right
(phase-diff for time), gradient descent finds the solution trivially and it extrapolates. When
the primitive is wrong, even a representable solution stalls and caps. A program-synthesis
system should therefore **search over bindings/coordinates, not over token-circuits** — and the
deep net's job is to *propose the coordinate frame* in which the discrete program is short.
(This is the bitter-lesson-compatible translation of Ndea's "deep learning guides program
search": no hand-built DSL, but a *learned* coordinate frame that makes programs short.)

---

## 3 · Catastrophic distribution shift, reframed

`established` (data point #2) → `directional` (the general claim)

Data point #2 is *literally* a distribution-shift result. A value `v0` shifts from train range
[0,100] to OOD [1000,1100]; the target (total change) is shift-invariant. Same architecture,
only the input representation differs:

- **absolute** (running values): in-dist climbs to ~0.74, **OOD pinned at chance the entire
  run**, generalization gap opens immediately and never closes.
- **delta** (increments): in-dist → 1.00 and **OOD identical at every step** — gap ≈ 0 from
  initialization, because in delta space *train and OOD are the same distribution*.

> **The headline: a catastrophic distribution shift is often an artifact of the representation,
> not the data.** The shift lived entirely in `v0`; the delta representation does not contain
> `v0`, so the shift *cannot reach it*. (Makushkin's thesis: a non-stationary signal with a
> stationary derivative becomes stationary the moment you represent the derivative.)

**The cloud-level generalization.** Catastrophic distribution shift — and, we conjecture,
catastrophic *forgetting* — is the disease of **knowledge bound to a non-invariant coordinate**.
When the world moves that coordinate, every binding anchored to it breaks at once. The cure is
representational, not optimizational (data point #4 confirms: optimizer fixes cannot close a
representational gap): **bind knowledge to invariant coordinates** (deltas, relative frames,
object-relative position) so the shift becomes *invisible*. A model that learns the invariants
does not forget, because new tasks do not move those coordinates.

**Why ARC-AGI-3 is the ideal probe for this.** `commitment` Each level **deliberately shifts the
distribution** (new layout, new mechanic) while holding the *invariants* (Core Knowledge) fixed.
So the per-level structure of ARC-AGI-3 **is a distribution-shift curriculum**. A model that
binds to surface pixels will fail catastrophically at each level boundary (absolute-style); a
model that binds to invariant Core-Knowledge coordinates will transfer (delta-style). The
replica lets us measure data point #2's signature — *does the generalization gap open at the
boundary, or stay ≈ 0?* — in an interactive setting, which is exactly the data the project wants
to contribute to the distribution-shift problem in deep learning.

---

## 4 · The experimental questions (what the replica is for)

`open`

1. **Binding-channel sweep on an interactive task.** Take one ARC-AGI-3 game and feed an agent
   the state three ways — raw pixel grid (token), pixel grid + object-id channel (partial
   coordinate), full slot/coordinate encoding (object-bound). Predict the same ordering as data
   point #1: tokens grok-and-cap, coordinates learn-and-transfer.
2. **The level boundary as a distribution-shift assay.** Log the per-level generalization gap
   (train-layout vs held-out-layout accuracy *within* a level's mechanic). Absolute-style
   (gap opens) vs delta-style (gap ≈ 0) is diagnosable from the first evals.
3. **Composition = the A∘B test, in-environment.** LockPath L3 composes two mechanics taught in
   L1 and L2 separately. Does an agent that solved L1 and L2 solve L3 *by recombination* (program-
   centric) or fail / need to relearn (value-centric)? This is RWM's decisive experiment with a
   concrete substrate.
4. **Skill-acquisition efficiency curve.** Chollet's actual metric: actions-to-competence per
   novel level. Plot it per agent; the slope is the thing we are really trying to bend.

## 5 · Falsifiable predictions

`theoretical`

- **P1 (channel dominance).** At matched information and capacity, the object/coordinate-bound
  encoding beats the token encoding on OOD layouts of the *same* mechanic — and the gap is
  largest exactly where information must travel (a new position, a composed mechanic).
- **P2 (shift invisibility).** An agent whose state is Δ-encoded / object-relative shows
  gap ≈ 0 across a level boundary that only changes absolute layout; a pixel-token agent shows a
  gap that opens at the boundary and persists.
- **P3 (composition needs structure).** No purely value-centric (interpolative) agent solves L3
  from L1+L2 without retraining on L3; a coordinate-binding agent does. (If a token agent *does*
  generalize to L3, P3 is falsified and the binding thesis is weaker than claimed.)
- **P4 (no free lunch on absent info).** If a needed invariant is in *no* channel, no agent
  closes the boundary gap — a representability wall, not a learnability delay (the `integer`
  arm's lesson).

## 6 · What exists now / what's next

`established` (built) — the **replica** (`arc_agi_3/`): faithful harness + `LockPath` (4 levels,
L3 composes two mechanics), 10 passing tests incl. a BFS proof every level → WIN, an oracle
trace, and a replay widget. `chollet_connection.md` (the program-centric framing). This doc.

**Next (no training on this machine — implement, push, user runs on GPU; Mistake #36):**
1. A minimal *learning* agent (the research vehicle), with a swappable **state-encoder** so the
   binding channel is the experimental knob (question 1 / P1).
2. A held-out-layout generator per mechanic, to measure the level-boundary gap (question 2 / P2).
3. Port the strongest binding (2-D PoPE / object slots / Δ-dynamics) from RWM once question 1
   picks a winner.

> **The through-line.** RWM asked these questions about *time* on synthetic streams and got
> clean answers (channel decides learnability; the right representation dissolves the shift).
> ProgramSynthesis asks the *same* questions about *objects, space, physics, and agency* on an
> interactive benchmark built precisely to require them — and feeds the answer back as the
> binding RWM's transformer experiments point to. One thesis, two arenas.
