# Plan — object-ROTATION invariance / equivariance

Goal: recognise a known object presented at a NOVEL orientation, and infer its orientation (pose). 2D (SO(2)) first.
Mechanism source: **"Orientation Invariant Sensorimotor Object Recognition Using Cortical Grid Cells"** (Frontiers 2021,
PMC8825787; the follow-up to Lewis 2019). Full findings: `notes/tbt_object_frame_and_bootstrap_research.md` §D. Design home:
ARCHITECTURE.md §8 (the TRANSFORM primitive). Grounds: [[reference_tbt_pose_invariant_recognition]],
[[reference_operator_as_group_representation]].

## The mechanism (Numenta 2021), mapped onto our substrate
Rotation is **not** plain SDR overlap (a rotated object's location codes don't overlap the original) and **not** a bolted-on
3D pose particle filter. It is three things we mostly already have, composed:
1. **Multi-orientation grid modules.** The location code = grid modules pre-tuned to orientations spread over 360° and
   ORDERED. Ours (`GridEncoder`) currently has modules per (scale, axis) — only axis-aligned (0°/90°). Extend to modules at
   `θ_i = i·360/N`: a module at orientation θ reads the phase of `(x,y)` projected onto the θ-rotated basis.
2. **The rotation operator = a CIRCULAR-BUFFER shift over the orientation-ordered modules.** "Rotate the object by ω" =
   cyclically shift which module feeds which downstream slot — a *mental rotation* of the whole location code. This is the
   TRANSFORM primitive again (a shift = a group action), now ACROSS modules (the orientation buffer) rather than WITHIN a
   module (the phase). It composes with SE(2): TURN shifts the heading ring (self), object-rotation shifts the
   orientation-module buffer (object) — both SO(2) shifts on our `ModularOperator` machinery.
3. **Recognition = a bounded SCAN over candidate orientations, ranked by fit.** For each ω (N buffer shifts): apply the
   rotation operator to the sensed feature-at-location stream, and score the fit against stored objects (the pooler's
   overlap-match / least location-layer ambiguity). The winning ω = the inferred pose; the matched identity = recognition.
   A cross-correlation over the orientation ring — bounded (N), not a free search. **Location disambiguation within each ω is
   the pooler's existing UNION/overlap** (reused unchanged).

Why this route (not the alternatives): equivariance + a bounded scan gives pose AND preserves discriminability, is what
Numenta validated in 2D, and REUSES our `GridEncoder` + `ModularOperator` + pooler. Plain overlap can't (loses invariance⟂
discriminability); a lossy displacement-MAGNITUDE descriptor (L5) gives identity-only invariance and is a possible cheap
COMPLEMENT, not the primary. The Monty pose-hypothesis particle filter is the 3D (SO(3)) answer — deferred.

## The rotation centre — TBT does NOT find one; it MARGINALISES over locations
TBT never computes a geometric centroid. The frame origin is an ARBITRARY anchor (Lewis 2019: a random starting phase = the
origin, explicitly not the centre). Recognition is uncertain about WHERE on the object the sensor currently is and holds ALL
candidate current-locations at once — the **"union of locations"** (multiple bumps per module) that path-integrate together
and narrow. So "which point is the rotation about" is MARGINALISED OUT: you consider every anchor point simultaneously, and
the consistent `(current-location, orientation)` hypothesis wins. The centre is implicit in the surviving hypothesis, never
found. The criterion is PREDICTION ACCURACY (the orientation/location with least location-layer ambiguity / best fit).
**Consequence for us:** the anchor need NOT be the centre, and we do NOT add a centre search; we pair the orientation scan
with a LOCATION UNION and let fit pick `(location, orientation)`. This makes the **location-union upgrade a PREREQUISITE for
GENERAL rotation** (rotated about any point, entered anywhere). A first slice with CONTROLLED entry (consistent anchor point)
can defer the union; the general case cannot.

## Symmetry — a feature the scan EXPOSES, handled by keeping the ambiguity
A symmetric object gives TIES: a square → 4 equally-best orientations, a circle → a continuum. That is CORRECT — pose is
genuinely undetermined up to the object's symmetry group. The plan:
- **Identity is still recognised** — the object MATCHES at all those orientations, so symmetry affects POSE only, not identity.
- **Pose is returned as the tied SET (the symmetry orbit), never forced to one angle** — this is why orientation belief must be
  a POPULATION, not a scalar ([[reference_population_code_belief]]): it collapses to the symmetry group, not a point.
- **NEAR-symmetry is broken by more evidence** — if a distinguishing feature exists, more sensations resolve it; if the
  symmetry is exact, the ambiguity is fundamental and is KEPT. Mirror Monty's `required_symmetry_evidence` guard (avoid
  early over-commitment under symmetry). The union/population representation is exactly what holds this ambiguity.

## Phased plan (each slice: built → wired from `agent.py` → exercised by a test; suite stays green)
- **R1 — multi-orientation `GridEncoder`.** Extend the encoder to N orientation-ordered modules (per scale). Test: encodes/
  decodes 2D location; nearby locations overlap; the modules are ordered by orientation (the buffer's index space). Pure
  encoder work, no agent change yet.
- **R2 — the rotation operator (circular-buffer shift).** A `ModularOperator`-style rotation that shifts the orientation
  buffer by ω. Test EQUIVARIANCE: `apply(encode(loc), rotate_ω) == encode(R_ω · loc)` — rotating the code equals rotating the
  location (up to the centre caveat). Reuse the operator machinery; do NOT hand-code a rotation matrix on coordinates.
- **R3 — orientation scan in recognition.** Extend `Column.perceive` / the pooler: for a possibly-rotated object, scan the N
  orientations (apply the rotation operator), score each with the pooler, pick the best → recognise the object AND report its
  pose ω. Test ROTATION-INVARIANT RECOGNITION (the crown): learn an object at ω=0, recognise it at a NOVEL ω, report ω;
  same-features-different-arrangement stays distinct (the frame is still load-bearing). Wire from `agent.py`.

## FINDING (R3, 2026-07-14) — the oriented grid makes ROTATION exact but TRANSLATION path-integration INEXACT
On an axis-aligned grid a unit move changes a module's projection by exactly 1 → a constant INTEGER cell shift, which is what
`ModularOperator` learns. On an ORIENTED grid a unit move changes module i's projection by `cos θ_i` — **not an integer** —
so the rounded phase shift varies with position and **no constant integer per-module shift represents a translation**. (The
real grid model has CONTINUOUS phases, where this is fine; our discrete rings are the simplification that bites.)
Consequence, and why the build is shaped this way: the ORIENTED column takes its location by **sensory fix** (`locate`), and
the **axis-aligned nav column keeps dead-reckoning** — two frames, each exact at what it does (consistent with "always
multi-column, each its own frame"). OPEN: unifying them (continuous/fractional phases so translation is exact on an oriented
grid, or path-integrate axis-aligned and re-encode for rotation-matching). Do NOT paper over it with rounded shifts.

## THE SUBSTRATE QUESTION — is L6a on the wrong CODE? (raised 2026-07-14; decides how far R4+ has to go)
The R3 finding above is not a rotation problem; it is a **coding** problem, and it generalises. Framing that earns its keep:

**SDR/HTM is an EFFECTIVE ("IR") description of a spiking substrate** — take spikes, time-average to rates, threshold to
binary, keep only WHICH cells are active. Like any effective theory it has a **regime of validity**: it is excellent where
information lives in the IDENTITY of the active set (category, object, semantic similarity) and it breaks where information
lives in MAGNITUDE or CONTINUOUS POSITION. That is exactly our seam — L4/L2·3 (identity, binding, recognition) work on binary
SDRs; **L6a is a METRIC quantity and that is where it broke.** We did not hit a bug; we used the IR theory outside its regime.
Corollary: permutations are the only group actions the coarse description can represent EXACTLY — which is why "rotation
exact, translation not" felt arbitrary but wasn't.

**RATE AND PHASE ARE NOT ALTERNATIVES** (corrected 2026-07-14 — "phase, not rate" was a category error). The continuous grid
phase is **carried BY** graded rates: a continuous-attractor bump's centre-of-mass sits *between* cells precisely because
neighbouring cells fire at graded ratios. **Rate is the representational MECHANISM; phase is the QUANTITY.** And rate does a
second, distinct job we **already depend on** — MAGNITUDE (evidence / confidence / reliability). Our own code is full of it:
the pooler's `_match` fraction, the orientation scan's `hits` count, `ModularOperator`'s vote tallies, the `_Readout`'s
softmax weights, belief-as-a-population ([[reference_population_code_belief]]). **We already run a HYBRID** — binary SDRs plus
float magnitudes computed *outside* the code. So the refinement is NOT "phase instead of rate"; it is: **the metric sector
needs GRADED values, and the metric quantity happens to be CYCLIC — which is why phase algebra (shifts) is its natural form.**
Graded rates on a ring ALONE fix exact translation (a continuous centre-of-mass, no rounding); FHRR/FPE is an ALGEBRAIC route
to the same continuity that additionally buys clean binding. Both are "go continuous"; they differ in the algebra.

**THE THREE ROLES — keep them straight (this is the real lesson):**
1. **IDENTITY** — *which* cells are active → **binary SDR is RIGHT** (capacity, noise tolerance, set semantics, and HTM's
   dendrites assume binary presynaptic activity). L4 / L2·3. Working; do not touch.
2. **MAGNITUDE** — evidence, confidence, reliability → **graded / rate-like**. We already use it, in floats outside the code.
   Honest question for later: should belief live *in* the representation rather than beside it?
3. **METRIC position** — a continuous, CYCLIC quantity → graded rates (CAN bump) / spike timing (phase precession) /
   algebraic phase (FHRR). **This is the one L6a quantises, and the only place the seam actually bites.**

Three lines converge on the cyclic/phase FORM of role 3:
- **Neuroscience:** theta **phase precession** — a place/grid cell fires at progressively earlier theta phases across its
  field, so cell identity gives the COARSE field and the **phase carries precisely the fractional remainder our rounding
  destroys**. Oscillatory-interference grid models integrate velocity *in phase* directly.
- **ML:** steerable / circular-harmonic representations — rotation IS a phase shift (the same trick as the "continuous basis"
  fix above).
- **VSA (the one that should interest us most):** Fourier-HRR / **fractional power encoding** / **Spatial Semantic Pointers**
  (Plate; Frady & Sommer; Komer & Eliasmith) encode a CONTINUOUS value as a phase, where **translation = binding = phase
  addition — exact, continuous, any vector**; SSPs reportedly also reproduce grid-cell-like hexagonal codes.

**The candidate:** a **phasor/complex VSA location code** for L6a — exact continuous translation AND rotation (both are phase
operations), keeping distributed representation + algebraic binding. Note we **already plan a VSA** (the thalamus's
content⊗location bind, ROADMAP Phase 5), so this could **unify L6a's metric with the thalamus's binding in ONE algebra** —
a simplification, not a bolted-on second substrate.

**Discipline — a MATCHED description, not a wholesale swap.** Nobody does chemistry with QCD: an effective theory's value is
what it discards. Refine ONLY the sector where it breaks (L6a's metric); KEEP binary SDRs for identity (L4/L2·3), where they
are strictly better — capacity, noise tolerance, set semantics, and HTM's dendritic learning assumes binary presynaptic
activity. A global substrate swap would dissolve the L4/L2·3 stack that currently works.

**Honesty flags (do not over-believe this):**
- The IR/UV analogy is **heuristic** — no controlled expansion, no small parameter, no actual RG flow from spikes to SDRs.
  It is an intuition pump, not a derivation.
- "Better" is job-relative. The claim is NOT "SDRs are wrong"; it is "**the metric wants a phase code; identity wants a
  sparse binary code**" — one system can have both, matched at the seam.
- The hard part is exactly that seam: can an SSP/phasor location still drive **HTM's binary dendrites** at the L4 interface?
  Verify before betting: that FPE/SSP translation-as-binding is exact in the form we need, that rotation stays clean in it,
  and that the L4 hand-off survives.

**DRIFT FALSIFIER — RUN 2026-07-14. RESULT: 3b REFUTED at arbitrary orientations; N=4 is EXACT.**
30 steps of learned unit path-integration, measuring code fidelity `overlap(pi_code, encode(truth))/w` AND whether the
multi-scale consensus still DECODES to the truth:
- **Axis-aligned (control):** overlap **1.00**, decode exact at every step. Learned shift `[1,0,1,0,…]`. ✓
- **Oriented N=8:** overlap collapses to **0.50 by step 2**; decode drifts **LINEARLY** — off by 1, 2, 4, 6, 8, 10, 13 at
  steps 2, 3, 5, 10, 15, 20, 30. **First decode failure at step 2.** Error correction does **NOT** bound it.
  - **WHY (the mechanism, not just the number):** the rounding error is **SYSTEMATIC** — the same rounded lie applied every
    step, coherently — not random. Sreenivasan–Fiete multi-scale/CRT redundancy corrects **independent** errors; it cannot
    correct a **coherent bias**, so the consensus drifts *with* it. Error correction was never the right tool here.
  - **The 0.50 plateau is the signature:** the N=8 set CONTAINS the exact subset {0°,90°,180°,270°} (integer projections)
    plus the inexact diagonals {45°,135°,225°,315°} (±√2/2). Exactly HALF the modules stay correct → overlap 0.50. The
    mechanism is confirmed, not inferred.
- **Oriented N=4 {0°,90°,180°,270°}:** overlap **1.00**, decode exact at every step. Learned shift `[1,0,6,0,…]` = (+1, 0,
  −1, 0) per scale — all exact integers. **Translation AND 90° rotation both exact, in ONE frame, fully SDR-native.**

**VERDICT (the decision is now empirical, not aesthetic).** Binary SDRs survive in L6a **iff the group we need is the
lattice's space group** (integer moves + 90° rotations) — and then **N=4 is exact and the two-frame seam DISAPPEARS**
(one column path-integrates AND rotates). For **arbitrary angles**, binary is **refuted** — no amount of error correction
rescues it — and the phase code (3a) is forced. So the question to answer before building R4+ is not about codes, it is:
**what rotation resolution do the target environments actually require?** 90° (grid worlds — ARC and most grid benchmarks,
where N=4 is exact and general) or arbitrary (continuous/robotic — where 3a is mandatory).

## Status: R1 ✅ R2 ✅ R3 ✅ (2026-07-14) — `test_oriented_grid`, `test_rotation_operator`, `test_rotation_recognition`
R3 result: an asymmetric object learned at ω=0 is recognised at ALL 8 novel orientations with the exact pose reported; a
4-fold symmetric object is recognised with ties `{0,2,4,6}` = its symmetry orbit (pose undetermined up to 90°, not forced).

## DECISION (2026-07-14, Cipher): ARBITRARY ANGLES ARE REQUIRED → L6a GOES CONTINUOUS
Target environments after ARC-AGI-3 are **Danganronpa and other games**: 3D, free camera ⇒ **continuous rotation, continuous
movement, SO(3) not SO(2)** (and Danganronpa is already the project's generality litmus,
[[feedback_epistemic_value_is_prediction_error]]). The falsifier showed binary lattice codes are exact ONLY for the lattice's
space group; that group is not available here. **So N=4 is out as the answer, and binary is refuted for L6a's metric.**

**What makes this tractable — continuity is needed for INTEGRATION, not BINDING.**
L4 does not need a continuous location; it needs a DISTINGUISHABLE, overlap-bearing one. So:
- **L6a STATE = continuous** (a coordinate / phase vector) → translation and rotation are exact, for any vector and any angle.
- **The SDR = a per-fixation READ-OUT** of that state, for L4 binding + L2/3 pooling.
- **Quantisation at the read-out is BOUNDED and NON-ACCUMULATING** — it is a fresh encode each fixation, never applied
  repeatedly. That is exactly the property the drift experiment proved we lack today. **L4/L2·3 stay untouched.**
The feared "can a phasor drive HTM's binary dendrites?" seam largely dissolves: it doesn't have to.

**This ALIGNS with our own reference, it does not depart from it.** [[reference_operator_as_group_representation]] says motion
is "a LEARNED group-representation MATRIX acting on the location CODE" (Gao 2021) — a CONTINUOUS code. Our permutation
operator was always the lattice-restricted SPECIAL CASE. The TRANSFORM primitive survives; only its representation changes
(a learned group action on a continuous state, still discovered from observation, still not hand-coded).

**Honest consequence — R1/R2 are likely SUPERSEDED.** With a continuous state, rotation is "rotate the state (any angle),
re-encode" — no orientation modules, no circular buffer, no N-cap. The multi-orientation `GridEncoder` and `RotationOperator`
were scaffolding to make rotation a permutation on a QUANTISED code; that problem dissolves. **R3's SCAN survives** — propose
candidate poses, rank by PREDICTION FIT, ties = the symmetry orbit — and it is the part that generalises to SO(3), where the
pose space is 3-D and the union/evidence pooler (R4) stops being optional. Decide keep-vs-retire in the design step, per
[[feedback_decisive_full_cutover]] (do not leave a superseded parallel path lying around).

## BAKE-OFF RUN 2026-07-14 — A vs B-continuous. **RESULT: A wins on a MEASURED basis.**
First, a correction the bake-off forced: **"modular phases" ≠ "quantised phases."** The drift experiment refuted only the
QUANTISED variant. **B-continuous** (float φ per module + oriented modules + the circular buffer) has EXACT translation
(`φ += d·u(θ_i)`, no rounding) and EXACT rotation at k·Δ — it would have PASSED the drift test. R1/R2 were **not** broken.

So both designs were tested on the deciding question: after best-effort un-rotation of an OFF-GRID rotation, does the
recovered location still produce the code the model was learned at (else L4 bursts and recognition fails)?

| radius | A match / err | B N=8 | B N=16 | B N=32 | B N=64 | B N=128 |
|---|---|---|---|---|---|---|
| 1 | 1.00 / 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 / 0.00 | 0.86 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 / 0.00 | 0.64 | 0.79 | 1.00 | 1.00 | 1.00 |
| 8 | 1.00 / 0.00 | 0.36 | 0.64 | 0.86 | 1.00 | 1.00 |
| 16 | 1.00 / 0.00 | 0.36 | 0.64 | 0.79 | 0.93 | 1.00 |

**The analytic prediction `N > 2π·r` was confirmed exactly** — first-working-N is the next power of two above the threshold at
every radius (r=1→6.3/8, 2→12.6/16, 4→25.1/32, 8→50.3/64, 16→100.5/128). The mechanism predicted the numbers.

**VERDICT.** B is exact and TBT-faithful **for small objects at coarse angles**, but its module count must grow **linearly
with object radius**; at r=16 that is 128 orientations × 4 scales = **512 modules / ~6k bits for ONE object**, unbounded in
object size — and in SO(3) the orientation set on a sphere scales **quadratically** for the same angular resolution. The
target environments (Danganronpa-class) do not bound object size, so **B has no viable operating point**. **Adopt A.**
R1/R2 are therefore retired **not because they are wrong but because they are cost-unbounded** — a measured cut, not an
inferred one.

## CUT-OVER TO A — **DONE 2026-07-15. Suite 46 green.**
Executed per [[feedback_decisive_full_cutover]] (build the replacement and DELETE the superseded in the SAME move; no
transitional two-stack). What landed:
- **L6a's state is CONTINUOUS** — `column.py` `_pose = ((x, y), heading°)`; the grid SDR is a per-fixation **read-out**
  (`_code`), so its quantisation is bounded and never accumulates.
- **`operator.MotionOperator`** replaced `ModularOperator` — it learns each action's **body-frame** displacement + heading
  change (a running mean over observed pose triples) and `apply`s it through the current heading. Position- AND
  heading-invariant by construction; still *discovered from observation*, never hard-coded (Gao 2021).
- **Non-abelian SE(2) got SIMPLER, not harder.** The keyed `(action, heading)` design + the heading-ring operator are GONE:
  the body-frame delta makes non-commutativity structural. New capability the discrete design could not have — `FORWARD`
  generalises to a heading **never observed** (taught at 0°/90°, correct at 37°). The deferred `ConjunctiveEncoder` tensor is
  **moot**: heading is a float.
- **Rotation is exact at ANY angle.** The R3 scan was KEPT and adapted — "apply rotation" moved from a module permutation to a
  state rotation, so candidates are no longer confined to multiples of 360/N (verified at 37°, 113.5°, 244.25°).
- **DELETED:** `ModularOperator`, `RotationOperator`, `GridEncoder(orientations=)`, `test_oriented_grid`,
  `test_rotation_operator` — the suite count fell 54 → 46 by design (rotation is now tested end-to-end through *recognition*,
  which is the property we actually wanted, rather than through the machinery that happened to implement it).
- The two-frame seam disappeared: ONE column, exact translation AND exact continuous rotation.

**A finding the cut-over exposed — pose precision is set by the LEVER ARM, not the substrate.** Ties come in two kinds, and
`recognize_rotated` now reports both honestly rather than claiming they are all symmetry: an **exact** tie is the symmetry
orbit, while a **resolution** tie reflects that separating ω from ω+δ means resolving the arc a feature travels (≈ r·δ). So
precision ≈ (grid resolution / object radius) — **measured: radius 2 → ±14°, radius 8 → ±3°** (`test_rotation_recognition`).
This is the law any real sensor obeys, and it is the *opposite* of the retired design's failure mode: here a bigger object
resolves pose BETTER, where the module-count wall got worse with radius and no object size could widen it.

## R4 — evidence-based recognition over a hypothesis population (the pose is SOLVED, not scanned)

**MECHANISM CHECK FIRST (2026-07-15, [[feedback_check_tbt_accuracy_per_step]]) — and the plan HAD drifted.** This doc has
said "R4 = the union/evidence pooler" since R1, and the cut-over section above still describes recognition as a *scan over
candidate poses*. Re-reading the primary mechanism ([[reference_tbt_pose_invariant_recognition]],
[[reference_tbt_layers_4_23]]) says something stronger, in Numenta's own words:

> Recognition = INCREMENTAL evidence accumulation, NEVER recomputed: INIT **solves** pose from the first sensation … each
> later step computes the displacement, **rotates it by the hypothesised rotation**, predicts the next location, compares
> features, ADDS evidence; terminate when one hypothesis dominates.

> You recognize an unseen orientation because you **SOLVE** for the rotation; you don't recall it.

**So the scan is not a simplification of the mechanism — it is a different (worse) mechanism.** Monty never samples angles.
R3's scan was the faithful implementation of the *2021 orientation paper*, which is discrete-by-construction; Monty is the
part of TBT that actually solves continuous pose, and R4 is where we adopt it. Correcting the doc before building, as the rule
requires.

**Why solving is possible with our (non-morphological) features.** Monty solves rotation from a **single** sensation because
its features carry a local frame (surface normal + principal curvatures) — align sensed frame to stored frame, read off the
rotation. Our features are colour-at-location: *non-morphological*, no intrinsic orientation
([[reference_tbt_feature_definition]]). The general substitute is the one this doc already identified: *"for polyominoes the
orienting cue is the inter-cell **displacement geometry**"*. Two sensed points suffice — a rotation is fixed by one
corresponding displacement pair:
- the object is presented at `(ω, t)` ⇒ every model point ℓ appears at `p = R(ω)·ℓ + t`;
- so for two fixations, `p₁ − p₀ = R(ω)·(ℓ₁ − ℓ₀)` — the translation cancels;
- ⇒ **ω = angle(p₁ − p₀) − angle(ℓ₁ − ℓ₀)**, exact, closed-form; then **t = p₀ − R(ω)·ℓ₀**.

The correspondence `(ℓ₀, ℓ₁)` is what is *hypothesised*; the pose is *derived* from it and then *verified* by prediction.
Nothing samples, nothing is tabulated. Two honest prunes come free from the geometry (a rotation is an **isometry**, which is
the group structure we already committed to in ARCHITECTURE §8 — not a domain prior): `|ℓ₁ − ℓ₀|` must equal `|p₁ − p₀|`, and
a zero displacement leaves ω genuinely undetermined.

**The hypothesis population = the union, narrowed.** This is the 2017/Lewis-2019 union under a different name: a sensed
feature activates *the union of locations where it occurs*, movement path-integrates the whole union by the ONE operator, and
each new sensation prunes what it fails to predict. A hypothesis is `(object, ω, t)` — Monty's **(object ID, pose)**, pose =
location + orientation, continuous. Evidence per fixation = the model's own prediction fit (L4 predicts the feature at the
hypothesised object-frame location) graded by L2/3's identity support — *feature match adds, mismatch subtracts*.

**What R4 delivers that R3 cannot:**
| | R3 (scan) | R4 (solve + evidence) |
|---|---|---|
| entry point | must share the object's anchor | **anywhere** — `t` is solved too |
| pose values | only the sampled ones | **exact**, closed-form |
| cost in SO(2) | O(samples) | O(&#124;corresponding pairs&#124;), distance-pruned |
| cost in SO(3) | **cubic in resolution** — the blocker | 3 non-collinear points solve it; **no resolution axis at all** |
| ambiguity/symmetry | a tie list over sampled angles | the surviving **population** |

**Ownership (`reference_model_ownership_map` — extend the owner, never build beside it).** The object model is DISTRIBUTED,
never one layer's data structure: L4 = features, L6a = locations, L2/3 = identity. So: the **L4→L6a associative link**
(feature → the union of (identity, location) where it was sensed) is the piece Lewis 2019 requires and we have never built —
it lives in the `Column`, learned Hebbian-style in `perceive`. Identity support is the **pooler**'s (`ColumnPooler.support`).
Path integration is the **operator**'s, unchanged. **NB — this is emphatically NOT the retired rotation table** (`shape →
next_shape` per orientation), which memorised the *answer* per pose; this stores the *object* (learned online, once, in its
own frame) and *derives* the pose. The distinction is the whole point of the pivot.

**Scope + the deletion (per [[feedback_decisive_full_cutover]]).** R4 SUBSUMES R3 (a shared anchor is the special case `t=0`),
so `recognize_rotated` + the `candidates` sampling are DELETED in the same move — no two-stack.

**Consequence to accept honestly:** the lever-arm resolution law measured above bounded the *scan's* discrimination. With the
pose solved from continuous positions there is no sampling to bound, so that test goes with the scan. The law itself is not
repealed — it re-appears as **pose error ≈ position noise / lever arm** the moment sensing is noisy. Deferred with sensor
noise, not silently dropped.

### R4 — **BUILT 2026-07-15. Suite 48 green.**
`Column.recognize` (+ `Hypothesis`, `_solve`, `_evidence`, the `_link` L4→L6a union, `ColumnPooler.support`); `recognize_rotated`
and the `candidates` sampling DELETED in the same move. All six properties tested end-to-end through `agent.py`:
novel orientation → pose solved exactly; **arbitrary off-grid angles exact** (37°, 113.5°, 244.25°, 359.9°); **the crown —
rotated AND translated far from the learned frame, entered on a different feature each time, recovering ω *and* the object's
origin** (the retired scan could not do this at all); an **ambiguous seed narrowed by evidence** (two objects sharing two
features at the same separation are indistinguishable for two fixations — neither the seed nor the isometry prune separates
them; the third fixation's burst refutes the wrong one); symmetry → the exact 4-fold orbit; one fixation → no hypothesis.

### The defect R4 EXPOSED (measured, not suspected) — L2/3 never revises at LEARNING time
Writing the ambiguity test the obvious way — two objects sharing feature 1 **at the same location** — failed, and the cause is
a real bug, not a test artifact. Measured: learning two such objects mints **ONE** identity, a chimera holding all six
features; both objects then "recognise" as object 0 with full evidence. R4 is faithfully reporting a corrupted model.

**Root cause** (`pooler.py`): `if self.active and (learn or self._match(...) >= self.persist_frac)` — during learning,
persistence is **unconditional**. L2/3 commits to an identity at the first fixation and never revises, so every later fixation
of the second object is reinforced into the first object's identity. At fixation 1 the two ARE indistinguishable, so
recognising A is *correct inference*; the bug is the absence of **revision** when fixation 2 contradicts it.

**Why the obvious patches fail** (worked through, so the next attempt does not re-derive them):
- *Make persistence conditional on support.* A code never yet bound to anything supports nothing, so `match = 0` for every
  object — but that is **ignorance, not refutation** (absence of evidence ≠ evidence of absence). Treating it as refutation
  mints a new object per fixation: duplication, the very bug the pooler was built to fix.
- *Refute on an L4 burst.* L4 is **object-agnostic** — shared across every object in one frame — so it cannot refute an object
  hypothesis. It bursts iff *no* object has a feature there.
- *Mint on the burst instead of deferring.* Breaks the current two-phase structure (pass 1 trains L4 while everything bursts;
  the pooler only learns from pass 2), and duplicates.

**The fix — the same evidence machinery R4 just built, applied at learning time.** Refutation needs the object's **extent**,
which the column now has in `_link`: under hypothesis A this fixation lands at (3,0), A has nothing at (3,0) ⇒ A is refuted.
That is exactly the replay. So **learning = recognise, then bind**: buffer the episode (Monty's Buffer), ask `recognize()`
whether a known object explains it, reinforce that identity if so, else mint and bind the sweep to the new one. One mechanism,
no parallel path — and it makes L2/3 hold a genuine **union** that narrows, which is what the theory says it does. Cost: the
learning API moves from per-fixation `perceive` to episode-level commitment; `test_l23_pooling` + `test_object_centric` are the
guard rails.

### R5 — **BUILT 2026-07-15. Suite 51 green. The merge is gone (2 objects → 2).**
`Column.commit` (buffer → `recognize` → reinforce-or-mint-and-bind), `ColumnPooler.mint`/`bind` public, and `pool` reduced to
pure INFERENCE — the `learn or …` branch that caused the merge is DELETED, not patched around. `_evidence` folded into
`_replay(identity, ω, origin, learn)`, which now SCORES and BINDS in one traversal.

**The API collapsed rather than grew**, which is the sign the decomposition was right:
- **LEARN** = `start_object` → `sense_sweep` × n → `commit`
- **INFER** (canonical pose, online) = `start_object` → `perceive` per fixation
- **RECOGNISE** (unknown pose) = `start_object` → `sense_sweep` × n → `recognize`

`perceive` lost its `learn` flag (it is inference, full stop) and `reset_object`/`perceive_object` were DELETED as superseded.
One traversal (`_replay`) serves scoring and binding, so **learning inherits pose-invariance for free**: studying a known
object at a novel pose reinforces it instead of minting a duplicate, and binds in the object's OWN frame so rotated
coordinates never enter the model (tested at 90° and 217°, then re-recognised at 45° to prove the model is unharmed).

Two guards keep the refutation honest, and both are tested:
- a **PARTIAL** sweep of a known object is all match, no contradiction ⇒ recognise + reinforce, never fragment. (A rule of
  "anything not fully explained is new" would shatter every object ever seen partially.)
- an **all-BURST** sweep is *unlearned*, not *new* ⇒ defer minting until L4 predicts something. A burst code is
  location-agnostic; binding it would teach "feature → object" (the feature-only trap) — the pooler's burst rule at episode scale.

**A degenerate test R4/R5 exposed.** `test_relative_arrangement_is_load_bearing` used P=[7,8] vs Q=[8,7] to prove the frame is
load-bearing. On a line, **Q is literally P rotated 180°** — so a pose-invariant recogniser is RIGHT to call them one object at
two poses, and the old test was asserting a *feature* of the pre-R4 code. The intent was sound, the example was degenerate:
fixed to P=[7,8,9] / Q=[8,7,9], which are genuinely unrelated by any rotation (P rotated 180° reads 9,8,7). The orientation is
not lost when identity and pose factor — it is *reported*, which is strictly more information than two identities.

## R6 — SO(3): the orientation is a MATRIX, and 2-D was the special case

**MECHANISM CHECK FIRST ([[feedback_check_tbt_accuracy_per_step]]) — and the sources are more specific than this plan was.**
Two independent memories name the representation outright:
- [[reference_tbt_pose_invariant_recognition]]: *"pose = location + orientation (**three orthonormal vectors**), CONTINUOUS
  (rotation matrices/quaternions, never discretized)"* — Monty's pose is **3-D-native and matrix-valued**.
- [[reference_operator_as_group_representation]] (Gao 2021): motion = *"a learned **group-representation matrix** acting on the
  location CODE"*.

So SO(3) is **not** "add a dimension to a 2-D mechanism." Our scalar `heading°` was an SO(2)-only encoding of a thing both
sources say is a matrix — the same category of shortcut as the discrete grid code the last cut-over removed, and it fails the
same way: a scalar angle simply cannot name a 3-D rotation (SO(3) is 3-DOF and non-abelian). The honest move is to **stop
special-casing the representation**: orientation becomes an n×n rotation matrix, and *degrees* become a 2-D READ-OUT
(`to_angle`), exactly as the grid SDR became a read-out of the continuous pose. One code path serves n=2 (ARC) and n=3
(Danganronpa-class), which is the generality the whole rotation thread was for.

**The pose solve generalises without changing shape.** Monty solves rotation by ALIGNING FRAMES (sensed local frame → stored
local frame). We have no morphological features, so we build the frame from **displacement geometry** — the same substitution
R4 already made, just with more vectors:
- an object at `(R, t)` puts model point ℓ at `p = R·ℓ + t`; differences cancel `t`, giving `R·(ℓᵢ−ℓ₀) = pᵢ−p₀`;
- **n−1 independent displacements determine R**: Gram-Schmidt them into an orthonormal frame on each side (`U` sensed, `V`
  model), completing the last axis by the generalized cross product so both frames are right-handed, then **R = Uᵀ·V**. This
  is the TRIAD method — 2-D needs 1 displacement (2 fixations, R4's `atan2` difference is exactly this at n=2), 3-D needs 2
  (3 fixations, non-collinear).
- **No angular-resolution axis, so nothing cubes.** This is precisely what design A bought and the bake-off predicted.

**What the generalization gives for free (the sign that it is right, not a port):**
- **Rigidity pruning gets STRONGER and simpler.** R4 pruned on one distance; the general form requires EVERY pairwise distance
  to be preserved, which subsumes the distance check *and* the angle-between-displacements check in one rule. Still the group
  structure (an isometry), still not a domain prior.
- **CHIRALITY falls out.** `R = Uᵀ·V` is always a proper rotation (det +1), so a MIRRORED object cannot be explained by any
  hypothesis — the evidence refutes it. Reflections are not in SO(3), and a chiral object's mirror is a genuinely different
  object. No special case: the existing evidence rule does it.
- **Non-commutativity becomes REAL.** In SO(2) rotations commute, so `test_operator_se2`'s non-abelian claim rests on
  translation-vs-rotation. In SO(3) the rotations themselves do not commute (yaw∘pitch ≠ pitch∘yaw) — a property SE(2) could
  not exhibit, and the operator gets it from `R' = R·ΔR` with no new code.
- **`wrap` disappears.** Angle wrap-around was an artifact of the scalar encoding; matrices compose without it.

**Honest approximations, noted not hidden:** averaging rotation matrices entrywise and re-orthonormalizing (Gram-Schmidt) is
the CHORDAL projection onto SO(n), biased toward the first axis — exact when observations agree (our case) and fine for tight
clusters; the Karcher/Fréchet mean (or an SVD polar projection) is the refinement if sensor noise ever makes it matter. And
`_solve` uses the first n fixations whose displacements are independent; a DEGENERATE sample (all collinear in 3-D) genuinely
leaves a 1-parameter family of poses, which a discrete population cannot express — so we return nothing rather than invent an
angle, the same honesty as the single-fixation case (the continuum is the circle-symmetry case this doc already flagged).

### R6 — **BUILT 2026-07-15. Suite 57 green.**
`operator.py` is now dimension-generic: the pose is `(position, R)` with an n×n rotation MATRIX, and the module carries the
rotation algebra the solve needs (`eye`/`rotate`/`compose`/`invert`/`from_angle`/`to_angle`/`cross`/`gram_schmidt`/
`frame_from`/`solve_rotation`/`orthonormalize`). `Column._pin_rotation` picks the `dims` fixations that pin R down;
`_solve` is the TRIAD method at any n. The scalar `heading°` and `wrap` are DELETED; `Agent(dims=2|3)` picks the space.
`test_operator_se2` → `test_operator_non_abelian` (it covers SE(2) AND SE(3) now).

Proven end-to-end: an object learned ONCE in its own frame, recognised at arbitrary 3-D orientations (about z, y, x, and a
**tilted** yaw∘pitch∘roll axis) AND translated, entered on a different feature each time, rotation recovered EXACT;
**chirality** (a mirrored object is its own identity — no rotation relates them); the **collinear limit** (returns nothing, as
it must); and SE(3) **rotations that do not commute** (YAW;PITCH faces (0,0,−1), PITCH;YAW faces (0,1,0)) — the property SE(2)
structurally cannot show.

**A REAL BUG the chirality test caught — the learning bar was arbitrary AND corrupting.** `commit` used
`evidence ≥ ½·fixations`. A 4-point chiral pair scored 3 matches − 1 refutation = **2**, and the bar was 0.5×4 = **2** — it
merged *by a hair*. Two things were wrong, and the fix is one rule:
- the tolerance was **arbitrary**: whether a contradiction was forgiven depended on how many OTHER fixations happened to
  agree, so a 3-point pair differing in one point stayed distinct while a 4-point pair differing in one point merged;
- worse, the tolerance was **corrupting**: `_replay(learn=True)` binds every predicted fixation, so "recognise despite a
  contradiction" BINDS the contradicting fixation into the object — which is precisely how R5's merge happened. Tolerance here
  does not degrade gracefully, it silently rewrites the model.
So the bar is now **"nothing REFUTES it"** (`Hypothesis.refuted`): threshold-free, and safe. A PARTIAL view is all match and no
contradiction ⇒ still recognises; one genuine contradiction ⇒ a different object, however much else agrees. Tolerating k
contradictions is a question for a sensor-NOISE model, deferred with noise itself — with an exact sensor a contradiction is
decisive. NB this made the ORIGINAL chiral pair pass; the fix was in the mechanism, not in choosing a friendlier object.

## R7 — the EMERGENT learning-time boundary (a sweep that crosses objects splits itself)

**MECHANISM CHECK ([[feedback_check_tbt_accuracy_per_step]]).** [[reference_tbt_segmentation_and_grouping]] is unambiguous:
*"the module doesn't explicitly segment objects… it relies on feature and morphology mismatch to implicitly detect
boundaries"*, and *"'two proto-things are really one' / 'one is really two' is just what the EVIDENCE concludes — not a re-run
of a segmenter."* So the boundary signal is **refutation**, which R4–R6 already built (`Hypothesis.refuted`). Nothing new is
needed to *detect* it. What is new is the **interpretation**.

**THE REAL DIFFICULTY (why TBT leaves this open).** Drop the caller's "this episode is one object" cue and a prefix match plus
a contradiction becomes genuinely AMBIGUOUS:
  (a) a **boundary** — the sweep covered object A and then entered B; or
  (b) a **different object** — the whole sweep is one object that merely SHARES a prefix with A.
Both are consistent with the same evidence. This is not a gap in our code — it is the segmentation problem itself, and R5
depends on reading (b) (`test_shared_features_do_not_merge`) while R7 wants to read (a). Committing to either blindly breaks
the other: always-split turns the R5 pair into "P plus a 1-cell blob" and re-mints that blob every pass; never-split is the
status quo.

**THE MODEL ITSELF DISAMBIGUATES — did the prefix EXHAUST the object?** You leave an object when you reach its EDGE. If the
prefix visited every location in A's model, then A *ended* and what follows is something else → boundary (a). If the prefix
covered only PART of A, the sweep never reached A's edge, so "A then something" is a worse parse than "one object that shares
A's prefix" → different object (b). The extent is already in the column — `_link` holds every (identity, object-frame
location) — so this is the model answering, not a prior. Checked against every case: the R5 shared pair (prefix = 2 of P's 3 →
no split ✓), a known object followed by another known one (✓ split), a known object followed by a NOVEL one (✓ split, then
mint), and cold start (no hypothesis → mint the whole sweep ✓).

**THE HONEST LIMIT, and it is the same thread as gravity.** A PARTIAL sweep of A that then wanders into a novel B is absorbed
as one blob: the prefix never exhausted A, so nothing licenses the split. And a wholly novel scene mints one blob — correctly,
since *"the object is a RECOGNITION construct"*: with no model there is no object. Splitting a blob needs a second signal, and
the same memory names it: *"Best grouping cue we already have: COMMON FATE (what moves together)"*. That is the DYNAMICS slice
(the operator over object poses) — so cold-start segmentation and "any unsupported object falls" are one thread, not two.

### R7 — **BUILT 2026-07-16. Suite 59 green.**
`Hypothesis.refuted` → `refuted_at` (the FIRST refuted index — one field answering both "is it refuted?" and "where?");
`Column._extent`/`_exhausts`; `commit` splits and recurses. Measured (all three cases):
- **A known + a NOVEL B, swept continuously with ONE onset** → the boundary is found, A is reinforced, B is minted, and B is
  then recognisable ALONE (`[1,1,1]`) **despite never having been marked as an object**.
- **both known, swept continuously** → library stays at 2; no spurious blob.
- **cold start (nothing known)** → one blob. The honest limit, exactly as the theory predicts.

**A contract violation this exposed.** A minted object now anchors its frame at **its own first fixation** (not `_anchor`) —
required, or a split remainder would be modelled at coordinates no later sweep could match. `test_l23_pooling` had been
`locate()`-ing at ABSOLUTE coordinates, which the old anchor silently made equivalent; under the correct rule its learning
(object-relative) and its inference (absolute) no longer agreed. The test was relying on a contract violation the old code
masked, so it now presents its objects in object-relative coordinates — which is what every other caller already did.

### R8 — ONLINE POSE-SOLVING. **BUILT 2026-07-16. Suite 60 green.**
The fourth time a representation shortcut failed at the same seam, and the prediction in memory was right: *"expect the next
one at the online path assuming its place on an object."* MEASURED before: online, the same object shifted to (7,3) read
`[-1,-1,-1]` and entered mid-object `[-1,-1]`, while the buffered path solved the identical presentation. AFTER: `[0,0,0]`
and `[0,0]`.

`perceive` now narrows a live `(object, pose)` population per fixation (`_pop`/`_narrow`) — Monty's evidence-based LM, never
recomputed. `_fit` is the ONE scoring rule; `_replay` loops it over a buffer, `_narrow` calls it per live hypothesis, so
batch and online are not two mechanisms. Seeding needs `dims` fixations, and before that the L4→L6a UNION is the evidence
(Lewis 2019) — a feature only one object carries names it; a shared feature is honestly ambiguous.

**Three things DELETED as superseded** (the shortcut was load-bearing in more places than it looked):
- L2/3's support-only `pool()` + `persist_frac` + `which()`. It could answer "which object does this code support?" but never
  "…and where am I on it?", so it *required* the sensor to already sit at the object's origin. The population subsumes it;
  `support` survives as the population's weight.
- the boundary's **re-anchoring** — `perceive` no longer teleports the frame on recognition failure; the next object's pose
  is solved. The re-anchor existed only to make the assumption true.
- `Column.object_id` — asking L2/3 for its settled state after `perceive`/`commit` already RETURNED what they concluded was a
  second route to one answer (`label_of`).

**A bug this surfaced in review, worth recording:** `_fit` first scored support and returned early on a low score, which meant
a freshly MINTED identity (support 0 by definition) could never be bound — learning silently produced nothing. Binding must
precede scoring. Caught by the suite going 18-red, not by inspection.

**An honest change to what a test may claim.** `test_relative_arrangement_is_load_bearing` (P=[7,8,9] / Q=[8,7,9]) now reads
`[-1,-1,0]` / `[-1,-1,1]` where it once read `[0,0,0]`. That is the improvement, not a regression: feature 7 belongs to both
objects and after two fixations [7,8] is still explained by P at 0° AND Q at 180°, so only the third fixation separates them —
which is exactly the test's claim (features can never tell them apart; the arrangement does). The old `[0,0,0]` came from the
ASSUMPTION breaking the tie, not from evidence.

### R9 — OBJECT DYNAMICS + COMMON FATE + the ART cut-over. **BUILT 2026-07-16. Suite 67 green.**
- **The operator over OBJECT poses** (`Column.dynamics`, an L5 engine): from ONE demonstration of a shove it predicts at
  positions never demonstrated, on the same object ROTATED, and on a DIFFERENT object never seen to move — fed by the poses
  `recognize` SOLVES, so perception → dynamics is end-to-end. Object motion is EXTRINSIC (measured: an intrinsic operator
  sends a 90°-turned block 90° off the shove).
- **Common fate** (`_common_fate_groups` + `look_again`) groups a look by MOTION with no model, and `_commit_split` makes the
  grouping PERSIST — the ART orienting RESET: a scene split by motion is >1 object, no identity may claim two groups, so parts
  are RECRUITED fresh. This CLOSES R7's cold-start blob: a blob learned from two things always seen together TEARS into its
  parts once one moves (measured, under continuous motion; library stays bounded, each part recognisable alone).
- **The un-binding answer is DON'T** (four literatures converge — ART recruit, latent-cause new-state, Xu&Carey motion-first,
  the size principle): ART **CHOICE** (`_choice` = matched/(α+|model|)) keeps a torn-off part winning over the blob, which
  dies of disuse; `perm_dec` stays dead. **The one subtlety the build exposed:** choice/vigilance must be counted at the
  FEATURE-AT-LOCATION level (`_replay`'s `matched`, via L2/3 `support`), NOT raw L4 cells — burst-binding inflated an L4-cell
  receptive field so a piece tied its parent blob at 0.5. The pooler's L4-cell `choice`/`match`/`receptive_field` were built
  (ART cut-over) then DELETED one commit later as the wrong granularity — honest thrash, recorded.
- **Still open** (unchanged): the operator's KEY (discovered condition, not a named action); and re-committing a fused,
  now-static multi-object scene is not handled (the game-loop should track individuated parts separately, not re-fuse them).

### R10 — L5PT DISPLACEMENT / relations. **BUILT 2026-07-16. Suite 71 green.**
`Column.relate` (the relative pose of B in A's frame — `location+location→the relation`, the inverse of the operator;
position- AND orientation-invariant by construction) + `observe_relation`/`relation_of` (a relation is a displacement that
stays STABLE as the pair moves — assumed fixed from the first view, DISSOLVED on independent motion; the assume-then-correct
philosophy, [[feedback_prefer_generalize_then_correct]]). Fed by the poses `recognize` SOLVES, so perception → relations is
end to end. Measured: the relative pose of a rigid pair is invariant across every position and shared orientation, confirms as
a relation, and breaks when one object moves alone. This EXTENDS common fate ("moves together → one thing") to "fixed relative
pose → a relation", and is the substrate the support-override reads. NB L5 = IT + PT (user reminder 2026-07-16): this is the
PT thick-tufted displacement role; the L5IT associative integrator is deferred.

## NEXT (in order)
1. **The context-gated OVERRIDE = gravity AND walls, one slice** (ARCHITECTURE §8/§9): the operator is the regular FREE KERNEL
   (everything falls; the push moves), and a LOCAL RELATIONAL CONTEXT predicts the exception (supported; blocked). The context
   now EXISTS (`relate`/`relation_of`); the remaining work is the GATING — when the relation holds, suppress the kernel's
   prediction. A table stopping a fall and a wall stopping a push are the SAME mechanism.
2. **The operator's KEY, discovered rather than given.** Gravity's key is a CONDITION, not an action; and WHICH relation gates
   WHICH delta is the discovery problem ([[feedback_subgoal_types_from_dynamics]], [[reference_l5_operator_kinds]]). "Every
   object falls alike" is a hypothesis the world can refute (feathers), which today's operator states by keying on nothing.
3. **Morphological features** — a feature carrying its own local frame would seed the pose from ONE fixation (Monty's actual
   path), instead of needing n. Not a prerequisite; a strict improvement in fixations-to-recognition.
4. **Sensor noise** — the deferred home for: tolerating k contradictions (a likelihood model), the Karcher/SVD rotation mean
   (vs today's chordal Gram-Schmidt projection), and the lever-arm precision law (pose error ≈ position noise / object radius).

## Deferred (noted, not invented)
- **The LOCATION UNION for GENERAL rotation.** R3's crown test uses CONTROLLED entry (a consistent anchor point), which needs
  no union. "Rotated about any point, entered anywhere" needs the union of candidate current-locations (which marginalises the
  centre — see above) + pose belief as a population. This is the same "union/evidence pooler" upgrade flagged for ambiguous
  objects — do it once, it serves both. **R4, and now the top of the queue.**
- ~~**Continuous orientation** (interpolate between the N discrete buffer positions)~~ — **RESOLVED by the cut-over**, not by
  interpolation: there are no buffer positions to interpolate between. Any ω is representable; sampling is a free choice, so
  coarse-to-fine refinement needs no substrate change. What *bounds* useful sampling is the lever-arm resolution above.
- **Morphological (oriented) features.** NOT a prerequisite for the scan (colour-at-location + the location scan suffices —
  correcting an earlier claim). Oriented features would SEED the pose from the first sensation (fewer ω to scan) and are the
  general case — later.
- **3D (SO(3))** — the STATE already generalises (a 3-DOF rotation is just state; this was design A's whole argument), but the
  SCAN goes cubic ⇒ needs R4's evidence/population path. Monty's multi-hypothesis territory.
- **Coupling to the emergent onset** — first encounter defines the canonical ω=0; later encounters scan. Decide whether
  `start_object` also resets orientation.

## Does this need L5? — NO (and where the boundary is)
*(Written pre-cut-over; the CONCLUSION survives it unchanged — rotation still lives entirely in L6a's state + the L2/3 scan.
Read "the rotation operator / circular buffer" below as "rotating the continuous pose".)*
R1–R4 need only: the location frame (encoder/L6a), rotation (applied to L6a's state), and the scan + union (L2/3 pooler). No
L5 layer is built. Why: the 2021 mechanism modifies **grid path integration** (*location + movement → location*, L6a +
operator) — it changes how a MOVEMENT maps into the location code. **Displacement cells (L5) are the INVERSE** (*location +
location → the movement/relation*), used for object COMPOSITION; rotation never asks that. (ARCHITECTURE §8 nominally assigns
the operator to "L6a and L5", but the L5 half is the displacement CONTENT, which rotation does not touch.) L5 becomes required
only for: (a) the Route-1 complement
(rotation-invariant identity via displacement MAGNITUDES); (b) object COMPOSITION (sub-objects at relative displacements);
(c) **ACTIVE disambiguation** — when the scan TIES on a symmetric object, the smart move is to MOVE to a tie-breaking
location (L5PT motor + goal generation). This plan handles symmetry PASSIVELY (hold the ambiguity as a population), so (c) is
the natural sequel, not a prerequisite.

## Hard rules
- No symbolic machinery: the rotation is a group action on L6a's state; the pose is INFERRED by a scan ranked on the model's
  own prediction fit — never hand-coded rotation logic, never a per-object angle table. Reuse `GridEncoder` +
  `MotionOperator` + the pooler union.
- Verify each step's mechanism against the 2021 paper (feedback_check_tbt_accuracy_per_step) before building; correct this doc
  first if it drifts. **This bit us productively:** the 2021 circular buffer IS the faithful reading of that paper, and we
  built it — but the paper's mechanism is DISCRETE-rotation only (its own stated limitation), so fidelity had to yield to the
  measured requirement (arbitrary angles, unbounded object size). Faithful-to-the-paper ≠ right-for-the-target.
- Keep the suite green throughout; each R-slice is a vertical slice driven by `agent.py` and exercised by a test.
