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

## NEXT (in order)
1. **R4 — the union/evidence pooler** on the continuous substrate: general rotation ("entered anywhere" — the replay currently
   assumes the sweep shares an anchor with the learned object), ambiguity, and symmetry-as-POPULATION rather than a tie list.
2. **SO(3).** The state generalises unchanged (a 3-DOF rotation is still just state), but a naive scan goes **cubic** — so R4's
   evidence path is the prerequisite, exactly as the bake-off predicted.

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
