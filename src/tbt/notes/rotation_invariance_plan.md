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

**DECISION PROCEDURE (do the cheap falsifier FIRST).** Run the **3b drift experiment**: long path-integration on the oriented
grid, measuring whether the multi-scale co-prime code's **error correction** (Sreenivasan & Fiete redundancy) BOUNDS the
per-step rounding drift or lets it accumulate. If bounded → binary survives in L6a, we keep the permutation operator, and
none of the above is needed. If it accumulates → the phase-code study is forced, and R4+ should be planned on that basis.

## Status: R1 ✅ R2 ✅ R3 ✅ (2026-07-14) — `test_oriented_grid`, `test_rotation_operator`, `test_rotation_recognition`
R3 result: an asymmetric object learned at ω=0 is recognised at ALL 8 novel orientations with the exact pose reported; a
4-fold symmetric object is recognised with ties `{0,2,4,6}` = its symmetry orbit (pose undetermined up to 90°, not forced).

## NEXT (in order)
1. **The 3b DRIFT EXPERIMENT — the gating falsifier.** Long path-integration on the oriented grid; does multi-scale
   error correction BOUND the rounding drift or does it accumulate? Cheap, and it decides the substrate question above
   before any more machinery is built on the answer.
2. **Then either:** R4 (the union pooler → general "entered anywhere" rotation) on the current binary substrate if drift is
   bounded — or the **phase-code study** (FPE/SSP; the L4-interface seam) if it is not.

## Deferred (noted, not invented)
- **The LOCATION UNION for GENERAL rotation.** R3's crown test uses CONTROLLED entry (a consistent anchor point), which needs
  no union. "Rotated about any point, entered anywhere" needs the union of candidate current-locations (which marginalises the
  centre — see above) + pose belief as a population. This is the same "union/evidence pooler" upgrade flagged for ambiguous
  objects — do it once, it serves both. R4.
- **Continuous orientation** (interpolate between the N discrete buffer positions) — R3 is discrete-N first.
- **Morphological (oriented) features.** NOT a prerequisite for the scan (colour-at-location + the location scan suffices —
  correcting an earlier claim). Oriented features would SEED the pose from the first sensation (fewer ω to scan) and are the
  general case — later.
- **3D (SO(3))** — no single ring; Monty's multi-hypothesis territory.
- **Coupling to the emergent onset** — first encounter defines the canonical ω=0; later encounters scan. Decide whether
  `start_object` also resets orientation.

## Does this need L5? — NO (and where the boundary is)
R1–R4 need only: the multi-orientation grid (encoder/L6a), the rotation operator (EXTENDS the existing L6a path-integration
operator), and the scan + union (L2/3 pooler). No L5 layer is built. Why: the 2021 mechanism modifies **grid path
integration** (*location + movement → location*, L6a + operator) — the circular buffer changes how a MOVEMENT maps into bump
shifts. **Displacement cells (L5) are the INVERSE** (*location + location → the movement/relation*), used for object
COMPOSITION; rotation never asks that. (ARCHITECTURE §8 nominally assigns the operator to "L6a and L5", but the L5 half is the
displacement CONTENT, which rotation does not touch.) L5 becomes required only for: (a) the Route-1 complement
(rotation-invariant identity via displacement MAGNITUDES); (b) object COMPOSITION (sub-objects at relative displacements);
(c) **ACTIVE disambiguation** — when the scan TIES on a symmetric object, the smart move is to MOVE to a tie-breaking
location (L5PT motor + goal generation). This plan handles symmetry PASSIVELY (hold the ambiguity as a population), so (c) is
the natural sequel, not a prerequisite.

## Hard rules
- No symbolic machinery: the rotation IS the operator (a shift); the pose is a cross-correlation (a scan), not hand-coded
  rotation logic or a per-object angle table. Reuse `GridEncoder` + `ModularOperator` + the pooler union.
- Verify each step's mechanism against the 2021 paper (feedback_check_tbt_accuracy_per_step) before building; correct this doc
  first if it drifts.
- Keep the suite green throughout; each R-slice is a vertical slice driven by `agent.py` and exercised by a test.
