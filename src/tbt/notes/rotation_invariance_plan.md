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
