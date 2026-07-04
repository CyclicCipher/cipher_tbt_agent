# HIPPOCAMPUS — the allocentric map + the egocentric↔allocentric gain field

*Design + build plan. Rewritten 2026-07-03 (branch `p4-collapse`) to the SDR model; supersedes the 2026-06-29 draft
(which predated the P0–P3 SDR collapse and referenced now-deleted files: `hippocampus.py`, `l6_grid.py`, `sensor._delta`,
`TARGET_ARCHITECTURE.md`). Companion to `ARCHITECTURE.md` (§8 "from the vector to the action", §10 P4(a)). Grounds
`reference_hippocampus`, `reference_tbt_frames_and_hippocampus`, `project_marker_exposes_hippocampus_prereq`.*

## 0. What this module is, and why it is not the column

A cortical column's reference frames are **egocentric / object-centred** — a frame attached to the thing being sensed. An
**allocentric, world-anchored** map is, by necessity, the **hippocampal formation** (place cells, entorhinal grid cells,
head-direction cells, the retrosplenial transform). Anything else that builds a world map is a **parallel system** (rule 1).

This document exists because the P4 location-collapse (`ARCHITECTURE.md` §10 P4(a)) tried to build the allocentric pose
**inside `CorticalColumn`** — a conjunctive `position ⊗ heading` SDR path-integrated by a per-heading operator. That is the
misplaced parallel system. The allocentric machinery is extracted here into a **hippocampus module**, separate from the
column, which the column reads (a top-down location prior) and writes (recognition + efference).

## 1. The mechanism (researched 2026-07-03 — the constraints are firm)

Three populations linked by one gain field:

1. **Head-direction ring attractor.** A bump on a ring, updated by **angular velocity** carried by the **turn's efference
   copy** (+ vestibular/proprioception). A turn shifts the bump by the turn angle — *angular path integration*. A ring
   attractor with asymmetric velocity-gated connectivity; the substrate is **conjunctive head-direction × angular-velocity
   cells** (Ajabi et al. 2023; "The Brain Compass", Page & Jeffery 2018; Sargolini 2006). **Separate** from position.
2. **Allocentric position** (entorhinal **grid** = metric ruler, hippocampal **place** = location). Path-integrated by
   self-motion; **accumulates drift unless reset by egocentric sensing** — re-seeing a landmark corrects it (loop closure).
3. **The gain-field coupling (retrosplenial).** Head-direction **gain-modulates** the transform between the egocentric
   (parietal) and allocentric (medial-temporal) frames — one machinery run **both ways**:
   - **IN (path integration):** rotate the **one** egocentric self-motion displacement **by** head-direction, then add to
     allocentric position. FORWARD is *one* displacement rotated — **not** a separate operator per heading.
   - **OUT (navigation):** rotate the allocentric goal vector **by −(head-direction)** into the egocentric frame; the motor
     picks **TURN** (goal not ahead) or **FORWARD** (goal ahead). Reorient-then-advance falls out of the transform.
   (Bicanski & Burgess 2018; Byrne, Becker & Burgess 2007; Hasselmo et al. 2023.) Dimension-general: an SO(3)
   head-direction rotates a 3-D displacement — **no centre of rotation** (`reference_operator_as_group_representation`).
4. **Reafference self/world split.** The efference copy predicts the self-induced global flow; **subtract it** (flow
   parsing, Rushton & Warren) — the **residual is world-motion** (exafference). This is how "did *I* move or did the
   *object* move" is answered — the same efference-copy comparison, not a largest-component heuristic (§4c of ARCHITECTURE).

The one thing the literature leaves genuinely open (the reviews call it "largely the realm of theory") is the **exact
learned form** of the gain field — settled in implementation, constrained by the above.

## 2. Why now — the gain field dissolves two P4 blockers (2026-07-03 finding)

Building (a) inside the column with a conjunctive per-heading operator worked mechanically (OrientationGame 4/8) but forced:
- **a directed explorer** to sample FORWARD in *every* heading (each `dp[heading]` learned independently), and
- **a depth-2 rollout** to reorient toward the goal.

Both are **symptoms of the missing gain field**. With it: **one** forward observation (in any heading) + the HD ring gives
every heading for free (coverage problem gone); and OUT-navigation is the inverse transform, not a rollout (myopia gone).
So the conjunctive-per-heading operator is itself a leaked parallel to retire; this module replaces it.

## 3. Reuse — most pieces already exist (assemble, do not reinvent)

- **HD ring** = `encoders.ScalarEncoder(periodic=True)` bump; the **turn operator** already learns the ring shift `dh`
  (`operator.ModularOperator`) — that piece is correct; keep it as the angular-velocity update.
- **Allocentric grid/place SDR** = `encoders.GridEncoder` (multi-scale rings = grid modules); value over it =
  `l6_sr.SuccessorFeatures` (`V=w·ψ`, generalises across overlap).
- **The one egocentric displacement** = the learned operator (`operator.py`) — but **one** vector, rotated by HD, not a
  per-(action,heading) table.
- **Loop closure / landmark reset** = `l23_object` recognition → identify a stable object → correct the drifting place.
- **Reafference** = the efference copy already threaded (`arc_sdk` `_last_a` → `sensor.read` → `column.perceive`); add the
  flow-cancellation (subtract the efference-predicted global shift; residual = world-motion) that today's largest-component
  heuristic stands in for.

## 4. Build plan (each stage suite-green; fast offline reproductions; NavGame → OrientationGame the gate)

- **H1 — HD ring + allocentric grid as SEPARATE codes.** Split the conjunctive `pos⊗head` belief into a grid SDR (position)
  and a periodic HD ring; the turn updates the ring (angular velocity), sensing corrects both. NavGame (one heading) is the
  degenerate case; suite green.
- **H2 — the gain-field coupling (IN).** Path integration = `position += R(head-direction)·(one egocentric displacement)`.
  Learn the *single* forward displacement once; derive every heading by rotation. Retire the per-heading `dp` coverage.
  Gate: OrientationGame path-integration faithful after FORWARD is seen in **one** heading only.
- **H3 — the gain-field coupling (OUT) = the navigator.** Action = rotate the goal vector by −head-direction → TURN if not
  ahead, else FORWARD. Retire the depth-2 rollout. Gate: OrientationGame solves; NavGame stays 8/8 (translation = the
  identity-heading special case).
- **H4 — reafference self/world split.** Cancel the efference-predicted global flow; the residual is world-motion. Retire
  the largest-component "self" heuristic (`sensor._mover_cloud`). Gate: a moving distractor object does not corrupt the
  agent's own path integration.
- **H5 — landmark anchoring / loop closure.** Re-seeing a recognised object resets the drifting place, so the map stays
  world-anchored over a long run.

## 5. Honest risks / open

- **The learned gain-field form.** The exact online rule that ties the one displacement to the HD rotation is the open
  piece; keep it constrained (a rotation acting on one displacement), not a free per-heading table (that *is* the thing
  being retired).
- **Single organ vs per-column.** TBT says each column replicates grid machinery for its *object* frames; the *global*
  allocentric map binding columns is the hippocampal organ. For our single/few-column agent, this module IS that map;
  cross-column binding (many frames → one world) is the deferred hard part.
- **Module boundary (rule 1/3).** The hippocampus must be a real module the column reads/writes — not a second copy of L6.
  The column's L6 stays its *local* object-centric frame; the hippocampus is the *global* one, broadcast as a top-down
  location prior (the `thalamus.read_location` channel already exists — reuse it, do not invent a fabric).
