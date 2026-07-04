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

## 1b. What the map stores at a place — a POINTER to the model, not the stimulus (researched 2026-07-03)

The hippocampus has TWO jobs: **self-localisation** (where am *I* — §1, the gain field, built) and the **allocentric object
map** (what is *where* in the world — not yet built). This section is about the second: what is stored at a place.

**A pointer (index), not the raw stimulus, and not the generic model.** *Hippocampal indexing theory* (Teyler & Rudy
2007): "the hippocampus itself does not contain the content of an experience but it does provide an **index** that allows
the content to be retrieved" — indices are "sets of **pointers** to cortically stored representations." The raw sensory
field and the object *model* live in **neocortex** (our COLUMN — L2/3's recognised "car"); the hippocampus stores, at each
allocentric place, a **pointer to the recognised object** bound to that place — the **what–where conjunction** (conjunctive
item–place coding, Komorowski 2009; place cells modulated by objects — "misplace cells"). A partial cue reactivates the
cortical model (pattern completion). Concretely this IS the `thalamus.bind` register **`R = Σ content ⊗ place`**: `content`
= a pointer to the recognised object (its L4/L23 code), `place` = the allocentric grid code. Indexing theory ≈ that register.

**Variation = the model/index split (Complementary Learning Systems; McClelland & O'Reilly).** The *generic* model — what a
car IS, its parts and invariances — is **neocortical** (slow, distributed, semantic). The *specific instance* — THIS car,
here, in this pose/state — is the **hippocampal index** (fast, sparse, **pattern-separated**, so this-car-here stays
distinct from that-car-there). So the instance's variation is stored as (a) a distinct pattern-separated index and (b) the
**pose/state bound to the object's frame** (the column solves the pose; the hippocampus binds object+pose at the place) —
never raw pixels, never a copy of the model.

**Why this is the right substrate (and a design law):** the object-pointer map **RECURS where the raw pixel frame churns** —
a car that reshapes or recolours is still "car-pointer at slot X", ONE stable state, while the 64×64 field changes every
step. So the hippocampal map stores **object-pointers-at-places, corrected by recognition — never raw stimuli** (that would
reintroduce the non-recurring per-pixel state, rule-5). This is the old H1 "allocentric object map", now with a mechanism.

**How the index is encoded as an SDR — identity vs features use TWO mechanisms (researched 2026-07-03).** The index is an
SDR (the brain's currency), and it is `content-pointer ⊗ place` — the two factors carry the two kinds of distinction:
- **Different FEATURES (red car vs blue car) = the CONTENT SDR, where OVERLAP = SIMILARITY.** The object code is a
  composite/conjunctive SDR ≈ the union of its feature SDRs (`shape("car") ⊕ colour("red")`): red- and blue-car SHARE the
  shape bits (→ "both cars") and DIFFER on the colour bits (distinguishable). Colour is a CONTINUOUS space → a per-channel
  `ScalarEncoder` (not a category — `encoders.py` already), so red overlaps orange more than blue (graded similarity). The
  binding problem (don't misbind red-car + blue-truck as red-truck + blue-car — "illusory conjunctions") is solved by
  binding features **through their shared place/token**, not a global feature pool.
- **Same-type IDENTITY (two IDENTICAL red cars) = the SPATIOTEMPORAL INDEX, NOT features.** Identical objects have identical
  content SDRs → features cannot individuate them; they are individuated by the **object file / token** addressed by its
  **place**, not any label (Kahneman & Treisman object files; Pylyshyn FINST — the identical-cans-on-a-shelf case). So
  **place is the individuator**, and the **dentate gyrus PATTERN-SEPARATES** — distinct instances get ORTHOGONAL sparse
  indices even when their content overlaps. The token persists as the object moves (spatiotemporal continuity = object
  permanence / tracking).

So in our terms the index SDR = `content ⊗ place`: `content` = the column's recognised object + features (overlap =
similarity), `place` = the `GridEncoder` code (individuation, pattern-separated). Two identical red cars = the SAME
content-pointer at two DIFFERENT places = two distinct tokens. (`reference_tbt_feature_definition` — colour is the
non-morphological, peripheral feature; this says how it and identity enter the index.)

## 1c. How the columns and the hippocampus communicate — entorhinal-IN, thalamus-OUT (restored 2026-07-03)

Two ASYMMETRIC routes (from `legacy docs/THALAMO_CORTICAL_ARCHITECTURE.md` §5/§6; the anatomy is `reference_hippocampus`):

- **IN (columns → hippocampus): the entorhinal gateway — cortico-cortical, NOT thalamic.** Cortex → entorhinal superficial
  → hippocampus → entorhinal deep → cortex is the binding loop; TBT says the cortex *replicated* the entorhinal grid into
  every column's L6, so columns are evolved-from-entorhinal and the hippocampus is the one structure that binds *across*
  them into a single world. The hippocampus reads each column's outputs DIRECTLY: **L2/3 recognition** (which object + its
  sensed pose — the landmark for loop closure / the reafference correction), the column's **egocentric position** estimate,
  and the **efference copy** (the last action — drives path integration). In our module: `observe(action, sensed_pos,
  sensed_head)` IS the entorhinal-IN.
- **OUT (hippocampus → columns): the thalamic context route.** Hippocampal context + head-direction reach cortex via the
  **nucleus reuniens** (HC↔mPFC) and the **anterior thalamic nuclei** (Papez: HC → mammillary → anterior thalamus →
  retrosplenial head-direction). The allocentric frame is broadcast back as a **top-down prior through `thalamus.py`** — and
  the channel already exists: **`read_location`** (content → location = "where is object X in the world" = the task-column-
  sets-a-goal-state-in-the-spatial-column loop). We REUSE the location channel, not invent a fabric. In our module:
  `here()` / `location_sdr()` IS the thalamus-OUT.

  ⚠ **`thalamus.py` is OLD (pre-SDR) — review before reusing.** It is torch-based, operates on dense place/content codes
  and node INDICES (`content_col.place_code(node)`, `L4.E @ …`), and predates the SDR encoders + successor features. Before
  wiring the OUT broadcast (and the H6 `content ⊗ place` object index) through it, re-vet it against the current SDR model —
  the `bind`/`read`/`read_location` SHAPE is right (it IS the index register), but the representation it binds must be the
  SDRs (`GridEncoder` place, the L4/L23 content SDR), not the old dense node-indexed codes. Single-column, OUT can be read
  directly meanwhile.

The reciprocal **control loop** (transthalamic, BG-gated): TOP-DOWN a goal-state (a target place/object) the column
vector-navigates to (MD-thalamus latches it as context); BOTTOM-UP the **achieved state** or **prediction-error/exafference**
updates the map (loop closure) and *learns* the dependency; LATERAL is L2/3 consensus voting (multi-column recognition, not
the control loop). **Single-column now:** the OUT broadcast can be read directly (the hippocampus location IS the belief);
the `thalamus.read_location` route earns its keep once we are genuinely multi-column.

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
- **The allocentric object map (what–where index)** = `thalamus.bind`'s register `R = Σ content ⊗ place` (§1b) — `content`
  = a POINTER to the object recognised by `l23_object` (its L4/L23 code), `place` = the `GridEncoder` code. Store pointers,
  never raw stimuli. (This is a SEPARATE hippocampal function from self-localisation; the module built so far does only the
  latter.)

## 4. Build plan (each stage suite-green; fast offline reproductions; NavGame → OrientationGame the gate)

*Self-localisation core (H1–H3) is BUILT in `hippocampus.py` (isolation-tested: one-observation generalisation, non-abelian
reorient-then-advance, abelian nav); the remaining work is WIRING it into the loop and then the object map + reafference.*

- **H1 — HD ring + allocentric grid as SEPARATE codes.** ✅ core built (`Hippocampus`: `grid` + `hd`). The turn updates the
  ring (angular velocity), sensing corrects both. NavGame (one heading) is the degenerate case.
- **H2 — the gain-field coupling (IN).** ✅ core built (`path_integrate`: `position += R(head)·ego`). One forward observation
  derives every heading by rotation (isolation-tested); retires the per-heading `dp` coverage on wiring.
- **H3 — the gain-field coupling (OUT) = the navigator.** ✅ core built (`navigate`: rotate the goal by −head → TURN/FORWARD;
  reorient-then-advance). Retires the depth-2 rollout on wiring.
- **H3w — WIRE into the loop.** Feed the hippocampus from `column.perceive`'s recognised pose + the efference (entorhinal-IN,
  §1c); read the belief back from `hippocampus.here()` (thalamus-OUT). Delete the conjunctive `_pose_code` + per-heading
  operator, the directed-coverage explorer, and the rollout. Gate: NavGame 8/8, OrientationGame solved, suite green.
- **H4 — reafference self/world split.** Cancel the efference-predicted global flow; the residual is world-motion. Retire
  the largest-component "self" heuristic (`sensor._mover_cloud`). Gate: a moving distractor object does not corrupt the
  agent's own path integration.
- **H5 — landmark anchoring / loop closure.** Re-seeing a recognised object resets the drifting place, so the map stays
  world-anchored over a long run.
- **H6 — the allocentric OBJECT map (what–where index, §1b).** Store, at each place, a POINTER to the object `l23_object`
  recognised (the `thalamus.bind` register `content ⊗ place`) — never raw stimuli. This is the map that RECURS where the
  pixel frame churns, and the substrate multi-object scenes / Sokoban plan over. Gate: an object that reshapes/recolours in
  place is ONE stable state at its slot; a moved object updates its slot.

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
