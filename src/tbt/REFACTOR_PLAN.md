# TBT folder refactor — making the layers do their real TBT jobs

*Draft for review, 2026-06-29. Companion to `TARGET_ARCHITECTURE.md` (the north star) and the memory
`reference_tbt_layers_4_23.md` (the researched cell/structure facts this plan implements). Nothing here is
executed until signed off. The governing constraint: **ONE model, no parallel systems** — every layer is a
view of the same column; the refactor removes a parallel object-memory, it must not add one.*

> **Structural principle (governs every phase).** `column.py` is a **container for the layers and a
> coordinator of information flow** — between the layers, and out to the thalamus / other columns. It does
> **not** hold functionality that belongs to a layer. The test: **math or state → a layer; routing between
> layers / the inter-column interface → the column.** This means bare math now sitting in `column.py`
> (`_sr_frame`, `_sparsify_topk`, the `loc_*` path-integration belief, the predict-then-compare) moves *into*
> the owning layer (L6 / L4); the column only wires sense → L4 binds feature at L6 location → L2/3 accumulates
> evidence → returns id, and exposes `content_code`/`place_code` to the thalamus.

---

## 0. Why a refactor (the incoherence, measured from the code)

The column composes four layers, but the **live agent uses one of them.** `agent.py` calls exactly
`col.observe`, `col.predict`, `col.motor`, `col.graph` — all of which resolve to **L5** (the per-action
operator + its edge graph). Concretely:

- **L6 is updated but never read.** `col.observe` runs `self.sr.observe(...)` (OnlineSR / TD-SR) every step,
  but `col.predict` only calls `L5.predict`, and the value planner (`agent._transitions`) reads `col.graph`
  (= `L5.edges`). The successor representation — the location frame, the whole point of L6 — feeds nothing.
- **L4 / L2/3 are dormant live.** `L4.bind` / `L23.pool` / `S` are touched only by the offline VSA path
  (`learn_domain` / `consolidate` / `refresh`), which **nothing in `src/` calls** (no test, no `arc_sdk`).
  The live state is a raw egocentric patch tuple — not a feature-at-location, no identity, no pose.
- **Recognition is a side-library, run from scratch every step.** `recognize.py`'s `Recognizer` keeps a
  PARALLEL object library (`self.models`), used only by `arc_sdk._object_barriers` to label walls — and
  `recognize_object` re-recognises the whole shape each call (no persistence). This is both the "two object
  memories" incoherence the audit found AND the per-step slowness the user hit live.

So the redesign's thesis (from the neuroscience): **an object model is not a library — it is a graph
distributed across the layers.** L6 holds the locations, L4 the features at them, L5 the displacements
between them, L2/3 the pooled identity + the evidence that settles it. Rebuilding the layers IS dissolving
`recognize.py` into them, and wiring the live agent to the result.

---

## 1. The layer contract (the neuroscience is the spec)

For each layer: the **cell type**, the **structure it forms**, and the **API that follows from it**. Sources:
Lewis 2019 (L6/L4), Hawkins 2019 *Framework…Grid Cells* (L5 displacement, L2/3 object), Numenta 2017 +
Monty/TBP 2024 (predictive L4, lateral voting, evidence accumulation). All recorded in
`reference_tbt_layers_4_23.md`.

### L6 — grid cells: the reference frame itself
- **Cell/structure:** grid-cell modules = an **object-centric LOCATION**, path-integrated by movement (a
  per-module transform `M_i`: the activity bump shifts with the sensor). L6 *forms the metric/space* — the
  "where." It is the substrate every other layer indexes into.
- **The honest two-frames point.** "Location frame" is used at two levels, and we must not conflate them:
  - **Object metric frame** (sensor level): a continuous 2-D Euclidean frame *per object*, with **pose
    `(θ, t)`** — this is what recognition/rotation/voting need. Today this lives *only inside* `recognize.py`.
  - **Navigational / relational SR frame** (scene level): the **topology-general SR-eigenframe** over the
    transition graph (our `OnlineSR`) — reachability + value, grid-like on metric space, correct on
    rings/trees. This is our genuine advantage over Monty (it navigates abstract spaces).
  These are **two instances of one mechanism at two hierarchy levels** (object-id-at-pose becomes a *feature*
  in the higher frame), not two systems. The refactor makes the **object metric frame a first-class L6 mode**
  (it is currently smuggled inside the recogniser), and keeps `OnlineSR` as the navigational frame — and
  finally **reads it** (NEED / place-code similarity in `reward._need`, which is currently stubbed to 1.0).
- **API:** `loc_reset/move/sense/where` (have, discrete graph path-integration) + a metric-frame read for
  recognition (new: the fovea location *in the object's frame*, supplied to L4/L5/L2-3). `l6_grid.py`
  (the innate hex prior, off) stays shelved; `OnlineSR` stays as the navigational frame.

### L4 — feature-at-location: inherits L6's structure, forms none of its own
- **Cell/structure:** L4 binds a **sensed feature** (driver) to the **L6 location** (modulatory/predictive):
  the location *depolarises* the L4 cells that should fire there, the sensed feature confirms which wins. L4 =
  "what, at the where L6 supplies." It builds **no locations of its own.**
- **What this fixes:** the live state stops being a raw 7×7 patch tuple and becomes a **(feature-id, location)
  pair**. The feature-id is a **label-free, online-grown code for the local patch** — exactly what
  `retina.Retina.feature()` already does and what L4's sparse codebook `E` is for. So **L4 owns the patch→id
  vocabulary** (absorb `retina.feature`'s codebook) and the feature-at-location bind/readout.
- **API (mostly have, re-seated):** `bind(feature, place)` and `readout(S, place)` stay (sparse SDM codes —
  the capacity argument is sound). **Add** `encode(patch) -> feature_id` (the label-free codebook, moved in
  from the retina) and make the *prediction* explicit: `predict_feature(place, object) -> feature` (given
  where I am + the active L2/3 object, what do I expect to sense) — this is where the column's
  predict-then-compare actually belongs (today it lives only in `agent.py` over opaque states).

### L5 — displacement cells: the difference operator over L6's space (NOT a duplicate)
- **Cell/structure:** thick-tufted cells = **displacements** = the vector *between two L6 locations*
  (`location + location → movement`), complementary to grid (`location + movement → location`). Modular like
  grid but for relative vectors. The **same cells are motor output AND the compositional object sent to higher
  regions AND the thalamic driver.**
- **Status:** mostly built — `disp`/`recolor` (KIND-general delta), `edges` (exceptions), `motor`, `driver`.
  The **gap the research exposes:** the **pose/rotation displacement** lives in `recognize.py`
  (`rot`, `align_rotations`, `ObjectModel.cells_at = R(θ)·model + t`). That is displacement-cell math —
  *the rotation that aligns one local patch's neighbour-vectors onto another's*. It belongs in **L5**.
- **API:** keep `observe/predict/successors/motor/driver/disp/recolor`. **Add** the pose operators
  `rotation_between(model_disps, sensed_disps) -> [θ]` (was `align_rotations`), `apply_pose(cloud, θ, t)`
  (was `cells_at`), `local_disps(...)` (the patch's neighbour-vectors = the sensed "feature pose"). Retire the
  archived matrix `learn`/`apply` (offline-only; unused live) unless a demo needs it.

### L2/3 — object identity: the graph-memory, settled by recurrence + lateral voting
- **Cell/structure:** a **stable sparse code = the object**, unchanged as the sensor moves, **pooling** over
  the L4 feature-at-location sequence, **settled by recurrent self-bias + LATERAL (inter-column) voting.** It
  forms the *identity manifold*; it holds **no metric frame of its own** (that is L6/L5's job).
- **This is where `recognize.py` dies and is reborn distributed.** The Monty mapping:
  - `ObjectModel` (a stored graph: node locations + each node's local patch + signature) **= an L2/3
    graph-memory entry** — but its *locations* are L6, its *features* are L4, its *edges* are L5 displacements.
  - `Recognizer.models` (the library) **= L2/3's set of learned objects** (the parallel library, now seated
    where objects belong).
  - `Recognizer.observe/best` (the **incremental** `(object, pose)` evidence loop) **= L2/3's recognition**,
    and it becomes **persistent across steps** (carry the hypothesis set; accumulate; never recompute from
    scratch) — this is the fix to the live slowness.
  - `recognize.vote` (pose-aware pooling by world-pose) **= L2/3's `vote`** (replacing the
    `NotImplementedError`) — the **CMP lateral channel**.
- **The unification (resolves design-Q1).** Don't choose "graph-memory vs VSA superposition" — **use both, one
  per role**: the **graph** (nodes/edges/poses) is the object's *structure* (for recognition + pose); each
  object's *content* (feature-at-location) is stored via **L4 `bind`/`pool`** — i.e. the old `L23.S`
  superposition becomes the **per-object** content store inside a graph-memory entry, read back by
  `L4.readout`. `recognize.models` and `L23.S` were the two object memories; **they become one** —
  the graph-memory entry, structure + content together. No parallel system.
- **API (rewrite):** `learn(object) / recognise(sensation_stream) -> (id, pose, evidence) / observe(loc,
  feature) / best() / vote(neighbours)`. The old `pool`/`revise` stay as the *within-object* content
  mechanism (no longer a global blob).

### Inter-column interactions (the directive's "don't forget")
- **Lateral voting (CMP) — buildable NOW, in the object metric frame.** Columns sensing different parts of the
  same object independently solve the **same world pose `(θ, t)`**; pooling hypotheses by world-pose IS the
  consensus (this is exactly `recognize.vote`, and it works because the *object* frame is shared Euclidean).
  Seat it in `L2/3.vote`. This is genuinely *not* deferred — it is the shared-frame case.
- **Thalamic driver / binding — have.** `L5.driver` emits the feed-forward "this changed by that" message;
  `thalamus.bind/read` does cross-column VSA conjunction (content⊗location). Keep; wire to L4/L5 cleanly.
- **DEFERRED, honestly:** voting across columns that learned **different SR navigational frames** needs
  learned cross-frame registration (TARGET_ARCHITECTURE §"Inter-column messaging"). Object-recognition voting
  does **not** hit this (shared metric frame); scene/navigational voting does. We deliver the first, defer the
  second, and say so.

---

## 2. File-by-file disposition

| File | Disposition |
|---|---|
| `l6_sr.py` (`OnlineSR`) | **Keep.** The navigational frame. Newly **read** by `reward._need` (place-code similarity) so L6 stops being write-only. |
| `l6_grid.py` (innate hex) | **Keep, shelved** (off). Unused live; the SR is the substrate. |
| `l4_feature_location.py` | **Extend.** Absorb the retina patch→id codebook (`encode`); add `predict_feature`. Keep sparse `bind`/`readout`. |
| `l5_displacement.py` | **Extend.** Add the pose operators (`rotation_between`/`apply_pose`/`local_disps`) moved from `recognize.py`. Retire the archived matrix `learn`/`apply` if no demo needs them. |
| `l23_object.py` | **Rewrite.** Becomes the graph-memory + persistent incremental `(object,pose)` evidence + `vote` (CMP). Absorbs `recognize.py`. Keep `pool`/`revise` as within-object content. |
| `recognize.py` | **DELETE** (the directive). Its math → L5; its library + evidence + vote → L2/3. `test_recognize.py` → `test_l23.py`, re-pointed at the column faculty. |
| `recurrence.py` | **Delete** (superseded online per TARGET_ARCHITECTURE; unused in `src/`). Belief = predict↔correct evidence in the column. |
| `column.py` | **Slim to a coordinator** (the structural principle above). Move `_sr_frame`/`_sparsify_topk` → L6, the `loc_*` path-integration belief → L6, the predict-then-compare → L4; delegate recognition to L2/3 (persistent). **Delete** the offline VSA path (decision 1); **archive** conditional-dynamics, its purpose folding into L5 (decision 2). |
| `retina.py` | **Fold `feature()`/codebook into L4**; keep `salient_cells`/`dominant_region` (the change/fovea primitives the sensor uses). |
| `sensor.py` | **Re-seat.** Emit **(feature-id, location)** to the column (feature via L4.encode, location via L6 path-integration) instead of a raw patch tuple. Keep the 7b/7c recurrence+navigation fixes intact. |
| `agent.py` | **Thin re-wire.** State becomes feature-at-location; `predict`/compare routes through L4/L2-3; otherwise unchanged (the loop is sound). |
| `arc_sdk.py` | **Simplify the barrier seam.** `_object_barriers` uses L2/3 **persistent** recognition (recognise-once-carry), killing the per-step from-scratch cost. |
| `behavior.py` | **Keep.** `ObjectBehaviour` (revisable barrier-ness keyed on recognised identity) is correct; it just consumes L2/3 ids now. |
| `perceive.py` (`ObjectField`) | **Demote to a sensor organ** (Phase 4, see below). Keep `background`/`segment`/`components`/`content_sig` + the change channel as a cheap *revisable* proto-object proposal; **move** stable-id permanence, `max_jump` matching, and contact-splitting **into the column** (recognition + path-integration). It is currently hand-coded cognition (`reference_tbt_segmentation_and_grouping`). |
| `thalamus.py`, `basal_ganglia.py`, `reward.py` | **Keep.** `reward._need` gains the SR read. |
| `factorize.py`, `residual.py` | **Keep for now** (residual's `_find_predicate` backs conditional-dynamics). Dissolution into L6/L5 is a *later* step, not this refactor. |

---

## 3. Execution order (each phase ends suite-green; debug on FAST OFFLINE reproductions, never longer live runs)

**Phase 1 — L5 pose operators. ✅ DONE 2026-06-29.** Moved `rot`/`local_disps`/`align_rotations`/`cells_at`
geometry into `l5_displacement.py`, **shaped as group-actions** (`pose_between`/`apply_pose`/`local_disps`,
SO(2) the spatial plug-in, abstract-domain note in-code) and exposed as L5's API (staticmethods).
`recognize.py` now imports them from L5 (one home, no second copy); `ObjectModel.cells_at` → `apply_pose`.
Added 4 direct L5 pose tests. *Gate met:* suite 49→**53 green**, recognise tests unchanged.

**Phase 2 — L4 feature-at-location. ✅ DONE 2026-06-29.** L4 now owns the label-free content codebook
(`encode`, grows past capacity — no hard wall), the rotation-INVARIANT feature descriptor (`invariant_sig`,
moved from `recognize.py` — the ventral 'what', complementary to L5's equivariant `local_disps`), and
`predict_feature(S, place)` (the predict half of predict-then-compare, seated where the feature lives).
`recognize.py` imports `invariant_sig` from L4. New `test_l4_feature_location.py` (6 tests). *Gate met:* suite
53→**59 green**. (NB the retina's `feature()` codebook is now redundant — it gets retired when the sensor routes
through `L4.encode` in Phase 4.)

**Phase 3 — L2/3 graph-memory + persistent evidence + vote (the library discard). ✅ DONE 2026-06-29.**
Rewrote `l23_object.py` as the object/identity layer: `ObjectGraph` (was `ObjectModel`) + `objects`
graph-memory (was `Recognizer.models`) + the **persistent** incremental evidence session
(`start`/`sense`/`best`, never recomputed) + one-shot `identify`/`recognize` + `vote` (CMP, method + module fn).
Kept the legacy VSA `S`/`pool`/`revise` as the within-object content store. `column` drops the `Recognizer`
member and **routes** `learn_object`/`recognize_object`/`identify_object` to L2/3 (coordinator, not worker).
**`recognize.py` DELETED**; `test_recognize.py` → `test_l23_object.py` (+ a `vote`-method test); `test_behavior.py`
re-pointed at `L23_Object`. *Gate met:* suite 59→**60 green**. (NB the live loop still calls `recognize_object`
one-shot per object/step — the persistence API exists but the slowness fix lands when Phase 4 gives each tracked
object its own persistent session.)

**Phase 4 — demote `ObjectField` to a pure sensor organ; the object becomes a COLUMN construct.** (Re-scoped
2026-06-29 after the segmentation research — `reference_tbt_segmentation_and_grouping`. The original "wire
feature-at-location + per-object sessions" was building on sand: `ObjectField` is a hand-coded tracker doing the
column's job — stable-id permanence, `max_jump` nearest-matching, contact-splitting. TBT's principle: **the
object is owned by the column's recognition, not the sensor** — boundaries emerge from prediction MISMATCH, not
a segmenter.) **The key refinement (from the proto-object research, `reference_tbt_segmentation_and_grouping`):
the proto-object PROPOSAL stays — it is biologically real (Rensink's volatile, pre-attentive, parallel
proto-objects; the brain HAS this stage, Monty omits it) — but it must be STATELESS. The harness-ery is the
MEMORY we bolted on (`_last`/ids/`max_jump`/contact-split = attention binding proto-objects across time = the
COLUMN's job). So: keep the stateless proposal, move the binding/permanence/boundaries to the column.** Staged:
- **4a — the stateless sensor floor + feature-at-location.**
  - **✅ The feature-at-location seam — DONE 2026-06-29.** The egocentric patch is now encoded through the
    column's `L4.encode` (injected via `Sensor(encode=...)`, wired in `arc_sdk` before the first read), so the
    live state is **`(feature_id, position)`** — L4's job, not a raw pixel tuple. The relabeling is injective →
    the transition graph is isomorphic → behaviour preserved. *Gate met:* suite **60→61 green**; offline on real
    cn04 frames: states are feature-at-location, L4 codebook grew to 44, and raw-vs-encoded over the same action
    sequence gave **identical** distinct-state counts (66=66) — recurrence untouched.
  - **✅ Stateless colour-aware proposal + tracker removed — DONE 2026-06-29 (the binding surgery).**
- **✅ 4b — objecthood moves into the column — DONE 2026-06-29.** The surgery turned out to be mostly
  **deletion**: the hand-coded tracker was providing permanence that **recognition (identity) + the fovea
  path-integration (the controllable object) already provide**, so removing it *is* moving objecthood to the
  column. Concretely: `segment` is now **colour-aware** (a colour boundary = a candidate border — which is what
  lets contact-split go: a mover bumping a different-colour wall no longer merges); `ObjectField` is a **stateless
  proto-object proposer** (no `_last`/ids/`max_jump`/contact-split/`predict`/`_dist`); the barrier faculty caches
  recognition by **shape signature** so a recurring shape is recognised **once** (the O(carry) fix). The `predict`
  hook turned out **obsolete**, not something to drive — no consumer needs cross-frame id stability (config_state
  snapshots, click-slots sort by size/pose, barriers key on recognition). *Gate met:* suite **61 green**; offline:
  barrier game 2143 steps → **1** recognize call (was ~1/step), 6/6 solved, 1 bump; cn04 still feature-at-location.
  **Deferred:** recognition-driven split/merge of *multi-colour* objects (colour-aware over-segments them) reduces
  to the **compositional hierarchy** (object-id-as-feature, TARGET_ARCHITECTURE step 8) — not needed for the
  demotion to be correct/green, and the failure mode (over-segment) is milder + recoverable vs the old merge.

**Decisions — RESOLVED (2026-06-29, neuroscience-backed):** (i) **colour** — *colour-aware* volatile proposal
(borders from contrast incl. colour; good for ARC) feeding recognition where colour is a *weak asymmetric L4
feature, never identity/role*; the two stages don't conflict, revisability is the safety net. (ii) **top-down /
4c** — folded INTO 4b (one mechanism), done now, not deferred; it is slightly beyond Monty but squarely
brain-faithful (border-ownership + feedback).

**Phase 5 — cleanup + docs.** Delete `recurrence.py`; resolve §4 decisions; update `TARGET_ARCHITECTURE.md`
(build-order 1 "recognition" → done-distributed; the two-frames reconciliation) and the memories
(`reference_tbt_layers_4_23`, `project_neocortex_missing_components`). *Gate:* green.

Inter-column voting lands in **Phase 3** (shared object frame). SR-navigational cross-frame voting stays
deferred. A single live run is reserved for **measuring** the end result, once, at a fixed budget.

---

## 4. Decisions — RESOLVED (2026-06-29)

1. **Delete the vestigial offline VSA path** (`learn_domain`, `consolidate` w/ `eigh`, `recall`, `infer`,
   `anchor`, `add`-arithmetic) from `src/tbt/column.py`. The arithmetic/language demos keep their own copy in
   `experiments/…/precursor/`. (Removes the "which object memory" confusion.)
2. **Archive the conditional-dynamics faculty** (`observe_effect`/`learn_dynamics`/`predict_effect`). Its
   *purpose* — predicting context-dependent change — **belongs in the layers, not a column faculty**: it is
   L5's edge-exceptions (the per-`(s,a)` override) + a learned **precondition** on the change. Pull the standalone
   methods out of the column; fold the function into L5 (and keep `residual._find_predicate` available to L5 for
   the precondition search). Not a from-scratch rebuild this refactor — the archive note + the L5 seam.
3. **Keep the explicit Euclidean `(θ, t)` object frame** for the visual-spatial column — **as a deliberate
   spatial *specialization*, not a universal claim** (see §6). The SR-eigenframe remains the general,
   topology-general construct; Euclidean is what it looks like on 2-D pixel space, which is what ARC is.

---

## 5. Risks / what this does and doesn't buy

- **Does:** one object memory (no parallel library); the live agent becomes an actual column (L4/L6 carry
  signal, not just L5); recognition becomes persistent (fixes the slowness); lateral voting exists.
- **Does not, yet:** complete a real ARC level (still the milestone everything is judged against — this
  refactor is *necessary structure*, not a guaranteed score), nor solve heterogeneous-frame navigational
  voting, nor fold factorize/residual into L6/L5.
- **Sharpest risk:** the live agent currently *works* (49/49, runs clean) on the L5-only path. Re-seating the
  state onto feature-at-location must preserve the 7b/7c recurrence+navigation gains. Phase 4's offline
  reproduction is the guard; if recurrence regresses, the feature-id granularity (patch→id) is the dial.

---

## 6. Neuroscience notes for later (cross these bridges when we reach them)

*Recorded because the Euclidean choice (decision 3) is a pragmatic specialization, and being careful now saves
a rebuild later. Durable copy in the memory `reference_tbt_frames_and_hippocampus`.*

### 6a. The Euclidean object frame is the *spatial special case*, not a universal TBT claim
- TBT asserts reference frames are universal across cortex (objects, concepts, math, language) but is
  **agnostic about the geometry** — grid-cell-like *location coding* everywhere, **not** 3-D Euclidean
  everywhere. Monty hard-codes 3-D Euclidean + SO(3); that is exactly why Monty does only physical objects.
- **Our SR-eigenframe is the more general construct** (Stachenfeld 2017: grid cells = SR eigenvectors).
  Euclidean/hexagonal *falls out* of the SR on open 2-D space; on a ring/tree/relational graph the same SR
  gives the correct **non-Euclidean** frame. So we use the general thing (SR) and note that ARC's 2-D pixel
  domain merely *looks* Euclidean — hence the explicit `(θ, t)` is an efficiency choice for the visual column.
- **Extension to abstract columns:** "pose" generalizes from rotation (SO(2), the spatial instance) to **a
  transformation in the domain's own learned symmetry group** (a permutation, a scaling, a relational
  re-anchoring). L5's displacement cells are already this general object — a displacement is "the relation
  between two locations in whatever frame," and `rotation_between`/`apply_pose` are the *spatial instance* of
  "re-express displacements under a group element." For an abstract column the group is **learned from the
  action-orbit structure** (the `factorize.py` direction), not assumed SO(2). **Design guard:** keep L5's pose
  API shaped as "apply a (learned) group element to displacements," so the spatial SO(2) version is a plug-in,
  not a hard floor.

### 6b. Do we need a hippocampus? Not yet — and we know exactly where one would become required
- In the brain, navigable maps — spatial **and** abstract — have a **hippocampal/entorhinal** seat (place+grid
  cells; the Tolman-Eichenbaum Machine unifies them and covers relational/abstract structure; Constantinescu
  2016 found grid-like codes for abstract concept space in human EC). TBT's bolder claim: the neocortex
  **replicated** the grid machinery **per column**, so cortex doesn't call the hippocampus for each frame.
- Standard division (complementary learning systems): **HPC = one global, allocentric, episodic, one-shot map**
  (binds many views into "where am I in the world", fast); **cortex = many local object/sensor-centric model
  frames** (slow, general). Same grid math, different scope/timescale.
- **Verdict for us: no separate hippocampus needed now.** Monty (TBP) has none and gets recognition + navigation
  from cortical LMs alone; and our **`OnlineSR` already plays the global-map role** (absorbed into L6's
  navigational frame). The refactor needs no hippocampus.
- **The two HPC functions we are approximating — the bridges to watch:**
  1. **One-shot episodic memory** (remember a *specific* layout instantly, pre-consolidation). Our online
     learning is fast but not single-shot-episodic. A game that needs verbatim recall of a past configuration
     would need this.
  2. **A single unified allocentric frame binding all columns' local frames** (loop closure; "here again from a
     new view"). **This is identical to the deferred cross-frame voting problem** — the hippocampus *is* the
     brain's cross-frame registration. So if heterogeneous-frame navigational voting becomes necessary, the
     principled implementation is a **hippocampus-like shared global frame every column registers into**, not
     pairwise learned column-to-column transforms. That is the moment a hippocampus stops being optional.
