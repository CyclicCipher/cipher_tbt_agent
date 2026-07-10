# MICROCIRCUIT — the minicolumn-native cortical loop (L4 · L2/3 · L5)

*Companion to `ARCHITECTURE.md` (the single source of truth). The plan to build the canonical cortical microcircuit so the
column runs the **full TBT loop** — localization AND content-prediction — not just the localization half it runs today
(`column.py` audit, 2026-07-05). The **minicolumns×cells HTM temporal memory is L4 and L2/3** (the input + object layers);
**L5 is NOT that substrate** — it is the **displacement-cell / motor-output / thalamus-driver** layer (see §2; verified
against Hawkins et al. 2019). So "minicolumn-native" names the L4/L2-3 redesign; L5 stays the displacement/motor layer and
L6 the grid-cell location. Grounded in the M5 HTM temporal memory (`sequence.py`) + the minicolumn structure: Hawkins &
Ahmad 2016 (*Why Neurons Have Thousands of Synapses*); Numenta "Columns" papers (Hawkins, Ahmad & Cui 2017, *A Theory of How
Columns…*; Lewis et al. 2019, *Locations in the Neocortex*); the layer roles from Hawkins et al. 2019 (*A Framework for
Intelligence and Cortical Function Based on Grid Cells*). Obeys the five dev rules (ARCHITECTURE §7). It COMPLETES `ARCHITECTURE §10` P2 (the one
prediction, in its full content form) + wires P3 (sequence memory) into the layers + adds the inter-layer feedback P4
implies.*

---

## 1. What's wrong now (the diagnosis)

The live loop (verified against `column.perceive`/`forward`, 2026-07-05) closes a faithful **localization** loop but
NOT the **content/prediction** loop:

1. **L4 is a vestige.** Content comes from the retina's `view_sdr` (M3), not bound in L4 at the L6 location. The
   canonical "L4 binds the sensed feature at L6's location" (§4c) is not wired; `L4_FeatureLocation` (an int codebook)
   is unused in the loop and surfaced as dead code during the classical-geometry cleanup.
2. **The layers are three incompatible mechanisms, not one substrate.** L4 = int codebook; L2/3 = a point-cloud
   evidence recogniser (M4); L5 = float efference tuples/matrices. §5's "ONE temporal-sequence-memory mechanism
   instantiated per layer" is aspirational — the M5 HTM TM exists but is a detached component, wired into nothing.
3. **There is NO top-down (apical) channel and NO output-layer pooler.** Nothing feeds back down, and there is no
   location-invariant object layer. So: no **frame selection** (the pooler + voting that pick which learned frame is
   active — §"one active frame"; this is L2/3's job, Stage 3), no top-down **confirm/bias/mismatch** (the apical tiebreak),
   no **hierarchy / modes**, no **top-down attention**.
4. **No content prediction.** `forward` merely ECHOES the given content (valid only for self-motion reafference). The
   column cannot predict a feature that CHANGES, nor the feature at an unvisited location.
5. **The column models static shapes at poses.** No dynamics / behaviors / order-config rules, even though `Behavior`
   (M5) exists — because it is a bolt-on, not the object layer's native operation.
6. **Recognition is not prediction.** L2/3 recognises via a separate evidence mechanism over a raw cloud, not by
   pooling a *predictive* L4 — so recognition and prediction are two systems, not one.
7. **The self / figure-ground is a connectivity HARNESS.** `column._attend_self` + `retina.connected_figure` group the
   self by a hand-coded **4-connected flood-fill** of the cells that moved — a hard-coded proxy for objectness (rule 4
   violation). It is NOT defensible as a "cohesion prior": in TBT motion-gated cohesion is EMERGENT, not a primitive.
   Monty (verified, arXiv 2507.04494) determines object membership by **reference-frame consistency, not
   connected-components** — "features that cohere under a single reference-frame transformation and consistent movement
   vectors belong together"; a new object "initializes a new reference frame." So the flood-fill must dissolve into the
   recognition loop (Stage 5d). It fails the moment objects aren't solid blobs (diagonal/dashed/occluded/touching).

## 2. The mechanism (TBT-accuracy check — verify before building, §11.3)

The canonical cortical microcircuit. Each cell has THREE dendrite channels (M5 built only the basal one):
- **PROXIMAL (feedforward)** — the "what"; drives which minicolumns are active.
- **BASAL / distal (lateral)** — CONTEXT; depolarises cells into the **predictive** state (the M5 mechanism).
- **APICAL (feedback)** — TOP-DOWN bias from the layer/region above.

Layer roles (Numenta "Columns"; Lewis grid-cell framework):

- **L6 — location.** Grid cells; the current location in the object's frame; path-integrated by L5's efference. **Already
  built** (`hippocampus`, M1). One reusable coordinate system (§"single cohesive reference frame").
- **L4 — the sensorimotor INPUT layer.** Minicolumns = the feed-forward feature (the content SDR). Cells-per-column give
  context. **Basal context = the L6 location ⊕ L5 efference** (so the same feature at a different location / under a
  different movement is a different cell). **Apical = the L2/3 object** — a pure TIEBREAK that CONFIRMS / biases L4's
  BASAL prediction and detects mismatch; it does NOT by itself select the object (see the correction note below). L4 IS the
  object's **feature-at-location map**: query it with a location (L6), narrowed by the object (L2/3) → the predicted
  feature there. A *burst* = the sensed feature was not BASALLY predicted = surprise / new-feature (apical never bursts).
- **L2/3 — the OUTPUT / object layer (the "column pooler").** Proximal = the active L4 cells. It **temporal-pools** them
  into a STABLE, location-INVARIANT object SDR (a slowly-decaying union, re-pooled on L4 surprise — reuse M5's pooling
  idea). Lateral = **voting** across columns + within-column stability. Apical = higher-region feedback (hierarchy;
  deferred). Recognition = which pooled object SDR the current L4 activity matches (**associative recall / overlap**, the
  M4 primitive, now over the L4 representation). It feeds **back to L4 (apical)** to bias/confirm L4's predictions — but
  the OBJECT is SELECTED HERE (this pooling + lateral voting), NOT by the apical channel. **THIS is where frame selection
  lives** (correction below).
- **L5 — displacement cells + motor output + thalamus driver.** Verified against Hawkins et al. 2019 (*A Framework for
  Intelligence…*, Frontiers): L5 (specifically the **thick-tufted / L5b** neurons) are **displacement cells** — the **vector
  difference between two location representations** ("what pose-delta gets from A to B"), used for BOTH movement (the self's
  per-action displacement) AND **object COMPOSITION / relations** (relating two objects' reference frames). It is a
  grid-cell-like MODULE/population code — **NOT** an L4-style feature-minicolumn sequence memory (a correction: the HTM
  minicolumns×cells temporal memory is L4 and L2/3, not L5). The same L5b neuron's axon SPLITS → subcortical **motor
  output** + the **thalamus DRIVER** (the transthalamic route by which L5 drives the higher region's L4 — how much of the
  hierarchy is actually built, not the pure L2/3 apical the rest of this doc assumes). **Partly built** (M1): our
  `l5_displacement.py` `self.eff` is the minimal displacement cell — the SELF's per-action pose-delta (`(dx, dy, dθ)`), read
  by L6 for path integration. **Gaps (not an HTMLayer rebuild):** (a) the displacement-cell GENERALITY — displacements
  between ARBITRARY locations/objects for composition/relations (retired with the symbolic path), as a module code; (b) the
  thalamus-driver route. Behaviors / config-dependence do NOT live here — they are **sequence memory** (basal context on the
  L4/L2-3 TM); L5 supplies the displacement, not the sequence.

**The loop, per sensorimotor step:**
```
   L5 motor emits action a  ─► efference copy
                                  │
   L6 path-integrates location by a  ─►  new location SDR ─┐
                                                           ▼ (basal)
   L2/3 object ──(apical, tiebreak)──►  L4: narrow the BASAL (location) prediction of the feature
                                  ▲ (proximal)
   retina view_sdr ─────────────►  L4: the SENSED feature fires the predicted cells (or BURSTS = surprise)
                                  │
                                  ▼ (proximal, up)
   L4 active cells ───────────►  L2/3: POOL → the stable object SDR ──► vote (lateral) ── SELECTS the object/frame
                                  │
                                  └──(apical, down)──►  biases/confirms L4's next prediction (tiebreak, never gates)  ⟲
```
Pose = the L6 location (+ orientation/anchor) combined with the object identity — **not a separate solve** (this is the
key migration: M4's virtual-rotation pose inference becomes "which L6 anchoring makes L4's predictions match," i.e. pose
lives in L6, the object in L2/3 is location-invariant). **Is this how a real column does it?** Yes — Numenta's input/output
two-layer circuit is exactly this; the location-invariant object + location-variant input is the core of "Columns."

> **⚠ CORRECTION (2026-07-06, from the Stage 1 research — Numenta `apical_tiebreak_temporal_memory`, Hawkins & Ahmad
> 2016).** An earlier draft of this doc (and my framing of it) OVERSTATED the apical channel as "gating" / "frame
> selection." The literature is unambiguous: **an active basal segment predicts a cell; an active apical segment alone
> does NOT.** Apical is a pure TIEBREAK — among the cells a column *basally* predicts, it narrows to the apically-supported
> subset if any exist, else keeps ALL of them; it never fires a cell, never causes or prevents a burst, and its learning
> is downstream of the basal winner. Consequently apical **cannot split a cell that basal made shared**, so it does NOT
> disambiguate two identical-prefix sequences — the OBJECT is selected by the **L2/3 pooler + lateral voting** (Stage 3),
> and the apical to L4 only CONFIRMS / biases / detects-mismatch. This reassigns the load-bearing "frame selection" from
> Stage 1 (apical) to Stage 3 (L2/3), and simplifies Stage 1 to the faithful tiebreak (BUILT 2026-07-06, `sequence.py`,
> `test_apical_is_a_pure_tiebreak_on_basal_predictions`).

## 3. What it unlocks (why, beyond biological accuracy)

1. **Context-dependence via SEQUENCE MEMORY → modeling CHANGE, not just static structure** (the big one): L4 content
   dynamics (a toggling/cycling feature), L2/3 object **behaviors** (open/close, a machine cycle, a patrol), motor skills as
   action SEQUENCES over L5's displacement primitives, and **order/config-dependence** (Sokoban: config as basal context) —
   all one mechanism, the HTM temporal memory with the right basal context (L4/L2-3), NOT a per-layer rebuild; the mechanics
   most ARC-AGI-3 games actually are.
2. **A top-down (apical) channel** we currently lack entirely: **confirm/bias toward an expected sequence + detect
   mismatch** (a tiebreak — the frame itself is SELECTED by the L2/3 pooler, not the apical), **hierarchy / "which
   behavior" modes** (Jiang & Rao, via the higher-region apical), **top-down attention / active sensing**.
3. **Sensorimotor imagination**: predict the feature at an UNVISITED / occluded location from the object map → **object
   permanence** + **planning by simulated sensing** ("if I move there, do I see the goal-feature?") without physically
   moving — the RHAE efficiency lever.
4. **A dense intrinsic learning signal**: bursting = prediction error at every layer, every step — self-supervision
   instead of relying on the sparse ARC score; the epistemic-value term (§8) becomes the burst rate.
5. **Unification**: recognition, prediction, and the active representation collapse into ONE thing (you recognise *by
   predicting correctly*); voting becomes a native lateral-settling dynamic that scales to the multi-column sheet (§3b).

## 4. What we're going to build

- **`sequence.py`**: add the **apical channel** to the HTM TM (a pure TIEBREAK — narrows the BASAL prediction to the
  apically-supported subset if any, else keeps all; never fires a cell or bursts) — the "apical tiebreak temporal memory"
  (Numenta htmresearch). The one new primitive. **BUILT 2026-07-06.**
- **L4** (rewrite `l4_feature_location` → the sensorimotor input layer): a minicolumn TM, proximal = content SDR, basal =
  L6 location ⊕ L5 efference, apical = L2/3 object; predicts feature-at-location; queryable at unvisited locations.
- **L2/3** (evolve `l23_object` → add the column pooler): pool L4 → a stable location-invariant object SDR via the
  canonical `encoders.SpatialPooler` over the L4 cells (§7a) + temporal persistence (§7b); recognition by overlap vs the
  library union; feed back to L4 (apical); keep `vote`. Pose read from L6.
- **`column.py`**: rewire `perceive`/`forward` to run the microcircuit loop; keep the L6/hippocampus/navigation
  localization loop intact.
- **Retire** the subsumed machinery once green: the L4 int codebook; M4's evidence/virtual-rotation pose solve *if* the
  L6-anchor pose inference subsumes it (else keep virtual rotation as the L6-anchor search).

## 5. The build plan (dependency-ordered; suite-green + one paper-test per stage; branch for risk)

**Stage 0 — this doc + the accuracy check.** (This commit.) One prediction, one substrate, no parallel systems.

**Stage 1 — the apical channel on the HTM TM. ✅ DONE (2026-07-06).** Extended `SequenceMemory` with an apical top-down
input as a pure TIEBREAK: among a column's BASALLY-predicted cells, narrow to the apically-supported subset if any exist,
else keep ALL of them; apical never fires a cell and never causes/prevents a burst; apical learning is downstream of the
basal winner. Isolated test (`test_apical_is_a_pure_tiebreak_on_basal_predictions`): a column with two basally-predicted
cells tagged with distinct feedback resolves the SAME basal prediction differently by the top-down input, falls back to
all-basal on mismatch, and never bursts; `apical=None` is exactly the basal TM. **Correction landed here:** the initial
"apical gates / disambiguates objects" plan was wrong (an active apical segment alone can't predict a cell — Hawkins &
Ahmad 2016), so object disambiguation moved to Stage 3 (the L2/3 pooler). LOW risk (extended M5, touched nothing live).

**Stage 2 — L4 as the sensorimotor feature-at-location layer (isolated prototype). ✅ DONE (2026-07-06).** First factored
the shared HTM cell/segment mechanism into `htm.HTMLayer` (minicolumns × cells, distal basal+apical segments, the Hebbian
rule, the connected/potential match, the basal-determined winner, the apical tiebreak) and re-seated `SequenceMemory` on
it (behaviour-preserving, M5 tests unchanged) — so there is ONE mechanism, not a parallel copy (rule 1). Then built
`l4.L4Layer` on it: minicolumns = feature-SDR bits, basal = the L6 grid-SDR of the LOCATION, apical = the object (tiebreak).
`test_l4.py` (1 test): learns a synthetic object's map, RECALLS the feature at each mapped location as a pure query
(**imagination** — predict what is there before sensing it), predicts (near) nothing at a far/unmapped location, and
BURSTS when a feature is sensed where a different one was learned (surprise / object-mismatch), not when it matches. NOT
wired into the live loop. NB per the Stage-1 correction, MULTI-object feature disambiguation is NOT L4's job (the apical
tiebreak can't suppress a different-object feature at the same location) — it is L2/3's (Stage 3); the prototype is
single-object. `basal = L6 location ⊕ L5 efference ⊕ history` (dynamics) is the next enrichment, at wiring time.

**Stage 2.5 — the canonical HTM spatial pooler. ✅ DONE (2026-07-06).** Before the column pooler, made the ONE spatial
pooler (`encoders.SpatialPooler`) HTM-faithful — the feed-forward proximal primitive L4 and L2/3 both stand on (§7a). It
was a reasonable SP but missing three pieces the notes require: **separate `syn_inc`/`syn_dec`**, a **stimulus threshold**,
and **dead-column revival** (overlap-duty-cycle permanence bump — "boosting is required… without it only a tiny number of
minicolumns contribute", notes 112). Reused, not duplicated (rule 1). `test_encoders.py`: fixed sparsity across input
density, determinism, overlap=similarity, learning-converges-to-a-stable-code, homeostasis (≥80% of columns used, no dead
columns). Suite 74 passed / 6 xfailed / 1 xpassed. LOW risk (only `test_encoders` + docs referenced the SP).

**Stage 3 — L2/3 as the column pooler. ✅ built 2026-07-06, ❌ DELETED 2026-07-09** (it was the recogniser for the abandoned
swap — see 5b; `L23_Object` is the recogniser). Historical record: rewrote `l23_pooler` from the union-only prototype (built
on a MISREADING of the pooler — it only captured the temporal-union half) to the faithful Numenta **ColumnPooler** on
`htm.HTMLayer` (§7b): objects as ASSIGNED stable SDRs, proximal synapses grown one-shot from each object's cells to its L4
feature-at-location cells (GROW-ONLY — a new `punish=False` on the shared `_learn_segment`, since every feature-location of
an object is positive evidence; punishing the inactive ones would erode the object's own union), inference =
UNION-then-NARROW. `test_l23_pooler.py` drives it through the REAL front-end `GridEncoder` (L6) → `L4Layer` → pooler, and it
(a) RECOGNISES each object of a small shape library, (b) is LOCATION/ORDER-INVARIANT — the same identity from any order of
its feature-locations, and (c) NARROWS — I and T sharing a bar stay a tie until the distinguishing cell, and distinct
objects get well-separated SDRs (no false merge). **Discovery (why the real L4 is load-bearing, not bypassable):** feeding
the pooler the raw FACTORED grid bits FAILS — an object with cells in column 1 AND row 0 falsely "covers" location (1,0);
L4's CONJUNCTIVE binding (a distinct cell per feature-at-location) is exactly the fix, so the pooler must union L4 CELLS,
not grid bits. **Scope split (correction to the earlier wording):** Stage 3 subsumes M4's *identity/recognition* machinery
(union + associative recall + narrowing); M4's *pose* machinery (virtual rotation) becomes the L6 **anchoring search** at
wiring — so "recognise at an unseen POSE" is a **Stage 4** item (§6 Risk 2), not Stage 3. `test_l23_object` (the M4
recogniser) stays GREEN — the live recogniser until Stage 5 retires it. Suite 77 passed / 6 xfailed / 1 xpassed.

**Stage 4 — wire the L4 feature-at-location CONTENT MAP into the column. ✅ DONE (2026-07-06).** The additive half of "wire
the microcircuit": `perceive` now drives `l4.L4Layer` (`self.L4map`) — each self-cell's COLOUR bound at its OBJECT-FRAME
location (world cell mapped through the RECOGNISED pose, `R(−θ)·(cell−t)`, `_bind_content`), so L4 learns the object's
feature-at-location map. This gives the NEW capabilities L4 exists for — **content PREDICTION / imagination**
(`predict_content_at_world`: the colour L4 expects at a location, before sensing it → object permanence) and a **content
BURST** (`_content_burst`: colours sensed where a different one was learned = surprise, the dense learning signal, §3.4).
Kept STRICTLY behind the tested contracts (perceive still returns `view_sdr` + the hippocampus pose; `forward` still echoes
content; `_pred_error` stays the pose residual) — the localization loop is UNTOUCHED and the L4 path only ADDS `_content_burst`
+ the imagination query. Pose comes from the M4 recogniser (`L23.best()`); using it (not an ad-hoc centroid) keeps the
object frame a recognition product (rule 1). `test_column_microcircuit.py` (2 tests): the column IMAGINES the object's
colours at its cells from the learned map, and the content burst RISES when a cell's colour changes. Additive, revert the
one `perceive` call on any regression (M1 lesson). Suite 79 passed / 6 xfailed / 1 xpassed — games (MockLive/sokoban/
collectall/NavGame) + the pose tests all green.

**Stage 5 — CORRECTED: no recognizer swap. `L23_Object` stays; the L6/hippocampus modernization (5c) + the segmentation
dissolve (5d) remain.** The original framing ("promote the pooler to the live recogniser, retire M4") was **wrong and is
abandoned** — `L23_Object` already IS the faithful TBT recognizer (5b, below). What remains valid under Stage 5: **5c** —
finish modernising the hippocampus itself (M1⊕M2 modernised the localization BELIEF — population bumps, `_ring_shift` path
integration, the superposition update — but left the OPERATOR hard-coded, the pure-query path classical, navigation a greedy
heuristic, and the SR value WRITE-ONLY; `hippocampus.py` audit, 2026-07-06); and **5d** — dissolve the self/figure-ground
flood-fill (§1.7) into the recognition loop (TBT motion-gated cohesion is EMERGENT — reference-frame consistency). Neither
5c nor 5d touches the recognizer.

**Stage 5a — isolated pooler pose-recognition experiment. ✅ built (2026-07-06), now ORPHANED by the 5b abandonment.**
`column.recognize_pose(cloud)` infers pose by a virtual-rotation search over the grid substrate (Lewis 2022): for each
candidate orientation, `_canonical_cells` brings the cloud into the object frame (rotate by −θ, bbox-min-normalise —
lattice-EXACT at 90° multiples), each cell's L4 is read by pure INFERENCE (`l4.observe(learn=False)`), and the orientation
that EXPLAINS the most cells wins. `test_pose_recognition.py` (3): recognises {I,T,S,L} at their canonical pose, at any 90°
rotation + translation, and recovers the asymmetric L's orientation. It worked — but it was built to feed the recognizer
SWAP, which is abandoned (5b), and `L23_Object` already does pose-invariant recognition (virtual rotation) FAITHFULLY. **So
`recognize_pose`/`learn_pose_object`/`_canonical_cells`/`self.pool` + `l23_pooler.py` + `test_pose_recognition.py` were
DELETED (2026-07-09), and `htm._learn_segment`'s `punish` flag (added for the pooler) reverted.** (The Stage-4 L4 content
map — `L4map`/`_bind_content`/`predict_content_at_world` — STAYS; it is live in `perceive`, not pooler code.)

**Stage 5b — ABANDONED: there was never a recognizer to replace (2026-07-09).** The premise "retire `L23_Object`, swap in
pooler recognition" was WRONG. Studying the TBT **theory** (Hawkins, Ahmad & Cui 2017, *A Theory of How Columns of Neurons
Enable Learning the Structure of the World* — the sensorimotor inference model) shows what recognition in a column actually
IS: the **output layer** holds a stable object population that persists over movements; the first feature-at-location invokes
a **UNION** of every object consistent with it; each **movement** senses a new feature-at-location and **NARROWS** the union
(the cube/wedge example); evidence **accumulates persistently across movements and never restarts**; location predicts the
next feature (output→input feedback); orientation is inferred over grid-cell modules (Lewis 2019, virtual rotation).
**`L23_Object` (`l23_object.py`) is a direct, faithful implementation of exactly this** — a persistent evidence session over
(object, orientation-bin) hypotheses, UNION (associative-recall shortlist) → NARROW (path-integrate + re-verify + prune),
pose by virtual rotation, cross-frame tracked/candidate persistence. It is not a rough approximation to improve on; it IS the
model. (So the recurring name "M4" just means `L23_Object`, the recogniser from the M4 SDR-migration step — stop calling it
"M4".)

The Stage-5a pooler pose-recognition (`recognize_pose`/`pool`) and `L23_Object` are **two implementations of the same
output-layer theory**, not a bad-vs-good pair to swap between. Every swap attempt wired recognition as **single-frame,
from-scratch** classification (reset the pool + re-search all orientations every frame), which discards the two things the
theory says recognition IS — **persistent accumulation** and **movement-driven prediction**. That is why it both regressed
the games AND blew up on cost (a full search × ~3/frame × ~1500 frames × a bloating pool —
[[feedback_slow_run_means_catastrophic_failure]]). **Reverted to `16dac1a`; `L23_Object` remains the live recognizer.** The
Stage-5a `recognize_pose`/`pool` + `l23_pooler.py` + `test_pose_recognition.py` (+ the pooler test) were **DELETED** — they
existed only to feed the abandoned swap.

**The one valid kernel that survives (§1.6 "recognition is not prediction"):** *recognition should BE prediction* — the 2017
output→input feedback, where the column recognises BY predicting the next feature-at-location correctly (a matched L4
content-map prediction reinforces the hypothesis; a burst weakens it). If ever pursued, that is an **ADDITION** to
`L23_Object`'s evidence structure (feed the L4 prediction into its evidence update), **NOT** a replacement of it. Deferred,
and only if it earns its place. **Lesson recorded:** don't discard validated machinery for a re-derivation that "gets the
same result" — study the theory the machinery already implements first.

**Stage 5c — the L6/hippocampus remnants, RE-SCOPED (2026-07-09) through the "don't reinvent working machinery" lens.** The
audit found four; two of them are the same reinvention trap as the recognizer swap and are DEFERRED:
1. **Learn the gain-field operator** (`_rot(head)·ego` → learned `M(action)`). **DEFERRED — reinvention trap.** `_rot(head)`
   is the *correct* SO(2) rotation; learning it just re-derives a known-correct matrix. The only payoff is future
   SO(3)/non-abelian (the L6_NONABELIAN strand), which is NOT a current-game need. Do it only when a non-abelian/3-D domain
   actually demands it, not for faithfulness alone.
2. **One operator, one code path** (`predict()`'s point-arithmetic → the SDR operator). **DEFERRED — marginal churn.**
   `predict()`'s cheap point read-out already AGREES with `path_integrate`; "unifying" makes a working query slower for the
   same result. A cheap correct read-out is defensible; leave it (revisit only if it ever disagrees).
3. **SR-value navigation (the navigation ⊕ value overhaul) — THE REAL WORK.** `navigate` (`:299`) is a greedy `atan2`-cone
   heuristic with **no obstacle handling** and it never READS the `sf` value (trained by `learn_location_value`,
   write-only). The SR/value POTENTIAL FIELD (greedy-on-`V` over the grid SDR, obstacles as SR-warped reachability —
   `reference_vector_navigation` / `reference_obstacle_as_transition_cost` / `reference_eigenoptions_subgoals`) ADDS a
   capability (a *better* result, not the same one) → not reinvention. Closes the ~10×-oracle efficiency gap + the parked
   NavGame issue. HIGH risk (load-bearing for every game) — study the vector-nav/SR theory FIRST, build behind `navigate`'s
   signature, judge on the games; `navigate` currently WORKS (solves, just inefficiently), so this is an efficiency add,
   not a bug fix.
4. **Vestige cleanup. ✅ DONE (2026-07-09).** Removed the dead dual head-resolution (`hd`/`head_sdr` + the `n_head` param +
   the orphaned `ScalarEncoder` import); the belief head ring is `_Q` alone. (The `board`-sized `_decode` is NOT a deletable
   vestige — it is the live position read-out; generalising its fixed board is a separate, non-urgent item.) Suite 76
   passed / 6 xfailed / 1 xpassed.

**Stage 5d — dissolve the segmentation HARNESS into the recognition loop (retire `connected_figure` + `_attend_self`).**
The §1.7 fix: motion-gated cohesion is EMERGENT in TBT, not a flood-fill. Object membership = **reference-frame
consistency** (Monty, arXiv 2507.04494) — a cell belongs to the object whose reference-frame TRANSFORMATION (L5
displacement) explains it, so the grouping is *the recognition loop*, not a separate connected-components step:
- **The self** falls out by **reafference** (von Holst) — the cells whose observed motion matches the efference-PREDICTED
  displacement (already the stated principle in `l5_displacement.py`; make it the actual selector, not the flood-fill).
- **Other objects** = cells sharing a consistent (non-self) displacement / one reference-frame transformation; a cell the
  current object's model can't place is a **boundary** (an L4 prediction **burst**, §3.4) → a different object.
- **A new object** = observations no existing reference frame explains → instantiate a new frame (the unsupervised
  recognise-or-create `L23_Object.add_if_novel` already does).
Depends on L5 displacement + L2/3 recognition (`L23_Object`) being LIVE — it's why we did NOT retire `connected_figure` earlier.
**Honest scope:** Monty largely *dissolves* bottom-up segmentation (it senses one patch on a surface); our WHOLE-FRAME
input still needs grouping-by-shared-displacement, but the CRITERION is the displacement-cell/reference-frame consistency
above, never connectivity. Acceptance: the self + movers are grouped with no `connected_figure`, incl. a non-solid
(diagonal / occluded / touching) object where the flood-fill fails; games stay green.

**Stage 6 — validate the NEW capabilities (the payoff).** New tests + wiring: content dynamics (a toggling feature),
object behavior (open/close via the L2/3 sequence), config-dependence (Sokoban push as basal context), imagination
(predict an occluded feature). These are the point of the whole build.

## 6. Acceptance & risks (the paper test, per stage)

- **Every stage**: explainable in one §1–§6 sentence; obeys the five rules; TBT-accurate mechanism (re-checked at the
  stage, not just here). Suite-green throughout; a git branch for the risky stages (4–5).
- **Risk 1 — the recognition-core rewrite.** M4 works; Stage 3/4 subsume it. Mitigate: build L2/3-pooler in isolation
  (Stage 3) and keep M4's tests as the acceptance bar before swapping.
- **Risk 2 — pose migration to L6.** M4 gives (quantised) pose in the recogniser; the new design infers pose as the L6
  anchoring that makes L4's predictions match. Likely QUANTISED (the grid modules give discrete anchors — consistent with
  M4's decision). Open question: exact-continuous vs nearest-anchor; resolve empirically, relax tests to the native
  semantics if needed (the M4 precedent).
- **Risk 3 — small-scale convergence.** The M5 tiny-SDR convergence cost recurs; tune params per stage, relax exposure in
  tests where the mechanism is faithful but slow (documented, not gamed).
- **Non-goal (this phase)**: the multi-column SHEET + full heterarchy (§3b C–E) and the higher-region apical hierarchy —
  the microcircuit is ONE column's loop; the sheet reuses it later.
- **Parked L5 strands (deferred, not forgotten — §2 L5):** (a) the **displacement-cell GENERALITY** — displacements
  between ARBITRARY locations/objects (object composition + relations), retired with the symbolic path, to return as a
  module code (ARCHITECTURE P3 relations); (b) the **L5b thalamus-DRIVER route** — the transthalamic path by which L5
  drives the higher region's L4 (how the cortical hierarchy is actually built, vs the pure-apical hierarchy this doc
  assumes). Both fold into the hierarchy/heterarchy phase, not this one.

## 7. The pooler family — spatial pooling (SPACE) vs. the column pooler (TIME)

Both L4 and L2/3 have a feed-forward **proximal** stage, and both are, at bottom, **spatial poolers** over their input —
so before building the L2/3 column pooler (Stage 3) we first made the ONE canonical spatial pooler HTM-faithful
(`encoders.SpatialPooler`, 2026-07-06; validated in `test_encoders.py`). The two "poolers" pool along DIFFERENT axes, and
the column pooler *contains* a spatial pooler as its first stage — that is the whole relationship. (Sources:
`notes/htm notes.txt` 97–151; Numenta *Overview of the Temporal Pooler*, htmresearch wiki; Cui, Ahmad & Hawkins 2017
*The HTM Spatial Pooler*; Hawkins, Ahmad & Cui 2017 *A Theory of How Columns…*.)

### 7a. The spatial pooler (SP) — pools over SPACE (one instant)

- **What it does.** A raw input SDR of ANY density → a **fixed-sparsity** SDR of active **minicolumns** that preserves
  semantic overlap (near inputs → near codes). It normalises sparsity and learns, online, a stable representation of its
  input space. This is HTM's general learned SENSORY encoder (`ARCHITECTURE §P5`) — used where no analytic encoder fits.
- **How it works — the algorithm** (order is **BOOST → INHIBIT → LEARN**, notes 110):
  1. **overlap** — each column's connected synapses onto the active input: `overlap = ((perm ≥ connected) & potential) @ x`;
  2. **stimulus threshold** — a column needs `overlap ≥ stimulus_threshold` connected active inputs to be *eligible*
     (noise rejection);
  3. **boosting** (homeostasis, *before* inhibition) — scale eligible overlaps by `exp(boost·(target_density −
     active_duty))` so **under-used columns are favoured** and no clique hogs the code (notes 108/112 — "boosting is
     required… without it only a tiny number of minicolumns contribute");
  4. **inhibition** — **k-winners-take-all**: the top `w` boosted-overlap columns fire (global inhibition = one
     neighbourhood; topological receptive fields are a deferred refinement, notes 117–119);
  5. **learning** (Hebbian, winners only) — on each winner's *potential* synapses: `+syn_inc` where the input bit is 1,
     `−syn_dec` where 0 (**separate rates**, notes 105); clip [0,1] — the pattern is carved into the permanences;
  6. **duty cycles + dead-column revival** — track `active_duty` (drives boosting) and `overlap_duty`; any column whose
     `overlap_duty` falls below `min_overlap_duty` has **all** its potential permanences bumped up, so a permanently-silent
     column re-connects and joins the code (notes 112).
- **Data types.** `potential` bool `[n_cols × n_inputs]` (the fixed potential pool — which inputs a column MAY see);
  `perm` float `[n_cols × n_inputs]` ∈ [0,1] (connected iff `perm ≥ connected` AND in the pool); `active_duty` /
  `overlap_duty` float `[n_cols]` (EMA homeostatic traces); output = `SDR(n_cols, winners)` at fixed sparsity `w`.
- **Where.** `encoders.SpatialPooler` — the ONE spatial pooler (rule 1); it is BOTH the general sensory encoder AND the
  feed-forward proximal stage of the column pooler (7b). Validated: fixed sparsity across input density, determinism
  (`learn=False`), overlap=similarity, learning-converges-to-a-stable-code, and homeostasis (≥80% of columns used, no
  dead columns).

### 7b. The column pooler (L2/3 object layer) — pools over TIME (a sequence)

- **What it does.** A **sequence** of L4 feature-at-location patterns (as the sensor moves over ONE object) → **ONE
  stable, location-INVARIANT** object SDR. The object code stays put while the L4 input changes; recognition = which
  learned object the active SDR matches (associative recall / overlap — the M4 primitive, now over the L4 representation).
- **How it works — the algorithm (Numenta ColumnPooler; Hawkins, Ahmad & Cui 2017, "Columns"):**
  1. **assigned object SDRs** — each object is a fixed sparse set of L2/3 cells, chosen once and HELD active while the
     object is explored (this held-stable activity is what makes the code location-invariant);
  2. **proximal growth** (the feed-forward stage — a spatial-pooling overlap over L4, §7a) — the held object cells grow
     proximal synapses to the L4 cells of every feature-at-location sensed, FAST (one-shot: `init_perm ≥ connected`), so
     afterwards ANY of the object's feature-locations drives (part of) its SDR;
  3. **union-then-narrow inference** — the FIRST sensation activates *every* object with enough proximal support (a UNION
     of candidates); each further sensation INTERSECTS the active set with the still-supported cells, so the representation
     **narrows** to the one object consistent with ALL sensed features (a contradiction / new object re-unions);
  4. **lateral** — the within-session narrowing IS one column's temporal stability; inter-column **voting** (the M4 `vote`)
     is the *sheet* (§3b; deferred). *(Alternative realization — the union temporal pooler: let the object SDR EMERGE from
     a persistence-held SP + Hebbian instead of assigning it — same location-invariance, but slow-converging; we build the
     assigned-SDR ColumnPooler for reliable FEW-SHOT online learning, no tiny-SDR convergence tail.)*
- **Data types.** `objects` = `name → frozenset` of L2/3 cell ids (the assigned stable SDR); per object cell ONE proximal
  segment `{L4 cell: permanence}` (reusing `htm.HTMLayer`'s ONE segment machinery — the same `_learn_segment` /
  `_connected_match` as L4 and the TM, rule 1); `active` = the currently narrowing object-cell set.
- **Where.** This §7 is retained as reference *theory* (SP vs column pooler); the concrete `l23_pooler` implementation was
  built (Stage 3) then **DELETED** (2026-07-09) with the abandoned swap — `L23_Object` is the recogniser. The proximal
  overlap step IS the spatial-pooling of §7a (differing only in learning regime: one-shot object-assigned growth vs the SP's
  competitive Hebbian).

### 7c. The relationship (the answer to "is the column pooler just spatial pooling?")

**No, and yes.** The column pooler is NOT a separate mechanism from spatial pooling — it **uses a (modified) spatial
pooler as its feed-forward first stage**, then adds temporal/union pooling + lateral voting for stability *over time*.
Same word "pool," **orthogonal axes**: the SP pools over the input SPACE at one instant (→ fixed-sparsity minicolumns);
the column pooler pools a temporal SEQUENCE into one stable object. This is also why **L4 and L2/3 both have a proximal
stage** — both are spatial poolers over their feed-forward input, differentiated by their *context*: L4's basal context is
the L6 **location** (feature-at-location); L2/3's is **temporal persistence + lateral voting** (a location-invariant
object). One primitive (the SP), two axes, three layers.
