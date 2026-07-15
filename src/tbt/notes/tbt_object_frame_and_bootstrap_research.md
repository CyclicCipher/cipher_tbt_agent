# TBT object-centric frame anchoring + new-object bootstrap — research findings (2026-07-14)

Salvaged from a deep-research run (search + primary-source claim-extraction completed; final auto-synthesis did not).
Claims below are extracted from primary sources (papers + Monty docs) with direct quotes. This doc is the durable record so
the study is not re-run. Feeds the object-centric-frame / unsupervised-object-boundary build (ARCHITECTURE §8, STATUS "Next").

## Sources
- **Lewis, Purdy, Ahmad & Hawkins 2019**, "Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using
  Cortical Grid Cells" (Frontiers Neural Circuits 13:22; PMC6491744). — the grid-cell frame + path integration.
- **Hawkins, Ahmad & Cui 2017**, "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World"
  (Frontiers 11:81; PMC5661005). — the L4/L2·3 two-layer object model.
- **Hawkins, Lewis, Klukas, Purdy & Ahmad 2019**, "A Framework for Intelligence and Cortical Function Based on Grid Cells"
  (PMC6336927). — displacement cells; grid cells in every column; allocentric vs egocentric.
- **Monty / Thousand Brains Project**: arXiv 2412.18354 (2024) + 2507.04494 (2025, Neural Computation) + docs
  (evidence-based-learning-module, unsupervised-continual-learning, how-learning-modules-work).

## A — How the object-centric frame is ANCHORED
**Biological/Numenta model (Lewis 2019) — EMERGENT, and this is the one we implement:**
- The anchor IS a fresh arbitrary phase: *"At the start of training on a new object, each module in the location layer
  activates a bump at a random phase."* Grid frames have **no origin**; the random phase itself is the origin → translation
  invariant by construction.
- Path integration updates each module: `φ_move_i = (φ_sense_i + M_i·d) mod 1`; every module gets the **same** movement
  vector `d`; `M_i` is a fixed per-module matrix (scale + 60° hex orientation). "Anchor" = *"selecting which grid cells in
  each module should be active at the current location."*
- Location is **allocentric** (object-relative), decoupled from the body frame: *"In 'what' regions cortical grid cells
  represent locations that are allocentric, in the location space of objects; in 'where' regions ... egocentric, in the
  location space of the body."* (Framework 2019).
- **Displacement cells** (Framework 2019, L5 thick-tufted): encode the position-invariant delta between two grid locations;
  same physical displacement → same displacement cell. Used for object composition.
- **HARD LIMIT (verified by the verification pass — an overreach was caught + refuted):** the 2019 model gives
  **translation invariance ONLY, NOT orientation**: *"Our model does not yet have a representation of orientation. As a
  result, our model will only recognize an object if the object is at its learned orientation ... The model assumes it
  receives movement vectors that are in the reference frame of the object."* Rotation invariance needs a separate
  head-direction-cell analog (unbuilt). ⇒ our SE(2) heading ring is exactly the missing piece, but object-rotation
  invariance is a bigger, separate problem — keep deferred.

**Monty implementation — HAND-CODED, do NOT copy:** object models are explicit **3D Cartesian graphs**, grid cells/SDRs
deferred (*"object models are currently based on explicit graphs in 3D Cartesian space ... may be substituted with more
powerful, albeit more inscrutable neural components."*). Frame "anchoring" = hypothesis-space INIT over ALL objects × ALL
stored locations; the first sensation's **pose features** (surface normal + principal curvature) prune rotations (~2 per
location, 180° curvature ambiguity). Pose is **solved/inferred, not recalled**. New frame on a novel object = *"the LM
initializes a new reference frame ... a 3D Cartesian coordinate space where sensed features are associated with the
currently active internal location ... updated by integrating the movement vector (path integration / dead reckoning)."*

## B — How a BRAND-NEW object is bootstrapped (and over-minting avoided)
**Numenta 2017 model:** novelty signal = **L4 minicolumn BURST** (*"If this part of the object has not yet been learned,
there will be no predictions in the sensory layer and every cell in these mini-columns will become active."*). On a new
object, L2/3 selects a **sparse set of cells that PERSIST** while the sensor moves, and feedforward from the changing L4 →
the unchanging L2/3 is reinforced (pooling many feature-at-location obs into ONE model). Recognition = a **union** of
candidate objects that **converges** to one over movement (*"The output converges to a single object representation over
time as the object is explored"*) — this convergence is what prevents per-sensation over-segmentation.
- **BUT the object boundary is NOT emergent here** — it is an external reset: *"During training, we reset the output layer
  when switching to a new object."* And the allocentric location is an **external input**, not yet derived. So 2017 does
  NOT solve the unsupervised boundary.

**Monty:** the mint trigger is **recognition failure**, a *negative* signal (no dedicated detector): *"If all objects have
only negative evidence we do not know the object ... and the LM creates a new model for it in memory."* Terminal states:
`no_match` → new graph; `match` → update the existing model (no duplicate); `time_out`. Gated by **hand-set thresholds**
(`x_percent_threshold`, `object_evidence_threshold`, `θ_converge`, `min_lms_match`) — flagged as engineering choices, not
emergent. Anti-duplication is at the **point** level (a hand-coded similarity threshold: *"a new point ... only forms a new
point if it is sufficiently displaced in 3D Cartesian or feature space from existing points"*). **Object-level
over-segmentation is an acknowledged OPEN problem** (*"multiple objects merged into one ... or multiple models learned for
one object ... tracked in mean_objects_per_graph / mean_graphs_per_object"*), and during **learning** Monty often uses the
**ground-truth** pose (*"uses the episode's ground truth rotation to transform the newly learned observations"*) — i.e. the
fully-unsupervised learning boundary is NOT cleanly solved; it is hand-coded/supervised.

## C — Is frame-anchoring the SAME behavior as new-object creation? — VERDICT: YES (in the 2019 grid model)
- **Lewis 2019 — one operation, EMERGENT:** *"By choosing random starting points within modules, unique location spaces can
  be defined for each environment ... The initial random starting point thus implicitly defines a unique location space for
  each environment."* Allocating a **fresh random grid phase** simultaneously (a) anchors the object's reference-frame origin
  AND (b) individuates the object (a distinct location space = a distinct object). **Frame-anchor and object-identity are the
  same act.** ← confirms the hypothesis.
- **Monty — coupled but hand-gated:** `no_match` → *both* a new object AND a new reference frame, in one decision — but the
  decision is gated by tuned evidence thresholds, not an emergent signal.
- **2017 model — NOT unified emergently:** location is external input; new-object is an external reset; an unpredicted burst
  just picks a winner cell among existing reps.

## Implications for OUR build (emergent only — no harness/symbolic segmenter)
1. **Object-centric frame = re-anchor the grid phase per object.** We already have grid location + path-integration
   operators; "reset the origin per object" = a fresh phase = the arbitrary origin. Gives translation invariance for free.
   Object-*rotation* invariance is a separate, harder problem (needs a head-direction / pose-hypothesis mechanism) — DEFER.
2. **One behavior triggers both** (per Lewis 2019): the SAME event — recognition failure — should (a) mint a fresh L2/3
   identity AND (b) re-anchor a fresh frame origin. Our `ColumnPooler` mint + a grid re-origin, fired by one signal.
3. **The signal is recognition FAILURE, not a per-sensation burst.** L4 burst = local surprise; the OBJECT boundary = the
   accumulated failure of any known identity to be supported (the union never settles). Our pooler already computes this
   (no object's match ≥ recognize_frac). The convergence/persistence dynamics are the over-minting guard — mint one identity,
   pool to it while consistent.
4. **HONEST GAP — do NOT invent machinery TBT lacks.** The fully-unsupervised object boundary *during learning of
   consecutive novel objects* is NOT cleanly solved in TBT (2017 = external reset; Monty = tuned thresholds + ground-truth
   during learning). So: implement the emergent core (fresh-phase anchor coupled to mint-on-recognition-failure); keep the
   learning-time boundary minimal/honest rather than hand-coding a symbolic segmenter; and treat Monty's tuned thresholds as
   the heuristics they are, not something to port.

## D — Object-ROTATION invariance (added 2026-07-14; Numenta SOLVED this — do not reinvent)
Source: "**Orientation Invariant Sensorimotor Object Recognition Using Cortical Grid Cells**" (Frontiers
in Neural Circuits 2021, PMC8825787; the direct follow-up to Lewis 2019, reviewed by Hawkins). Lewis 2019 explicitly lacked
orientation; this paper adds it. Findings (fetched, quoted):
- **Plain SDR overlap alone does NOT recognise a rotated object.** Location disambiguation uses the UNION property (candidate
  locations path-integrate + narrow), but ORIENTATION needs an explicit shift-and-rank: *"The most plausible orientation for
  an arbitrary sensorimotor sequence is chosen after sequentially evaluating each possible object orientation."*
- **The mechanism = a CIRCULAR BUFFER over orientation-ordered grid modules (a "mental rotation"), NOT a 3D pose particle
  filter.** Grid modules are pre-tuned to orientations spread over 360° and ORDERED; shifting which modules connect
  downstream rotates how movement maps to bump shifts: *"By ordering the modules according to their orientations beforehand,
  a simple circular buffer can be used to constrain the connections to grid cell modules."* This "effectively rotates how
  movement displacements transform into grid-cell representations" = a mental rotation of the representation.
- **Recognition = a bounded SCAN over candidate orientations, ranked by fit.** For each orientation, replay the movement
  through that buffer configuration; the winner has the LEAST location-layer ambiguity (fewest simultaneously-active bump
  cells) / highest firing rate. A cross-correlation over the orientation ring, NOT a search over free hypotheses.
- **This VALIDATES the equivariance/shift route** (rotation = a cyclic shift = the operator; pose = a bounded scan; location
  = the union) and REFUTES the framing that rotation needs a bolted-on particle filter. It reuses grid + operator + union.
- Caveats it does NOT remove: needs the rotation CENTRE (our object-centric anchor supplies it); is clean in 2D (SO(2), one
  ring) — 3D (SO(3)) is where Monty's multi-hypothesis earns its keep; discrete orientation resolution (N buffer positions).

**Corrected implication for our build:** object-rotation is NOT "hard/deferred forever" — it is **multi-orientation grid
modules + a circular-buffer rotation operator + an orientation scan**, reusing our `GridEncoder` (extended to orientation-
spread modules), the `ModularOperator`/SE(2) machinery (rotation = a shift over the orientation buffer), and the pooler's
union (location). Plan: `notes/rotation_invariance_plan.md`.
