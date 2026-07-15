# STATUS.md — the live map of the TBT agent

The single answer to *"what is in the codebase and what is wired?"* Derived from the code, per `RULES.md` #1 — if this
doc and the code ever disagree, the code wins and this doc is fixed. Not a plan, not history; just current reality.

## State: STARTING OVER (2026-07-09)

The previous system is archived under `src/tbt/Legacy - DO NOT USE OR IMPORT!/` — **reference only, never imported.**
It became an un-navigable tangle: mechanisms built and then un-wired in successive "collapse" rewrites and never
re-integrated (basal ganglia, thalamus, forward model), plus four conflicting source-of-truth docs. We are rebuilding
from scratch, **vertical slice first**, under `RULES.md`.

The legacy code is for **reference so we don't redo solved work** (encoders, HTM temporal memory, SR features, the
recognizer, the operator geometry all exist there and are sound in isolation) — but nothing is copied without going
through RULE 5 (does it exist / extend the owner) and RULE 3 (it isn't "done" until wired + exercised end-to-end).

## North star
A Thousand-Brains agent that plays ARC-AGI-3-style games end-to-end (perceive → predict → plan → act → win), learned,
no per-game code. **Design + rationale: `ARCHITECTURE.md`. Discipline: `RULES.md`. This doc: current wired state only.**
The system is **ALWAYS MULTI-COLUMN** — at least one sensory column *and* one PFC/task column, never single-column, no
exceptions (`ARCHITECTURE.md` §5.1). **First concrete target:** win one trivial replica level end-to-end through that loop.

## What is BUILT + WIRED in the new system
FIRST VERTICAL SLICE DONE (2026-07-10): everything is now reachable from the live entry point — STANDALONE is empty, the
RULES.md #2 goal.

| module | one-line job | wired? | its test |
|--------|--------------|--------|----------|
| `agent.py` | live entry point / ROOT — composes a SENSORY + a TASK `Column` (the sensorimotor SCAN), a DECISION loop (`decide`/`reward`), a SPATIAL nav slice (`learn_move`/`locate`/`path_integrate`/`where`), the L4↔L6a loop (`sense_at`/`predict_feature`: feature at location — order-invariant), L2/3 pooling (`reset_object`/`perceive_object`), OBJECT-CENTRIC recognition (`start_object`/`perceive`: re-anchor per object → translation-invariant; emergent boundary), SE(2) NON-ABELIAN path integration (`set_pose`/`learn_pose_move`/`path_integrate`/`pose` — non-commutative, continuous heading in DEGREES; ARCHITECTURE §8), AND POSE-INVARIANT recognition (`sense_sweep`/`recognize` — buffer a sweep, SOLVE the pose, return the hypothesis population); the `step(obs)→action` game loop is still a stub | ROOT | `test_column_arithmetic`, `test_bg_thalamus`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_se2`, `test_object_centric`, `test_rotation_recognition` |
| `column.py` | the cortical COLUMN — 5 layers (L4, L2/3, L5IT, L5PT, L6a) as one `HTMLayer` each, wired per its §12 table; `observe` drives L4 (arithmetic); a SPATIAL column (`location=GridEncoder`) also gains L6a's CONTINUOUS pose state `_pose=((x,y),heading°)` + its TRANSFORM engine (`MotionOperator`), with the grid code a per-fixation READ-OUT (`_code`), the L4↔L6a feature-at-location loop (`sense_at` depolarises L4 by the location → location-specific firing), L2/3's `ColumnPooler`, OBJECT-CENTRIC recognition (`start_object`/`perceive` — one recognition-failure event re-anchors the frame AND mints the identity), SE(2) pose integration, the **L4→L6a associative link** (`_link`/`_union_for` — a feature recalls the UNION of (identity, location) where it occurs; Lewis 2019), + POSE-INVARIANT recognition (`recognize`/`Hypothesis`/`_solve`/`_evidence` — the pose is SOLVED from displacement geometry, never scanned; ties = indistinguishable poses); the full §13 `step` counterstream is the target | WIRED | `test_column_arithmetic`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_se2`, `test_object_centric`, `test_rotation_recognition` |
| `operator.py` | the TRANSFORM primitive (ARCHITECTURE §8) — `MotionOperator` LEARNS each action's BODY-frame displacement + heading change (running mean) and `apply`s it to the CONTINUOUS pose through the current heading; position- AND heading-invariant by construction (one observation generalises everywhere, incl. headings never seen), non-commutative for free (FORWARD;TURN ≠ TURN;FORWARD) with no keying, no ring, no discretisation. The second primitive alongside `htm.py`'s ASSOCIATE. Plus `rotate`/`wrap` — plain geometry the pose scan applies. **2026-07-15 CUT-OVER:** `ModularOperator` + `RotationOperator` DELETED — a discrete SDR state makes every action a PERMUTATION, which only represents group elements mapping the code's lattice onto itself (measured: quantised phases drift LINEARLY; exact off-grid rotation needs N > 2π·radius modules). Obstacle override deferred | WIRED | `test_operator_path_integration`, `test_operator_se2`, `test_rotation_recognition` |
| `pooler.py` | L2/3 TEMPORAL POOLING (ARCHITECTURE §8) — the STABLE object-IDENTITY that pools ONLY the PREDICTED (non-burst) L4 stream (`ColumnPooler`: Hebbian feedforward L4→identity + persistence + overlap-recall); a BURST = novelty → persist while learning / the boundary signal at inference. `support(l4, identity)` grades ONE named identity (what recognition-by-evidence reads). ASSOCIATE in a pooling regime (decoupled stable output the `HTMLayer` can't do), NOT a third primitive. Identity ONLY (structure in L4/L6/L5). **KNOWN DEFECT (measured 2026-07-15, next slice):** learning-time persistence is UNCONDITIONAL, so L2/3 commits at fixation 1 and never REVISES → two objects sharing a feature-at-location silently MERGE into one chimeric identity. Cross-column VOTING (thalamus) deferred | WIRED | `test_l23_pooling`, `test_object_centric`, `test_rotation_recognition` |
| `basal_ganglia.py` | the value-driven SELECTOR — OpAL Go/NoGo + dopamine-RPE + STN commitment (reference_basal_ganglia); the ONE place competing options are arbitrated (rule 4). MoE column-allocation deferred | WIRED | `test_bg_thalamus` |
| `thalamus.py` | the inter-column ROUTER/GATE — relays a column's percept to the selector + gates the BG's winner to the motor; content⊗location binding (place-value / voting) deferred | WIRED | `test_bg_thalamus` |
| `htm.py` | the ONE cortical-layer mechanism — HTM sequence memory (proximal SDR-in via SP, basal context, apical, learn/predict/burst); a layer = one instance + a declared (proximal-in, context-in, apical-in, target-out) wiring; `depolarize(context)` lets an EXTERNAL context (L6a location) drive which cell fires (sensorimotor feature-at-location), not just the recurrent sequence | WIRED | `test_htm` |
| `encoders.py` | SDR transduction library (`SDR` + Scalar/Category/Grid/Multi/Conjunctive/SpatialPooler) — data ↔ overlap-bearing SDR. `GridEncoder` is axis-aligned modular phases, and after the cut-over is a pure READ-OUT of L6a's continuous pose (encode-per-fixation ⇒ its quantisation is bounded and NEVER accumulates). The `orientations=N` multi-orientation variant was REMOVED with `RotationOperator` (see `operator.py`) | WIRED | `test_encoders` |

**Generalization investigation — RESOLVED 2026-07-09 (now wired into the column, above).** Two durable results,
full detail in memory `project_place_invariance_needs_factored_state` + `reference_htm_canonical_pipeline`, plain-English
writeup in `ARCHITECTURE.md` §7:
1. **The substrate is VALIDATED.** The canonical HTM pipeline **encoder → SpatialPooler → HTMLayer → SDRClassifier** reaches
   **96% exact** next-value on a repeating count. Earlier "0% / can't generalize" scares were a NON-CANONICAL harness — no
   SP, and decoding the TM's *predictive cells* through the encoder-INVERSE (blur/lag). Use a trained classifier over the
   *active* cells; grade by EXACT decode (overlap grading is fooled by persistence). `HTMLayer.observe` gained a
   `learn=False` inference mode (kept).
2. **The place-invariance wall + its fix.** A per-place model generalizes carries WITHIN the trained place-value structure
   (held-out numbers: non-carry 100%, carry 86%) but is **0%** on a novel place-VALUE — plain SP/TM have no weight-sharing
   across positions (a known HTM limit). Fix = the Monty/TBT reference-frame move: ONE shared model walked across
   place-LOCATIONS with content place-invariant and carry as an inter-location STATE → hundreds **100%** (monolithic 0%).
   A FACTORED recurrent state channel closes it (100%) where PURE temporal memory can't (37%). ReSU investigated + DROPPED
   (temporal encoder, not spatial invariance).

Suite: **48 passed** (~20s; `test_column_arithmetic` is the ~16s end-to-end column test, the rest — `test_bg_thalamus`,
`test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_se2`, `test_object_centric`,
`test_rotation_recognition` — are fast). The count fell from 54 to 46 at the 2026-07-15 cut-over (`test_oriented_grid` +
`test_rotation_operator` were deleted with the discrete-rotation code they covered), then rose to 48 with R4's recognition
tests — rotation is now tested end-to-end through *recognition*, which is the property we actually wanted, rather than through
the machinery that implemented it. Run `python src/tests/test_reachability.py` for the wired map; the 20 legacy test files are
archived under `Legacy - DO NOT USE OR IMPORT!/tests/`.

## Next
Wired so far: the **forward-model** slice (sensory ⊕ task `Column`, place-invariance win ≈ **100%** on a place never trained,
`test_column_arithmetic`); the **decision** slice (basal-ganglia value SELECTION + thalamus relay/gate, learns a
context→action map from reward, `test_bg_thalamus`); and the **path-integration** slice (L6a's `MotionOperator` TRANSFORM
primitive — an action's effect learned in a small region dead-reckons correctly into NEVER-VISITED positions,
`test_operator_path_integration`); the **L4↔L6a loop** (predict the FEATURE at the path-integrated location —
ORDER-INVARIANT: a 3×3 object learned in one order is predicted 9/9 traversed in another, `test_feature_at_location`); and
**L2/3 pooling** (the L4 stream → a STABLE object identity — recognised invariant to fixation order, ONE identity per object,
`test_l23_pooling`); and the **NON-ABELIAN SE(2) operator** (heading-dependent motion — FORWARD;TURN ≠ TURN;FORWARD, dead-
reckons a full path into novel poses, and — after the cut-over — generalises to headings NEVER OBSERVED, `test_operator_se2`);
and **POSE-INVARIANT recognition** (an object learned once, in its own frame, is recognised ROTATED to any angle AND TRANSLATED
anywhere, ENTERED on any feature — with the pose SOLVED in closed form from displacement geometry, not scanned, and
indistinguishable poses reported honestly as a population, `test_rotation_recognition`). Each is minimal-but-real; each
subsystem's RICHER role is deferred to a task that needs it. Thicken from here (RULES.md #4 — always keep it runnable,
ARCHITECTURE.md §3/§5.1):
1. **Join the two slices into one loop on a small GAME** — perceive (column) → predict / roll out → SELECT (BG) → GATE
   (thalamus) → act → reward. This needs a **TD value critic** (`reward.py` — multi-step value; the RPE the BG currently
   fakes as a centered immediate reward) + a tiny nav/decision env, and is where the BG + thalamus earn their full keep.
2. **The spatial column's remaining frontier.** DONE on this thread: L6a path integration (the operator), the L4↔L6a
   feature-at-location loop, L2/3 pooling, the NON-ABELIAN SE(2) operator, the OBJECT-CENTRIC frame + emergent boundary (one
   recognition-failure event re-anchors the frame + mints the identity; translation-invariant, arrangement-sensitive), and
   **ROTATION invariance via the 2026-07-15 CONTINUOUS cut-over** (L6a's state is a continuous pose; the grid code is a
   read-out ⇒ SO(2) is exact at ANY angle, and the continuous-heading TENSOR form is MOOT — heading is just a float), and
   **R4 pose-invariant RECOGNITION** (the pose SOLVED from displacement geometry over a hypothesis population; free entry,
   free translation; ambiguity + symmetry as the surviving population).
   What remains, in order:
   **(a) L2/3 REVISION at learning time — the next slice, and load-bearing.** MEASURED DEFECT: learning-time persistence is
   unconditional, so L2/3 commits to an identity at fixation 1 and never revises → two objects sharing a feature-at-location
   MERGE into one chimeric identity (two objects → one). Recognition is only as good as the model it reads, and real ARC
   objects will share feature-at-location codes constantly. Fix = the same evidence machinery R4 built, applied at learning:
   buffer the episode, ask `recognize()` whether a known object explains it, else mint + bind. Diagnosis + the patches that
   DON'T work are worked through in `notes/rotation_invariance_plan.md`.
   (b) the context-gated obstacle OVERRIDE (a wall blocks the move — read from local context; `operator.py`);
   (c) **SO(3)** — the state generalises unchanged; R4 removed the cubic scan (solving needs 3 non-collinear points, not an
   angular resolution), so this is now mostly `_solve` in 3-D;
   (d) the fully-unsupervised LEARNING-time boundary (TBT itself leaves it open — currently a minimal episode cue);
   (e) the SR (ROADMAP 3b) accumulating the operator into value. Cross-column VOTING over these identities is the thalamus (#3).
3. **The thalamus's binding role** (content ⊗ location across two columns → place-value / cross-column VOTING) + the factored
   state via L4's basal `context=` channel; then **L2/3 recognition/voting**, **L5 motor**, **hippocampal rollout**. Each
   added as a task exercises it, driven by `agent.py`. Nothing counts until imported from `agent.py` AND the agent plays more
   than before (RULES.md #3). No single-column experiments, ever (ARCHITECTURE.md §5.1).

## How to answer "where are we?"
Run the reachability test (once it exists); it reports the wired module map. Keep this table equal to that output.
