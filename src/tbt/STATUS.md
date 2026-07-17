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
| `agent.py` | live entry point / ROOT — composes a SENSORY + a TASK `Column` (the sensorimotor SCAN), a DECISION loop (`decide`/`reward`), a SPATIAL nav slice (`learn_move`/`locate`/`path_integrate`/`where`; `dims=2\|3` picks the space — an ENVIRONMENT property), the L4↔L6a loop (`sense_at`/`predict_feature`: feature at location — order-invariant), NON-ABELIAN SE(n) path integration (`set_pose`/`learn_pose_move`/`path_integrate`/`pose` — non-commutative, orientation as a continuous MATRIX; ARCHITECTURE §8), and the OBJECT surface: `start_object` (onset: re-anchor + fresh identity + clear buffer), `perceive` (INFER one fixation → the stable identity; emergent boundary), `sense_sweep` (buffer a fixation), `commit` (LEARN the buffered episode), `recognize` (POSE-INVARIANT recognition → the hypothesis population); the `step(obs)→action` game loop is still a stub | ROOT | `test_column_arithmetic`, `test_bg_thalamus`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`, `test_object_centric`, `test_rotation_recognition` |
| `column.py` | the cortical COLUMN — 5 layers (L4, L2/3, L5IT, L5PT, L6a) as one `HTMLayer` each, wired per its §12 table; `observe` drives L4 (arithmetic); a SPATIAL column (`location=GridEncoder`) also gains L6a's CONTINUOUS pose state `_pose=((x,y),heading°)` + its TRANSFORM engine (`MotionOperator`), with the grid code a per-fixation READ-OUT (`_code`), the L4↔L6a feature-at-location loop (`sense_at` depolarises L4 by the location → location-specific firing), L2/3's `ColumnPooler`, the OBJECT-CENTRIC frame (`start_object` — one recognition-failure event re-anchors the frame AND starts a fresh identity), SE(2) pose integration, the **L4→L6a associative link** (`_link`/`_union_for` — a feature recalls the UNION of (identity, location) where it occurs; Lewis 2019), POSE-INVARIANT recognition in ANY dimension (`recognize`/`Hypothesis`/`_pin_rotation`/`_solve` — the pose is SOLVED from displacement geometry via the TRIAD method, never scanned, so SO(3) costs no more than SO(2); ties = indistinguishable poses), ONLINE recognition over a narrowed hypothesis POPULATION (`perceive`/`_narrow`/`_pop`/`_union_identities` — solves its place on the object per fixation; `_fit` is the ONE scoring rule both the batch and online paths use), **L5 OBJECT DYNAMICS** (`dynamics`/`learn_object_move`/`predict_object_move` — the same TRANSFORM primitive applied to an OBJECT's pose rather than the sensor's; ONE demo generalises to any position/orientation/object), and EPISODE-level learning (`commit` — recognise-then-bind, the bar being "nothing REFUTES it" rather than a score; `_replay` scores AND binds in ONE traversal, so learning inherits pose-invariance; and a sweep that crosses an object BOUNDARY SPLITS ITSELF and recurses — `_exhausts`/`_extent` tell a boundary from a shares-a-prefix object by asking whether the prefix reached the object's EDGE, so no boundary cue is needed from the caller); the full §13 `step` counterstream is the target | WIRED | `test_column_arithmetic`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`, `test_object_centric`, `test_rotation_recognition` |
| `operator.py` | the TRANSFORM primitive (ARCHITECTURE §8/§9), DIMENSION-GENERIC — `MotionOperator` LEARNS each action's displacement + rotation (running mean) in the frame that holds it INVARIANT, and `apply`s it to the CONTINUOUS pose `(position, R)`. **Two ways the group acts, an empirical fact about the action:** `ego=True` acts on the RIGHT (`p'=p+R·d`, `R'=R·ΔR`) — a BODY's own motion, "forward from where I face", non-commutative for free, and in 3-D the ROTATIONS stop commuting too (yaw∘pitch ≠ pitch∘yaw), which SE(2) cannot exhibit; `ego=False` acts on the LEFT (`p'=p+d`, `R'=ΔR·R`) — an OBJECT's motion, which does not care which way the object faces (measured: an intrinsic operator sends a 90°-turned block 90° off the demonstrated shove). Either way, position/orientation-invariant by construction — one observation generalises everywhere. The second primitive alongside `htm.py`'s ASSOCIATE. Plus the rotation algebra the pose solve uses: `eye`/`rotate`/`compose`/`invert`/`from_angle`/`to_angle`/`cross`/`gram_schmidt`/`frame_from`/`solve_rotation` (TRIAD)/`orthonormalize`. **Two 2026-07-15 CUT-OVERS, same lesson twice:** `ModularOperator`+`RotationOperator` DELETED (a discrete SDR state makes every action a PERMUTATION — only lattice-preserving group elements; measured: quantised phases drift LINEARLY, exact off-grid rotation needs N > 2π·radius modules); then the scalar `heading°` DELETED (an SO(2)-only encoding — SO(3) is 3-DOF, and both Monty and Gao 2021 say the orientation IS a matrix), so degrees are a 2-D READ-OUT and `wrap` is gone. Obstacle override deferred | WIRED | `test_operator_path_integration`, `test_operator_non_abelian`, `test_rotation_recognition` |
| `pooler.py` | L2/3 — the STABLE object-IDENTITY (ARCHITECTURE §8). Three jobs: **`support(l4, identity)`** grades ONE named identity (associative recall — the WEIGHT the column's (object,pose) population is scored with); **`settle(identity)`** holds what the column concluded; **`mint`+`bind`** LEARN, driven by `Column.commit` at an EPISODE boundary (the fix for the measured MERGE — per-fixation commitment could not revise, so two objects sharing a feature-at-location fused into one chimeric identity). The support-only `pool()`/`persist_frac`/`which()` are DELETED (2026-07-16): they could answer "which object does this code support?" but never "…and where am I on it?", so they silently required the sensor to sit at the object's learned origin — the column's pose-solving population subsumes them. ASSOCIATE in a pooling regime (decoupled stable output the `HTMLayer` can't do), NOT a third primitive. Identity ONLY (structure in L4/L6/L5). Cross-column VOTING (thalamus) deferred | WIRED | `test_l23_pooling`, `test_object_centric`, `test_rotation_recognition` |
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

Suite: **66 passed** (~18s; `test_column_arithmetic` is the ~16s end-to-end column test, the rest — `test_bg_thalamus`,
`test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`,
`test_object_centric`, `test_rotation_recognition`, `test_object_dynamics` — are fast). Count history, 2026-07-15: 54 → **46** at the continuous
cut-over (`test_oriented_grid` + `test_rotation_operator` deleted with the discrete-rotation code they covered) → **48** with
R4's recognition tests → **51** with R5's (the merge regression, the partial-view guard, learning a rotated known object
without duplicating it) → **57** with R6/SO(3) (`test_operator_se2` renamed `test_operator_non_abelian`, now covering SE(3)
too: 3-D orientations, a tilted axis, chirality, the collinear limit, and rotations that do not commute) → **59** with R7's
emergent learning boundary (a continuous sweep over two known objects, and over a known object into a novel one) → **60** with
online pose-solving (the same object recognised online, placed anywhere and entered mid-object) → **64** with `test_object_dynamics`
(ONE demo of an action on an object → correct at unseen positions, on a rotated object, and on an object never seen to move).
Rotation is tested
end-to-end through *recognition* — the property we actually wanted — rather than through the machinery that implements it. Run
`python src/tests/test_reachability.py` for the wired map; the 20 legacy test files are archived under
`Legacy - DO NOT USE OR IMPORT!/tests/`.

## Next
Wired so far: the **forward-model** slice (sensory ⊕ task `Column`, place-invariance win ≈ **100%** on a place never trained,
`test_column_arithmetic`); the **decision** slice (basal-ganglia value SELECTION + thalamus relay/gate, learns a
context→action map from reward, `test_bg_thalamus`); and the **path-integration** slice (L6a's `MotionOperator` TRANSFORM
primitive — an action's effect learned in a small region dead-reckons correctly into NEVER-VISITED positions,
`test_operator_path_integration`); the **L4↔L6a loop** (predict the FEATURE at the path-integrated location —
ORDER-INVARIANT: a 3×3 object learned in one order is predicted 9/9 traversed in another, `test_feature_at_location`); and
**L2/3 pooling** (the L4 stream → a STABLE object identity — recognised invariant to fixation order, ONE identity per object,
`test_l23_pooling`); and the **NON-ABELIAN SE(2) operator** (heading-dependent motion — FORWARD;TURN ≠ TURN;FORWARD, dead-
reckons a full path into novel poses, and — after the cut-over — generalises to headings NEVER OBSERVED, `test_operator_non_abelian`);
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
   and **R5 EPISODE-level learning** (`commit`: buffer → recognise → bind, which killed the measured MERGE where two objects
   sharing a feature-at-location fused into one chimeric identity; learning now inherits pose-invariance, so a known object
   met at a novel pose is reinforced, not duplicated), **R6 SO(3)** (orientation is a MATRIX, so 2-D was the special case:
   ONE code path recognises objects at arbitrary 3-D orientations, with chirality and SE(3) non-commuting rotations falling
   out of the group structure — and no angular resolution to sample, so SO(3) costs no more than SO(2)), and **R7 the EMERGENT
   LEARNING BOUNDARY** (a continuous sweep SPLITS ITSELF: a known object is reinforced and a novel neighbour is minted and
   becomes recognisable alone, WITHOUT the caller marking anything — the boundary is refutation, and `_exhausts` reads it).
   and **ONLINE POSE-SOLVING** (`perceive` narrows a live (object, pose) population per fixation — Monty's evidence-based LM
   — so it recognises an object entered ANYWHERE at ANY pose: shifted to (7,3) went `[-1,-1,-1]` → `[0,0,0]`, entered
   mid-object `[-1,-1]` → `[0,0]`; L2/3's support-only `pool()` and the boundary's re-anchoring DELETED as superseded), and
   **the OPERATOR OVER OBJECT POSES** (`Column.dynamics` — an L5 engine fed by the poses `recognize` SOLVES, so perception →
   dynamics is end-to-end: from ONE demonstration of a shove it predicts correctly at positions never demonstrated, on the
   same object ROTATED (90°, 217°), and on a DIFFERENT object never seen to move — the §7 lesson on objects. Object motion is
   EXTRINSIC, measured: an intrinsic operator sends a 90°-turned block 90° off the shove it was shown).
   What remains = the rest of **ROADMAP Phase 3b, OBJECT DYNAMICS + RELATIONS (ARCHITECTURE §9)** — inserted BEFORE the
   value/BG rework, because a critic with no forward model of OBJECTS has nothing to plan over:
   (a) **COMMON FATE — HALF BUILT (2026-07-16), and the half that is missing is named.** BUILT: `_common_fate_groups`
   (+ `look_again`) groups a look by MOTION with NO model — a scene swept as one episode, never told there are two things,
   groups `[[0,1,2,3]]` while static and `[[0,1],[2,3]]` the moment one part moves; a rigidly-moved scene stays ONE group.
   Its two REFUSALS were forced by measurement (unguarded it shattered a symmetric object, and auto-rolling the buffer
   fragmented a chiral pair): it refuses when a feature REPEATS in a look, and it must be TOLD the scene is the same one —
   "the previous EPISODE" is not "the previous LOOK". NOT BUILT: making a grouping PERSIST as objects — the moved part lands
   where L4 never sensed it (its mint defers), the next static look re-groups as one, and splitting an ALREADY-LEARNED blob
   needs UN-BINDING, which the pooler cannot do (bind only strengthens). The general fix for the refusals is the one `_key`
   needs anyway: motion should NARROW A POPULATION of correspondences, not be read off a dict;
   (b) **the operator's KEY, DISCOVERED not given** — today the caller names an action; gravity's key is a CONDITION. "Every
   object falls alike" is a hypothesis the world can refute (feathers), which today's operator states by keying on nothing;
   (c) **L5 DISPLACEMENT CELLS + the context-gated OVERRIDE = gravity AND walls, ONE slice** (§9): displacement is
   `location + location → the relation` ("resting on"; also compositional objects); then the operator is the free kernel and
   a local relational context predicts the exception. A table stopping a fall and a wall stopping a push are ONE mechanism.
   Then: MORPHOLOGICAL features (seed the pose from ONE fixation — Monty's path — instead of needing n; an improvement, not a
   prerequisite); the SR (ROADMAP 3c) accumulating the operator into value; cross-column VOTING = the thalamus (#3).
   DEFERRED to a MOVING sensor: the ego→allo transform, so an extrinsic law stays invariant when the observer moves — the
   HIPPOCAMPUS's job, not a column's ([[reference_tbt_frames_and_hippocampus]]); ARC's static frame does not need it.
3. **The thalamus's binding role** (content ⊗ location across two columns → place-value / cross-column VOTING) + the factored
   state via L4's basal `context=` channel; then **L2/3 recognition/voting**, **L5 motor**, **hippocampal rollout**. Each
   added as a task exercises it, driven by `agent.py`. Nothing counts until imported from `agent.py` AND the agent plays more
   than before (RULES.md #3). No single-column experiments, ever (ARCHITECTURE.md §5.1).

## How to answer "where are we?"
Run the reachability test (once it exists); it reports the wired module map. Keep this table equal to that output.
