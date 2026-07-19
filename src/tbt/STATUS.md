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
| `agent.py` | live entry point / ROOT — composes a SENSORY + a TASK + a NAV + a COMPOSITIONAL/SCENE `Column` (the last built lazily; the override is genuinely MULTI-COLUMN — `place_object` routes recognised (object-id, pose) up via the thalamus, `learn_behavior`/`predict_behavior` run the state-conditioned dynamics there), the sensorimotor SCAN, a DECISION loop (`decide`/`reward` — the reward now trains a real TD VALUE CRITIC whose δ drives the actor; `reward(r, next_context, done)` for multi-step), a SPATIAL nav slice (`learn_move`/`locate`/`path_integrate`/`where`; `dims=2\|3` picks the space — an ENVIRONMENT property), the L4↔L6a loop (`sense_at`/`predict_feature`: feature at location — order-invariant), NON-ABELIAN SE(n) path integration (`set_pose`/`learn_pose_move`/`path_integrate`/`pose` — non-commutative, orientation as a continuous MATRIX; ARCHITECTURE §8), and the OBJECT surface: `start_object` (onset: re-anchor + fresh identity + clear buffer), `perceive` (INFER one fixation → the stable identity; emergent boundary), `sense_sweep` (buffer a fixation), `commit` (LEARN the buffered episode), `recognize` (POSE-INVARIANT recognition → the hypothesis population); the `step(obs)→action` game loop is still a stub | ROOT | `test_column_arithmetic`, `test_bg_thalamus`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`, `test_object_centric`, `test_rotation_recognition`, `test_override`, `test_key_discovery` |
| `column.py` | the cortical COLUMN — 5 layers (L4, L2/3, L5IT, L5PT, L6a) as one `HTMLayer` each, wired per its §12 table; `observe` drives L4 (arithmetic); a SPATIAL column (`location=GridEncoder`) also gains L6a's CONTINUOUS pose state `_pose=((x,y),heading°)` + its TRANSFORM engine (`MotionOperator`), with the grid code a per-fixation READ-OUT (`_code`), the L4↔L6a feature-at-location loop (`sense_at` depolarises L4 by the location → location-specific firing), L2/3's `ColumnPooler`, the OBJECT-CENTRIC frame (`start_object` — one recognition-failure event re-anchors the frame AND starts a fresh identity), SE(2) pose integration, the **L4→L6a associative link** (`_link`/`_union_for` — a feature recalls the UNION of (identity, location) where it occurs; Lewis 2019), POSE-INVARIANT recognition in ANY dimension (`recognize`/`Hypothesis`/`_pin_rotation`/`_solve` — the pose is SOLVED from displacement geometry via the TRIAD method, never scanned, so SO(3) costs no more than SO(2); ties = indistinguishable poses), ONLINE recognition over a narrowed hypothesis POPULATION (`perceive`/`_narrow`/`_pop`/`_union_identities` — solves its place on the object per fixation; `_fit` is the ONE scoring rule both the batch and online paths use), **L5 OBJECT DYNAMICS** (`dynamics`/`learn_object_move`/`predict_object_move` — the same TRANSFORM primitive applied to an OBJECT's pose rather than the sensor's; ONE demo generalises to any position/orientation/object), **L5PT EFFERENCE** (`efference`/`apply_efference`/`to_world` — the world-frame self-motion emitted as the efference copy + broadcast to peers so a moving sensor's self-motion is subtracted, not read as object-motion; the moving-sensor fix, `Agent.broadcast_efference`) + **L5PT DISPLACEMENT / RELATIONS** (`relate`/`observe_relation`/`relation_of` — the relative pose between two objects, `location+location→the relation`; a relation = a displacement STABLE as the pair moves, assumed then dissolved on independent motion), the **COMPOSITIONAL / SCENE** role (`place_object`/`state_of`/`learn_behavior`/`predict_behavior` + `_scene_objects` — objects-as-features one level up; the STATE-CONDITIONED override: the free kernel `(action,∅)` + per-CUE **Rescorla-Wagner** corrections (`_cue_weights`/`_cue_prediction`/`_cue_lr`) that DISCOVER the true condition and BLOCK spurious correlates by prediction error — measured `w(support)=0.998`, `w(neighbour)=0.002`; geometry-keyed, generalises across objects), and EPISODE-level learning (`commit` — recognise-then-bind, the bar being "nothing REFUTES it" rather than a score; `_replay` scores AND binds in ONE traversal, so learning inherits pose-invariance; and a sweep that crosses an object BOUNDARY SPLITS ITSELF and recurses — `_exhausts`/`_extent` tell a boundary from a shares-a-prefix object by asking whether the prefix reached the object's EDGE, so no boundary cue is needed from the caller); the full §13 `step` counterstream is the target | WIRED | `test_column_arithmetic`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`, `test_object_centric`, `test_rotation_recognition`, `test_object_dynamics`, `test_l5_displacement`, `test_override`, `test_key_discovery` |
| `operator.py` | the TRANSFORM primitive (ARCHITECTURE §8/§9), DIMENSION-GENERIC — `MotionOperator` LEARNS each action's displacement + rotation (running mean) in the frame that holds it INVARIANT, and `apply`s it to the CONTINUOUS pose `(position, R)`. **Two ways the group acts, an empirical fact about the action:** `ego=True` acts on the RIGHT (`p'=p+R·d`, `R'=R·ΔR`) — a BODY's own motion, "forward from where I face", non-commutative for free, and in 3-D the ROTATIONS stop commuting too (yaw∘pitch ≠ pitch∘yaw), which SE(2) cannot exhibit; `ego=False` acts on the LEFT (`p'=p+d`, `R'=ΔR·R`) — an OBJECT's motion, which does not care which way the object faces (measured: an intrinsic operator sends a 90°-turned block 90° off the demonstrated shove). Either way, position/orientation-invariant by construction — one observation generalises everywhere. The second primitive alongside `htm.py`'s ASSOCIATE. Plus the rotation algebra the pose solve uses: `eye`/`rotate`/`compose`/`invert`/`from_angle`/`to_angle`/`cross`/`gram_schmidt`/`frame_from`/`solve_rotation` (TRIAD)/`orthonormalize`. **Two 2026-07-15 CUT-OVERS, same lesson twice:** `ModularOperator`+`RotationOperator` DELETED (a discrete SDR state makes every action a PERMUTATION — only lattice-preserving group elements; measured: quantised phases drift LINEARLY, exact off-grid rotation needs N > 2π·radius modules); then the scalar `heading°` DELETED (an SO(2)-only encoding — SO(3) is 3-DOF, and both Monty and Gao 2021 say the orientation IS a matrix), so degrees are a 2-D READ-OUT and `wrap` is gone. Obstacle override deferred | WIRED | `test_operator_path_integration`, `test_operator_non_abelian`, `test_rotation_recognition` |
| `pooler.py` | L2/3 — the STABLE object-IDENTITY (ARCHITECTURE §8). Three jobs: **`support(l4, identity)`** grades ONE named identity (associative recall, per fixation — the burst-INDEPENDENT quantity ART's choice/vigilance are counted from in `Column._replay`); **`settle(identity)`** holds what the column concluded; **`mint`+`bind`** LEARN, driven by `Column.commit` at an EPISODE boundary (the fix for the measured MERGE). Config for ART: `alpha` (choice, the size-principle bias toward the smallest model — read by `Column._choice`) + `rho` (vigilance, the deferred sensor-noise knob; ρ=1 today = the `refuted_at` bar). DELETED: the support-only `pool()`/`persist_frac`/`which()` (2026-07-16 — silently required the sensor at the object's origin; the pose-solving population subsumes them), and the L4-CELL `choice`/`match`/`receptive_field` (wrong granularity — burst-binding inflated the receptive field; ART is counted at the feature-at-location level instead). ASSOCIATE in a pooling regime, NOT a third primitive. Identity ONLY. Cross-column VOTING (thalamus) deferred | WIRED | `test_l23_pooling`, `test_object_centric`, `test_object_dynamics`, `test_rotation_recognition` |
| `basal_ganglia.py` | the value-driven SELECTOR — OpAL Go/NoGo + dopamine-RPE + STN commitment (reference_basal_ganglia); the ONE place competing options are arbitrated (rule 4); the actor learns from the CRITIC's δ, with tonic-DA `ρ`=`critic.rho()` gating explore/exploit. **Phase 4: DISTRIBUTED per-bit read-off** — `WG`/`WN` keyed on `(bit, action)`, `Go=Σ` over active bits, so SDR OVERLAP generalises value across contexts (measured: generalises to an unseen colour via shared shape bits; the old exact-key could not). Eligibility trace + MoE column-allocation deferred | WIRED | `test_bg_thalamus`, `test_value_critic` |
| `reward.py` | the VALUE CRITIC (ARCHITECTURE §3; ROADMAP 3c) — a linear TEMPORAL-DIFFERENCE value over the state SDR (`ValueCritic`: `value`/`learn`→δ = `r+γV(s′)−V(s)`/`rho`). Its δ is the dopamine-RPE the basal-ganglia actor trains on — the SAME delta rule as Rescorla-Wagner cue competition, `_Readout`, and the BG (reuse, not new machinery). Buys a real BASELINE (`δ=r−V`) + BOOTSTRAPPING (delayed reward propagates backward). Linear = the ROUTINE/leaf critic (a relational V* needs the ROLLOUT, which uses this at the leaf); the SR form (`V=M·R`, fast re-tuning) deferred. TD step normalised by \|active bits\| or it diverges | WIRED | `test_value_critic`, `test_bg_thalamus` |
| `thalamus.py` | the inter-column ROUTER/GATE/REGISTER (deterministic — no learner) — `relay` (percept→BG context), `gate` (real DEFAULT-OFF disinhibition of the BG winner; None ⇒ nothing enacted), `project` (transthalamic relay of a recognised (object-id, pose) UP to the compositional column), and **Phase 5 the content⊗location BINDING**: `bind` (Smolensky tensor product), `bundle` (support Counter, overlap=agreement), `read` (unbind at a location, support ≥ min_support). Serves PLACE-VALUE (exact roundtrip) + cross-column CMP VOTING (min_support=k = k columns agree) | WIRED | `test_bg_thalamus`, `test_override`, `test_key_discovery`, `test_thalamus_binding` |
| `htm.py` | the ONE cortical-layer mechanism — HTM sequence memory (proximal SDR-in via SP, basal context, apical, learn/predict/burst); a layer = one instance + a declared (proximal-in, context-in, apical-in, target-out) wiring; `depolarize(context)` lets an EXTERNAL context (L6a location) drive which cell fires (sensorimotor feature-at-location), not just the recurrent sequence | WIRED | `test_htm` |
| `encoders.py` | SDR transduction library (`SDR` + Scalar/Category/Grid/Multi/Conjunctive/SpatialPooler) — data ↔ overlap-bearing SDR. `GridEncoder` is axis-aligned modular phases, and after the cut-over is a pure READ-OUT of L6a's continuous pose (encode-per-fixation ⇒ its quantisation is bounded and NEVER accumulates). The `orientations=N` multi-orientation variant was REMOVED with `RotationOperator` (see `operator.py`) | WIRED | `test_encoders` |
| `hippocampus/map.py` | the HIPPOCAMPUS's EC/place core — `WorldMap`, the forkable allocentric world-STATE the rollout simulates in (DESIGN §2, slice 1 of the four-part build). Binds the agent's self-location (nav L6a pose) + objects at world poses (scene column) + the frame boundary into ONE state; `snapshot`/`move_agent` fork it and path-integrate the agent under the column's SHARED learned operator (borrowed by reference — the fast-forkable-state / slow-shared-model split, `reference_hippocampus`); `place`/`remove`/`anchor` (loop closure) mutate a fork; `key` = the visited-pruning signature. Assembled by `Agent.world_state`; DERIVED from the columns, not a parallel store | WIRED | `test_world_map` |
| `hippocampus/replay.py` | the ROLLOUT — model-based planning IN the world-map (DESIGN §2/§3, slice 2). `WorldModel` = the learned forward model over a world-state, COMPOSING the agent's path-integration (map operator) with each object's STATE-CONDITIONED dynamics (the compositional column's RW cue competition — the agent included as a relatum, so a PUSH reads the agent's pre-move position); `step` unrolls one action, `learn` folds one observed transition in. `Rollout` = value-guided BFS over world-states with VISITED-PRUNING (O(states), not the 2^K flat-action product) — shortest goal-reaching path, else toward the best-VALUE leaf (critic as heuristic); `greedy` = the 1-step baseline it beats. Wired via `Agent.world_model`/`Agent.plan`. Measured: a learned PUSH (Sokoban) solved by going AROUND to the far side (`S,W,E,E`) where a greedy step can't; the EZ-V2 sampled search is the branching scale-up | WIRED | `test_replay` |
| `hippocampus/ca3.py` | the CA3 autoassociative ATTRACTOR (DESIGN §2/§3, slice 3) — ONE recurrent structure doing BOTH the one-shot EPISODIC store AND pattern COMPLETION (Rolls/Treves). `store` = one-shot Hebbian (co-active bits strengthen the recurrent weight); `complete` settles from a PARTIAL/NOISY cue to the whole stored pattern by GROW-then-PRUNE (monotonically add bits co-occurring with ≥ `theta`·\|active\|, then one prune) — so even a 1-bit cue / 2-token pattern recovers where a strict recompute would OSCILLATE (found by the slice-6 orchestrator test); noise still drops in the prune. Carries §3½ one region up: partial cue COMPLETES, noise DROPS, an AMBIGUOUS cue stays ambiguous (the UNION, no confabulation), a NOVEL cue recalls nothing. Sparse Hopfield/SDM over hashable bits (ints or (object,cell) episode tokens). Wired via `Agent.remember_scene`/`recall_scene` (a glimpse of one object recalls the whole scene — the maze-wall case). Heavy pattern OVERLAP cross-talks — that is DG's job (slice 4), not a flaw | WIRED | `test_ca3` |
| `hippocampus/dg.py` | the dentate gyrus — PATTERN SEPARATION → chart keys (DESIGN §2/§3, slice 4). `separate(signature)` maps an environment signature (set of input bits) to a sparse, orthogonalized CHART KEY, REUSING the k-WTA `SpatialPooler` (a fixed sparse projection + winner-take-all IS separation). DETERMINISTIC (`learn=False`) so the same environment returns the same key; DECORRELATES (distinct envs → near-disjoint keys, measured 0/24 overlap; a similar view keeps 11/24 — graded). Closes CA3's overlap seam: overlapping raw signatures cross-talk in CA3, their DG keys are recalled cleanly. Wired via `Agent.chart_key`; the base for multi-chart REMAPPING (slice 5) | WIRED | `test_dg` |
| `hippocampus/__init__.py` | the `Hippocampus` ORCHESTRATOR (DESIGN §2/§3, slice 6, the LAST) — composes the subfields into ONE region behind a single agent handle: EPISODIC (`remember`/`recall` = CA3), REMAPPING (`chart_key`/`visit` = DG+CA3+CA1), PLANNING (`plan` = replay over a cortex-assembled world+model). Holds the stateful memory; the world-state assembly (nav pose + scene objects) is the cortex→hippocampus bridge kept on the agent. `Agent.__init__` now holds ONE `self.hippocampus` (the per-region `ca3`/`dg`/`remapper` fields folded behind it — clean cutover); `remember_scene`/`recall_scene`/`chart_key`/`visit_environment`/`plan` all delegate | WIRED | `test_hippocampus` |
| `hippocampus/featurize.py` | the world-state → SDR FEATURISER for the value critic — `WorldFeaturizer.encode(world)` maps a `WorldMap` (agent self-location + objects at places) to an OVERLAP-BEARING SDR of `(entity, axis-bit)` bits. Position is a per-axis METRIC `ScalarEncoder` (overlap decreases monotonically with distance), NOT the modular `GridEncoder` — the grid ALIASES over distance and would break value smoothness (`reference_sdr_regime_and_phase_codes`: SDR great for IDENTITY, breaks for METRIC). Closes replay.py's leaf-value seam: `Agent.value_of` = the `ValueCritic` scoring the featurised world (the rollout leaf, defaulted in `plan`); `Agent.learn_value` trains it on world-state transitions by TD. Measured: adjacent states share 17/18 place-bits, far states 0; a trained critic pulls the rollout toward a goal beyond the horizon where an untrained one gives no plan | WIRED | `test_featurize` |
| `hippocampus/ca1.py` | the CA1 COMPARATOR + multi-chart REMAPPING (DESIGN §2/§3, slice 5). `CA1.compare(observed, recalled)` = the match/novelty detector (Lisman/Hasselmo): MATCH iff the recall explains every observed bit (observed ⊆ recalled) — the §3½ rule one region up, so a PARTIAL view matches (absence) and a CONTRADICTED bit mismatches (novelty). `Remapper` composes CA3 (recall) + CA1 (compare): revisit → RECALL the chart, a novel/CHANGED environment → MINT a new one. Comparison runs on CONTENT tokens (subset-preserving), NOT DG keys (k-WTA breaks the subset relation) — DG separates full signatures at the index layer, composed in slice 6. Wired via `Agent.visit_environment`. Measured: glimpse recalls chart 0, `A`+contradicting-`X` remaps (novelty 0.25) | WIRED | `test_ca1` |

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

Suite: **125 passed** (~28s; `test_column_arithmetic` is the ~16s end-to-end column test, the rest — `test_bg_thalamus`,
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
   and **COMMON FATE + the ART cut-over** (2026-07-16): `_common_fate_groups` (+ `look_again`) groups a look by MOTION with NO
   model (a scene swept as one episode, never told there are two things, groups `[[0,1,2,3]]` static and `[[0,1],[2,3]]` the
   moment one part moves; a rigid move stays ONE group; it REFUSES on a repeated feature or a mismatched look). `commit` then
   makes the grouping PERSIST via `_commit_split` — the ART orienting RESET: a scene split by motion is >1 object, no identity
   may claim two groups, so parts are RECRUITED fresh, closing R7's cold-start blob (a blob learned from two things always
   seen together TEARS into its parts once one moves; measured: library stays bounded under continuous motion, each part
   recognisable alone). The un-binding answer is DON'T — ART **CHOICE** (`_choice` = matched/(α+|model|), the size principle)
   keeps a torn-off part winning over the blob, which dies of disuse; `perm_dec` stays dead. Choice + vigilance are counted at
   the FEATURE-AT-LOCATION level (`_replay`'s `matched`, via L2/3 `support`), NOT raw L4 cells — burst-binding inflated an
   L4-cell receptive field and tied a piece with its parent (the pooler's L4-cell `choice`/`match`/`receptive_field` were
   DELETED as the wrong granularity).
   and **L5PT DISPLACEMENT / relations** (`relate`/`observe_relation`/`relation_of` — `location+location→the relation`, the
   relative pose between two objects, position/orientation-invariant; a relation is a displacement STABLE as the pair moves,
   "resting on"/"part of"; assumed from the first view, dissolved on independent motion — measured. NB L5 = IT + PT; this is
   the PT thick-tufted role, the L5IT integrator deferred).
   and the **CONTEXT-GATED OVERRIDE = gravity + support, MULTI-COLUMN** (2026-07-16): the first COMPOSITIONAL slice — a second
   `Column` (`Agent._scene`) fed recognised (object-id, pose) from the sensory column via `Thalamus.project`, running
   STATE-CONDITIONED dynamics keyed on `(action, STATE)` (`state=∅` = free kernel; a supported object's non-null state gets its
   own keyed effect; unlearned states fall back to the kernel). Measured: supported stays, free falls, a NEW object generalises
   (object-independent, geometry-keyed state), support removed → falls again (assume-then-correct). No `if supported`. Honours
   §5.1 + exercises the thalamus binding role.
   and the **operator's KEY, DISCOVERED** (2026-07-18) — Rescorla-Wagner CUE COMPETITION over the state's features (`_cue_weights`):
   each cue learns its correction by prediction error; the BLOCKING effect (Kamin) gives a spurious co-present correlate ~0
   weight, and contingency zeroes a correlate that sometimes appears without the effect. Measured: `w(support)=0.998`,
   `w(neighbour)=0.002`, a neighbour-only block FALLS (rejected as a condition). RW = the delta rule the `_Readout`/BG-RPE
   already run (reuse); emergent, no `if supported`. Incremental (converges over views), which relaxed `test_override`'s
   exact assertions to "converges to".
   **Phase 3b DONE. Then the VALUE CRITIC (ROADMAP 3c, `reward.py`) ✅ DONE (2026-07-18)** — a linear TD value over the state
   SDR whose δ (`r+γV(s′)−V(s)`) replaces the faked `2r−1` (the SAME delta rule as the RW cue competition above, reused). The
   bandit still learns (now with a baseline), and a DELAYED-reward corridor is solved (value propagates backward, which the
   faked RPE could not do). This is the 9th wired module.
   **Then Phase 4 — the BASAL GANGLIA redesign ✅ DONE (2026-07-18):** the OpAL actor now reads value off the ACTIVE BITS
   (`WG`/`WN` per `(bit,action)`), so value GENERALISES across overlapping contexts (measured: transfers to an unseen colour
   via shared shape bits, which the exact-key actor could not); ρ from the critic wired. Eligibility trace deferred.
   **Then Phase 5 — the THALAMUS redesign ✅ DONE (2026-07-18):** the content⊗location VSA register (`bind`/`bundle`/`read`,
   wired via `Agent.vote_consensus`) — place-value roundtrip + cross-column CMP voting (min_support=k); `gate` is now a real
   default-off disinhibition. The decision loop is now a full ACTOR-CRITIC (critic δ → generalising per-bit actor → default-off gate).
   What remains, in order:
   (a) **Phase 6 — THE FULL FOUR-PART HIPPOCAMPUS (IN PROGRESS, plan: `hippocampus/DESIGN.md`).** Decided up front (not the
   single-chart rollout alone): the EC-map / DG / CA3 / CA1 subfields + replay, in the `hippocampus/` subpackage. Build order
   map → replay → CA3 → DG → CA1+remapping → orchestrator; each wired from `agent.py` + exercised.
   **Slice 1 (`hippocampus/map.py`) ✅ DONE (2026-07-18)** — `WorldMap`, the forkable allocentric world-STATE, path-integrated
   under the shared operator; `Agent.world_state` assembles it. Removes the rollout block: nothing before held a coherent,
   forkable "world right now, including me".
   **Slice 2 (`hippocampus/replay.py`) ✅ DONE (2026-07-18)** — the ROLLOUT (`WorldModel` forward+learn, `Rollout` BFS with
   visited-pruning + best-value-leaf fallback + `greedy` baseline), wired via `Agent.world_model`/`Agent.plan`. Solves a
   delayed goal where a 1-step greedy can't, incl. a LEARNED push (Sokoban) solved by going around (`S,W,E,E`) — object
   dynamics driven INSIDE the rollout, no hand-coded physics. The critic-as-leaf-heuristic hook is present (`value=`); training
   it on world-state SDRs (a world→SDR featuriser) is folded into the full loop. This is the imagined-future widget's substrate.
   **§3½ occlusion/partial-view invariant (DESIGN §3½, `reference_recognition_under_occlusion`): mint on REFUTATION, never on
   incompleteness. Column STRICT-SUBSET falsifier ✅ DONE (2026-07-18, `test_partial_recognition`):** a partial view (even
   rotated/translated) recognises as the WHOLE and does not mint; a CONTRADICTING feature is refuted + mints. It caught a real
   gap — `recognize` computed vigilance (`refuted_at`) but never applied it, so a contradicting view survived when it was the
   sole hypothesis; fixed by filtering the population to `refuted_at is None or _exhausts(h)` (refuted-within-extent = a
   contradiction, excluded; refuted-after-exhausting = a boundary, kept, which `commit` needs). CA1 absence-vs-contradiction
   is built with slice 5.
   **Slice 3 (`hippocampus/ca3.py`) ✅ DONE (2026-07-18)** — the CA3 autoassociative attractor (one-shot store + partial/noisy
   cue completion; ambiguous→union, novel→nothing), wired via `Agent.remember_scene`/`recall_scene`. Falsifier first-class (`test_ca3`).
   **Slice 4 (`hippocampus/dg.py`) ✅ DONE (2026-07-18)** — dentate gyrus pattern SEPARATION → chart keys (reuses the k-WTA
   `SpatialPooler`): deterministic, decorrelates (distinct envs → 0/24 key overlap, similar view → 11/24 — graded), and closes
   CA3's overlap seam. Wired via `Agent.chart_key`.
   **Slice 5 (`hippocampus/ca1.py`) ✅ DONE (2026-07-18)** — the CA1 comparator (`compare` = MATCH iff observed ⊆ recalled) +
   `Remapper` (CA3 recall + CA1 compare → recall a known chart or mint a new one). The §3½ absence-vs-contradiction falsifier
   is now first-class at the scene level (`test_ca1`): a partial view RECALLS, a contradicted view REMAPS. Mechanism refinement
   recorded (DESIGN §2): the comparison runs on CONTENT tokens, not DG keys (k-WTA breaks the subset relation). Wired via
   `Agent.visit_environment`.
   **Slice 6 (`hippocampus/__init__.py`, the `Hippocampus` ORCHESTRATOR) ✅ DONE (2026-07-18) — THE FULL FOUR-PART HIPPOCAMPUS
   IS COMPLETE.** map ⊕ replay ⊕ CA3 ⊕ DG ⊕ CA1 composed behind ONE agent handle (`self.hippocampus`); the per-region
   `ca3`/`dg`/`remapper` fields folded behind it (clean cutover, existing tests validated the delegation). End-to-end
   (`test_hippocampus`): one agent routes PLANNING (replay to a goal) + EPISODIC recall (glimpse → whole scene) + REMAPPING
   through the single handle. The orchestrator test also caught + fixed a CA3 2-token oscillation (grow-then-prune completion).
   **The world→SDR value FEATURISER ✅ DONE (2026-07-18, `hippocampus/featurize.py`, `test_featurize`)** — closed replay.py's
   leaf-value seam: `Agent.value_of`/`learn_value` let the `ValueCritic` score + learn world-STATES (metric `ScalarEncoder`
   place code, not the aliasing grid), and `plan`'s leaf now defaults to the trained critic.
   **BUILD ALL OF L5 HONESTLY (user, 2026-07-18) — no deferral, no inert placeholders** (`notes/l5_efference_broadcast_design.md`).
   Corrected from an earlier "defer L5IT + keep the declared-but-undriven `layers["L5IT"]`/`layers["L5PT"]`" — that is dead
   weight that LIES about what it does, and "not needed for ARC" is not a reason to half-build a brain region. Four steps, each
   wired + falsified, suite green throughout:
   **Step 1 (L5PT EFFERENCE) ✅ DONE (2026-07-18, `test_efference`)** — `Column.efference(action)` (world-frame self-motion) +
   `Column.apply_efference(Δ)` (peer flow-parsing) + `Column.to_world(ego)` (egocentric→world) + `Agent.broadcast_efference`.
   Measured: the body moves E, a STATIC object at (20,10) → the peer recovers (20,10) WITH the efference, misreads (19,10)
   WITHOUT; a moving object nets to its true world-motion. The moving-sensor fix's substrate — grounded by [[reference_layer5_role]]
   (displacement = motor = efference copy = inter-column message), [[reference_hippocampus]] (the world-map is the allocentric
   destination, now BUILT).
   **Step 2 (L5IT → BG) ✅ DONE (2026-07-18)** — mechanism check REVISED the plan (the old "L5PT motor / fold the `_Readout`" was
   wrong: the `_Readout` is peripheral BY DESIGN — value-readout, value-free cortex; L5PT's motor is already operator+efference+BG).
   The real scattered-L5 was L5IT: `Column.striatum` DRIVES `layers["L5IT"]` so `decide`/`reward` route the BG's cortical input
   through a column's L5IT (the striatal projection), not a raw-L4 relay. **HONEST caveat:** a single-stimulus decision makes the
   frozen L5IT projection TRANSPARENT (bursts the same cells L4 would) — the correct anatomical path + a driven L5IT, not new
   behaviour; genuine L5IT POOLING bites only for a multi-fixation object identity → BG (the OPEN increment).
   **L5PT placeholder ✅ DELETED (2026-07-18)** — the inert `layers["L5PT"]` HTMLayer is gone; the `MotionOperator`/`dynamics`/
   `relate`/`efference` ARE L5PT's mechanism (an HTMLayer is the wrong tool for continuous displacement), stated at the dict.
   **OPEN:** the genuine multi-fixation L5IT integration (object identity → BG, an object-based decision task) is the real
   exercise of L5IT, not yet demonstrated. **Step 4:** reconcile ARCHITECTURE §9/§12/§15. See [[reference_l5_operator_kinds]].
   After L5: the loop/brain object (move `decide`/`scan` OUT of the agent) → the THIN agent; the `step(obs)→action` game loop; the
   two-pane imagined-future WIDGET. Deferred within the hippocampus (DESIGN §4): theta/replay timing, graded remap.
   Still after the hippocampus: the MOTOR region (L5 emit + decode; move the readout OUT of the agent); the loop/brain object
   (move `decide`/`scan` OUT of the agent); the THIN agent; the `step(obs)→action` game loop. Plus a full multi-sensory-column
   recognition-BY-VOTING task consuming the Phase-5 register.
   (b) **object-dynamics refinements** (not blocking): abstraction across GEOMETRY VARIANTS (overlap recall over relation SDRs,
   the `_key` refinement everywhere) + intrinsic-feature cues; a WALL (partial-effect cue) = the same RW mechanism; the SR
   form of the critic (`V=M·R`, fast reward-re-tuning) for a task that needs it; MORPHOLOGICAL features; the eligibility trace.
   DEFERRED to a MOVING sensor: the ego→allo transform ([[reference_tbt_frames_and_hippocampus]]); ARC's static frame needs none.
   DEFERRED to a MOVING sensor: the ego→allo transform, so an extrinsic law stays invariant when the observer moves — the
   HIPPOCAMPUS's job, not a column's ([[reference_tbt_frames_and_hippocampus]]); ARC's static frame does not need it.
3. **The thalamus's binding role** (content ⊗ location across two columns → place-value / cross-column VOTING) + the factored
   state via L4's basal `context=` channel; then **L2/3 recognition/voting**, **L5 motor**, **hippocampal rollout**. Each
   added as a task exercises it, driven by `agent.py`. Nothing counts until imported from `agent.py` AND the agent plays more
   than before (RULES.md #3). No single-column experiments, ever (ARCHITECTURE.md §5.1).

## How to answer "where are we?"
Run the reachability test (once it exists); it reports the wired module map. Keep this table equal to that output.
