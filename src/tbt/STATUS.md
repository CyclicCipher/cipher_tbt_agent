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
| `agent.py` | live entry point / ROOT — composes a SENSORY + a TASK + a NAV + a COMPOSITIONAL/SCENE `Column` (the last built lazily; the override is genuinely MULTI-COLUMN — `place_object` routes recognised (object-id, pose) up via the thalamus, `learn_behavior`/`predict_behavior` run the state-conditioned dynamics there), the sensorimotor SCAN, a DECISION loop (`decide`/`reward` — the reward now trains a real TD VALUE CRITIC whose δ drives the actor; `reward(r, next_context, done)` for multi-step), a SPATIAL nav slice (`learn_move`/`locate`/`path_integrate`/`where`; `dims=2\|3` picks the space — an ENVIRONMENT property), the L4↔L6a loop (`sense_at`/`predict_feature`: feature at location — order-invariant), NON-ABELIAN SE(n) path integration (`set_pose`/`learn_pose_move`/`path_integrate`/`pose` — non-commutative, orientation as a continuous MATRIX; ARCHITECTURE §8), and the OBJECT surface: `start_object` (onset: re-anchor + fresh identity + clear buffer), `perceive` (INFER one fixation → the stable identity; emergent boundary), `sense_sweep` (buffer a fixation), `commit` (LEARN the buffered episode), `recognize` (POSE-INVARIANT recognition → the hypothesis population); and the **thin-agent GAME LOOP** `step(FrameData)→action` (`transduce`→discover self + MOVERS from motion→learn the self's operator AND each mover's PUSH dynamics online (`_learn_dynamics`)→credit the reward to discover the goal, bare-feature (nav) OR `(mover,landmark)` RELATIONAL (`goal_mem`)→route movers to the scene (`_route_movers`)→`_act`: PRAGMATIC rollout to the goal (nav, or PUSH a mover onto a landmark) else EPISTEMIC world-CONFIG novelty; walls are NOT a remembered set: `_static_cells` records what PERCEPTION sees at each untracked cell and L5 PREDICTS what pressing into that feature does, so one bump generalises to every cell of that feature — no game semantics) + `imagine` (the plan unrolled through the model, for the widget), driving LockPath (nav transfer) AND Push (the go-around, at oracle) end-to-end | ROOT | `test_column_arithmetic`, `test_bg_thalamus`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`, `test_object_centric`, `test_rotation_recognition`, `test_override`, `test_key_discovery`, `test_perceive`, `test_game_loop`, `test_push` |
| `column.py` | the cortical COLUMN — 5 layers (L4, L2/3, L5IT, L5PT, L6a) as one `HTMLayer` each, wired per its §12 table; `observe` drives L4 (arithmetic); a SPATIAL column (`location=GridEncoder`) also gains L6a's CONTINUOUS pose state `_pose=((x,y),heading°)` + its TRANSFORM engine (`MotionOperator`), with the grid code a per-fixation READ-OUT (`_code`), the L4↔L6a feature-at-location loop (`sense_at` depolarises L4 by the location → location-specific firing), L2/3's `ColumnPooler`, the OBJECT-CENTRIC frame (`start_object` — one recognition-failure event re-anchors the frame AND starts a fresh identity), SE(2) pose integration, the **L4→L6a associative link** (`_link`/`_union_for` — a feature recalls the UNION of (identity, location) where it occurs; Lewis 2019), POSE-INVARIANT recognition in ANY dimension (`recognize`/`Hypothesis`/`_pin_rotation`/`_solve` — the pose is SOLVED from displacement geometry via the TRIAD method, never scanned, so SO(3) costs no more than SO(2); ties = indistinguishable poses), ONLINE recognition over a narrowed hypothesis POPULATION (`perceive`/`_narrow`/`_pop`/`_union_identities` — solves its place on the object per fixation; `_fit` is the ONE scoring rule both the batch and online paths use), **L5 OBJECT DYNAMICS** (`dynamics`/`learn_object_move`/`predict_object_move` — the same TRANSFORM primitive applied to an OBJECT's pose rather than the sensor's; ONE demo generalises to any position/orientation/object), **L5PT EFFERENCE** (`efference`/`apply_efference`/`to_world` — the world-frame self-motion emitted as the efference copy + broadcast to peers so a moving sensor's self-motion is subtracted, not read as object-motion; the moving-sensor fix, `Agent.broadcast_efference`) + **L5PT DISPLACEMENT / RELATIONS** (`relate`/`observe_relation`/`relation_of` — the relative pose between two objects, `location+location→the relation`; a relation = a displacement STABLE as the pair moves, assumed then dissolved on independent motion), the **COMPOSITIONAL / SCENE** role (`place_object`/`state_of`/`learn_behavior`/`predict_behavior` + `_scene_objects` — objects-as-features one level up; the STATE-CONDITIONED override: the free kernel `(action,∅)` on `MotionOperator` + per-CUE corrections through the ONE L5 **`behavior.Transform`** (`_cues`/`_act_sdr`/`_cue_prediction` — state cues PROXIMAL so they sum, the action BASAL so the correction is per-action; the read-out's delta rule IS Rescorla-Wagner, so there is no second copy of the rule) that DISCOVER the true condition and BLOCK spurious correlates by prediction error — measured `w(support)=0.998`, `w(neighbour)=0.002`; geometry-keyed, generalises across objects), and EPISODE-level learning (`commit` — recognise-then-bind, the bar being "nothing REFUTES it" rather than a score; `_replay` scores AND binds in ONE traversal, so learning inherits pose-invariance; and a sweep that crosses an object BOUNDARY SPLITS ITSELF and recurses — `_exhausts`/`_extent` tell a boundary from a shares-a-prefix object by asking whether the prefix reached the object's EDGE, so no boundary cue is needed from the caller); the full §13 `step` counterstream is the target | WIRED | `test_column_arithmetic`, `test_operator_path_integration`, `test_feature_at_location`, `test_l23_pooling`, `test_operator_non_abelian`, `test_object_centric`, `test_rotation_recognition`, `test_object_dynamics`, `test_l5_displacement`, `test_override`, `test_key_discovery` |
| `operator.py` | the TRANSFORM primitive (ARCHITECTURE §8/§9), DIMENSION-GENERIC — `MotionOperator` LEARNS each action's displacement + rotation (running mean) in the frame that holds it INVARIANT, and `apply`s it to the CONTINUOUS pose `(position, R)`. **Two ways the group acts, an empirical fact about the action:** `ego=True` acts on the RIGHT (`p'=p+R·d`, `R'=R·ΔR`) — a BODY's own motion, "forward from where I face", non-commutative for free, and in 3-D the ROTATIONS stop commuting too (yaw∘pitch ≠ pitch∘yaw), which SE(2) cannot exhibit; `ego=False` acts on the LEFT (`p'=p+d`, `R'=ΔR·R`) — an OBJECT's motion, which does not care which way the object faces (measured: an intrinsic operator sends a 90°-turned block 90° off the demonstrated shove). Either way, position/orientation-invariant by construction — one observation generalises everywhere. The second primitive alongside `htm.py`'s ASSOCIATE. Plus the rotation algebra the pose solve uses: `eye`/`rotate`/`compose`/`invert`/`from_angle`/`to_angle`/`cross`/`gram_schmidt`/`frame_from`/`solve_rotation` (TRIAD)/`orthonormalize`. **Two 2026-07-15 CUT-OVERS, same lesson twice:** `ModularOperator`+`RotationOperator` DELETED (a discrete SDR state makes every action a PERMUTATION — only lattice-preserving group elements; measured: quantised phases drift LINEARLY, exact off-grid rotation needs N > 2π·radius modules); then the scalar `heading°` DELETED (an SO(2)-only encoding — SO(3) is 3-DOF, and both Monty and Gao 2021 say the orientation IS a matrix), so degrees are a 2-D READ-OUT and `wrap` is gone. Obstacle override deferred | WIRED | `test_operator_path_integration`, `test_operator_non_abelian`, `test_rotation_recognition` |
| `pooler.py` | L2/3 — the STABLE object-IDENTITY (ARCHITECTURE §8). Three jobs: **`support(l4, identity)`** grades ONE named identity (associative recall, per fixation — the burst-INDEPENDENT quantity ART's choice/vigilance are counted from in `Column._replay`); **`settle(identity)`** holds what the column concluded; **`mint`+`bind`** LEARN, driven by `Column.commit` at an EPISODE boundary (the fix for the measured MERGE). Config for ART: `alpha` (choice, the size-principle bias toward the smallest model — read by `Column._choice`) + `rho` (vigilance, the deferred sensor-noise knob; ρ=1 today = the `refuted_at` bar). DELETED: the support-only `pool()`/`persist_frac`/`which()` (2026-07-16 — silently required the sensor at the object's origin; the pose-solving population subsumes them), and the L4-CELL `choice`/`match`/`receptive_field` (wrong granularity — burst-binding inflated the receptive field; ART is counted at the feature-at-location level instead). ASSOCIATE in a pooling regime, NOT a third primitive. Identity ONLY. **Cross-column VOTING ✅ BUILT 2026-07-27 (H1), on the corrected locus** — `lat` (peer L2/3 cell → own identity cell) + `learn_lateral`/`vote`, driven by `Column.link`/`cast_vote`/`receive_votes`: the direct LATERAL cortico-cortical link (L3 cells in different columns Hebbian-linked, arXiv:2507.05888), NOT the thalamus as this file and `ARCHITECTURE` §8 used to say. `_drive` is the ONE recall for both wirings, and it COUNTS converging afferents rather than unioning: measured, two identities sharing a single cell out of 2048 made a peer's vote recall the other object in full (one shared cell fans out to all 40 of its cells) so every candidate scored 1.0 — set-union recall gives one spurious synapse the authority of forty | WIRED | `test_l23_pooling`, `test_lateral_voting`, `test_object_centric`, `test_object_dynamics`, `test_rotation_recognition` |
| `basal_ganglia.py` | the value-driven SELECTOR — OpAL Go/NoGo + dopamine-RPE + STN commitment (reference_basal_ganglia); the ONE place competing options are arbitrated (rule 4); the actor learns from the CRITIC's δ, with tonic-DA `ρ`=`critic.rho()` gating explore/exploit. **Phase 4: DISTRIBUTED per-bit read-off** — `WG`/`WN` keyed on `(bit, action)`, `Go=Σ` over active bits, so SDR OVERLAP generalises value across contexts (measured: generalises to an unseen colour via shared shape bits; the old exact-key could not). Eligibility trace + MoE column-allocation deferred | WIRED | `test_bg_thalamus`, `test_value_critic` |
| `reward.py` | the VALUE CRITIC (ARCHITECTURE §3; ROADMAP 3c) — a linear TEMPORAL-DIFFERENCE value over the state SDR (`ValueCritic`: `value`/`learn`→δ = `r+γV(s′)−V(s)`/`rho`). Its δ is the dopamine-RPE the basal-ganglia actor trains on — the SAME delta rule as Rescorla-Wagner cue competition, `_Readout`, and the BG (reuse, not new machinery). Buys a real BASELINE (`δ=r−V`) + BOOTSTRAPPING (delayed reward propagates backward). Linear = the ROUTINE/leaf critic (a relational V* needs the ROLLOUT, which uses this at the leaf); the SR form (`V=M·R`, fast re-tuning) deferred. TD step normalised by \|active bits\| or it diverges. **`GoalMemory`** = the SAME delta rule applied to reward CONTINGENCY: `credit(features, r)` competes co-present features (`w += lr·(r − Σw)`) so the reward-linked FEATURE climbs while spurious ones decay; `goal()` = the argmax feature. Feature-based ⇒ POSITION-INVARIANT, so the goal transfers across levels where the critic's absolute-position value cannot (the score→goal→plan loop; `reference_goal_setting_priority_map`) | WIRED | `test_value_critic`, `test_bg_thalamus`, `test_game_loop` |
| `thalamus.py` | the inter-column ROUTER/GATE/REGISTER (deterministic — no learner) — `relay` (percept→BG context), `gate` (real DEFAULT-OFF disinhibition of the BG winner; None ⇒ nothing enacted), `project` (transthalamic relay of a recognised (object-id, pose) UP to the compositional column), and **Phase 5 the content⊗location BINDING**: `bind` (Smolensky tensor product), `bundle` (support Counter, overlap=agreement), `read` (unbind at a location, support ≥ min_support). Serves PLACE-VALUE (exact roundtrip) + a k-of-n agreement read (`min_support=k`). **LOCUS DEBT (2026-07-22):** that agreement read is used as cross-column VOTING, but TBT puts voting in **direct LATERAL cortico-cortical** connections (L3, and also L5b + a subset of L6 — arXiv:2507.05888), NOT the thalamus; the thalamus's real roles are the ego→allo relay transform and the TRANSTHALAMIC HIERARCHICAL route (which `project` correctly serves — it is how compositional child-of structure is learned). The binding math is sound; its LOCUS is wrong and voting should move to a lateral column↔column path | WIRED | `test_bg_thalamus`, `test_override`, `test_key_discovery`, `test_thalamus_binding` |
| `htm.py` | the ONE cortical-layer mechanism — HTM sequence memory (proximal SDR-in via SP, basal context, apical, learn/predict/burst); a layer = one instance + a declared (proximal-in, context-in, apical-in, target-out) wiring; `depolarize(context)` lets an EXTERNAL context (L6a location) drive which cell fires (sensorimotor feature-at-location), not just the recurrent sequence. Also the canonical pipeline's THIRD stage, vector-valued: **`PopulationReadout`** (active cells → a continuous vector; each cell contributes a learned preferred vector and their SUM is the decode — the population vector of motor cortex and of the grid/place/HD populations, learned by the delta rule with the error shared over the active cells). So an assembly is an IDENTITY code and a METRIC code at once; there is no dichotomy to work around | WIRED | `test_htm` |
| `encoders.py` | SDR transduction library (`SDR` + Scalar/Category/Grid/Multi/Conjunctive/SpatialPooler) — data ↔ overlap-bearing SDR. `GridEncoder` is axis-aligned modular phases, and after the cut-over is a pure READ-OUT of L6a's continuous pose (encode-per-fixation ⇒ its quantisation is bounded and NEVER accumulates). The `orientations=N` multi-orientation variant was REMOVED with `RotationOperator` (see `operator.py`) | WIRED | `test_encoders` |
| `hippocampus/map.py` | the HIPPOCAMPUS's EC/place core — `WorldMap`, the forkable allocentric world-STATE the rollout simulates in (DESIGN §2, slice 1 of the four-part build). Binds the agent's self-location (nav L6a pose) + objects at world poses (scene column) + the frame boundary into ONE state; `snapshot`/`move_agent` fork it and path-integrate the agent under the column's SHARED learned operator (borrowed by reference — the fast-forkable-state / slow-shared-model split, `reference_hippocampus`); `place`/`remove`/`anchor` (loop closure) mutate a fork; `key` = the visited-pruning signature (position collapsed to the CELL — at finer resolution a converging model's fractional deltas make every near-identical state a distinct node and pruning stops working); `static` = {cell: feature}, RAW PERCEPTION rather than a verdict, read by `occupant`; `move_agent` is FREE motion that knows nothing about obstacles, because whether a press goes anywhere is the forward model's PREDICTION. Assembled by `Agent.world_state`; DERIVED from the columns, not a parallel store | WIRED | `test_world_map`, `test_game_loop` |
| `hippocampus/replay.py` | the ROLLOUT — model-based planning IN the world-map (DESIGN §2/§3, slice 2). `WorldModel` = the learned forward model over a world-state, COMPOSING the agent's path-integration (map operator) with each object's STATE-CONDITIONED dynamics (the compositional column's RW cue competition — the agent included as a relatum, so a PUSH reads the agent's pre-move position); `step` unrolls one action, `learn` folds one observed transition in. **`snap`** = an injected grid read-out that quantises each object's predicted pose to integer cells — discretises the state space (else a fractional half-learned push proliferates positions and the search explodes) and makes a one-observation push read out as a FULL cell step. **RIGID-BODY COUPLING** in `step`: an object is solid — the agent is BLOCKED if it walks into one that does not move out of the way, so a box is an obstacle to navigate AROUND and only a LEARNED push lets the agent advance (this is what stops the rollout imagining a walk-through and shoving the box the wrong way). `Rollout` = value-guided BFS over world-states with VISITED-PRUNING (O(states), not the 2^K flat-action product) — shortest goal-reaching path, else toward the best-VALUE leaf (critic as heuristic); `greedy` = the 1-step baseline it beats. Wired via `Agent.world_model`/`Agent.plan`/`Agent.imagine`. Measured: a learned PUSH (Sokoban/Push) solved by going AROUND to the far side where a greedy step can't; the EZ-V2 sampled search is the branching scale-up | WIRED | `test_replay`, `test_push` |
| `hippocampus/ca3.py` | the CA3 autoassociative ATTRACTOR (DESIGN §2/§3, slice 3) — ONE recurrent structure doing BOTH the one-shot EPISODIC store AND pattern COMPLETION (Rolls/Treves). `store` = one-shot Hebbian (co-active bits strengthen the recurrent weight); `complete` settles from a PARTIAL/NOISY cue to the whole stored pattern by GROW-then-PRUNE (monotonically add bits co-occurring with ≥ `theta`·\|active\|, then one prune) — so even a 1-bit cue / 2-token pattern recovers where a strict recompute would OSCILLATE (found by the slice-6 orchestrator test); noise still drops in the prune. Carries §3½ one region up: partial cue COMPLETES, noise DROPS, an AMBIGUOUS cue stays ambiguous (the UNION, no confabulation), a NOVEL cue recalls nothing. Sparse Hopfield/SDM over hashable bits (ints or (object,cell) episode tokens). Wired via `Agent.remember_scene`/`recall_scene` (a glimpse of one object recalls the whole scene — the maze-wall case). Heavy pattern OVERLAP cross-talks — that is DG's job (slice 4), not a flaw | WIRED | `test_ca3` |
| `hippocampus/dg.py` | the dentate gyrus — PATTERN SEPARATION → chart keys (DESIGN §2/§3, slice 4). `separate(signature)` maps an environment signature (set of input bits) to a sparse, orthogonalized CHART KEY, REUSING the k-WTA `SpatialPooler` (a fixed sparse projection + winner-take-all IS separation). DETERMINISTIC (`learn=False`) so the same environment returns the same key; DECORRELATES (distinct envs → near-disjoint keys, measured 0/24 overlap; a similar view keeps 11/24 — graded). Closes CA3's overlap seam: overlapping raw signatures cross-talk in CA3, their DG keys are recalled cleanly. Wired via `Agent.chart_key`; the base for multi-chart REMAPPING (slice 5) | WIRED | `test_dg` |
| `hippocampus/__init__.py` | the `Hippocampus` ORCHESTRATOR (DESIGN §2/§3, slice 6, the LAST) — composes the subfields into ONE region behind a single agent handle: EPISODIC (`remember`/`recall` = CA3), REMAPPING (`chart_key`/`visit` = DG+CA3+CA1), PLANNING (`plan` = replay over a cortex-assembled world+model). Holds the stateful memory; the world-state assembly (nav pose + scene objects) is the cortex→hippocampus bridge kept on the agent. `Agent.__init__` now holds ONE `self.hippocampus` (the per-region `ca3`/`dg`/`remapper` fields folded behind it — clean cutover); `remember_scene`/`recall_scene`/`chart_key`/`visit_environment`/`plan` all delegate | WIRED | `test_hippocampus` |
| `perceive.py` | the peripheral RETINA (the game-loop bridge, step 1) — `segment(grid)` turns a game FRAME (colour grid) into OBJECTS (same-colour 4-connected components: colour ⊕ cells ⊕ top-left anchor), Core-Knowledge OBJECTNESS with NO semantics read (the mechanic is inferred from colour + score, like the human in `play.py`); the deeper COMMON-FATE / RECOGNITION grouping refines it. `SelfTracker` DISCOVERS the controllable ROOT (the 'self') from MOTION — the object that moves with the action — never colour-as-self (`feedback_bitter_lesson`; `reference_l5_operator_kinds`). Wired via `Agent.transduce`/`observe_self`/`self_color`; verified on the real LockPath frame (recovers wall boundary + agent + goal + block/pad at true positions; the self discovered as it moves) | WIRED | `test_perceive` |
| `hippocampus/featurize.py` | the world-state → SDR FEATURISER for the value critic — `WorldFeaturizer.encode(world)` maps a `WorldMap` (agent self-location + objects at places) to an OVERLAP-BEARING SDR of `(entity, axis-bit)` bits. Position is a per-axis METRIC `ScalarEncoder` (overlap decreases monotonically with distance), NOT the modular `GridEncoder` — the grid ALIASES over distance and would break value smoothness (`reference_sdr_regime_and_phase_codes`: SDR great for IDENTITY, breaks for METRIC). Closes replay.py's leaf-value seam: `Agent.value_of` = the `ValueCritic` scoring the featurised world (the rollout leaf, defaulted in `plan`); `Agent.learn_value` trains it on world-state transitions by TD. Measured: adjacent states share 17/18 place-bits, far states 0; a trained critic pulls the rollout toward a goal beyond the horizon where an untrained one gives no plan | WIRED | `test_featurize` |
| `hippocampus/ca1.py` | the CA1 COMPARATOR + multi-chart REMAPPING (DESIGN §2/§3, slice 5). `CA1.compare(observed, recalled)` = the match/novelty detector (Lisman/Hasselmo): MATCH iff the recall explains every observed bit (observed ⊆ recalled) — the §3½ rule one region up, so a PARTIAL view matches (absence) and a CONTRADICTED bit mismatches (novelty). `Remapper` composes CA3 (recall) + CA1 (compare): revisit → RECALL the chart, a novel/CHANGED environment → MINT a new one. Comparison runs on CONTENT tokens (subset-preserving), NOT DG keys (k-WTA breaks the subset relation) — DG separates full signatures at the index layer, composed in slice 6. Wired via `Agent.visit_environment`. Measured: glimpse recalls chart 0, `A`+contradicting-`X` remaps (novelty 0.25) | WIRED | `test_ca1` |
| `region.py` | a cortical REGION = one `Column` + the declared WIRING that says what it is (`proximal` source, `frame` source, `target`), plus `Hierarchy` holding them and `edges()` reporting the region→region links. The same argument `htm.py` makes for a LAYER, one level up: biology fixes the role by CONNECTIVITY inside a uniform microcircuit, not by a different algorithm. A REGION is not a MODALITY — a modality has a transducer at the periphery; a higher region's proximal drive is another region's OUTPUT, which is why "an event modality" was a category error and PFC needs no special case. Live edge: `sensory → scene` | WIRED | `test_l23_pooling` |
| `successor.py` | the LEARNED frame — `SuccessorFrame`, an online TD successor representation over an ARBITRARY transition graph of any hashable state (`observe`/`code`/`similarity`/`value`/`states`). `GridEncoder` is the frame you are GIVEN and works because physical space is metric; a task space has no coordinates to hand it, so this is the frame you LEARN. Deliberately NOT: eigendecomposition (O(n³) online, and the eigenpurpose built on it was dropped), a matrix operator over SR rows (built once, failed measurably), orthogonalisation (SR codes are MEANT to be correlated — that overlap IS the topology). It does not replace the grid: physical space keeps its metric prior. Measured: similarity falls monotonically with graph distance; `V = M·R` re-values a MOVED goal with no relearning. Fed the agent's joint `_world_key` transitions — the H0 setup below | WIRED | `test_successor`, `test_h0_factorisation` |

**H0 — ANSWERED 2026-07-27: one frame over the joint state does NOT factorise ⇒ the second column is justified.**
The legacy `HETERARCHY_PLAN` makes this the gate on everything above it ("H0 gates everything — run it first"): feed ONE
frame the joint `(position, configuration)` transitions and find out whether the two separate INSIDE it before allocating a
column for the task factor. `Agent.task_frame` is fed `_world_key` precisely so the live loop runs the experiment.
Measured (`test_h0_factorisation`), in a world built as the BEST case — position and configuration moving independently, a
true Cartesian product, as separable as a world can be made:
  * **State count multiplies:** 32 joint states for 8 positions × 4 configurations, where factored needs 8+4=12.
  * **No shared position code:** the same position under a distant configuration overlaps **0.279**, LOWER than a genuinely
    DIFFERENT position at the same configuration (**0.705**). Collecting a key moves you further, as the frame sees it, than
    walking does. On the LIVE LockPath loop the same cell under two configurations shares **0.000** — literally nothing.
  * **No shared operator:** the code-change for the SAME step across configurations aligns at **0.325** (1.0 = one operator).
    Mechanism: a row is supported on successors, and the successors of `(p,c)` all carry `c`, so the supports are disjoint by
    construction. This is `project_place_invariance_needs_factored_state` one level up.
  * **No transfer:** learn the corridor exhaustively at one configuration, take ONE step at a second, ask for the route —
    **V = 0.0000**. The counterfactual arm (a frame indexed by position ALONE, same experience) has it at 0.123.
  * **LIVE LockPath:** 92 joint states = 40 cells × 7 configurations; **29 of 40 cells** were re-entered under another
    configuration and stored afresh each time.
WHY THE EIGEN PHRASING WAS NOT USED. The plan says "does the SR *eigenframe* factorise", written when the frame was assumed
to be a decomposition; ours is not, on the record. The test is run on the CODE where the agent reads it — also the sounder
instrument, since a product graph has DEGENERATE eigenvalues, an eigenspace has an arbitrary basis, and "is this eigenvector
separable" then asks about the basis numpy returned. **And the spectral route would not rescue it anyway:** the state count
is multiplicative in EXPERIENCE, you can only decompose states you have VISITED, so a decomposition describes experience
already had and cannot supply experience not had. ⇒ **H1 (inter-column communication) → H2 (the task column) stand.** The
open problem the counterfactual arm names honestly: its split was made BY HAND, so it measures the CEILING factoring buys,
not a mechanism that earns it. WHO DECIDES THE SPLIT is what H1/H2 must deliver rather than assume.

**H1 — BUILT 2026-07-27: two peer columns reach consensus over direct lateral links.** The step the legacy plan puts before
any task column: prove inter-column messaging works. **The plan's substrate was WRONG and correcting it was part of the
step** — it says route the hypotheses "through the thalamus" and call `L23.vote`; per arXiv:2507.05888 voting is direct
LATERAL cortico-cortical, the thalamus carries the separate HIERARCHICAL route, and `L23.vote` never existed here (a legacy
name). We had logged that locus error on 2026-07-22 as debt and left the thalamic version standing; this is the cut-over.
  * **Result (`test_lateral_voting`):** column A glances a feature consistent with {X, Y}; column B, at another vantage point
    on the same object, one consistent with {Y, Z}. Each reports −1 alone. After ONE lateral round both settle on Y — drive
    **0.5066 for Y against 0.0500** for the rival, ~10× — with neither column ever seeing enough on its own.
  * **The link is LEARNED, not shared.** Two columns mint their own random SDRs for the same object, so nothing makes A's
    code resemble B's; co-experience turns that into a correspondence by the same Hebbian rule the feedforward synapses use.
    Tested: peers that learned the objects but never attended them TOGETHER reach no consensus — silence is not a vote.
  * **The vote MODULATES, never drives.** It re-ranks identities a column already holds live and cannot add one, so a column
    with an unambiguous view keeps it however loudly a peer disagrees (tested). Without that guard, columns converge on
    whichever spoke first — which looks like agreement and carries no information.
  * **CMP speed-up, measured:** alone a column needs a 2nd fixation to break the ambiguity; with a peer, one fixation plus
    one vote. Evidence gathered in PARALLEL across the sensor array rather than serially by one sensor.
  * **CUT in the same move:** `Agent.vote_consensus` → `read_register` (what it actually is), the thalamus's "cross-column
    vote tally" docstrings corrected, and the two voting tests in `test_thalamus_binding` DELETED rather than kept alongside.
    The register itself survives on its real merits (place-value, the L4 surface `_sense_frame` writes).
  * **HONEST SCOPE:** exercised at the COLUMN level, not yet in the live game loop — the loop has no peer columns to vote
    (`sensory`/`nav`/`scene` are hierarchically related, not peers viewing one object). Its first real use is a sensory
    ARRAY over a 64×64 ARC frame, where one column sweeping serially is exactly the cost voting removes.

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

Suite: **150 passed** (~28s; `test_column_arithmetic` is the ~16s end-to-end column test, the rest — `test_bg_thalamus`,
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
   The real scattered-L5 was L5IT: `Column.striatum` is the column's cortex→BG projection, so `decide`/`reward` route the BG's
   cortical input through L5IT, not a raw-L4 relay. **BOTH inert L5 HTMLayer placeholders (`L5IT`, `L5PT`) DELETED from the dict:**
   both are PROJECTIONS / engines (IT/PT are projection-neuron classes), not HTMLayers — L5IT = `striatum` (relay the
   representation), L5PT = the `MotionOperator`/`dynamics`/`relate`/`efference` engines. No dead layer advertising a role it can't do.
   **HONEST caveat:** for a single-stimulus decision the L5IT projection is TRANSPARENT (relays the L4 percept); it does NOT
   integrate — the INTEGRATION is L2/3's pooler (validated in `test_object_centric`/`test_rotation_recognition`).
   **Object→action decision DROPPED as a detour (2026-07-18):** L5IT is a projection, so the object-based decision is a valid
   COMPOSITION of (validated pooling) + (this projection) + (validated BG), not a new L5 mechanism. Building it surfaced that
   recognition is STABLE (verified: `A→B→A` gives the SAME identity, 40/40) — the apparent "instability" was a pytest/numpy
   k-WTA tie-break artifact (byte-identity test, not a recognition bug). Needs a stable object LABEL as the BG context; parked.
   **Step 4 (docs):** reconcile ARCHITECTURE §9/§12/§15 with L5-as-projection. See [[reference_l5_operator_kinds]].
   **THE THIN-AGENT GAME LOOP (IN PROGRESS) — drive the real replica games (`src/tasks/`: LockPath/Sokoban/Tetris/…, the harness
   + BFS oracle + human traces; RHAE = levels ÷ action_counter).** The agent plugs into `Environment.reset`/`step(action,coords)`;
   `Agent.step(FrameData)→action` composes perceive → world-state → hippocampal rollout → act + online learners.
   **Step 1 (perception bridge) ✅ DONE (2026-07-18, `perceive.py`, `test_perceive`)** — the retina (`segment`) + the discovered
   self (`SelfTracker`, motion not colour), wired via `Agent.transduce`/`observe_self`/`self_color`.
   **Step 2 (the `Agent.step` loop) ✅ DONE (2026-07-18, `test_game_loop`)** — the agent plays LockPath L0 END-TO-END through its
   own perceive→learn→plan→act loop: `step(FrameData)→action` = `transduce` → discover self + `learn_pose_move` (the operator
   learns each action's Δ from experience — never assumed) → NOVELTY-directed rollout (`plan` toward the nearest unvisited cell,
   the goal being hidden) → act. Walls are PREDICTED impassable by L5 from what is perceived at the cell (`WorldMap.static`; formerly a bumped-cell set —
   reachability reshaping, `reference_obstacle_as_transition_cost`) — no game semantics read. Measured: solves L0 in ~42 actions
   (oracle-optimal ~9; the 4.7× is exploration overhead) across seeds. `Agent.new_level` resets per-level map memory; the learned
   model persists (cross-level transfer).
   **Step 3 (GOAL/REWARD from the sparse score) ✅ DONE (2026-07-18, `reward.GoalMemory`, `test_game_loop`)** — the score
   DISCOVERS the goal and the discovery TRANSFERS. `GoalMemory` credits the FEATURE of the object the self reached by the delta
   rule (Rescorla-Wagner cue competition, `reference_cue_competition_key_discovery` — the SAME rule as the critic): reaching the
   goal AT reward drives its colour→1, reaching non-goals at r=0 decays theirs→0. Because the goal is an object FEATURE (colour),
   it is POSITION-INVARIANT — it names the goal wherever it sits, which the absolute-position `ValueCritic` cannot. `Agent._act`
   is then ONE EFE planner (`reference_efe_and_epiplexity`, `reference_goal_setting_priority_map`): PRAGMATIC (rollout to the
   visible goal-featured object) when the goal is known, EPISTEMIC (novelty) when it isn't. Measured on two nav levels with goals
   at DIFFERENT cells: L0 explore 42 actions (discovers the goal) → L1 goal-directed **8 = oracle-optimal** (nothing positional
   carried; the discovered feature did) → WIN.
   **Sokoban Slice A (the PUSH) ✅ GROUNDED (2026-07-21, `touch.py`+`modality.py`+`behavior.py`, `test_push`/`test_behavior`/
   `test_touch`/`test_modality`)** — the DEGENERATE scaffolding of commit `bf04c9e` (scene-column RW push + a rollout SNAP + a
   hand-coded rigid-body coupling, crutches #3/#4/#5/#9) is DELETED and re-seated on TOUCH + a learned BEHAVIOR model:
   - **Touch modality** — `Agent(modalities=[vision(), touch()])` via the `modality.py` factory (a sense = `(transduce, feature,
     location, pose_source)`; the column + connections are modality-INVARIANT). The SKIN (`touch.py`, body surface + per-face
     contact) is the AGENCY signal: only motions the self FELT itself cause teach the dynamics (REAFFERENCE — a box shoved by
     another box is never mis-attributed to the agent).
   - **`behavior.Transform`** — REPLACED `ObjectBehavior`/`ContactDynamics` on 2026-07-22 (both DELETED). The L5 change model is
     now an ordinary cortical layer read out as a metric quantity: `cells = HTMLayer(proximal=cues, basal=situation)`,
     `delta = PopulationReadout(cells)`. No YIELD/RESIST/PASS enum, no per-object dispatch, no `T` matrix with an identity
     prior, no "sticky yield" repair — three named behaviours with three code paths became three learned deltas out of one
     mechanism. Two facts per felt interaction: `("into", felt)` = the CORRECTION to the body's own free motion (solidity is
     that correction) and `("of", felt)` = the felt thing's own delta. The body is corrected and the felt thing absolute
     because only the body has an efference-copy baseline.
   - **OCCASION-SETTING GATE ✅ BUILT (2026-07-22)** — the dynamics is a press-following BASE plus OCCASION overrides,
     combined by SELECT not SUM (`Agent._dynamics_delta`/`_learn_delta`/`_press`/`_world`/`_confident`/`_cue`/`_beyond`).
     This REPLACED the two-frame CO-SUM (deleted) and folds in the learnable-novelty paper (`reference_learnable_novelty`):
     the split criterion is epiplexity read as a LEVEL — `behavior.Transform.confident(cues, param)`, HTM's own
     connected-segment test = "has reusable structure been committed for this cue yet". No new module, no invented threshold,
     no few-states prior; the guard the gate needed IS the substrate's notion of a learned prediction.
       * BASE — press-frame, one-shot, direction-general (bold press-following, `feedback_prefer_generalize_then_correct`).
         Learned only from FREE contacts. From one east push it predicts all four directions exactly (the two-frame co-sum
         hedged 0.5/0.5, which is what regressed Push — not-generalizing is the thing the thesis warns against).
       * BACKDROP occasion — minted when the body is IMPEDED (did not advance by its full efference: wall/edge/mover behind),
         keyed on what is behind, ALWAYS (even the first contact), so an obstructed push never poisons the base "it stays".
         Once confident it SHIELDS the base ⇒ no dilution. The board EDGE is a real backdrop (`_beyond`, `WorldMap.occupant`
         return "edge" off-board), so edge-obstruction can't masquerade as open space.
       * WORLD occasion on the object — minted when a FREE contact refutes press-following (moves world-fixed however pushed:
         a balloon). The `IMPEDED` fact cleanly separates "a wall stopped it" (backdrop) from "wrong frame" (world).
   - **`WorldModel` contains no physics**: free path integration + the corrections the cortex has learned.
   - **Push ✅ 8/8 at oracle-6** (was L0-only; both former `test_push` xfails now pass). VERIFIED offline: base one-shot in all
     4 directions; no dilution over 20 obstructed pushes; unseen backdrop (the pad) inherits the base; a one-off backdrop
     corrupts neither the base nor a different backdrop; a balloon rises in directions it was never pushed. The run is ~60×
     faster (0.5s vs 32s) — the model no longer emits fractional deltas that blew up the rollout's visited-pruning (which is
     now cell-granular in `WorldMap.key`).
   - **LEARNABLE NOVELTY ✅ BUILT on the agent's OWN model (2026-07-22) — `_visited` is DELETED.**
     (Retracted en route, recorded so it is not re-tried: a "deterministic no-op" claim; a frozen-reservoir spectral-determinant
     estimator; and a shadow frame-model `epiplexity.py` — all WRONG, all deleted. See `reference_learnable_novelty`.)
     Epiplexity is `S_T := |P*|`, the program length of the best compute-bounded model, estimated by the AREA under a *learning*
     model's PREQUENTIAL loss curve (Finzi 2026 eq 8): LOW for random AND simple, HIGH only for complex-and-slowly-learnable.
     * **`reward.LearningProgress`** — fed by `Agent._model_loss`, which scores what the agent's OWN `WorldModel` predicted
       (L6a operator ⊕ L5 `Transform` ⊕ occasions) against what happened, BEFORE it learns from it. NO second model: a shadow
       predictor re-learns what this one knows and its errors are not the errors that make plans fail. Colour intact.
       `total()` = the eq-8 area `Σloss − N·floor` — NOT a rectified running sum, which turns EMA jitter into reward and
       re-admits the noisy TV (measured 5.09 on pure noise). Synthetic ordering: dark room 0.0 < noise 2.4 < instantly-mastered
       19 < learning 141 < slow/complex 201. Real fixture: the agent's model loss falls 0.175 → 0.0 on LockPath.
     * **Exploration = `Agent._unlearned_cells`** — go where the MODEL cannot yet predict, via the SAME `Transform.confident`
       primitive the occasion gate uses (zero new machinery). Epiplexity read as a LEVEL, prospectively. Unlike visitation
       (geometric bookkeeping outside the model, which keeps paying on an understood world) confidence GENERALISES — a
       never-visited cell whose features are all modelled pays nothing — and dies at mastery. It is also how a hidden goal is
       found: the goal is a perceptible FEATURE never interacted with, hence exactly what is not-yet-confident (verified on
       LockPath L0: wall confident=True, goal feature confident=False).
     * MEASURED: Push 8/8 at oracle-6 with its exploratory L0 FASTER than under `_visited` (mean ~18.3 → ~15.0 actions);
       LockPath L0 solved cold at mean 15.9 (oracle 8) over 8 seeds, range 15–19. Fewer actions is directly RHAE `(human/agent)²`.
     * **TONIC DOPAMINE ✅ WIRED (2026-07-22)** — `progress()` feeds `ValueCritic.tonic()`, so ρ tracks the average PAYOFF rate
       (extrinsic ⊕ epistemic), and the pragmatic-vs-epistemic choice MOVED INTO THE BG (`_act` proposes both drives; the BG
       selects on salience ⊕ learned Go/NoGo with ρ as the gain; the choice is trained by its payoff RPE). It had been an
       if/else ladder — arbitration outside the BG, violating ARCHITECTURE rule 4. Results preserved exactly (LockPath L0 15.9,
       Push 8/8 at oracle-6). HONEST: ρ has ~no leverage yet — its differential effect is ρ·[(Go+NoGo)_a − (Go+NoGo)_b] and both
       drives sit at ≈2.0 (1.998 vs 2.000), so it is ~0.002·ρ against a salience gap of ~1. Live but inert.
     * ⚠ **KNOWN, MEASURED: LockPath stalls after L0 (1 of 4 levels)** — pre-existing, verified against the prior commit. L1 is
       key+door: the goal is visible but a door blocks it, and `_goal_plan`'s cheap `_nav_inverse` read-out always returns a step
       toward it, so the pragmatic drive looks permanently available and walks into the door. Affordance-only salience fixes it
       (2 levels, L1 in 12) but costs Push its oracle (6 → 10), because the BG's drive value is dominated by whichever drive was
       active when the first reward landed — always exploration. **The fix both need: a per-drive payoff RATE (marginal value
       theorem) instead of a contingency that cannot fall.** Not shipped — a design step, not a knob.
     * NEXT after that: L2 is the block+pad Sokoban, needing a RELATIONAL goal discovered while a bare-feature goal is already
       believed.
     * ⚠ **CORRECTION 2026-07-27 — the CONJUNCTIVE win condition was DEAD CODE for seven commits.** `a67177c` inserted the new
       `_credit_goal`/`_goal_plan` ABOVE the old ones without deleting them, so Python bound the OLD definitions (the later of
       two same-named methods wins) and every run since executed the elemental credit + single-goal planner. Deleted the stale
       copies; the conjunction is now genuinely live. **It changes nothing measurable, and could not have:** LockPath is
       unchanged (L0/L1 at 15/19 and 19/19 actions over 3 seeds, score 2 — same as before), because the configural cue is only
       credited when a reward ARRIVES with both conjuncts holding, and L2 is never won, so no conjunction is ever credited. The
       machinery is credit-assignment for a win already achieved; it cannot produce the first one. That is exactly why no test
       caught the deletion — `test_learning_progress` exercises `GoalMemory` directly, and nothing downstream depended on the
       wiring yet (RULES.md #3: not done until wired AND exercised — this was neither, while STATUS claimed both).
       Observed after the L2 refutation: `goal_mem.goals()` is EMPTY — the belief was correctly dropped and nothing replaced it.
       **The L2 blocker is upstream of credit assignment**, in what the agent does before any win exists to learn from.
   The touch COLUMN (recognise-BY-touch, for OCCLUSION) is DEFERRED; the fully-observed push needs only the skin (agency) + the
   behavior model. In full observation contact is geometric; touch earns its place as the agency/reafference signal.
   **INVERSE-MODEL planner, step 1 (NAV) ✅ DONE (2026-07-21, `operator.utilities`, `BasalGanglia.select(salience=)`,
   `Column.striatum_proposal`, `Agent._nav_inverse`, `test_inverse_model`)** — forward and inverse are the TWO DIRECTIONS of the L5↔L6a loop
   (`notes/inverse_model_featurization_design.md`): `apply` turns an action into a displacement (L5→L6a); `utilities` turns a
   desired displacement back into the actions that produce it (L6a→L5), the SAME learned table read backwards — which is why the
   GCML's Hebbian `W` converges to exactly these action effects, so we READ ours off instead of associating a second copy. The
   goal VECTOR comes from the world-map, the column's **L5IT** projection (`striatum_proposal`) emits `(per-action drive,
   context)`, learned obstacles VETO, and the **BG** selects (priority = salience ⊕ value — the cortex proposes, the BG gates;
   `reference_goal_setting_priority_map`). The agent only ROUTES (`feedback_thin_shell_agent`). Measured: LockPath L1 goal-directed nav solved at **8 =
   oracle with the read-out handling ALL 8 actions** — the nav BFS is gone for goal-directed nav; exploration (novelty) and the
   RELATIONAL push still deliberate, which is what step 2 targets. O(actions)/step vs a search that grows with the space (the
   real win on 64×64 frames). Featurization settled: the code for `W` is per-entity COORDINATES (linear ⇒ additive, eq 11); the
   `WorldFeaturizer` bump code FAILS additivity and stays where it belongs (the value critic).
   NEXT: **inverse-model step 2 (OBJECTS)** — `T⁻¹·δ_obj` on `ObjectBehavior` → press direction → press position (the contact
   condition) → hand to the nav read-out, turning the relational push into a search-free plan. This is the step that EARNS or
   BREAKS the design; measure against the Push result we already hold (L1 at oracle 6, 12/12). Then **the EPISTEMIC drive** (`feedback_epistemic_value_is_prediction_error`, cold-start on an UNCONSTRAINED push);
   **Slice B** (the CONJUNCTIVE win `box-on-pad AND agent-on-goal`); **Tetris** (multi-cell + rotation + click); the real SDK.
   Deferred within the hippocampus (DESIGN §4): theta/replay timing.
   Still after the hippocampus: the MOTOR region (L5 emit + decode; move the readout OUT of the agent); the loop/brain object
   (move `decide`/`scan` OUT of the agent); the THIN agent; the `step(obs)→action` game loop. Plus a full multi-sensory-column
   recognition-BY-VOTING task consuming the Phase-5 register.
   (b) **object-dynamics refinements** (not blocking): abstraction across GEOMETRY VARIANTS (overlap recall over relation SDRs,
   the `_key` refinement everywhere) + intrinsic-feature cues; a WALL (partial-effect cue) = the same RW mechanism; the SR
   form of the critic (`V=M·R`, fast reward-re-tuning) for a task that needs it; MORPHOLOGICAL features; the eligibility trace.
   DEFERRED to a MOVING sensor: the ego→allo transform, so an extrinsic law stays invariant when the observer moves. **LOCUS
   CORRECTED 2026-07-22** — this is the **THALAMUS's** job, not the hippocampus's: TBP's long-range-connections paper
   (arXiv:2507.05888) states the ego-centric→allo-centric transformation of incoming sensory and motor data "occurs in the
   thalamus, and is a primary function of thalamic relay cells." Earlier text here (and
   [[reference_tbt_frames_and_hippocampus]]) assigned it to the hippocampus — wrong, and it matters because we defer it to
   exactly the moving-sensor case where it is needed. ARC's static frame still does not need it.
3. **The thalamus's binding role** (content ⊗ location across two columns) + the factored
   state via L4's basal `context=` channel; then **L2/3 recognition/voting**, **L5 motor**, **hippocampal rollout**. Each
   added as a task exercises it, driven by `agent.py`. Nothing counts until imported from `agent.py` AND the agent plays more
   than before (RULES.md #3). No single-column experiments, ever (ARCHITECTURE.md §5.1).

## How to answer "where are we?"
Run the reachability test (once it exists); it reports the wired module map. Keep this table equal to that output.
