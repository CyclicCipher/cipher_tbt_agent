# COLUMN_AUDIT — is each layer doing its TBT job, and are the layers connected as TBT specifies?

*2026-06-30. The benchmark is NOT a game score — it is **anatomical correctness**: each layer does its TBT job, and
the inter-layer messages run the predict-sense-update CYCLE. This audit is grounded in the LIVE-LOOP CALL GRAPH
(`agent.step` → `_choose` → `column`/`reward`/`sensor`), not in what the layers *could* do. It replaces the
score-driven task list. Spec sources: `TARGET_ARCHITECTURE.md`, `reference_tbt_layers_4_23`, `reference_layer5_role`,
`reference_grid_sr_eigenbasis`, `reference_tbt_frames_and_hippocampus`.*

## Build strategy — single correct COLUMN → communication → heterarchy (2026-06-30)
TBT = MANY copies of ONE column algorithm + communication, so the build order is a DEPENDENCY chain, not a preference:
1. **ONE CORRECT COLUMN first (the current priority).** A column's output IS the CMP message (pose + features +
   object-state); until the column is correct, that message is MALFORMED and every copy + every communication built on
   it inherits the flaw (`config_state` was exactly a malformed message). = finish the single-column cycle (C1–C5 below)
   + wire the built-but-BYPASSED faculties into the loop: **L2/3 RECOGNITION + voting**, the **GSG** (goal generation),
   and the **L6 FACTORISATION** capability (which decides whether a 2nd column is even needed).
2. **THEN inter-column + thalamus communication** — `thalamus.bind/read` (have) + CMP voting `L23.vote` (have, object
   frame); only meaningful once the column emits correct messages.
3. **THEN the heterarchy is easy** — two correct columns wired through correct communication, OR one column whose L6
   factors both. The SPATIAL+TASK split (the C4 integration), MultiKey/LockPath returning correctly factored, and the
   PFC-like task column all FALL OUT of 1–2. NB Mountcastle: a "PFC column" is the SAME algorithm fed task-state input,
   not a new mechanism; and `TARGET_ARCHITECTURE` says FACTOR within one column first (L6 eigen-subspaces), allocate a
   2nd column (basal ganglia) only when factors don't separate — so the heterarchy is the FALLBACK, not the first move.

**⇒ The single-column cycle is now COMPLETE and VERIFIED (C1–C5 + the GSG in the loop; see *Verification* below).**
The C4 INTEGRATION (the spatial/task heterarchy) remains DEFERRED to step 2/3 — the plan is written in
`HETERARCHY_PLAN.md` (spatial-map + task-map, communication via the thalamus + CMP voting). That is the next session.

## The root deviation (why every score-fix was brittle)
The live loop is: **`sensor` → `config_state` → `L5.edges` (the graph) + `reward.py`'s value sweep → `col.act` (motor)**.
That is a bare tabular-RL loop over a perception SHORTCUT. The column never runs `L6 → L4 → L2/3` to FORM its state — the
sensor hands it a ready-made `config_state` symbol. So three of four layers are bypassed and the TBT cycle does not run.
Tuning value/exploration/§3 on top of this was optimizing a bypassed organ — the source of the brittleness and of both
harnesses. **`config_state` is the original harness.** Correctness = dissolve it and run the real cycle.

## Reference-frame grounding — what L6 is OVER (TBT-checked 2026-06-30)
The frame is **OBJECT-CENTRIC, not sensor/egocentric** (Lewis/Hawkins 2019 *Locations in the Neocortex*, frontiersin
fncir.2019.00022; Monty 2025 arXiv 2507.04494): the sensor detects features EGOCENTRICALLY (the **S**/surface-patch
frame), but learning in egocentric coords is inefficient (relearn at every shifted position — *exactly* `config_state`'s
problem), so L6 grid cells represent the sensor's location RELATIVE TO THE OBJECT (the **M** frame) and the brain
CONVERTS egocentric→object-centric. (Monty's third frame **B** = body-centric, shared, for inter-column CMP.)
- **The chicken-and-egg (object needs a frame, frame needs an object) is broken by SELF-MOTION.** The efference copy —
  HOW you moved — is self-generated, known WITHOUT knowing the object (grid cells path-integrate "from self-motion cues
  without external landmarks"). A metric frame is laid down from MOVEMENT alone; the object EMERGES as the coherent set
  of (feature, displacement) relations inside it. Allocate a fresh frame on novelty; the ORIGIN IS ARBITRARY (the frame
  is defined by RELATIVE displacements); move + bind features at path-integrated locations; the coherent structure IS
  the object — a MATCH aligns to an existing frame (recognise), a mismatch allocates a new one (learn). Landmarks
  RE-ANCHOR the drifting path-integrated frame (grid cells re-anchor to task objects; Nature Neuro 2025 s41593-025-02054-6).
- **For us:** L6 = the **MOVEMENT-BOOTSTRAPPED location frame** — exactly the sensor's step-7c path integration (`_delta`
  = the efference, `coarse_pos` = the path-integrated position from an arbitrary origin, "snap to a sighting" = the
  landmark correction). The board/level is NOT presupposed — it EMERGES as the features-at-positions map (L7-A).
  `config_state` was wrong precisely because it defines the scene WITHOUT the movement-bootstrapped location — it skips
  the one primitive (self-motion) that makes an object-centric frame possible.

## Per-layer: TBT spec vs. ACTUAL (what the loop calls)
| layer | TBT job (spec) | ACTUAL in the live loop | gap |
|---|---|---|---|
| **L6** — location frame | the SR-eigenframe IS the column's "WHERE"; its place code is what L4 binds to and L5 path-integrates; grid cells = SR eigenvectors | `col.observe`→`sr.observe` LEARNS the SR every step; `sr.eigenpurpose` read once (the dead-zone). The place code `sr.code` is **never read as the location**; `col.value`/`reachable` (M1) and `col.feature_at` (M5-A) exist but are **unused**; `refresh` (builds `self.place`) is **never called**; `l6_grid` (innate) is **inert** | **learned but not READ as the location.** The column's "where" is the opaque `config_state`, not the L6 place code |
| **L4** — feature-at-location | bind a sensed FEATURE to the L6 LOCATION; PREDICT the feature-at-location (predict-then-compare seated here); the content codebook | only `encode` is used (via `col.feature_field`, the forward model's content). `bind`/`readout`/`predict_feature` run **only in `refresh` (uncalled)** + the M5-A `bind_at`/`feature_at` (tested-only) | **L4's core job does not run.** Only its codebook is borrowed; it never predicts a feature at a location in the loop |
| **L5** — displacement/motor/driver | position-invariant displacement (generalising operator) + motor output + efference copy + thalamus driver + the forward model | `observe`/`predict`/`successors`/`motor`/`field_step`/`observe_field`/`predict_field` — **all heavily used**; the workhorse | **mostly CORRECT** — but it operates over `config_state` symbols (the bypass), its displacement does NOT path-integrate L6 (the location loop is open), and `driver` is a single-column no-op |
| **L2/3** — object/identity | the STABLE object pooled over the L4 feature-at-location sequence, settled by recognition + lateral CMP VOTING; the object graph-memory | **not called in the loop at all.** The state = `config_state` (proto-object poses, canonicalised) bypasses recognition; `L23.pool` only in `refresh` (uncalled); `recognize`/`vote`/`disambiguation_goal` unused live | **entirely bypassed.** The object is a raw config tuple, not recognised/settled by L2/3 |

## The inter-layer CYCLE
- **Spec (Hawkins):** `L6 location → L4 predicts feature-at-location → sense + compare → L2/3 settles the object (lateral
  voting) → L5 emits the displacement → L6 PATH-INTEGRATES to the next location → repeat.` Predict-then-compare across
  layers is the single learning signal.
- **Actual:** `sensor → config_state → L5.edges + reward.sweep → motor.` No L6-read, no L4-predict, no L2/3, no
  L5→L6 path-integration. **The cycle does not run.**

## The corrected task list (foundation-up; each gated by a MECHANISM test, never a score)
- **C1 — L6 READ as the location substrate.** The column's "where" each step = the L6 place code (`sr.code`), exposed
  as the location the other layers use (the reads exist: `col.value`/`reachable`, `col._place_code`). *Test:* the
  location code encodes topology (nearby states → similar codes; the place-code test) and is the value the loop reads.
- **C2 — L4 ↔ L6 over the MOVEMENT-BOOTSTRAPPED location frame; DISSOLVE `config_state`.** L6 = the object-centric
  POSITION frame (the sensor's path-integrated `coarse_pos`, origin arbitrary — see *Reference-frame grounding*). Every
  step: L4 predicts the feature at the current L6 location, senses the egocentric local view, compares (the mismatch is
  the learning signal), binds it into the map — the object EMERGES. The STATE becomes egocentric-feature ⊗
  object-centric-position, not the `config_state` symbol. *Test:* `L4.predict_feature` at the location matches the
  sensed feature once learned; the error drives learning. **This is the core fix — the bypass is removed here.**
- **C3 — L5 → L6 path integration.** L5's chosen displacement (efference copy) path-integrates L6's location
  (`loc_move`/`path_integrate`); the location updates with each move (predict, then correct by a sighting). *Test:* the
  existing `loc_*` discrete-graph-tracking test, now driven by the loop's efference.
- **C4 — L2/3 over L4 ⊗ L6.** The object = the graph of (feature, location) pooled over the sensing sequence, settled
  by recognition + CMP voting; the state's "what" comes from L2/3, not the raw config. *Test:* the object is recognised
  and persists across the sequence; voting settles an ambiguous sighting.
- **C5 — the CYCLE runs end-to-end** (`L6→L4→sense→L2/3→L5→L6`), predict-then-compare driving cross-layer learning.
  *Test:* the loop runs the cycle on a small world and the per-layer predictions sharpen.

## Progress + the C2↔C4 coupling (2026-06-30)
- **C1 ✅** `column.locate` — L6 READ as the location (topology-encoding place code; mechanism-tested).
- **C2 ✅** `column.sense_at` + the live loop (`TbtPolicy`) — L4-over-L6 predict-then-compare over the movement-
  bootstrapped POSITION frame; `config_state` dissolved as the LIVE representation. **C2a ✅** — DG pattern-separation
  of the location code (the raw place code at γ≈0.95 was too DIFFUSE → a global bag; adjacent 0.987 vs antipode 0.977)
  → the cycle now recognises a multi-location object.
- **The C2↔C4 COUPLING (finding, the audit working):** the position-ONLY planning state is the correct NAV substrate
  (LockPath L0 **31** vs the joint state 254, near-oracle 8) BUT regresses BOARD-STATE games (MultiKey 2/2→**0/2**,
  LockPath 2/4→1/4) — it conflates the BOARD STATE (which keys are collected), which `config_state` had been secretly
  carrying. That board-state is the **OBJECT STATE = C4's job** (like Monty's "stapler open/closed"). So the planning
  state = **position (C2) + object-state (C4) TOGETHER**; C4 restores MultiKey/LockPath *correctly* (factored). Per
  [[feedback_dont_salvage_between_critical_steps]] do NOT revert C2 — C4 is the fix. **⇒ C4 is the next step.**
- **C4 ✅ (mechanism)** `column.object_state` — L2/3's dynamic object STATE (the frozenset of changed feature-at-locations
  from sense_at's surprise; `reset_object_state` per level). Mechanism-tested. **⇒ but its INTEGRATION is the HIERARCHY:**
  the planner must run over `(position, object_state)` to distinguish board-states, while the SR/L6 + `sense_at` need the
  frame over POSITIONS (clean place codes, the metric). One frame can't be both without the diffuse/joint problems
  returning. So C4's integration is a SECOND (task/relational) frame composed with the spatial one — exactly
  [[reference_hierarchy_substrate]]'s spatial-map + task-map heterarchy (spatial: SR+L5 over positions; task: value over
  `(position, object_state)`). `object_state` is the input to it. **The hierarchy is DEFERRED until the single column is
  correct** (see *Build strategy* above); it is where MultiKey/LockPath return, correctly factored (position from the
  spatial map, keys-collected from the task map) — but it is step 3, and we are mid-step-1.
- **C3 ✅ (verified)** `column.loc_reset/loc_move/loc_sense/loc_where` — L5's learned edge (the efference) path-integrates
  L6's location (dead-reckon by the displacement, snap-to-sighting correction). The location loop is CLOSED: the same L5
  operator that predicts the next state also advances the L6 belief. Mechanism-tested (`test_path_integration_*`,
  `test_full_cycle_end_to_end` (b)).
- **C4 ✅ (recognition wired)** `column.sense_object` — L2/3 RECOGNISES the sensed cloud (pose-invariant identity via the
  evidence recogniser + CMP voting) and binds THAT identity into the feature-at-location map, so the object is SETTLED by
  recognition, not a raw patch. Boundaries = recognition mismatch (`sensed_surprise`). In the loop via `agent.step(cloud=…)`.
  Mechanism-tested (`test_l23_recognition_wired_to_feature_at_location`, `test_agent_loop_maps_recognized_objects`).
- **C5 ✅** the CYCLE runs end-to-end through the agent loop — one walk learns BOTH L4-over-L6 (`feature_at` predicts the
  feature at each L6 location) AND L5→L6 path integration (`loc_move` dead-reckons by the learned displacement)
  (`test_full_cycle_end_to_end`).
- **GSG in the loop ✅** `agent._choose` consults `column.propose_goals` + `basal_ganglia.gate` every step — goal
  generation from the column's OWN uncertainty (ACT always; DISAMBIGUATION when L2/3 hypotheses compete), arbitrated by
  the BG. Plain nav → ACT (value policy unchanged); the reward-less dead-zone with an ambiguous object → DISAMBIGUATE
  (active recognition). Mechanism-tested (`test_gsg_and_basal_ganglia_select_the_goal_in_the_agent_loop`).

## Verification — each layer does its TBT job AND is connected (2026-06-30)
The bypass (`config_state`) is dissolved as the live representation; the real cycle runs. Per-layer, in the loop:
| layer | does its TBT job? | connected? | evidence (mechanism test) |
|---|---|---|---|
| **L6** location frame | ✅ READ as the "where": `locate` returns the SR-eigenframe place code, DG-sparsified so it encodes topology (adjacent ≈, antipode ≠) | ✅ L4 binds to it; L5 path-integrates it; `value`/`reachable` read it | `test_l6_is_read_as_the_location_substrate`, `test_the_cycle_recognizes_a_multi_location_object` |
| **L4** feature-at-location | ✅ PREDICTS the feature at the L6 location, compares, binds (predict-then-compare seated here) | ✅ over L6 locations; feeds L2/3 | `test_sense_at_is_l4_over_l6_predict_then_compare`, `test_feature_at_location_map_binds_and_reads_back` |
| **L5** displacement/motor | ✅ position-invariant displacement + motor + forward model | ✅ path-integrates L6 (C3); drives `col.act` | `test_l5_displacement.py` (7), `test_path_integration_is_discrete_graph_tracking` |
| **L2/3** object/identity | ✅ RECOGNISES the object (pose-invariant + CMP voting); tracks the dynamic object-STATE | ✅ settles over L4⊗L6; emits the GSG goal | `test_l23_object.py` (9), `test_object_state_tracks_the_dynamic_scene` |
| **GSG + BG** goal generation/arbitration | ✅ proposes goals from the column's uncertainty; BG arbitrates by EFE | ✅ in `agent._choose` each step | `test_gsg.py` (12) |
| **the CYCLE** | ✅ `L6→L4→sense→L2/3→L5→L6` runs end-to-end | ✅ one loop | `test_full_cycle_end_to_end` |

**Conclusion (AMENDED — see *Correction* below).** The single column's FACULTIES are individually TBT-shaped and
mechanism-green, and the core bypass (`config_state` as the *planning symbol*) IS dissolved on the live path (plan over
`pos`, `sense_at` live). That part holds. BUT the table above conflated "mechanism-tested in isolation" with "connected
on the LIVE drive path": four faculties (C3 path-integration, C4 object-state, the GSG, L2/3 recognition) are
tests-only live, and were built ALONGSIDE the old live substitutes rather than REPLACING them (parallel mechanisms). So
"every layer does its TBT job AND is connected in the live cycle" is OVERSTATED. The corrected precondition for the
heterarchy is in *Correction* — H0 does NOT come first; retiring the parallel mechanisms does.

## Correction (2026-06-30, session 2): mechanism-tested ≠ live-connected — the four gaps + the no-parallel-mechanisms objective
The *Verification* table asked "connected? ✅" and answered it with MECHANISM TESTS. That conflates two different claims:
(i) a faculty EXISTS, is wired to the other layers WITHIN the column, and passes a `src/tests` mechanism test; vs
(ii) it is CALLED on the LIVE drive path (`arc_sdk.choose_action` → `agent.step` → `_choose` → `column`). For four
faculties only (i) holds. Grep evidence: `col.loc_move/loc_where/loc_reset/loc_sense`, `col.object_state`, `col.locate`,
`col.value`, `col.reachable`, `col.sense_object`, `col.recognize_object`, `col.refresh`, `col.examine` are called ONLY
from `src/tests/` — never from `agent.py`/`arc_sdk.py`. The live loop exercises `observe` (SR+L5), `sense_at` (L4-over-L6),
`predict`/`motor`/`act` (L5), the field FM, and `sr.eigenpurpose`; it also *computes* `propose_goals`+`bg.gate` but does
not act on the result. **Wherever an implementation choice is ambiguous, resolve by TBT** — done per gap below.

| gap | TBT job (spec) | LIVE reality | resolution (by TBT) |
|---|---|---|---|
| **C3 — L6 path integration** | L5's efference-copy displacement path-integrates L6; the sensor is the egocentric **S-frame** | the LIVE position `pos` is path-integrated by the **SENSOR** (`sensor._delta`/`_coarse_pos`/`_update_fovea`); the column's `loc_*` (built "TBT-correct" by C3) is **tests-only** → TWO path integrators | L5 OWNS the efference/displacement, L6 path-integrates; the sensor emits feature + raw sighting ONLY ([[reference_tbt_reference_frame]], [[reference_layer5_role]]). ⇒ dissolve `sensor._delta` into L5; live pos = `col.loc_where()`. GROUNDING **P1** |
| **C4 — object-state** | L2/3's dynamic object-state reaches the value so board-states are distinguished | `_changed` is written by `sense_at`, but NO live caller reads `col.object_state()`; the planner plans **position-only** (the MultiKey 2/2→0/2 regression the audit itself flagged — STILL live) | the object-state must reach the value; whether via ONE L6 (factored eigen-subspaces) or a TASK column is exactly **H0**. ∴ making C4 live == running H0. GROUNDING **P4** |
| **GSG** | the column proposes goal-states from its uncertainty; the BG gates; the motor pursues the winner | `self.goal` is SELECTED but neither `_choose` branch dispatches on it → behaviour UNCHANGED (inert). Root cause (`MOTOR_REFACTOR §8.2`): the GSG resolves object-IDENTITY (graph-mismatch), which full-frame ARC makes covert/moot; and `arc_sdk` never passes `cloud` → no hypotheses ever compete → only ACT is proposed | the GSG's live role is COVERT recognition (free); its OVERT epistemic goal is DYNAMICS (`lp`, already in the EFE) + the GOAL/mechanic-HYPOTHESIS, which couples to the TASK column ⇒ the overt rebuild lands in the heterarchy (§3), not the single column. Downgrade "GSG in the loop ✅" → **computed but inert**. See `MOTOR_REFACTOR §8.6` |
| **L2/3 recognition/voting** | the object SETTLED by pose-invariant recognition + lateral CMP voting, bound into the map | `sense_object`/`recognize_object`/`vote` are wired to `agent.step(cloud=…)`, but `arc_sdk` passes `feature=`, never `cloud=` → the pose-invariant recognition faculty is **tests-only** live | covert recognition over the frame should feed the feature-at-location map (`MOTOR_REFACTOR §8.4`); until a cloud is produced live (perception), L2/3-recognition is not on the drive path. Rides on P1/§3 |

**The objective (user, this session).** Make the correct TBT loop the LIVE path in ALL situations, with NO parallel
mechanisms — this simplifies the code, prevents future confusion, and makes the heterarchy easier. That work is
`GROUNDING_PLAN.md`'s revised **P1–P4** (path-integration into the column → M2 BG arbitration → M1 sparse SR reads →
object_state live == H0), THEN §3 / the heterarchy. **H0 is not the first move; it re-enters at P4.** The single column
is *close* to correct — the faculties are right and mechanism-green — but "correct AND the live path AND singular" is the
bar, and four faculties are not yet the live path.

## DEFERRED to the heterarchy / policy tuning (they ride ON the now-correct column)
The **C4 INTEGRATION** (planning over `(position, object_state)` — where MultiKey/LockPath return correctly factored)
belongs to the spatial+task heterarchy (`HETERARCHY_PLAN.md`), NOT the single column. Value/exploration (dead-zone,
eigenpurpose), **§3** (the mechanic library + model-based rollout + commit), **M2** (BG channel arbitration), and every
game/oracle SCORE are policy/optimisation on top of the cycle — now unblocked (they ride on a correct column, so they
should stop being brittle), but sequenced after the heterarchy foundation.
