# COLUMN_AUDIT — is each layer doing its TBT job, and are the layers connected as TBT specifies?

*2026-06-30. The benchmark is NOT a game score — it is **anatomical correctness**: each layer does its TBT job, and
the inter-layer messages run the predict-sense-update CYCLE. This audit is grounded in the LIVE-LOOP CALL GRAPH
(`agent.step` → `_choose` → `column`/`reward`/`sensor`), not in what the layers *could* do. It replaces the
score-driven task list. Spec sources: `TARGET_ARCHITECTURE.md`, `reference_tbt_layers_4_23`, `reference_layer5_role`,
`reference_grid_sr_eigenbasis`, `reference_tbt_frames_and_hippocampus`.*

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

## DEFERRED until the column is correct (then they ride ON a correct column, and should stop being brittle)
Value/exploration (the dead-zone, the eigenpurpose), **§3** (the GSG mechanic library + model-based rollout + commit),
**M2** (the basal-ganglia channel arbitration), and every game/oracle SCORE. These are policy/optimisation on top of the
cycle — premature until C1–C5 land. (The grounding migrations M1/M3/M5-A already did the cheap, correct *reads*; C1–C5
turn them into the live cycle.)
