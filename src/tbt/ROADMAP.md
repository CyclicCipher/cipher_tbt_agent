# ROADMAP — the brain regions and the development plan

Born from the agent audit (2026-07-10): `agent.py` was impersonating several brain regions at once — deciding what to do,
computing value/RPE, controlling where to look, owning the readout. ARCHITECTURE.md §5.2 says the agent is **pure
plumbing**. This doc decomposes the brain into REGIONS (each with ONE job + an interface) and orders the development so
every region is built → wired → exercised (RULES #2/#3), never orphaned — the failure that caused the whole reset. It
elaborates ARCHITECTURE.md §3 (the planning loop) into concrete contracts + build-status + a phased plan; ARCHITECTURE.md
stays the design source, this stays the roadmap.

## 1. The regions

| region | owns (its ONE job) | speaks (interface) | status |
|---|---|---|---|
| **agent** (env shell) | env ↔ brain PLUMBING only | `step(obs) → action` | stub; must be THINNED (it impersonates the regions below) |
| **peripheral** (retina + effectors) | TRANSDUCTION: raw field ↔ feature-at-location SDRs; motor SDR ↔ effector command | `transduce(field) → [(feature, location)]`; `decode(motor_sdr) → command` | `encoders.py` (analytic decode) built; a RETINA (field → features) NOT built |
| **neocortex** (cortical columns) | the value-free WORLD MODEL — feature-at-location, location, operator, object, PREDICTION | `observe(feature, loc)`; `predict()`; `recognise()`; `motor() → SDR`; `goal` | `column.py`: L4 via `observe`; L5/L6/L2·3 present but NOT driven |
| **PFC / task column** | the GOAL-STATE + plan hierarchy (a column on an abstract task frame) | the same column interface, abstract frame | partial (the current "task column" is degenerate) |
| **old brain / value** | the DRIVES + the CRITIC (expected future reward) + DOPAMINE (the RPE) | `reinforce(reward)`; `value(state) → v`; `dopamine() → rpe` | NOT BUILT — the agent FAKES `rpe = 2r−1` |
| **basal ganglia** | SELECT by value (Go/NoGo disinhibition), trained by the dopamine RPE | `select(candidates) → winner`; `learn(rpe)` | `basal_ganglia.py` — from a FAILED stretch → **REDESIGN** |
| **thalamus** | ROUTE/BIND (content ⊗ location, cross-column VOTING) + GATE the selection to the motor + relay L5's driver | `bind(cols) → R`; `read(R, q)`; `gate(sel) → motor` | `thalamus.py` — minimal (relay + gate only) → **REDESIGN/EXTEND** |
| **hippocampus** | ROLLOUT — preplay/VTE sweep over the cortical model toward the goal — + episodic/allocentric binding | `rollout(goal, cortex) → trajectories` | NOT BUILT |
| **the loop / brain** | WIRE the regions; own the cognitive CYCLE (perceive → goal → rollout → select → gate → motor) | `perceive(features)`; `act(actions) → motor`; `reinforce(reward)` | NOT BUILT (the agent impersonates it) |

## 2. The interfaces (the CMP — who speaks what to whom)

The cycle, one turn, region by region (this is what `agent.step` should reduce to as plumbing):

```
env frame ─► agent ─► peripheral.transduce ─► neocortex.observe (perceive)         [sensory in]
                                              PFC column holds/updates the GOAL
              hippocampus.rollout(goal, neocortex) ─► candidate trajectories        [imagine]
              old_brain.value scores them ─► basal_ganglia.select ─► the winner     [value + select]
              thalamus.gate(winner) ─► neocortex.L5 motor SDR                        [gate → motor]
   agent ◄── peripheral.decode(motor SDR) ◄──────────────────────────────────       [motor out]
   env reward ─► agent ─► old_brain.reinforce ─► dopamine RPE ─► basal_ganglia.learn [reinforce]
```

Rules the contracts enforce: the neocortex NEVER sees a value; the value/BG NEVER hold the model (they read the column's);
the agent NEVER decides/values/looks — it only moves frames and rewards across the env boundary.

## 3. What development is needed

- **A. BUILD the missing regions:** the value critic + dopamine (`reward.py`); the hippocampus (rollout); the retina
  (field → features); the loop/brain object; the motor region (L5 emit + effector decode).
- **B. EXTRACT the misplaced responsibilities out of the agent** (the 2026-07-10 audit) into their regions: where-to-look →
  L5 motor + attention; state-carrying → task column + thalamus; the decision loop → the brain; explore rate → the BG's
  tonic-DA ρ; the RPE → the value critic; the readout/decode → the motor periphery; target-driven training → the columns'
  self-supervised prediction error.
- **C. REDESIGN the BG + thalamus** — they are from a failed stretch of the project; study their real neuroscience +
  computational role, then rebuild against the §2 interfaces and their real neighbours (a value critic, the columns).

## 4. Substrate decisions (from the deep study, 2026-07-10 — full report: `notes/bg_thalamus_value_research.md`)
**No ANNs anywhere** — backprop is the wrong shape for all three.
- **Basal ganglia → modified-HTM / SDR.** Learns by a LOCAL dopamine-gated **three-factor Hebbian** rule (pre × post ×
  scalar δ) + a short eligibility trace; D1/D2 opponent (OpAL). Our current `basal_ganglia.py` is the **tabular one-hot
  seed** of exactly this — the redesign WIDENS the key from exact-match to a per-bit SDR read-off (so value generalises
  across overlapping percepts), adds an eligibility trace + tonic ρ. An EXTENSION, not a rewrite.
- **Value critic (`reward.py`) → SDR-linear TD = the successor representation.** Value = `w·(W·φ)`, a read-off of a learned
  predictive map; the scalar δ = `r + γV(s′) − V(s)` is what the BG consumes; ρ = tracked average reward. REUSE the
  validated legacy `SuccessorFeatures` (`l6_sr.py`). The SR is the discounted RESOLVENT of the one-step path-integration
  OPERATOR (Phase 3a; ARCHITECTURE §8) — it READS OFF that transition, so the operator is the prerequisite, not a separate
  "SR wired into L6a." PROVEN CEILING (`project_linear_value_cannot_hold_sokoban`): linear value cannot hold relational V* →
  `value(φ)` must double as a ROLLOUT-leaf evaluator for those tasks.
- **Thalamus → DETERMINISTIC.** It mostly does NOT learn (fixed routing/gate; only slow modulatory-gain plasticity,
  decisions learned upstream). Route + **default-off disinhibition gate** (mirror GPi/SNr; disinhibit the BG winner) + a
  fixed **VSA content ⊗ location bind** for voting. No ANN/HTM inside; if gain-plasticity is ever needed, one learned scalar
  per channel driven by the BG's existing RPE.

## 5. The development plan (dependency-ordered; the runnable slices — arithmetic + bandit — stay GREEN throughout)

- **Phase 1 — DEEP STUDY.** ✅ DONE (`notes/bg_thalamus_value_research.md`; verdicts in §4).
- **Phase 2 — CONTRACTS.** Firm the region interfaces (§2) into ARCHITECTURE.md, informed by the study.
- **Phase 3a — THE PATH-INTEGRATION OPERATOR INTO L6a (the TRANSFORM primitive; ARCHITECTURE §8).** ✅ DONE (2026-07-14/15).
  L6a was a plain `HTMLayer` — but path integration is NOT sequence memory (a memorized per-position transition fails
  place-invariance, §7). `operator.py`, one concept: `MotionOperator.learn(pose, action, pose')` (the body-frame delta, as a
  running mean) + `apply(pose, action)` (map that delta through the current heading), wired into L6a so location is
  path-integrated by the efference copy. Delivered: the abelian core, the L4↔L6a feature-at-location loop, L2/3 pooling, the
  OBJECT-CENTRIC frame + emergent boundary, non-abelian SE(2), and SO(2) rotation-invariant recognition. **The design changed
  under measurement:** the operator was first a per-module cyclic shift on the `GridEncoder` code (a permutation), which the
  drift falsifier + bake-off refuted for anything off the lattice — so the 2026-07-15 cut-over made L6a's STATE continuous and
  the grid code a READ-OUT of it (ARCHITECTURE §8; `notes/rotation_invariance_plan.md`). Delivered beyond the original scope:
  pose-invariant recognition (the pose SOLVED, R4), episode-level learning (R5), SO(3) — the orientation is a MATRIX, so 2-D
  was the special case (R6), and the emergent learning boundary (R7). This is the transition the SR (3c) accumulates.
- **Phase 3b — OBJECT DYNAMICS + RELATIONS (ARCHITECTURE §9).** *Inserted 2026-07-16, BEFORE the value/BG rework: a critic
  with no forward model of OBJECTS has nothing to plan over, and this is also what ARC actually needs ("what does this action
  do to that block"). Three steps, in order:*
  1. **Online pose-solving — ✅ DONE (2026-07-16).** `perceive` now narrows a live (object, pose) population per fixation
     (Monty's evidence-based LM), seeded by SOLVING and, before a pose is solvable, by the L4→L6a union (Lewis 2019). The
     measured asymmetry is closed: the same object shifted to (7,3) went `[-1,-1,-1]` → `[0,0,0]`, entered mid-object
     `[-1,-1]` → `[0,0]`. One scoring rule (`_fit`) now serves both the batch and online paths; L2/3's support-only `pool()`
     and the boundary's re-anchoring are DELETED as superseded. The `(object, pose)` stream step 2 needs now exists.
  2. **The operator over OBJECT poses — ✅ DONE (2026-07-16).** `Column.dynamics` (an L5 engine — L6a path-integrates the
     SENSOR, this learns what an action does to a THING) + `learn_object_move`/`predict_object_move`. Fed by the poses
     `recognize` SOLVES, so perception → dynamics is end-to-end. From ONE demonstration: correct at positions never
     demonstrated, on the same object ROTATED (90°, 217°), and on a DIFFERENT object never seen to move — the §7 lesson on
     objects. Forced one real design fact: an object's motion is EXTRINSIC, not intrinsic (measured — an intrinsic operator
     sends a 90°-turned block 90° off the demonstrated shove). REMAINS: the column noticing frame-to-frame which poses moved
     together, by itself = **common fate** (the cold-start segmentation cue R7 lacks).
  3. **L5 DISPLACEMENT CELLS + the context-gated override** — `location + location → the relation` (the inverse of the grid;
     also compositional objects). Then the override: the operator is the free kernel, a local relational context predicts the
     exception. **A table stopping a fall and a wall stopping a push are ONE mechanism** — so gravity and obstacles are one
     slice, not two. Open within it: the operator's KEY must be DISCOVERED (gravity's key is a condition, not an action).
- **Phase 3c — VALUE CRITIC (`reward.py`). ✅ DONE (2026-07-18).** A linear TEMPORAL-DIFFERENCE value over the state SDR
  (`ValueCritic`: `value(state)`, `learn(state,r,next,done)→δ`, `rho()` = running-avg reward). Its δ = `r + γV(s′) − V(s)`
  replaces the faked `2r−1` in `Agent.reward` (now `reward(r, next_context, done)`); the basal-ganglia actor learns from δ.
  Measured: the immediate-reward bandit still learns (≥99%, now with a real baseline `δ = r − V`), AND a DELAYED-reward
  corridor is solved — value propagates backward as a γ-gradient (`V(s0..s3)=0.46,0.52,0.67,0.93`), which the faked RPE cannot
  do (it scores every non-terminal advance as −1). THE UNIFICATION: this δ is the SAME delta rule as Rescorla-Wagner cue
  competition (the operator's KEY), `_Readout`, and dopamine-RPE — one rule, reused (`reference_cue_competition_key_discovery`).
  One fix worth noting: the TD step is normalised by |active bits| or it diverges (effective rate = |state|·lr). DEFERRED
  refinements: the SUCCESSOR-REPRESENTATION form (`V=M·R`, fast reward-re-tuning + cheap planning; grid cells = SR eigenvectors)
  — a linear value cannot hold relational V* (`project_linear_value_cannot_hold_sokoban`), so relational tasks need the ROLLOUT
  (Phase 6+), which uses THIS critic at the leaf. That is why 3c precedes the rollout.
- **Phase 4 — REDESIGN THE BASAL GANGLIA.** Widen `OpponentActor` from exact-match keying to a per-bit SDR read-off
  (`W_G[a]`, `W_N[a]` over the percept SDR) + eligibility trace + ρ from the critic. Test: bandit + a harder selection task
  where SDR-overlap generalisation across contexts is required.
- **Phase 5 — REDESIGN THE THALAMUS.** Keep it deterministic: add the VSA content ⊗ location `bind`/`read` + make `gate` a
  real default-off disinhibition of the BG winner. Test: a place-value / cross-column-voting task.
- **Phase 6+ — THE BIGGER LOOP.** The motor region (L5 emit + decode; move the readout OUT of the agent); the hippocampus
  (rollout over the model, using `value` as the leaf); the loop/brain object (move `decide`/`scan` OUT of the agent); the
  THIN agent. Each a vertical slice, driven by `agent.py`, exercised end-to-end.

**Discipline:** nothing is "done" until wired + exercised (RULES #3); keep it runnable at every step (RULES #4); reuse via
RULES #5 (legacy `basal_ganglia`/`thalamus`/`l6_sr`/`hippocampus` are REFERENCE, rebuilt against the new interfaces, not
copied). `test_column_arithmetic` + `test_bg_thalamus` stay green throughout.
