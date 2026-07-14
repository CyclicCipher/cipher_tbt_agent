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
- **Phase 3a — THE PATH-INTEGRATION OPERATOR INTO L6a (the TRANSFORM primitive; ARCHITECTURE §8).** L6a is a plain
  `HTMLayer` now — but path integration is NOT sequence memory (a memorized per-position transition fails place-invariance,
  §7). Build `operator.py`, one concept: `ModularOperator.learn(loc, action, loc')` (per-module phase-delta voting) +
  `apply(loc, action)` (block-structured cyclic shift on the `GridEncoder` code, using `encoders.modules()`), wired into L6a
  so location is path-integrated by the efference copy. ABELIAN (translation) first; the conjunctive/non-abelian
  (heading-dependent) case + the context-gated obstacle override are noted-DEFERRED. Test: an action's effect learned at some
  positions generalises to a NOVEL position (the operator analogue of the place-invariance win) + dead-reckoning matches
  ground truth. This is the transition the SR (3b) accumulates — logically prior to it.
- **Phase 3b — VALUE CRITIC (`reward.py`).** The SR read-off critic (the SR = the discounted resolvent of the 3a operator,
  ARCHITECTURE §8): `value(φ)`, `dopamine(φ,r,φ′,done)→δ`, `rho()`. Rewire the bandit's RPE to come from it (replace the faked
  `2r−1`). Test: the bandit still learns, now with a real critic.
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
