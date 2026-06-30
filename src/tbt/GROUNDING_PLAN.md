# GROUNDING_PLAN — drain the cognition out of `agent.py` into the columns + basal ganglia (the thin-shell completion)

*Plan, 2026-06-30. Survives compaction. Motivated by: `agent.py` grew from a reafference LOOP into a SOLVER (a
hand-assembled `if flat→forward / elif reachable→exploit / else→eigenpurpose` arbitration, an inline SVD eigenpurpose,
a reachability BFS, field planning) — the HARNESS TRAP, again ([[feedback_thin_shell_agent]], [[feedback_one_model]],
[[feedback_bitter_lesson]], [[feedback_reuse_canonical_components]]). The brittleness we kept fighting (3-case
arbitration breaking tests, the eigenpurpose helping MultiKey but misleading LockPath L1 / the local NavGame) is the
SMELL of un-grounded glue: a pile of special-cased thresholds is robust only on the layout it was tuned on. Companion:
`L6_PLAN.md` (§6 points here), `reference_brain_planning`, `reference_eigenoptions_subgoals`, the oracle/human/agent
trace (this session). The fix the user named: make the COLUMNS natively capable of these behaviors, neuroscientifically
grounded, so the agent returns to a thin reafference loop.*

## 0. The principle (and why it likely cures the failures)
The agent should be the **active-inference / reafference LOOP** and nothing more: sense → predict → compare (surprise)
→ learn → **select (delegate)** → motor. Every behavior that is *cognition* — what to value, where reward is, which
grain decides, where to explore — belongs in the cortical layers + the basal ganglia, where it is ONE grounded
mechanism rather than N special cases. A hand-coded `if/elif` over value functions is exactly what the brain does NOT
do; the basal ganglia gates competing cortical channels by *learned reliability* (dopamine-RPE). Grounding the glue
should make selection ROBUST where the thresholds were brittle — the most likely cure for the regressions.

NB the honest split (so we don't over-claim): grounding the cognition makes the agent **correct + lean**; it does NOT
by itself fix the navigation NUMBERS, which need the location-anchored state (M5) because `config_state` has no metric.
Both migrations are needed; they reinforce.

## 1. What's cognition (migrates) vs. the loop (stays)
| in `agent.py` now | → home | neuroscience | status of the home |
|---|---|---|---|
| `_choose` 3-case arbitration | **basal ganglia** `gate` | BG selects among competing cortical channels (Go/D1 disinhibit winner, NoGo/D2 suppress rivals); dopamine-RPE credits the channel that predicted reward; explore/exploit = dopaminergic tone | `basal_ganglia.py` has `gate(options, values)` = value + learned affinity + RPE — built, used only for subgoals |
| `_eigenpurpose` + the SVD throttle | **L6** (`l6_sr`) | grid cells ARE the SR eigenvectors (Stachenfeld); eigenoption sub-goal drive = entorhinal grid + replay | `l6_sr.grid` exists; the readout is just in the wrong file |
| `_reward_reachable` (BFS) + the dead-zone flag | **L6 SR**, natively | `M[s,g] > 0` IS reachability (expected discounted future occupancy) | `l6_sr.value/values` BUILT then reverted |
| the prioritized sweep value | the **SR** (`V = M·R`) | the hippocampal predictive map; deep value = one dot product, no rollout (Dayan; reference_brain_planning) | `l6_sr.value` reverted; `reward.py` sweeps a tabular V |
| `reward._need` (flat 1.0) | **L6 SR** | NEED in prioritized replay (Mattar & Daw) IS the SR (relevance to the future) | placeholder |
| `_field_plan` / `field_features` / `_field_err` | **L5** | L5 = motor + the forward model (efference copy) | L5 has `predict_field`/`observe_field`/`field_step` |
| `config_state` as the agent's state | **L4 ⊗ L6** | predictive coding into the hippocampal canvas: feature-at-location (TEM); reference_brain_generative_model | L4 has `bind`/`readout`/`predict_feature`; only `refresh` (uncalled) uses them |

## 2. The grounded migrations (each: from → to → mechanism → validation)
**M1 — reachability + value + NEED collapse into READING THE SR.** The hippocampal SR row `M[s,:]` is the expected
discounted future occupancy from `s`: it natively encodes WHERE you can get to (reachability = `M[s,g] > 0`) and HOW
GOOD (`V(s) = Σ_g M[s,g]·R[g]`, one dot product — the deep value precomputed into the cached SR, no rollout). And the
prioritized-replay NEED is the same SR (relevance to the future). So THREE agent/reward hacks are one thing.
- *Do:* un-revert `l6_sr.value(s, R)` / `values(R)`; `column` exposes `reachable(s)=value(s,R)>0` and `value(s,R)`;
  DELETE `agent._reward_reachable` (the BFS) and the global dead-zone flag (the per-level behavior the trace demanded
  EMERGES: a fresh level's states have no SR path to `_GOAL` → not reachable → directed search); set `reward._need`
  from the SR (`M[current, s]` / place-code similarity).
- *Validate:* SR-reachability matches the BFS on the replica graphs; suite green; the dead-zone fires per-level.
- *Risk noted earlier:* `V=M·R` over EVERY state each step is O(states²) (it killed a test before). Keep the bounded
  prioritized sweep as the planner; use the SR only for the cheap SPARSE reads (reachability, NEED, the value of a few
  candidate states) — NOT a dense per-step solve. (reference_brain_planning: SR for cheap deep value, rollout sparingly.)

**M2 — arbitration → the basal ganglia (the keystone).** Replace the `if/elif` with `BG.gate`: each grain is a
CHANNEL that proposes (best action, value) — the tabular/SR-value channel, the L5 forward-model channel, the L6
eigenpurpose channel; the BG selects by value + LEARNED affinity (dopamine-RPE), so "which grain decides here" emerges
from reliability, not a threshold. This is the cure for the layout-dependence (the BG learns the eigenpurpose is
unreliable on LockPath-like layouts and down-weights it).
- *The one real design question (do the homework here):* the affinity must be CONTEXT-conditioned (a channel is
  reliable in SOME situations), but the context key must not be hand-coded domain features (bitter lesson). Grounded
  answer: the context is the cortical STATE pattern itself — coarsely, the value-landscape regime (e.g. a learned
  cluster of the place code, or the SR-value spread). Start with the coarsest faithful key and let the gate learn;
  resist enumerating regimes ([[feedback_subgoal_types_from_dynamics]]).
- *Validate:* dynamics games still route to the forward model, nav to the SR — but LEARNED; suite green; benchmark
  holds; the brittle-test regressions do NOT recur.

**M3 — eigenpurpose → L6.** Move `_eigenpurpose` (+ the SVD throttle, which belongs with the grid cache) into
`l6_sr.eigenpurpose(visits)` — the grid readout oriented toward the under-visited extreme. It becomes the L6
exploration CHANNEL the BG (M2) gates. *Validate:* MultiKey gain preserved, LockPath L1 not regressed (the BG
down-weights it where it misleads).

**M4 — forward-model value → L5.** Move `_field_plan`/`field_features`/`_field_err` into L5 (the forward model owns
its own value: pragmatic field-value + epistemic learning-potential). It becomes the dynamics CHANNEL. *Validate:* the
dynamics games (Toggle/Tetris/CollectAll) unchanged.

**M5 — the STATE → L4 ⊗ L6 (feature-at-location).** The original §6 (L6_PLAN L7-A..D): run `L4.bind`/`readout`/
`predict_feature` over the L6 grid location EVERY step, so the agent's state is the sensed FEATURE-at-LOCATION, not the
opaque `config_state`. This is the metric `config_state` lacks → vector navigation, cross-position generalization, and
cross-level transfer (the human's reusable navigation faculty; the oracle/human/agent trace showed L1 cost 12× *in
exploit mode* because nothing geometric transfers). Heaviest change (touches perception); separable from M1–M4.

## 3. The target `agent.py` (the thin loop that remains)
`step` (predict→compare→learn→**delegate select**→predict), `new_episode`, `complete` (episode boundary + reward
observe), `motor`. The select is `column.act(...)` consulting the channels + BG. Target ≈ 80–110 lines, no value
math, no arbitration, no SVD, no BFS. The column becomes the cognition; the agent becomes the body.

## 4. Sequencing (each suite-green; validate by the oracle metric AND the human play-traces)
1. **M1** (SR value/reachability/NEED) — cleanest, deletes the most glue, PURE REUSE, de-risks M2.
2. **M3 + M4** (turn the eigenpurpose and the field value into CHANNELS in their layers).
3. **M2** (the BG composes the channels — the keystone; this is what removes the hand-coded glue for good).
4. **M5** (the location-anchored state — the efficiency lever; can proceed in PARALLEL, different files).
Gate each on: the suite stays green for CORRECTNESS, and the oracle metric does not regress — but per
[[feedback_dont_salvage_between_critical_steps]], M2 is one change across several channels; judge it WHOLE, not mid-way.

## 5. What legitimately STAYS in the agent + open questions
- Stays: the reafference loop itself, the episode boundary (`complete`), the `_GOAL` terminal convention, the motor call.
- Open: the M2 context key (above) is the crux; the SR-value performance bound (M1 risk); whether the eigenpurpose
  survives as a channel or is subsumed once M5 gives a true metric (it may become redundant — re-evaluate after M5).
- The human play-traces (`src/play.py`, built alongside) ground HOW the BG should weight exploration vs. a salient
  target, and whether M5 alone closes the human gap (Findings B/C of the trace).
