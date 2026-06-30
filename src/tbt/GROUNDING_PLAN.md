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

**M5 — the STATE → L4 ⊗ L6, AND the full After-L6 reconnection (correct the column's predict-sense-update loop).**
Today's opaque `config_state` BYPASSES the layers — the column never runs the TBT cycle. Complete L6_PLAN §6 (L7-A..D),
which still applies and is what actually corrects the cortical column:
- **L7-A — L4↔L6 in the loop:** run `L4.bind`/`readout`/`predict_feature` over the L6 grid LOCATION every step; the
  state becomes the sensed FEATURE-at-LOCATION (the TEM canvas = structure × content), not `config_state`. Gives the
  metric `config_state` lacks → vector navigation + cross-position generalization (the trace: L1 cost 12× *in exploit
  mode* because nothing geometric transfers).
- **L7-B — L5→L6 path integration:** L5's chosen displacement (efference copy) path-integrates L6, so the operator
  reads/writes the LOCATION-anchored state and generalizes the same displacement anywhere.
- **L7-C — L2/3 over L4⊗L6:** the object = the graph of (feature, grid-location) pooled over sensing; recognition +
  lateral CMP voting predict the feature L4 should expect (object permanence under the grid frame).
- **L7-D — the STATE shift** from opaque config → location-anchored (grid place + features-at-locations).
The heaviest change (touches perception); it is the SUBSTRATE the mechanic layer (§3) rides on. Separable from M1–M4.

## 3. The MECHANIC LIBRARY + the hypothesis-test loop (the cross-game lever — researched 2026-06-30)
The human play-traces ([[reference_human_baseline_traces]]) prove the biggest RHAE lever is NOT navigation tuning: a
human plays at/near ORACLE once the mechanic is known (all overhead = DISCOVERY), and **cross-GAME mechanic transfer**
is decisive (LockPath L2 unsolvable cold → optimal 13 after Sokoban's push-block-onto-marker clicked). `MOTOR_REFACTOR
§8.2` independently found ARC's real OVERT uncertainty is **dynamics/RULE + goal**, not object identity. So above the
substrate the agent must DISCOVER, STORE, and RE-PROPOSE mechanics. This corrects two mappings I had wrong (Cipher
caught it) and answers "what if the truth isn't derivable from priors?".

- **What a MECHANIC is, neurally (corrected — NOT L2/3).** A mechanic = a **SCHEMA / latent cause = a cognitive map of
  TASK-relational structure**, abstracted over episodes and generalizing across contexts (Ghosh & Gilboa; Tse et al.).
  It lives in a **TASK-SPACE TEM module** — a SECOND map/column (PFC/OFC-like), composed with the spatial/object column
  via the HETERARCHY (long-range connections), per [[reference_hierarchy_substrate]] — NOT in L2/3 (which is single-OBJECT
  identity). The object behaviours/affordances it composes ("a block translates when pushed", "a pad is a marker") are
  TBT **object states/behaviours** (L5 operators + L2/3 object — TBT models exactly these). The mechanic is the
  RELATION: "movable object placed on salient marker → the score advances." (The L6 "location frame" is L6/SR; M5 wires
  the STATE to it — M5 is not itself the frame; the second correction.)
- **How a hypothesis is GENERATED — including BEYOND/AGAINST the priors (Cipher's challenge, now grounded).** The brain
  does NOT generate from a fixed prior. It holds priors over an **OPEN set of latent structures** (seeded by Core-
  Knowledge: objectness, movability-inferred-from-dynamics, goal-directedness) and ASSIMILATES observations into
  existing schemas by Bayesian inference. When prediction error is LARGE — an action's outcome or a score-change no
  current schema predicts — it **infers a NEW latent cause / SPLITS a state / mutates the schema** (Gershman–Norman–Niv;
  Redish: "tonically negative prediction errors → higher probability of creating a new state"; the HIPPOCAMPUS binds the
  prior-violating conjunction; lesions → everything forced onto one cause). So the priors SEED but never CAP the space;
  surprise EXPANDS it. That is the formal fix for "the truth contradicts the priors" — accommodation, not just
  assimilation. (Connects to [[project_discovery_program]] schema-mutation/MDL, [[reference_discovery_regime_transition]].)
- **The LOOP = the GSG, extended from identity to RULE/DYNAMICS.** The GSG generates a candidate goal-state HYPOTHESIS
  ("bring the movable object to the marker") from the task map + current uncertainty → the **basal ganglia gates** it
  (value + epistemic + urgency, dopamine-RPE; Cisek affordance competition) → the agent ACTS to test it → the **SCORE**
  confirms (reward-PE → raise the schema's affinity, consolidate) or refutes (no PE → drop for this context, or split a
  new latent cause). The current GSG (`MOTOR_REFACTOR` GD1–GD4: `L23.disambiguation_goal` graph-mismatch, `propose_goals`,
  `BasalGanglia.gate`) does this for object-IDENTITY — moot in full-frame ARC (§8.2). **EXTEND it to the rule/dynamics
  hypothesis-test**: graph-mismatch is domain-general (resolve disagreement between any two competing models — object OR
  rule); the message-shaped `GoalState` already exists. This is the GSG work `MOTOR_REFACTOR` left unfinished.
- **The EXECUTION = MODEL-BASED planning + COMMITMENT (the crux; [[reference_commit_to_test_a_hypothesis]], 2026-06-30).**
  Sokoban/LockPath give NO intermediate signal -- only full completion. So the agent must COMMIT to and execute a whole
  multi-step plan (push the block all the way to the marker) to TEST the hypothesis, BEFORE any reward. A coverage-
  shaping DIAGNOSTIC (reward = pads covered, from the true state) did NOT solve Sokoban L0 (3 seeds, both modes) --
  model-free shaping is the WRONG fix (it rewards coverage after the fact, but nothing makes the agent commit to the
  untested push). The brain does it by: (1) **model-based SIMULATION** -- the forward model rolls the plan out
  (hippocampal preplay / vicarious trial-and-error), so the plan's value comes from the IMAGINED goal back-propagated
  through the rollout, NOT reward-along-the-way; (2) the hypothesis as a **PRIOR PREFERENCE** (active inference -- min
  EFE toward preferred outcomes); (3) the **EPISTEMIC value** of testing the hypothesis is the drive to commit to an
  UNTESTED plan (EFE = pragmatic + epistemic; resolving the rule is worth it before you know it works); (4)
  **COMMITMENT** -- the BG gates the plan as ONE option with hysteresis (no per-step re-deliberation; PFC goal-
  maintenance), VTE only at deliberation then automation. So the loop's "ACTS to test it" is precisely: the forward
  model (**FM1-4**, the simulator) ROLLS OUT the GSG goal-hypothesis to value the plan toward the IMAGINED, UNVISITED
  config (the **achiever/`ValueLearner`** in feature space earns its keep here, unproven at this depth) → the **BG**
  COMMITS → the **score** confirms/refutes. All pieces exist; the wiring "GSG-hypothesis → rollout-to-value → commit →
  confirm" is §3's core build. Denser reward shaping is NOT the fix -- this is model-based, not model-free.
- **CROSS-GAME transfer + negative-transfer SAFETY (reverses "transfer OFF").** A confirmed schema PERSISTS across games
  (cortical consolidation), indexed by object-affordance so the same KIND of object retrieves it (Sokoban block ≡
  LockPath block). It is re-proposed as a HYPOTHESIS (a top-down prior), never asserted — the BG gates it, the score
  tests it; a wrong schema earns no reward-PE → its affinity drops → it's dropped for this game (or a new cause is
  split). So cross-game transfer is SAFE *because* it's tested. Transfer MECHANICS (tested schemas), not weights/graphs.
- **Bitter-lesson guardrail.** The hypothesis SPACE is general (Core-Knowledge priors + latent-cause expansion); the
  specific mechanics are GENERATED + TESTED, never hand-coded — "block→pad" in code is the bug. The GSG reads only the
  column's generic uncertainty + the score, never game colours/features ([[feedback_bitter_lesson]], [[feedback_subgoal_types_from_dynamics]]).
- **Build order (§3 increments, each suite-green; validate on Sokoban L0 + LockPath L2):**
  - **G-A — the goal-HYPOTHESIS generator.** From the frame + dynamics, detect MOVABLE objects (seen to move) and
    SALIENT MARKERS (rare/distinct static cells; the goal-cell + pad are instances), and propose the goal-state
    "movable object at marker location" (general — never reads a colour id). Extends `column.propose_goals`.
  - **G-B — ROLLOUT-to-value (the crux).** The forward model (FM1-4) imagines the plan to the proposed (UNVISITED)
    goal config; the achiever (`ValueLearner` in feature space) values the IMAGINED goal so the plan's value flows
    back through the rollout (model-based, no reward-along-the-way). This is the diagnostic's lesson made real.
  - **G-C — COMMIT.** The BG gates the chosen goal as ONE option with hysteresis (`MOTOR_REFACTOR` §7.3) so the
    multi-step push executes without per-step re-deliberation; the EPISTEMIC value of testing drives committing.
  - **G-D — CONFIRM + persist.** Completion confirms the schema (reward-PE) → store it indexed by object-affordance →
    re-propose on the next game (cross-game transfer); no completion refutes it → drop/split. Safe because tested.

## 4. The target `agent.py` (the thin loop that remains)
`step` (predict→compare→learn→**delegate select**→predict), `new_episode`, `complete` (episode boundary + reward
observe), `motor`. The select is `column.act(...)` consulting the channels + BG. Target ≈ 80–110 lines, no value
math, no arbitration, no SVD, no BFS. The column becomes the cognition; the agent becomes the body.

## 5. Sequencing (each suite-green; validate by the oracle metric AND the human play-traces)
SUBSTRATE first (it's what §3 rides on — the BG is the hypothesis selector, the SR the value, L4⊗L6 the execution frame):
1. **M1** (SR value/reachability/NEED) — cleanest, deletes the most glue, PURE REUSE, de-risks M2.
2. **M3 + M4** (turn the eigenpurpose and the field value into CHANNELS in their layers).
3. **M2** (the BG composes the channels — the keystone; ALSO the hypothesis SELECTOR §3 needs).
4. **M5 / After-L6** (L7-A..D — the location-anchored state; the metric lever; parallel-able, different files).
THEN the human-level lever:
5. **§3 — the task-space map + the rule/dynamics GSG + cross-game schema persistence** (the biggest RHAE win per the
   traces; the latent-cause expansion / beyond-priors lands here). It RIDES on M1 (value), M2 (selector), M5 (frame).
Gate each on: the suite stays green for CORRECTNESS, and the oracle metric does not regress — but per
[[feedback_dont_salvage_between_critical_steps]], M2 (and §3) is one change across several channels; judge it WHOLE.

## 6. What legitimately STAYS in the agent + open questions
- Stays: the reafference loop itself, the episode boundary (`complete`), the `_GOAL` terminal convention, the motor call.
- Open: the M2 context key (the crux); the SR-value performance bound (M1 risk); whether the eigenpurpose survives as a
  channel or is subsumed once M5/§3 give a true metric + relational structure (re-evaluate after M5).
- §3 open: how the task-space TEM module is learned ONLINE from a single playthrough (TEM normally learns structure
  offline over many envs — [[reference_hierarchy_substrate]] flags this); the affordance-index for cross-game schema
  retrieval; the latent-cause "split" threshold (surprise magnitude) without it thrashing.
- The human play-traces (`src/play.py`) ground HOW the BG weights exploration vs. a salient target, and whether M5
  alone closes the gap or §3 is required.

## Sources (the 2026-06-30 research)
- **Latent-cause inference / structure expansion:** Gershman, Norman & Niv 2015, *Discovering latent causes in RL*;
  Gershman & Niv 2010, *Learning latent structure: carving nature at its joints* (PMC2862793) — priors over an open
  structure space; large prediction error → state-splitting / new latent cause; dopamine gates it; hippocampus binds it.
- **Schemas:** Ghosh & Gilboa 2014, *What is a memory schema?*; Tse et al. 2007 — schema = abstracted relational
  knowledge, mPFC/hippocampus, assimilation vs accommodation; prediction error updates/creates schemas.
- **Task-space cognitive map / heterarchy:** Whittington et al. 2020 (TEM); Behrens et al. 2018 (*What is a cognitive
  map?*); Hawkins et al. 2019 (grid framework); TBP *Hierarchy or Heterarchy?* (arXiv 2507.05888) — via [[reference_hierarchy_substrate]].
- **TBT object behaviours/states + GSG:** Thousand Brains Project (arXiv 2412.18354; Thousand-Brains Systems arXiv
  2507.04494) — learning modules model object states/behaviours + compositional objects; each LM has a GSG.
- **Selection / goals:** Cisek 2007 (affordance competition); Adams/Shipp/Friston (*predictions not commands*) — via
  [[reference_gsg_goal_generation]].
