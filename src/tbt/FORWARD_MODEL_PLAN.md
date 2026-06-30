# FORWARD_MODEL_PLAN — columns that do generative forward modelling

*Plan, 2026-06-30. The pivot the cn04/ls20 measurements forced (HIPPOCAMPUS.md H1): structured-dynamics games need a
GENERATIVE forward model, not a tabular transition graph. Settled with neuroscience first (memory
`reference_brain_generative_model`) so the model is a COLUMN CAPABILITY, not a harness. Companion to
`TARGET_ARCHITECTURE.md` / `HIPPOCAMPUS.md`.*

## 0. Why — and the harness we avoided
The tabular loop learns a graph over RECURRING states. cn04/ls20 transform a high-dimensional scene every step, so no
state both recurs AND preserves the dynamics (cn04: full board 70/76 distinct; the only encoding that recurs throws
away the reshaping that is the whole game). The fix is to predict the TRANSFORMATION. A validated prototype — a
position-invariant, action-conditioned LOCAL RULE (`next = f(neighbourhood, action)`) — predicted **66% of cn04's
CHANGED cells** (r=2, action-conditioned; coverage ~1.0; action-conditioning ~doubled accuracy → actions genuinely
STEER the dynamics). **Local patterns repeat where the global board never does — why this generalises where tabular
state cannot.** But that prototype read a raw frame and wrote a raw frame = predicting into an EXTERNAL IMAGE = a
harness (deleted). The brain has no external image to write to. So we settled what it predicts INTO first.

## 1. The neuroscience verdict (what to predict INTO)
- **Predictive coding** (Rao–Ballard / Friston): each cortical level predicts the level BELOW in that level's own
  representation; error flows up. No single image — the hierarchy is the canvas.
- **The world-centred canvas = the hippocampal generative model (TEM, Whittington & Behrens 2020)**: factorise
  STRUCTURE `g` (medial-entorhinal GRID = where) × CONTENT `x` (lateral-entorhinal SENSORY = what); the HIPPOCAMPUS
  binds them (`p = g×x`); the objective is **predict the NEXT SENSORY OBSERVATION given position + action**,
  allocentric.
- **The cortex REUSED the entorhinal grid machinery** (Hawkins et al. 2019 — "mechanisms … that evolved for learning
  the structure of environments are now used by the neocortex to learn the structure of objects"). So a column models
  an object the way the entorhinal models an environment: **SAME algorithm, different frame.** ⇒ Build forward
  modelling ONCE as the column algorithm; the hippocampus is the same algorithm on the GLOBAL ALLOCENTRIC frame
  (Cipher's insight = established TBT). This is why we don't choose per-column vs hippocampus — it's build-order.

## 2. The mapping — we already have the structures
| Brain (TEM) | Ours | role in the forward model |
|---|---|---|
| MEC / grid — structure `g` (where) | **L6** (SR-eigenframe / grid) | the location frame predictions are indexed by |
| LEC / sensory — content `x` (what) | **L4** (feature-at-location) | what is predicted AT each location |
| hippocampus — binding `p = g×x` | **L2/3** + the **hippocampus** organ | binds feature⊗location; the allocentric global bind |
| generative prediction (next obs ∣ pos, action) | **L5** operator | predicts the next L4-feature at each L6-location |

**The forward model = L5's operator predicting, for each LOCATION in the L6 frame, the next L4 FEATURE, from the
local feature-context + action.** The validated CA mechanism is exactly this at cell grain (feature = colour =
L4's simplest encoding; location = cell; neighbourhood = L4-features at adjacent L6-locations).

## 3. What changes vs the current column
- Today L5 predicts an OPAQUE next state (tabular `edges`) or an OBJECT-level config transition (`disp`/`recolor`).
  Coarse — good for movement, blind to sub-object dynamics.
- New: L5 also predicts at the FINEST grain — **per-location feature from the local feature-neighbourhood + action.**
  This is the GENERALISING BASE for dynamics; the discrete `edges` stay the per-(state,action) EXCEPTIONS; the
  object-level `disp`/`recolor` is a COARSER abstraction of the same position-invariant operator (kept for movement
  efficiency; unifying grains into one hierarchy — object = a compressed local rule — is deferred).
- The prediction is a feature-at-location MAP (L4 over L6) written into the column's predictive state, compared to the
  actual next map → DENSE per-location prediction error = the learning + epistemic signal (not opaque state equality).

## 4. Build stages (each suite-green; offline reproductions first, per feedback_no_debug_by_extending_actions)
- **FM1 — the column predicts feature-at-location. ✅ DONE 2026-06-30 (suite 80, 4 new tests).** L5 gained
  `observe_field`/`predict_field`/`field_confidence` (the per-location operator: rule `(local feature-neighbourhood,
  action) -> next centre feature`, r=2); the column gained `feature_field` (frame -> L4 feature ids via `L4.encode`)
  + `observe_field`/`predict_field` routing. Lives in L5, reads L4, indexed by L6 — not a standalone, no pixel buffer.
  Validated: a synthetic hidden CA learned + predicted EXACTLY and ACTION-CONDITIONED (`test_forward_model.py`); cn04
  THROUGH THE COLUMN reproduces the prototype exactly — per-cell 0.9822, **0.658 on CHANGED cells**, rule bounded
  (~3.6k), 17 ms/step.
- **FM2 — the predictive loop at feature grain. ✅ DONE 2026-06-30 (suite 83, 3 new tests).** `Agent.step(..., frame=)`
  now does dense field predict-then-compare: L5's forward model predicts the next field, compared per location to the
  actual; the fraction of CHANGED cells mispredicted (`field_error`) is fed to `reward.observe_error` as the
  learning-progress signal (REPLACING the opaque binary state-surprise; binary stays the no-frame fallback, so it's
  backward compatible). Validated: synthetic learning-progress (error drops, epistemic value winds DOWN when mastered,
  and a non-local wrap is correctly treated as bounded noise — noise-robust); cn04 online through the loop, field_error
  0.72 → 0.56 (learning the dynamics live). The error is still indexed by the opaque tabular state (the interim seam);
  FM3 moves PLANNING into field space.
- **FM3 — planning by rollout (epistemic first). ✅ DONE 2026-06-30 (suite 86, 3 new tests).** `col.act` gained a
  per-action `bonus` channel (selection STAYS in the column's inverse-model motor); `Agent._field_plan(field, depth)`
  computes each action's EPISTEMIC value via the forward model (`1 - field_confidence` = learning potential; depth>1
  rolls out via `predict_field`), and `_choose` adds `beta·epistemic` as the bonus when a frame is present. So in a
  dynamics game (flat tabular value -- states never recur) the forward model DRIVES the agent to the action it
  understands least, winding down as each rule is pinned (handing off to pragmatic value). Validated: drives toward
  the unseen action, winds down when both learned, rollout depth composes. Cost: **depth-1 98 ms/step** (the sound
  default); depth-2 768 ms -> the deep/pragmatic rollout (FM4) must be SAMPLED (EZ-V2 Sequential-Halving), not full.
  (Opt deferred: one cell-pass querying all actions would cut depth-1 ~6x.) The PRAGMATIC term toward the score is
  FM4.
- **FM4 — the goal in feature space. ✅ DONE 2026-06-30 (suite 89, 3 new tests).** `Agent.field_value` (a
  `ValueLearner`) learns a GENERALISING value over field FEATURES (`field_features` = per-colour binned counts,
  game-agnostic), TD-trained online from the sparse score; `_field_plan` now returns `(pragmatic, epistemic)` per
  action (pragmatic = the field value of the predicted next field), combined in `_choose` as `pragmatic +
  beta·epistemic` -- the unified EFE drive (plan TOWARD the score while still drawn to the unlearned). KEY FIX (the
  terminal-credit / config-reward bug): the rewarded CONFIGURATION itself is credited (`update(feats, score_delta)`),
  not only the pre-goal config -- else greedy-on-V(next) climbs only to the pre-goal and stalls. Perf: `L5.field_step`
  returns predict+confidence in ONE pass (the planner's hot path) -> `_choose` stays ~100 ms/step at depth 1.
  Validated: the value DIRECTS planning toward the target from either side (grow below / shrink above); the value
  LEARNS from the score in-loop; END-TO-END on a grow-to-target env it scores ~3x a random walk at a hard target
  (30 vs 11). The loop is complete: learn dynamics (FM1) -> dense learning-progress (FM2) -> epistemic drive (FM3) ->
  pragmatic goal (FM4).
- **FM5 — the hippocampus inherits it (deferred).** Apply the SAME machinery to the GLOBAL ALLOCENTRIC frame (the
  hippocampus's L6 over the whole world) = allocentric world modelling. Build once (column) → free for the hippocampus.

## 4b. Unification — ONE model (the forward model subsumes tabular), 2026-06-30
The forward model and the tabular loop are NOT separate algorithms for separate games (Cipher's objection to a
per-game flag). They are arbitrated PER STATE by whether the **tabular value expresses a preference**: in
`agent._choose`, `_tab_spread = max−min` of the tabular action-values; if it is ~0 (the tabular loop is INDIFFERENT
— a dynamics game's novel states, or a recurring state with no learned reward) the FORWARD MODEL decides, else the
tabular value leads and the forward model stands down (and its costly rollout is skipped — the same signal is the
performance gate). One model, no per-game switch; validated by the live-loop still converging to oracle while cn04
auto-engages the forward model. The end goal is to **delete the tabular loop entirely** — the forward model is the
general world model; tabular was a fast exact-memoisation shortcut it subsumes (movement / pushing / blocking /
in-place transformation are all just field transformations):
- **Step 1 ✅ DONE** — deprecate the recognition-based `barriers` faculty (`behavior.py` + its 2 tests deleted; the
  policy wiring stripped). OBSTACLES are handled by the forward model NATIVELY: a blocked move is predicted as
  no-change, so the planner makes no progress there (`test_forward_model_predicts_a_blocked_move_as_no_change`). The
  barriers' object-identity generalisation becomes a FUTURE forward-model/feature improvement, not a faculty.
- **Step 2 — the forward model's VALUE grows up.** Its current value (per-colour COUNTS) can't represent a SPATIAL
  goal ("reach this cell" changes no count), and the rollout is shallow where tabular's value-sweeping was deep. So:
  spatial/relational field features + multi-step value bootstrapping, until the forward model subsumes the
  navigation/recurring-state games. VALIDATE against the PRE-RESET replicas pulled from git history (LockPath,
  MultiKey, Sokoban, Tetris, CollectAll, Toggle, partial-obs) — a far stronger bar than a synthetic env — plus cn04.
- **Step 3 — delete the tabular loop** (L5 `edges`, the SR-value sweeping in `reward.py`, `col.act`-over-graph, the
  inert `blocked` hook) once Step 2 clears the bar. The L4/L5-operator/L6/L2-3 layers STAY; only the discrete-state
  value loop goes. Delete-last (keep tabular as the safety net until the forward model passes), not delete-first.

## 5. Seating discipline (the not-a-harness contract)
Reads **L4** (feature-at-location), indexed by **L6** (the frame); lives in **L5** (the operator layer); writes the
**column's predictive state**. NO raw-pixel buffer — the `Sensor` still bridges frames→L4 features; the column
predicts in FEATURE space. This MODIFIES the one column (L5), it is not a parallel predictor (`feedback_one_model`).

## 6. Open / risks
- **Cost**: per-location prediction + rollout over 64×64 → the all-background skip + sampled/shallow rollout + (later)
  coarser grain / the object-level abstraction.
- **Grain**: cell-grain (feature = colour) fits cn04; richer L4 features (patches) for textured games — the L4 encoder
  already grows a vocabulary online.
- **Stochastic / partial dynamics**: the rule keeps a Counter of outcomes; a `confidence` (fraction unambiguous) gates
  the planner (don't trust an un-pinned context) and feeds learning-progress.
- **Unifying grains**: object-level `disp`/`recolor` and per-location coexist now; the hierarchy (object = compressed
  local rule, recruited on residual) is the later compression step.
