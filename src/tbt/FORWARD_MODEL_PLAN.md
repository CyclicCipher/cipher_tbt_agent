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
- **FM2 — the predictive loop at feature grain.** The agent's predict-then-compare compares predicted vs actual
  feature-maps (dense surprise), feeding learning-progress epistemic value at the feature level (reuse `reward.py`).
- **FM3 — planning by rollout.** Roll out action sequences through the forward model (sampled / shallow, EZ-V2
  Sequential-Halving to bound cost over 64×64; reuse the all-background skip), evaluate the predicted maps toward the
  goal, pick the best. Replaces/augments tabular-graph planning for dynamics games.
- **FM4 — the goal in feature space.** Associate score rises with feature-map configurations (reward over predicted
  maps); the goal = the scoring configuration; plan toward it. The harder half (the goal in a dynamics game).
- **FM5 — the hippocampus inherits it (deferred).** Apply the SAME machinery to the GLOBAL ALLOCENTRIC frame (the
  hippocampus's L6 over the whole world) = allocentric world modelling. Build once (column) → free for the hippocampus.

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
