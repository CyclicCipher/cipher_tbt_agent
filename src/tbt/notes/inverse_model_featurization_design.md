# notes/inverse_model_featurization_design.md — the featurization for an INVERSE-MODEL planner (design of record)

Status: DESIGN (2026-07-21), settled before any code. Companion to `gcml_neural_sampling_cognitive_maps.md` (the GCML paper)
and `touch_and_body_design.md` §7 (the behavior model). Answers the one question that paper leaves open for us: **it plans over a
single state VECTOR, ours is a STRUCTURED map (agent + objects at poses) — what representation makes `s* − s` meaningful?**

## 1. What `s* − s` actually requires

The inverse model `u = W(s* − s)` is only sound if the code satisfies the CML's eq 11: `s_{t+1} ≈ s_t + V·a`. That is not
bookkeeping — it is what makes a LARGE difference decompose into the SUM of the single-step changes `W` was trained on.

- **R1 ADDITIVITY** — an action's effect must be a FIXED vector ADDED to the state, independent of the current state.
- **R2 ENTITY-FACTORING** — the difference must say WHICH entity must move WHERE, not a blended sum.
- **R3 METRIC (not modular)** — the difference must point in a direction an action can reduce.
- **R4 STABLE IDENTITY** — entity *i* in `s*` must be entity *i* in `s` (colour-keyed today; recognition later = crutch #6).

## 2. The trap: our existing featurizer is the WRONG code for W

`hippocampus/featurize.WorldFeaturizer` (per-axis `ScalarEncoder` bumps) has R2/R3 but **FAILS R1**: a bump code SHIFTS rather
than ADDS, so the change caused by moving Δ depends on WHERE the bump was — `V·a` is not a fixed vector and eq 11 breaks.
`GridEncoder` fails harder (modular aliasing, `reference_sdr_regime_and_phase_codes`). The paper survives with grid cells ONLY
because large-scale grid states form a locally-flat, ~linear manifold, and `W` is trained on local diffs.

**Decision: the inverse model's state code is LINEAR IN POSITION — per-entity COORDINATES.**
`s = { entity → position }`, differences taken per entity. Keep `WorldFeaturizer` where it belongs (the VALUE critic, where
overlap-generalisation is the point); it is not the code for `W`. Two codes, two jobs — not parallel machinery.

## 3. WHERE it lives: forward and inverse are the two DIRECTIONS of the L5↔L6a loop

- **Forward model = L5 → L6a.** L5 holds the action's DISPLACEMENT (a position-invariant delta, `reference_l5_operator_kinds`);
  applying it updates L6a's location frame; L4 then reads the feature there (`reference_brain_generative_model`: predict the next
  feature from the next MOVEMENT).
- **Inverse model = L6a → L5.** A LOCATION DIFFERENCE selects the displacement — i.e. the motor command. L5 IS the motor-output
  layer (L5PT = motor command + efference copy + thalamus driver, `reference_layer5_role`), so "which displacement closes this
  gap" is L5's output being driven by the location frame. **Same associative table, read backwards** — which is exactly why
  `W`'s rows turn out to BE the action effects.
- The paper corroborates this in its own network, not by analogy: in Fig. 1a `W` is the set of **synapses from GRID cells onto
  the basis ACTION neurons** — a projection from the location/map layer onto the motor layer. Cortically that is L6a → L5.
- **Honest alternative:** classically, inverse models are placed in the CEREBELLUM / motor cortex (the motor-control literature
  the paper cites). We have no cerebellum; in the cognitive-map framing the inverse model is the map→motor projection, so the
  L5↔L6a loop is our locus. Worth remembering rather than pretending the cerebellar account does not exist.

**We therefore do not need to LEARN `W`.** The paper learns it by `ΔW = η·a·(s_{t+1} − s_t)^T` (a Hebbian outer product of
ACTION × STATE-CHANGE), which accumulates to each action's effect vector — and we already store those explicitly: the nav
`operator` holds every action's displacement (`move_of`), and `behavior.ContactDynamics` holds each object's change `T`. So the
inverse is `utility(a) = δ · effect(a)`: the SAME object as their `W`, read off the table we already own instead of associated
into a second one. (Learning it Hebbian would converge to the same matrix; do that only where the forward model is implicit.)

**Ownership (per `reference_model_ownership_map` — extend the owning layer, never build parallel machinery):**

| piece | TBT locus | ours |
|---|---|---|
| action → displacement (forward) | L5 operator on L6a | `operator.MotionOperator.apply` |
| **displacement → action (inverse)** | same L5↔L6a loop, reversed | `MotionOperator` — a new READ-OUT |
| object change (forward) | L5 dynamics, object-independent behavior frame | `behavior.ObjectBehavior` (`T`) |
| **desired object change → efference** | same behavior frame, inverted | `ObjectBehavior` (`T⁻¹`) |
| the difference `s* − s` | hippocampal GOAL-VECTOR cells (`reference_goal_setting_priority_map`, `reference_vector_navigation`) | the world-map / hippocampus |
| choose among utilities | **BASAL GANGLIA** + thalamic gate (the paper's WTA = "lateral inhibition") | `basal_ganglia.py`, `thalamus.py` |

The paper independently lands on our decision loop: it associates the mechanism with "the neural circuit formed by the
**hippocampus and the ventral striatum**" — hippocampus generating candidate trajectories, striatum evaluating them.

## 4. The hard part: our push VIOLATES additivity — resolve it PER ENTITY

"RIGHT" always moves the AGENT by a fixed delta (R1 holds), but moves the BOX only when the agent is behind it — a
state-DEPENDENT effect, so a flat `V·a` over the box's slot is wrong. A single flat `W` over the whole world-state therefore
cannot work for us; the paper never hits this because its actions have state-independent effects.

**Apply the inverse model per entity, at the level where effects ARE fixed:**
1. **Agent slot (navigation).** Effects are fixed ⇒ `utility(a) = δ_agent · delta_a` over the operator's learned deltas. This is
   exactly the paper's spatial case and REPLACES the nav BFS: a cheap sense-of-direction, O(actions) per step.
2. **Object slot (manipulation).** The required object displacement inverts through the LEARNED BEHAVIOR:
   `efference = T⁻¹ · δ_obj` (the object moves `T·efference`; invert it). The **contact condition** then fixes WHERE the body
   must be (behind the object along that efference) — a nav TARGET, handed back to (1).

So a relational goal resolves with NO search: `δ_box = pad − box` → invert `T` → press direction → press position → nav inverse
model drives the body there → press. **A two-level inverse-model plan.** The object level is object-INDEPENDENT (the same rule
for any object that yields), matching TBP's object-independent behavior frame and our operator's position-invariance.

## 5. Feasibility gating (the paper's affordance `G`)
- **Nav level:** the learned obstacles `_blocked` mask actions that cannot execute (our existing learned reachability;
  `reference_obstacle_as_transition_cost`). The paper's repulsive-force variant (eq 6/7) is the smoother alternative.
- **Object level:** the affordance of "press object O by e" is *can the body REACH the press position* — itself a nav query,
  so it reduces to (1). An object whose behavior is RESIST/UNKNOWN affords no press at all.

## 6. Where the rollout still earns its place (honest limits)
- **Routing an object around obstacles** (the box must turn a corner) is a nav-like problem in the OBJECT's space with its own
  obstacles; the same inverse model applies per-leg, but choosing the legs is sequencing — rollout as the fallback.
- **Chains** (box pushes box) and **conjunctive goals** (Slice B) are relational; a linear read-off will not resolve them —
  keep the rollout for those leaves (`project_linear_value_cannot_hold_sokoban`, `reference_brain_planning`: cheap default,
  sparing deliberation).
- **Non-positional changes** (objects appearing/vanishing, property changes) do not fit a position-displacement code — the
  compositional embedding (the paper's silhouette case) is the future answer, tied to the object/scene representation.
- **R4 identity** is a prerequisite: colour-keyed today (crutch #6); recognition-based identity is the real fix.

## 7. Build order implied
1. **Nav inverse model** — the L6a→L5 read-out on `MotionOperator`: `utility(a) = δ · (world displacement of a at this pose)`,
   with the goal-VECTOR `δ` taken from the world-map and the winner chosen through the **BG/thalamus gate** (masking with the
   learned `_blocked`), NOT a bare argmax. Use it for the pragmatic nav goal in `_act`, rollout as fallback. Smallest step, and
   it kills the nav BFS cost.
2. **Object inverse model** — `T⁻¹·δ_obj` on `ObjectBehavior` → press direction → press position (the contact condition) → hand
   to (1). Turns the relational push goal into a search-free plan. **This is the step that earns or breaks the design** — if the
   press position does not reconstruct cleanly on the open board, the two-level story is wrong; measure it against the Push
   result we already hold (L1 at oracle 6, 12/12).
3. Measure against the current rollout on Push/LockPath (per-step cost, actions-to-goal); keep the rollout as the fallback path.
