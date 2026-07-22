# notes/gcml_neural_sampling_cognitive_maps.md — the INVERSE-MODEL cognitive map (GCML), and what it fixes for us

**Paper:** Lin, Yang, Zhao, Pezzulo, Maass — *"Neural sampling from cognitive maps enables goal-directed imagination and
planning"*, Nature Machine Intelligence (2026), doi:10.1038/s42256-026-01254-4 (OPEN ACCESS; PDF in `../research papers/`).
Detailed notes taken 2026-07-21. **Why it matters:** it supplies the piece our hippocampus/columns are missing — a learned
**inverse model** that turns a cognitive map into a cheap, goal-directed, *compositionally-generalising* planner **without
search, without backprop, with only local plasticity** — and it is exactly TBP's acknowledged open problem ("using behaviors
to inform actions"). This is the strongest candidate yet for replacing our expensive/blowing-up BFS rollout.

## 0. Thesis in one line
A cognitive map (a learned FORWARD model relating actions to state changes) becomes a *planner* the moment you also learn its
**INVERSE model** `W` — a linear map from a *state difference* to *the action that reduces it*. Then planning = "at each step
move in the direction of the goal" (`u = W·(s* − s)`), and **imagination** = iterate that using the forward model's own
prediction instead of a real observation. Add noise → a generative model that **samples diverse goal-directed trajectories**
(reproducing hippocampal replay). Everything is learned online by delta/Hebbian rules; no deep nets, no backprop-through-time.

The three brain "tools" it integrates: **cognitive maps** (relational structure), **stochastic computing** (noise as a
resource for sampling), **compositional coding** (embeddings that generalise to never-seen goals).

## 1. Spatial version (grid-cell cognitive map)
- **State** `s ∈ ℝ^1000` = firing of 1000 grid cells at a location; each grid cell = sum of 3 planar cosine gratings (scales
  `λ~U(0.05,8)`, wavevectors 60° apart). The large-scale cells' PCA is a ~flat 2-D manifold → a metric with a *sense of
  direction*. (This IS our L6 SR/grid frame; `reference_grid_sr_eigenbasis`.)
- **Forward model (path integration):** 4 action-basis neurons → a velocity; `Δx = a_right − a_left, Δy = a_up − a_down` (eq 1).
- **Place cells:** ONE-SHOT learning (a BTSP-like plateau rule): `P_i ← s_t` (eq 2) — a place field is just a *stored grid
  pattern* at a visited location. (Our `Column._sr_frame` / place read-out.)
- **Inverse model `W`** (the novelty): learned by a **local Hebbian rule** mapping the grid-state DIFFERENCE to the action
  that caused it: `ΔW = η_w · a_t (s_{t+1} − s_t)^T` (eq 14).
- **Goal-directed action selection:** `a_t = η_a (W(s* − s_t) + ε)` (eq 3). `W` maps *(goal − current)* → the movement toward
  the goal. Noise `ε ~ N(0,σ)` makes it a **sampler** of diverse routes.
- **Imagination (forward sweep, no movement):** replace the real next observation with the forward model's prediction and
  iterate. Reproduces rodent replay incl. **generalisation to never-explored regions** (W is linear, trained on *local*
  diffs, applied to *large* goal diffs).
- **Obstacles WITHOUT remapping:** object-vector / barrier cells → **repulsive forces** added to the action:
  `f_i = δ_i/|δ_i|²`, `F = Σ f_i` (eq 6); `Δx' = Δx + γF_x` (eq 7). Reroutes around novel barriers with no map change —
  matches `reference_vector_navigation` (potential field) + `reference_obstacle_as_transition_cost`.

## 2. General version — the CML / GCML (any/high-dim space)
For non-spatial / high-dim domains, don't assume grid cells; LEARN the embedding.
- **Embeddings:** observation `s_t = Q o_t` (eq 8); goal `s* = Q o*` (eq 9); actions embed via `V`, predicted next state
  `ŝ_{t+1} = Q o_t + V a_t` (eq 10). Target: `Q o_{t+1} ≈ Q o_t + V a_t` (eq 11) — i.e. the action's embedding V·a IS the
  predicted state CHANGE. (Compare our `operator`/L5 "a position-invariant DELTA per action" — same idea, in an embedding.)
- **Forward-model learning (local delta rules, self-supervised prediction error):**
  `ΔV = η_v (s_{t+1} − ŝ_{t+1}) a_t^T` (eq 12), `ΔQ = η_q (ŝ_{t+1} − s_{t+1}) o_t^T` (eq 13). **These are the SAME delta rule
  we already lean on** (`reference_cue_competition_key_discovery`, the critic, `_Readout`) — approximate gradient descent with
  no backprop. If V is linear and eq 11 holds, a linear inverse model is *guaranteed to exist*.
- **Inverse model `W`** (eq 14, as above): a **universal, goal-conditioned value function** — `u_t = W(s* − s_t)` (eq 15)
  scores every action's usefulness for reaching *any* goal `s*`, **policy-independent, no per-goal retraining** (unlike RL /
  a fixed V). This is the direct answer to `project_linear_value_cannot_hold_sokoban`'s framing: don't store a scalar V(s);
  store W that turns *(goal−state)* straight into a graded action preference.
- **Affordance gate `G`** (learns which actions are executable): `ΔG = η_g (g_t − G s_t) s_t^T` (eq 17). **Eligibility**
  `e_t = ĝ_t ⊙ (u_t + ε)` (eq 18); **select** `a_t = WTA(e_t)` (eq 19) — winner-take-all (lateral inhibition = our BG/thalamus
  gate). Noise `ε` is what makes CML (deterministic) into **GCML** (a sampler).
- **Imagination / rollout by bootstrapping:** use the *imagined* state for the utility, `u_t = W(s* − ŝ_t)` (eq 16), and
  advance with the forward model `ŝ_{t+1} = ŝ_t + V a_t` (eq 20). No environment feedback. **Self-correcting homing:** even a
  bad noisy step is compensated because the next step again points at the goal. Diverse runs ≈ hippocampal replay diversity.
- **Cost:** each planning step is O(one WTA over actions), *independent of distance to goal* — no tree search. Fig. 6 shows it
  beats k-shortest-path search / MCTS-style methods on nodes-visited and replanning latency, and instantly re-plans when the
  goal or rewards change (just change the synaptic INPUT `s*`, no retraining).

## 3. Compositional generalisation (the ARC-relevant part)
Task: decide/produce a decomposition of a silhouette into building-blocks (BBs) — **NP-hard**, states never seen in training.
- Make `Q` (observation embedding) **compositional** (here Q≈identity over a binary pixel vector, plus an affordance that
  up-weights protruding pixels). Then the cognitive map has a *sense of direction* even toward goals never encountered.
- **`W` becomes an overlap detector:** with a saturating Hebbian update `W_{t+1} = min(W_t + a_t(s_t − s_{t+1})^T, 1)` (eq 21),
  W is binary and outputs, for a state-diff, *the pixel overlap between the diff and each BB* (0 if the BB isn't contained).
  So "which action helps" = "which building block fits the remaining difference" — read straight off the inverse model.
- **Context affordance by convolution:** `g2(Δ) = (k * (1+Δ)) ⊙ (−Δ)` with the 4-neighbour kernel `k` (eq 22/23); action
  `a_t = WTA(g1(W·g2(Δ)) + ε)` (eq 24). Add-vs-remove is just the sign of Δ (compose = inverse of decompose, same map).
- **Result:** trained on 5-BB silhouettes, it solves **8-BB** ones it never saw — strong compositional generalisation from a
  linear inverse model over a compositional embedding. This is the LID lesson (`reference_lid_locally_in_distribution`)
  realised concretely: a compositional state code makes each local action-choice in-distribution.

## 4. What this SOLVES for our architecture (the point)
Mapping to our known blockers (and TBP's open problems):

1. **The BFS rollout is expensive / blew up (the hang; the current slow L1 planning).** → Replace/augment `Rollout.plan`'s
   graph search with the **inverse-model policy**: pick actions by `W(goal − state)` (+ noise for exploration), advancing the
   world-state with our forward model. O(1)/step, no search, self-correcting, generalises. This is the highest-leverage import.
2. **We have a FORWARD model but no INVERSE model.** Our `operator` (path integration), `WorldModel`, and the new
   `ContactDynamics` all predict *state given action*; none answer *which action reduces (goal − state)*. `W` is precisely that
   missing head, learnable by one Hebbian rule alongside what we already learn. **This is TBP's unresolved "use behaviours to
   inform actions" (capability 5) — Numenta flagged it open; this paper answers it.**
3. **"Linear value can't hold relational V*"** (`project_linear_value_cannot_hold_sokoban`). → `W` is a *universal
   goal-conditioned* value (a policy, really), not a scalar V; it re-targets instantly by changing the goal input. Reframes
   our value story: keep the rollout for genuinely relational leaves, but let W give the cheap sense-of-direction default
   (matches `reference_brain_planning`: routine = cheap read-off, rollout = sparing fallback).
4. **Compositional generalisation to never-seen states (real ARC).** → A compositional observation embedding `Q` + linear `W`
   generalises to novel goals — the concrete mechanism behind our LID discussions.
5. **Obstacles as reshaped reachability, no remapping.** → Repulsive potential field (eq 6/7) is a cleaner `_blocked`/barrier
   story that needs no re-learning when an obstacle moves.
6. **Everything is online + local + no-backprop** — fully compatible with our NO-ANN hard rule; the delta rules (eq 12/13/17)
   are the same rule we already use, and `W` (eq 14) is one more Hebbian synapse set.

## 5. Concrete adoption path (proposal, not yet built)
- **Add an inverse-model head `W`** to the spatial/nav column (and later the scene/behaviour column): learn `ΔW = η_w a
  (s_{t+1} − s_t)^T` online from the SAME transitions we already observe (`s` = the grid/SR code, or the world-state
  featurisation). Then goal-directed action = `WTA(W(s* − s) + noise)`, gated by available actions (our BG/thalamus = the WTA).
- **Replace the novelty/pragmatic `plan` inner loop** (currently BFS) with the inverse-model sampler; keep the rollout only as
  a fallback for relational goals the linear W can't resolve. Directly targets the L1 slowness we are staring at right now.
- **Forward model = what we have** (`operator`/`WorldModel`/`ContactDynamics`); the paper's `V` is our per-action delta, and
  its bootstrap `ŝ_{t+1} = ŝ_t + V a` is our `WorldModel.step` — so the inverse model slots on top of existing machinery.
- **Compositional embedding** for ARC frames is the longer-horizon win (map a frame → a compositional code so `W` generalises
  to novel goals); tie to the object/scene representation.
- Open question for us: our world-state is a *structured* map (agent + objects), not a single vector — so `W(s* − s)` needs a
  featurisation where subtraction is meaningful (the `WorldFeaturizer` is a start; a compositional per-object code is better).

## 6. Key equations (verbatim, for implementation)
- Path-integration forward model: `Δx = (a_t)_right − (a_t)_left`, `Δy = (a_t)_up − (a_t)_down` (eq 1).
- Place field one-shot: `P_i ← s_t` (eq 2).
- Spatial action selection: `a_t = η_a (W(s* − s_t) + ε)` (eq 3), `η_a = 0.05`.
- Obstacle force: `f_i = δ_i/|δ_i|²`, `F = Σ_i f_i` (eq 6); `Δx' = Δx + γF_x`, `Δy' = Δy + γF_y` (eq 7), `γ=0.2`.
- Embedding / forward: `s_t = Q o_t` (8), `s* = Q o*` (9), `ŝ_{t+1} = Q o_t + V a_t` (10), target `Q o_{t+1} ≈ Q o_t + V a_t` (11).
- Forward-model plasticity (delta): `ΔV = η_v (s_{t+1} − ŝ_{t+1}) a_t^T` (12), `ΔQ = η_q (ŝ_{t+1} − s_{t+1}) o_t^T` (13).
- Inverse-model plasticity (Hebbian): `ΔW = η_w a_t (s_{t+1} − s_t)^T` (14). Saturating/binary variant: `W ← min(W + a(s_t −
  s_{t+1})^T, 1)` (21).
- Utility / value: `u_t = W(s* − s_t)` (15); imagined: `u_t = W(s* − ŝ_t)` (16).
- Affordance: `ΔG = η_g (g_t − G s_t) s_t^T` (17); eligibility `e_t = ĝ_t ⊙ (u_t + ε)` (18); select `a_t = WTA(e_t)` (19).
- Imagination bootstrap: `ŝ_{t+1} = ŝ_t + V a_t` (20).
- Hyperparams (graph task): dim ~1000, `η_q=0.1`, `η_v=η_w=η_g=0.01`, noise `α_ε=0.1`; robust across a wide range.

## 7. Honest caveats / where it stops
- The demonstrated inverse models are LINEAR (they work because the maps are ~linear / the embedding makes them so). Genuinely
  relational V* (Sokoban chains) may still need rollout — so this AUGMENTS, not replaces, the rollout.
- Spatial and abstract tasks used SEPARATE maps (by dimensionality, not by physical-vs-abstract) — a unified map is future work.
- It plans over a learned *state vector*; our world-state is structured, so the featurisation/embedding that makes `W(s*−s)`
  meaningful for multi-object scenes is the real integration work.
- Relation to us: `reference_lid_locally_in_distribution`, `project_linear_value_cannot_hold_sokoban`,
  `reference_brain_planning`, `reference_grid_sr_eigenbasis`, `reference_vector_navigation`, `reference_eigenoptions_subgoals`.
