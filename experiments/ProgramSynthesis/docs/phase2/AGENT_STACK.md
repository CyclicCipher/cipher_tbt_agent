# Agent stack inventory (2026-06-13)

A full map of the machine that perceives, learns, and acts in the replica — written to reason from the
whole rather than patch parts. The headline finding is at the bottom (§ "Where the thrashing lived").

## The loop (the spine — shared by every agent)

`WorldModelAgent.choose_action(frame)` (`agent/wm/agent.py`), inherited by all wm2 agents:

1. **GAME_OVER** → `wm.learn_death(prev)` then return `RESET`.
2. else if the agent **moved** (prev_action ≠ RESET, same level) → `wm.update(prev_grid, prev_action, grid, Δscore)`.
3. else if a **level was crossed** → `wm.infer_goal(prev)`, clear `_visited` + `dead_block_cells`.
4. `action = self._decide(frame)`  ← **the control policy; the only thing the subclasses change**
5. `self._after(...)` — record prev_*, increment `_visited[agent_pos]`.

Everything below feeds this loop. The agent sees only `frame.grid` (ints), `score`, `level`, `state`,
`available_actions`.

## Layers

### L1 · Perception — `perceive.py` · SOLID (prior-minimal)
- `changes` / `color_displacements` — frame diff.
- `discover_dynamics(buffer)` → `agent_color`, `move_model` (discovered agency, no single-cell prior).
- `discover_blockers` (standalone/tested; the agent actually learns blockers per-step in L2).
- `find_cells`, `color_at`.
- Prior floor: cells = (x,y,color); nearest-position metric; consistency. *Nothing seeded above that.*

### L2 · World model / schema — `world_model.py` → `DiscoveredWorldModel` · SOLID (A+B)
The single shared substrate every agent reads. Fields: `background, agent_color, move_model,
blocker_colors, goal_colors, contact_effect{trigger→removed}, pushable_colors, contacted,
win_contexts / reach_no_win_contexts (→ required_absent, the F_τ(C) context condition),
dead_block_cells, fail_positions`.
Induction: `update` (discovered agency + per-step move/blocker), `_learn_contact` (general
residual-signature: reappear⇒pushable, vanish⇒contact_effect), `infer_goal` (score⇒goal+win_context),
`learn_death`, `required_absent`, `goal_sufficient`, `resolved`. **Prior removals A (agency) + B
(rule-types) live here. This layer is not where the trouble is.**

### L3 · Planning — THREE implementations, all reading the same L2 fields
- **Typed (Phase-1, `planner.py`)**: `plan_to` (BFS over pos × opened-set), `plan` (to goal, routing
  *around* pushables), `plan_push_to` (Sokoban, with an *optimistic passthrough* probe for
  true-deadlock detection). Cheap, targeted. **Used by the 12/12 agent.**
- **Flat forward-sim (`plan.py`)**: `forward_step` (the learned transition model) + `plan_to_win`
  (full-state BFS over agent×movables×opened; `_is_win` with the removable-experiment). General, but
  the predicted **scaling wall** (re-searches every step).
- **Hierarchical (`hplan.py`)**: `hplan_to_win` — macros = discovered edges (reach-trigger /
  push-onto-target / reach-goal) composed in prerequisite order, each refined by a focused BFS.
  General, **fast** (seed 2 in 120 steps), deadlock = a macro with no refinement (`None`).
- Known latent bug in the forward/hier base-grid reconstruction: a movable *covering* an effect-target
  (block on pad) erases the target from `base`, so an already-satisfied condition isn't recognized.

### L4 · Control policy — `_decide` — FOUR implementations · **THE crux**
| agent | _decide structure | result |
|---|---|---|
| **`VolumeAgent`** (= Phase-1 `WorldModelAgent._decide`) | exploit *if `goal_sufficient`* → cover (push to required-absent, *optimistic-probe deadlock* → `dead_block_cells` + RESET) → explore-unresolved (*`_would_strand` guard*) → epistemic (contact unknowns) → experiment (reach goal to refute over-constraint) → coverage (*`_would_strand` guard*) | **12/12** |
| `ForwardPlanAgent` | plan → explore → reset | 2/12 |
| `HierarchicalPlanAgent` | plan(hplan) → explore → reset | 2–4/12 |
| `EFEAgent` | pragmatic(plan) → epistemic → recover, + `_block_landings` | 0/12 |

### L5 · Scoring — `score.py`: RHAE-proxy (oracle-baseline).

## Where the thrashing lived — the one finding that matters

**The 12/12 control policy works because of a tuned set of interacting GUARDS, not because of its
planner:**
- `goal_sufficient` — anti-fixation: don't walk to the goal when its context isn't met.
- `dead_block_cells` + `_would_strand` — deadlock **learning + avoidance**: after a stranding+reset,
  *refuse to wander the block back into the dead cell*. (Recovery is three coupled parts: **detect**
  (optimistic-probe), **learn** (`dead_block_cells`), **avoid** (`_would_strand`) — all three must agree.)
- coverable-vs-not — choose *cover* vs *experiment*.

**When I built the general planner (L3 — a real, good generalization), I simultaneously threw away the
entire L4 cascade and its guards, and tried to re-derive them piecemeal.** That conflated two
independent changes, and re-deriving the guards in a new control structure is what oscillated 4→2→0.
The wm2 agents kept only **detect + reset**, dropping **learn + avoid** — so they re-strand after every
reset. `EFEAgent`'s `_block_landings` is a *fourth, disconnected* learning store that doesn't feed the
planner or the avoidance.

**`VolumeAgent` (12/12) is the existence proof that the control policy can work.** The disciplined path
the inventory points to: **change ONE layer at a time, starting from the working agent** — not maintain
four parallel from-scratch control policies.

## Coupling map
- **Control (L4)** reads L2 fields, calls L3 planners, and triggers recovery (RESET + `dead_block_cells`).
- **Recovery** is split across layers: *detect* (L4 optimistic-probe / L3 hplan-None), *learn* (L2
  `dead_block_cells`), *avoid* (L4 `_would_strand`). The wm2 agents broke this triad.
- **Discovery of context-dependent effects** (block→pad) needs a *directed* push (epistemic), which the
  pure planner never does — it must be supplied by L4.

## Redundancy to retire
4 agents · 3 planners · 2 `_decide` shapes. The next work should converge on **one** control policy and
**one** planner, *derived from the working `VolumeAgent`*, not maintained as parallel WIP.

## Candidate disciplined next steps (one change at a time, each re-verified)
1. **Isolate the planner swap.** Start from `VolumeAgent`'s 12/12 cascade; replace only its planning
   calls with the hierarchical planner where it is a clean drop-in — keeping *every* guard. Verify the
   win-count holds before touching the control structure.
2. **Or port the missing guards** into `HierarchicalPlanAgent`: the deadlock *learn + avoid*
   (`dead_block_cells` + `_would_strand`) it currently lacks, plus fix the L3 covered-target base bug —
   but each as a separate, re-verified change.
3. Then, and only then, consider replacing the guard-cascade with a principled (continuous-EFE) policy —
   against the now-understood baseline, with the guards as the behaviours it must reproduce.
