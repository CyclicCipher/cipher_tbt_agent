"""hippocampus/replay.py — the ROLLOUT: model-based planning IN the world-map (DESIGN §2/§3, slice 2).

This is the payoff the map (`map.py`) unblocked: with a forkable world-STATE, the hippocampus can REPLAY the cortex's learned
forward model forward through it — the deliberative sweep (`reference_brain_planning`: rollout = sparing deliberation;
`reference_commit_to_test_a_hypothesis`: simulate toward an imagined goal, value at the leaf; `reference_exploration_replay`:
planning = forward sweeps). It is also the substrate for the two-pane imagined-future widget (`project_hippocampus_imagination_and_widget`):
the imagined trajectory either tracks reality or visibly diverges, and divergence IS the bug.

TWO objects:
- `WorldModel` — the learned FORWARD MODEL over a world-state. It COMPOSES (RULES.md #5, DESIGN §5) the agent's
  path-integration (the map's borrowed L6a operator) with each object's STATE-CONDITIONED dynamics (the compositional
  `Column` the hippocampus replays — Rescorla-Wagner cue competition, the push/gravity/override machinery). It re-derives
  nothing; the hippocampus does not own a physics.
- `Rollout` — value-guided model-based PLANNING: a graph search over world-states with VISITED-PRUNING, so it stays O(states
  reachable within the horizon) and never the 2^K flat-action product (`scaling_probe`'s wall). BFS ⇒ the first goal-reaching
  path is the shortest; if the goal is beyond the horizon it heads toward the best-VALUE leaf (the critic as the heuristic).
  The sampled-search refinement that scales branching (`reference_efficientzero_v2`: Gumbel + Sequential Halving) is the noted
  next step; this is the exact small-branching version.

The GOAL is a reward predicate over world-states, PASSED IN (learned/selected elsewhere — `reference_goal_setting_priority_map`),
never hand-coded into the planner (`feedback_bitter_lesson`). Pure stdlib.
"""

from __future__ import annotations

from collections import deque

from ..operator import add, norm, sub

_TOL = 1e-9


class WorldModel:
    """The learned forward model over a world-state — CONTACT-conditioned (`behavior.ContactDynamics`, the touch-grounded object
    behavior model; `notes/touch_and_body_design.md` §7). `step` advances one imagined action. Stateless over the map: it holds
    only the (shared, learned) behavior model, so a rollout forks the state and re-uses the one learned dynamics."""

    def __init__(self, contact) -> None:
        self.contact = contact  # a `behavior.ContactDynamics`: yield/resist/pass + the change T per object, learned from touch

    def step(self, world, action):
        """One imagined step (functional — returns a NEW world). The agent path-integrates; where its move lands ON an object
        (contact — computed GEOMETRICALLY here, since imagination has no skin), that object's LEARNED behavior fires: RESIST
        blocks the body (both stay), YIELD/PASS moves the object by the learned change (0 for pass) and the body advances.
        Solidity is the learned RESIST, not an assumption; the change T is exact one-shot, so there is no snap and no
        per-direction key (`behavior.py`)."""
        tentative = world.move_agent(action)
        if tentative.agent is None or not world.objects:
            return tentative
        eff = sub(tentative.agent[0], world.agent[0])          # the body's displacement for this action (0 if a wall blocked it)
        if norm(eff) <= _TOL:
            return tentative                                   # already wall-blocked → no forward contact
        acell = tuple(round(c) for c in tentative.agent[0])
        agent, objects = tentative.agent, dict(world.objects)
        for oid, pose in world.objects.items():
            if tuple(round(c) for c in pose[0]) == acell:      # the body presses into this object
                obj_disp, blocked = self.contact.predict(oid, eff)
                if blocked:
                    agent = world.agent                        # RESIST: the body cannot advance
                else:
                    objects[oid] = (add(pose[0], obj_disp), pose[1])   # YIELD/PASS: the object takes the learned change
        out = world.snapshot(); out.agent = agent; out.objects = objects
        return out


class Rollout:
    """Value-guided model-based planning in the world-map — the hippocampal forward sweep. `plan(world)` returns the action
    sequence to enact; `greedy(world)` is the 1-step baseline it must beat on a delayed goal."""

    def __init__(self, model, reward, actions, horizon: int = 12, value=None, max_expansions: int = 20000) -> None:
        self.model = model
        self.reward = reward                      # (world) -> float; > 0 at a goal state
        self.actions = list(actions)
        self.horizon = int(horizon)
        self.value = value or (lambda w: 0.0)     # (world) -> float; the critic at a non-goal leaf (0 = no heuristic yet)
        self.max_expansions = int(max_expansions)

    def plan(self, world) -> list:
        """The shortest action sequence reaching a goal within the horizon (BFS over world-states, visited-pruned, off-board
        agent states dropped); if no goal is reachable, the path toward the best-VALUE leaf. Empty ⇒ already satisfied, or
        nothing improves."""
        if self.reward(world) > 0:
            return []
        visited = {world.key()}
        queue = deque([(world, [])])
        best = (self.value(world), [])            # best-value leaf fallback when no goal is within the horizon
        expansions = 0
        while queue and expansions < self.max_expansions:
            node, plan = queue.popleft()
            if len(plan) >= self.horizon:
                continue
            for a in self.actions:
                nxt = self.model.step(node, a)
                if nxt.agent is not None and not nxt.in_bounds(nxt.agent[0]):
                    continue                      # keep the sweep on the board (the boundary as a container)
                k = nxt.key()
                if k in visited:
                    continue
                visited.add(k); expansions += 1
                nplan = plan + [a]
                if self.reward(nxt) > 0:
                    return nplan                  # shortest goal-reaching plan
                v = self.value(nxt)
                if v > best[0]:
                    best = (v, nplan)
                queue.append((nxt, nplan))
        return best[1]

    def greedy(self, world):
        """The 1-step degenerate: the action maximising immediate reward + leaf value (ties → first). The baseline a rollout
        must beat — on a DELAYED goal it has no gradient beyond one step and cannot solve what the sweep can."""
        best_a, best_s = None, float("-inf")
        for a in self.actions:
            nxt = self.model.step(world, a)
            s = self.reward(nxt) + self.value(nxt)
            if s > best_s:
                best_s, best_a = s, a
        return best_a
