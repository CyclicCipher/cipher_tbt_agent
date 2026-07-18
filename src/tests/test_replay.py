"""test_replay.py — the hippocampal ROLLOUT: model-based planning in the world-map (hippocampus/replay.py; DESIGN §2/§3, slice 2).

The map (slice 1) made the world-state forkable; replay is the payoff — fork it, run the learned forward model forward, and
search for the action sequence that reaches a goal. The claim (DESIGN §3.2): a DELAYED goal is solved by forward simulation
where a one-step GREEDY cannot. Three cases: (1) navigation to a goal that needs a turn; (2) a learned PUSH (Sokoban) where the
box only moves when the agent is behind it — object dynamics driven inside the rollout, no hand-coded physics; (3) the value
critic as the leaf heuristic when the goal is beyond the horizon.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                          # noqa: E402
from tbt.hippocampus.map import WorldMap             # noqa: E402
from tbt.hippocampus.replay import Rollout           # noqa: E402
from tbt.operator import eye                         # noqa: E402

ACTIONS = ["N", "S", "E", "W"]
_STEP = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _agent_at(pos) -> Agent:
    """An agent localised at `pos` that has learned the four unit moves (one exact demo each — the operator is a mean, so one
    observation is exact and position-invariant thereafter)."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a.set_pose((float(pos[0]), float(pos[1])), eye(2))
    for act, d in _STEP.items():
        a.learn_pose_move(act, ((0.0, 0.0), eye(2)), (d, eye(2)))
    return a


def _execute(model, world, plan) -> WorldMap:
    for act in plan:
        world = model.step(world, act)
    return world


def test_rollout_reaches_a_delayed_goal_a_greedy_step_cannot():
    """A goal two east + one north of the start: the rollout's forward search finds the multi-step path, while a 1-step greedy
    on the SPARSE goal has no gradient beyond one step and straight-lines past it."""
    a = _agent_at((10, 10))
    goal = (12.0, 11.0)
    reward = lambda w: 1.0 if _dist(w.agent[0], goal) < 0.5 else 0.0
    plan = a.plan(reward, ACTIONS, horizon=6)
    assert plan, "the rollout must return a plan"
    final = _execute(a.world_model(), a.world_state(), plan)
    assert reward(final) > 0, f"the rollout's plan must reach the goal; ended at {final.agent[0]}"

    roll = Rollout(a.world_model(), reward, ACTIONS, horizon=6)
    g, reached = a.world_state(), False
    for _ in range(6):
        g = roll.model.step(g, roll.greedy(g))
        reached = reached or reward(g) > 0
    assert not reached, "1-step greedy has no gradient to a delayed sparse goal — it must fail where the rollout succeeds"


def test_rollout_pushes_a_box_onto_a_target_via_learned_dynamics():
    """Sokoban in miniature, no hand-coded physics: the PUSH is LEARNED (the box moves east under 'E' only when the agent is
    immediately west of it — Rescorla-Wagner cue competition), and the rollout drives that learned dynamics inside the
    forward model. The agent starts NORTH of the box, so it must first go AROUND to the west side (a detour a greedy step
    would never take) before pushing twice."""
    a = _agent_at((5, 6))
    wm = a.world_model()                                   # builds the scene column that holds the object dynamics
    for _ in range(12):                                    # RW converges the push magnitude to ~1 cell over repeats
        before = WorldMap(((3.0, 5.0), eye(2)), {1: ((4.0, 5.0), eye(2))})
        after = WorldMap(((4.0, 5.0), eye(2)), {1: ((5.0, 5.0), eye(2))})
        wm.learn("E", before, after)                       # agent west-adjacent + E ⇒ box moves east

    a.place_object(1, ((5.0, 5.0), eye(2)))                # the scene: box at (5,5), agent at (5,6)
    target = (7.0, 5.0)
    reward = lambda w: 1.0 if _dist(w.objects[1][0], target) < 0.5 else 0.0

    plan = a.plan(reward, ACTIONS, horizon=10)
    assert plan, "the rollout must find a push plan"
    final = _execute(a.world_model(), a.world_state(), plan)
    assert reward(final) > 0, f"the plan must push the box onto the target; box ended at {final.objects[1][0]}"

    roll = Rollout(a.world_model(), reward, ACTIONS, horizon=10)
    g, reached = a.world_state(), False
    for _ in range(10):
        g = roll.model.step(g, roll.greedy(g))
        reached = reached or reward(g) > 0
    assert not reached, "a greedy step cannot position-then-push a delayed goal; the rollout must be what solves it"


def test_value_critic_guides_the_leaf_when_the_goal_is_beyond_the_horizon():
    """When the goal is unreachable within the horizon, the rollout heads toward the best-VALUE leaf — the critic as the
    heuristic (here a distance-to-goal stand-in). The plan must move the agent nearer the goal, exercising the value hook."""
    a = _agent_at((10, 10))
    goal = (20.0, 10.0)                                    # ten east — far beyond horizon 2
    reward = lambda w: 1.0 if _dist(w.agent[0], goal) < 0.5 else 0.0
    value = lambda w: -_dist(w.agent[0], goal)            # nearer = better (the critic's role at the leaf)
    plan = a.plan(reward, ACTIONS, horizon=2, value=value)
    assert plan, "with the goal beyond the horizon, the plan heads toward the best-value leaf"
    final = _execute(a.world_model(), a.world_state(), plan)
    assert _dist(final.agent[0], goal) < _dist((10.0, 10.0), goal), "the value heuristic must pull the plan toward the goal"


if __name__ == "__main__":
    ag = _agent_at((5, 6))
    m = ag.world_model()
    for _ in range(12):
        m.learn("E", WorldMap(((3.0, 5.0), eye(2)), {1: ((4.0, 5.0), eye(2))}),
                WorldMap(((4.0, 5.0), eye(2)), {1: ((5.0, 5.0), eye(2))}))
    ag.place_object(1, ((5.0, 5.0), eye(2)))
    tgt = (7.0, 5.0)
    p = ag.plan(lambda w: 1.0 if _dist(w.objects[1][0], tgt) < 0.5 else 0.0, ACTIONS, horizon=10)
    print(f"push plan: {p}")
    w = _execute(ag.world_model(), ag.world_state(), p)
    print(f"  box ended at {tuple(round(c, 2) for c in w.objects[1][0])} (target {tgt})")
