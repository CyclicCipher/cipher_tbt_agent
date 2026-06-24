"""Phase-2 replica test — prior-minimal agency + dynamics discovery (agent/wm2/perceive.py).

The Phase-1 agent was *told* the agent is a single translating cell. Here we DISCOVER it from a blind
random walk using only the floor priors (sensory (x,y,color) cells, nearest-position metric,
consistency/compression) — no single-cell rule, no contact, no rule-type vocabulary."""

import random

from arc_agi_3 import Environment
from arc_agi_3.core import GameAction
from arc_agi_3.games import LockPath

from agent.wm2.perceive import discover_blockers, discover_dynamics


def _collect_level0(seed=0, steps=120):
    env = Environment(LockPath())
    f = env.reset()
    rng = random.Random(seed)
    acts = [a for a in f.available_actions
            if a != GameAction.RESET and not a.requires_coordinates]
    obs = []
    for _ in range(steps):
        if f.level != 0:
            break
        a = rng.choice(acts)
        g0 = [row[:] for row in f.grid]
        f2 = env.step(a, None)
        if f2.level == 0:
            obs.append((g0, a, [row[:] for row in f2.grid]))
        f = f2
    return obs


def test_discovers_agent_and_move_model_from_scratch():
    obs = _collect_level0()
    agent, model = discover_dynamics(obs)

    assert agent == 2                                      # discovered, not seeded (LockPath agent)
    # the four cardinal unit moves, all present and distinct — discovered with no rule-type prior:
    deltas = set(model.values())
    assert deltas == {(1, 0), (-1, 0), (0, 1), (0, -1)}
    assert all(abs(dx) + abs(dy) == 1 for dx, dy in model.values())


def test_no_dynamics_from_empty_observations():
    agent, model = discover_dynamics([])
    assert agent is None and model == {}


def test_discovers_walls_as_blockers():
    obs = _collect_level0(steps=300)
    agent, model = discover_dynamics(obs)
    blockers = discover_blockers(obs, agent, model)
    assert blockers == {1}                                 # the wall — discovered, not seeded
