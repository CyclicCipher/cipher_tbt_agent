"""The ONE game agent (tbt.agent.Agent) playing through perception + the Neocortex control-loop planner.

Validates RECONNECT S3 — the factored relational PUSH (egocentric ⊗ absolute, one mover at a time, never the
agent × all-blocks joint) — end to end through the single agent, on Sokoban (push N blocks onto N pads, then
reach the goal). The world-model roles are INJECTED here; F's cold-start that would DISCOVER them from the score
is RECONNECT S2 (next). This test locks in that the Neocortex, wired as the agent's planner, solves the multi-pad
cover loop through the one play(env) loop — no per-game harness.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from perception.control import NeocortexPlanner  # noqa: E402
from perception.scene import Perception, WorldModel  # noqa: E402
from tasks import Environment  # noqa: E402
from tasks.games import Sokoban  # noqa: E402
from tasks.games.sokoban import C_AGENT, C_BLOCK, C_GOAL, C_PAD, C_WALL, _LEVELS  # noqa: E402
from tbt.agent import Agent  # noqa: E402


def _sokoban_world() -> WorldModel:
    """Sokoban's roles (INJECTED — RECONNECT S2 will discover these from the sparse score): the agent body, the
    block as the only pushable, walls block, the goal colour is the reach target, an uncovered pad is the
    required-absent (cover) term. No effects/doors/hazards."""
    return WorldModel(
        body=C_AGENT, pushable={C_BLOCK}, blocking={C_WALL}, death=set(),
        effects={}, adds={}, harmful=set(), goal_colors={C_GOAL}, required_absent={C_PAD},
    )


def _agent() -> Agent:
    world = _sokoban_world()
    return Agent(Perception(world), NeocortexPlanner(world, seed=0))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_neocortex_agent_solves_each_sokoban_level(level):
    """Each level in isolation (a one-level game), so a failure localises to that mechanic depth."""
    env = Environment(Sokoban(levels=[_LEVELS[level]]))
    out = _agent().play(env, max_steps=600)
    assert out.won, f"Sokoban L{level} not solved: {out}"


def test_neocortex_agent_solves_full_sokoban():
    """All three levels in sequence through the one play loop — win == cleared every level."""
    out = _agent().play(Environment(Sokoban()), max_steps=2000)
    assert out.won and out.levels == len(_LEVELS), out
