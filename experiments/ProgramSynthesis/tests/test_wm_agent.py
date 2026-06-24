"""Tests for the v0 symbolic world-model agent on the ARC-AGI-3 replica.

Symbolic + CPU, so it runs here directly (no GPU / Mistake #36 issue).
"""

from __future__ import annotations

import os
import sys

import pytest

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from arc_agi_3 import Environment, run_episode  # noqa: E402
from arc_agi_3.core import GameAction  # noqa: E402
from arc_agi_3.games import LockPath  # noqa: E402
from agent.wm import WorldModelAgent  # noqa: E402
from agent.wm.world_model import WorldModel  # noqa: E402
from agent.wm.planner import plan  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_agent_solves_at_least_one_level_by_discovery(seed):
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=seed)
    result = run_episode(env, agent, max_actions=4000)
    assert result.levels_completed >= 1


def test_agent_discovers_the_right_structure():
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=0)
    run_episode(env, agent, max_actions=4000)
    wm = agent.wm
    # Discovered from scratch — no LockPath knowledge was seeded.
    assert wm.background == 0                       # modal color
    assert wm.agent_color == 2                      # the object that moves with actions
    assert 3 in wm.goal_colors                      # inferred from the score signal
    assert 1 in wm.blocker_colors                   # walls block movement
    assert len(wm.move_model) >= 3                  # learned its own controls
    # The four learned deltas are the four unit directions (in some assignment).
    deltas = set(wm.move_model.values())
    assert deltas <= {(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)}


def test_agent_handles_game_over_with_reset():
    # If it ever dies, the only legal action is RESET — it must comply, not crash.
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=7)
    result = run_episode(env, agent, max_actions=2000)
    assert result.total_actions <= 2000            # ran to completion without error


def test_agent_induces_the_key_door_causal_rule():
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=0)
    run_episode(env, agent, max_actions=4000)
    # Discovered purely from surprise: contacting the key (4) opens the door (5).
    assert agent.wm.contact_effect.get(4) == {5}


def test_causal_rule_enables_deliberate_planning():
    g = LockPath()
    g.load_level(1)                                # key+door level, door closed
    grid = g.render()[0]
    mm = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
          GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}
    base = dict(background=0, agent_color=2, move_model=mm,
                blocker_colors={1, 5}, goal_colors={3})
    assert plan(grid, WorldModel(contact_effect={}, **base)) is None        # door blocks
    assert plan(grid, WorldModel(contact_effect={4: {5}}, **base)) is not None  # via key


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_agent_reliably_solves_two_levels(seed):
    # Epistemic exploration + the causal rule + the no-op fix -> L0 and L1 by understanding.
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=seed)
    assert run_episode(env, agent, max_actions=4000).levels_completed >= 2


def test_push_is_detected_from_a_translation():
    old = [[0, 2, 6, 0, 0]]                          # agent(2) beside block(6), empty beyond
    new = [[0, 0, 2, 6, 0]]                          # agent shoved the block one cell right
    wm = WorldModel(background=0, agent_color=2, move_model={GameAction.ACTION4: (1, 0)})
    wm.update(old, GameAction.ACTION4, new, 0)
    assert wm.pushable_colors == {6}


def test_goal_becomes_context_dependent():
    # Reaching the goal-color completes L0/L1 but not L2 (pad uncovered) -> the agent
    # induces a context condition on the goal triplet (F_τ(C)) from the sparse win signal.
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=0)
    run_episode(env, agent, max_actions=4000)
    wm = agent.wm
    assert wm.reach_no_win_contexts                  # it saw "reached goal, no win"
    assert 7 in wm.required_absent()                 # learned the pad's presence blocks the win
    # And un-fixation made it actually investigate the block/pad it used to ignore.
    assert {6, 7} <= wm.contacted and 6 in wm.pushable_colors


def test_goal_sufficient_gate():
    wm = WorldModel(background=0, agent_color=2, goal_colors={3},
                    win_contexts=[frozenset({1, 3})],
                    reach_no_win_contexts=[frozenset({1, 3, 7})])
    assert wm.required_absent() == {7}
    assert not wm.goal_sufficient([[3, 7]])          # pad present -> reaching goal won't win
    assert wm.goal_sufficient([[3, 0]])              # pad gone -> it would win


def test_agent_wins_the_full_game_by_discovery():
    # End to end, from scratch: navigate, key->door, push block onto pad (context),
    # avoid the hazard, compose all of it on the final level.
    env = Environment(LockPath())
    agent = WorldModelAgent(seed=6)
    result = run_episode(env, agent, max_actions=6000)
    assert result.won and result.levels_completed == 4
    assert agent.wm.pushable_colors == {6}           # learned to push the block
    assert 6 not in agent.wm.required_absent()       # over-constraint self-corrected away
