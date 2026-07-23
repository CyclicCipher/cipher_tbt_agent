"""test_push.py — the PUSH: the agent learns an object's dynamics online and plans a go-around with the hippocampal rollout.

Slice A of Sokoban, on the pure-push fixture (win = block-on-pad, no agent-goal conjunct). It exercises the pieces a nav level
cannot: (1) a MOVER — an object whose motion the agent does NOT directly control, discovered from motion; (2) its push DYNAMICS
learned as a touch-grounded BEHAVIOR (`behavior.ContactDynamics`): the block YIELDS (moves by the learned change `T`) when the
body presses into it, discriminated from RESIST/PASS by the prediction error between the operator's predicted body motion and the
actual outcome, with the SKIN grounding AGENCY (only self-caused motions teach) — no snap, solidity learned not assumed; (3) a
RELATIONAL goal — 'block on pad', discovered from the sparse score and position-invariant so it transfers; (4) the ROLLOUT
planning a GO-AROUND (navigate to the block's far side, then push) that a one-step value cannot find
(`project_linear_value_cannot_hold_sokoban`). Level 0 is a constrained slot solved by exploration (which learns the push + the
goal); level 1 is an open board — solved GOAL-DIRECTED at oracle cost, purely from the transferred relation + learned behavior.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import pytest                                     # noqa: E402

from tbt.agent import Agent                        # noqa: E402
from tasks.games.push import Push                  # noqa: E402
from tasks.harness import Environment              # noqa: E402
from tasks.oracle import solve_level               # noqa: E402


def _play(seed: int, budget: int = 400):
    """Play the two Push levels continuously through one Environment, returning the final frame, actions-per-level, and agent."""
    game = Push()
    env = Environment(game)
    fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=seed)
    per_level, base = [], 0
    for _ in range(budget):
        action, coords = agent.step(fd)
        fd = env.step(action, coords)
        if fd.score > len(per_level):
            per_level.append(fd.action_counter - base)
            base = fd.action_counter
        if fd.is_terminal() or fd.is_win():
            break
    return fd, per_level, agent


def test_agent_solves_push_and_discovers_the_relational_goal():
    """Cold start: the agent solves the constrained push level 0 by exploration, and in doing so discovers the block is a MOVER,
    LEARNS its dynamics by felt contact (the block moves when pressed; a wall/edge behind it BLOCKS it — an occasion, never
    coded), and discovers the goal is the RELATION (block-colour on pad-colour) — all from motion + the sparse score."""
    from tbt.operator import norm
    fd, per_level, agent = _play(seed=0)
    assert len(per_level) >= 1, "the agent must solve at least the first push level by exploration"
    assert 6 in agent._movers, "the block (colour 6) must be discovered as a mover from its motion"
    east = (1.0, 0.0)
    assert norm(agent._dynamics_delta("of", 6, None, east)) > 0.5, "the block's BASE dynamics must be learned (it moves when pressed)"
    assert agent.goal_mem.goal() == (6, 7), f"the goal must be the relation (block 6 on pad 7), got {agent.goal_mem.goal()}"


def test_go_around_push_transfers_at_oracle_cost():
    """The transfer: level 1 puts the block's pad on the far side, so nothing positional carries over — yet the rollout plans the
    GO-AROUND (navigate to the block's left, push it right onto the pad) goal-directed from the transferred relation, at oracle
    cost. Checked across seeds so it is the mechanism, not a lucky start."""
    probe = Push(); probe.load_level(1)
    oracle = len(solve_level(probe))                          # shortest level-1 solution (the go-around push)
    for seed in range(8):
        fd, per_level, agent = _play(seed=seed)
        assert fd.is_win(), f"seed {seed}: both push levels must be solved (WIN); ended {fd.state}, per_level={per_level}"
        assert per_level[1] == oracle, (
            f"seed {seed}: level 1 must be oracle-optimal goal-directed ({oracle}), took {per_level[1]} — the go-around push")


if __name__ == "__main__":
    probe = Push(); probe.load_level(1); oracle = len(solve_level(probe))
    fd, per_level, agent = _play(seed=0)
    print(f"Push: L0 (explore) {per_level[0]} actions discovers goal={agent.goal_mem.goal()} -> "
          f"L1 (go-around, goal-directed) {per_level[1]} actions (oracle {oracle}); WIN={fd.is_win()}")
