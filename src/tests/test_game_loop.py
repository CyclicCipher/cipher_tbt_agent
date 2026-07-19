"""test_game_loop.py — the thin-agent GAME LOOP: the agent plays a real replica level end-to-end (step 2 of the loop).

`Agent.step(FrameData) → action` composes the built regions — the retina (`transduce`), the discovered self (`SelfTracker`),
the L6a operator (`learn_pose_move`), and the hippocampal rollout (`plan`) — into one online interaction. On LockPath L0 the
goal is HIDDEN, so the agent explores by NOVELTY (rollout toward the nearest unvisited cell), learns the action semantics and
the walls from experience (no game semantics read), and reaches the goal. This is the first evidence the composed brain plays a
game, not a synthetic unit test.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                      # noqa: E402
from tasks.games.lockpath import LockPath        # noqa: E402
from tasks.harness import Environment            # noqa: E402


def _play(seed: int, budget: int = 300):
    game = LockPath()
    env = Environment(game)
    fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=seed)
    for _ in range(budget):
        action, coords = agent.step(fd)
        fd = env.step(action, coords)
        if fd.score >= 1:                                    # LockPath L0 completed (the agent reached the hidden goal)
            break
    return fd


def test_agent_solves_lockpath_L0_by_exploration():
    """The agent completes LockPath L0 through its own perceive→learn→plan→act loop — driven only by the colour frame and the
    score, discovering the self, the action displacements, and the walls from experience."""
    fd = _play(seed=0)
    assert fd.score >= 1, f"the agent must solve LockPath L0 by exploration; ended score={fd.score}, actions={fd.action_counter}"


def test_solves_across_seeds():
    """Not a lucky seed — the loop solves L0 from several starts (exploration is directed, so it always covers the room)."""
    for seed in (1, 2, 3):
        fd = _play(seed=seed)
        assert fd.score >= 1, f"seed {seed}: failed to solve L0 (score={fd.score}, actions={fd.action_counter})"


if __name__ == "__main__":
    fd = _play(seed=0)
    print(f"LockPath L0: score={fd.score} in {fd.action_counter} actions (oracle-optimal ~9 — exploration is step 2; "
          f"goal-directed efficiency is step 3)")
