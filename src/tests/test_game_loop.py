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
from tasks.oracle import solve_level             # noqa: E402


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


def _play_levels(seed: int, levels, budget: int = 400):
    """Play a custom multi-level game continuously through ONE Environment (the harness auto-advances levels), returning the
    final frame and the actions spent on each completed level — so cross-level transfer is measurable."""
    game = LockPath(levels=levels)
    env = Environment(game)
    fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=seed)
    per_level, base = [], 0
    for _ in range(budget):
        action, coords = agent.step(fd)
        fd = env.step(action, coords)
        if fd.score > len(per_level):                        # a level just completed — record its action cost
            per_level.append(fd.action_counter - base)
            base = fd.action_counter
        if fd.is_terminal() or fd.is_win():
            break
    return fd, per_level


# Two PURE-NAVIGATION levels whose goals sit at DIFFERENT cells (bottom-right, then top-right): nothing POSITIONAL can carry
# from one to the other, so solving the second efficiently can only come from a POSITION-INVARIANT goal (the discovered feature).
_NAV_TWO = [
    ["########", "#A.....#", "#......#", "#......#", "#.....G#", "########"],
    ["########", "#.....G#", "#......#", "#......#", "#A.....#", "########"],
]


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


def test_goal_discovered_from_score_transfers_across_levels():
    """Step 3 — the sparse score DISCOVERS the goal, and the discovery TRANSFERS. Level 0 is solved by exploration; on the way
    the reward credits the goal FEATURE (the goal object's colour) by the delta rule. Level 1's goal sits at a different cell,
    so nothing positional carries over — yet the agent goes STRAIGHT to it (the feature names the goal wherever it is), solving
    it at ~oracle cost, far under the exploration it took to find the goal the first time. This is the score→goal→plan loop."""
    fd, per_level = _play_levels(seed=0, levels=_NAV_TWO)
    assert fd.is_win(), f"both nav levels must be solved (WIN); ended state={fd.state}, score={fd.score}"
    assert len(per_level) == 2, f"expected two completed levels, got {per_level}"

    probe = LockPath(levels=_NAV_TWO); probe.load_level(1)
    oracle = len(solve_level(probe))                         # the shortest level-1 solution (open board ⇒ Manhattan distance)

    assert per_level[1] <= oracle + 3, f"level 1 must be ~oracle goal-directed (oracle {oracle}), took {per_level[1]}"
    assert per_level[1] < per_level[0], (
        f"goal-directed L1 ({per_level[1]}) must beat exploratory L0 ({per_level[0]}) — the transfer")


if __name__ == "__main__":
    fd = _play(seed=0)
    print(f"LockPath L0: score={fd.score} in {fd.action_counter} actions (oracle-optimal ~9)")
    fd, per_level = _play_levels(seed=0, levels=_NAV_TWO)
    probe = LockPath(levels=_NAV_TWO); probe.load_level(1); oracle = len(solve_level(probe))
    print(f"NAV transfer: L0 (explore) {per_level[0]} actions -> L1 (goal-directed) {per_level[1]} actions "
          f"(oracle {oracle}); WIN={fd.is_win()}")


def test_solves_the_key_and_door_level():
    """LockPath L1 — key + door — which the agent could not solve at all until 2026-07-23. The goal is VISIBLE but a door
    blocks it, and the mechanic (step on the key, all doors open) is never stated.

    What had to be true for this to work. The epistemic drive has to reach the key, which it does because the key is a feature
    the model has never interacted with (`_unlearned_cells`) — measured, the door opens ~9 steps into the level. And the
    PRAGMATIC drive has to be honest about being stuck: the goal sits behind the door, so the rollout must report it
    UNREACHABLE and hand over. A locally-greedy read-out cannot report that — it happily proposes a step that shrinks the goal
    vector while the goal is walled off — and the agent oscillated between two cells forever, pragmatic at one and epistemic
    at its neighbour."""
    game = LockPath()
    env = Environment(game)
    fd = env.reset()
    agent = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    per_level, base = [], 0
    for _ in range(400):
        action, coords = agent.step(fd)
        fd = env.step(action, coords)
        if fd.score > len(per_level):
            per_level.append(fd.action_counter - base)
            base = fd.action_counter
        if fd.score >= 2:
            break
    assert len(per_level) >= 2, f"the agent must clear L0 AND the key+door level L1, got {per_level}"
    assert per_level[1] < 60, f"and clear L1 without flailing (oracle 12), took {per_level[1]}"
