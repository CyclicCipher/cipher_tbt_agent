"""test_inverse_model.py — the INVERSE model: L6a → L5, a goal VECTOR read back into an action (no search).

`MotionOperator.apply` turns an action into a displacement (forward, L5→L6a); `utilities` turns a DESIRED displacement back
into the actions that produce it (inverse, L6a→L5) — the same learned table read backwards, which is why the GCML's Hebbian `W`
converges to exactly these action effects (`notes/inverse_model_featurization_design.md` §3). The agent reads the goal vector
through it, vetoes learned obstacles, and lets the BASAL GANGLIA select (priority = salience ⊕ value); when nothing reduces the
gap it returns None so the caller DELIBERATES with the rollout instead — cheap read-off by default, search sparingly.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                      # noqa: E402
from tbt.basal_ganglia import BasalGanglia       # noqa: E402
from tbt.operator import MotionOperator, eye     # noqa: E402
from tasks.core import GameAction                # noqa: E402

N, S, W, E = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4
_D = {N: (0.0, -1.0), S: (0.0, 1.0), W: (-1.0, 0.0), E: (1.0, 0.0)}


def _op() -> MotionOperator:
    """An operator that has learned all four unit moves (one exact demo each — the running mean is exact at n=1)."""
    op = MotionOperator(ego=True)
    for a, d in _D.items():
        op.learn(a, ((0.0, 0.0), eye(2)), (d, eye(2)))
    return op


def _agent_at(cell) -> Agent:
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    for act, d in _D.items():
        a.learn_pose_move(act, ((0.0, 0.0), eye(2)), (d, eye(2)))
    a.set_pose((float(cell[0]), float(cell[1])), eye(2))
    return a


def test_utilities_score_actions_by_how_far_they_close_the_gap():
    """The inverse read-out: the action whose learned displacement points along the goal vector wins, its opposite scores
    worst — a graded 'sense of direction', not a search."""
    u = _op().utilities(((0.0, 0.0), eye(2)), list(_D), (3.0, 0.0))     # goal is three EAST
    assert max(u, key=u.get) is E and min(u, key=u.get) is W
    assert u[N] == u[S] == 0.0, "a perpendicular move neither closes nor widens the gap"


def test_unlearned_actions_do_not_vote():
    """An action the operator has never observed contributes no utility (no evidence, no vote)."""
    op = MotionOperator(ego=True)
    op.learn(E, ((0.0, 0.0), eye(2)), ((1.0, 0.0), eye(2)))
    u = op.utilities(((0.0, 0.0), eye(2)), list(_D), (1.0, 0.0))
    assert set(u) == {E}, f"only the learned action may vote, got {set(u)}"


def test_basal_ganglia_selects_on_salience_the_cortex_proposes():
    """Priority = salience ⊕ value: with no learned value the BG follows the cortical proposal, and a `-inf` salience VETOES
    an action outright (the gate is where feasibility is enforced)."""
    bg = BasalGanglia(seed=0)
    assert bg.select((), 3, salience=[0.0, 5.0, 1.0]) == 1
    assert bg.select((), 3, salience=[float("-inf"), 0.5, float("-inf")]) == 1


def test_agent_navigates_by_the_inverse_model_and_defers_when_it_stalls():
    """The agent reads the goal vector through the operator and moves toward the target with NO rollout; an action the learned
    FORWARD MODEL says goes nowhere is vetoed; and when nothing reduces the gap it returns None so the caller falls back to
    deliberation. The veto is a prediction, not a remembered list of impassable cells -- so it GENERALISES: one press taught the
    model what feature 9 does, and every OTHER cell holding feature 9 is vetoed without ever being touched."""
    a = _agent_at((5, 5))
    assert a._nav_inverse((5, 5), (8, 5), list(_D)) is E, "should step EAST toward a target three east"

    a._surface = {(6, 5): 9, (5, 4): 9}                                 # feature 9 now sits east of here -- and north of here
    a._learn_delta("into", 9, None, (1.0, 0.0), (-1.0, 0.0))            # ONE press east into a 9: it gave nothing
    assert a._nav_inverse((5, 5), (8, 5), list(_D)) is not E, "an action the model predicts goes nowhere must be vetoed"

    assert a._nav_inverse((5, 5), (5, 5), list(_D)) is None, "already there ⇒ nothing closes the gap ⇒ defer to the rollout"
