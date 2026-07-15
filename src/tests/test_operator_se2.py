"""End-to-end test of NON-ABELIAN SE(2) path integration (ARCHITECTURE.md §8): FORWARD's effect DEPENDS on heading, so the
group is non-commutative.

SE(2) = R²⋊SO(2). After the 2026-07-14 cut-over the operator learns each action's **body-frame** displacement + heading
change from observation, and maps it to the world through the CURRENT heading. That makes non-commutativity structural
(FORWARD;TURN ≠ TURN;FORWARD) with **no keying, no heading ring, and no discretisation** — heading is a CONTINUOUS angle in
degrees, and one observation generalises to every position AND every heading (the body-frame delta is invariant to both).

RULES #3 acceptance: the agent dead-reckons heading-dependent motion.
"""

from __future__ import annotations

import math
import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent      # noqa: E402
from tbt.operator import rotate  # noqa: E402

TOL = 1e-6
STEP = 1.0                       # FORWARD advances one unit along the current heading
TURN = 90.0                      # TURN rotates 90° left (any angle would do — heading is continuous)


def _true(pose, action):
    """The world's (unknown-to-the-agent) dynamics."""
    (x, y), h = pose
    if action == "FORWARD":
        dx, dy = rotate((STEP, 0.0), h)
        return ((x + dx, y + dy), h)
    return ((x, y), (h + TURN) % 360.0)


def _close(a, b):
    (ax, ay), ah = a
    (bx, by), bh = b
    return abs(ax - bx) < TOL and abs(ay - by) < TOL and abs((ah - bh + 180.0) % 360.0 - 180.0) < TOL


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _teach(agent: Agent) -> None:
    """Observe both actions at a FEW poses. Because the operator stores the BODY-frame delta, this generalises to every
    position and every heading — including headings never observed."""
    for h in (0.0, 90.0):                       # deliberately only two headings — the rest must generalise
        for x in (9.0, 11.0):
            for y in (9.0, 11.0):
                pose = ((x, y), h)
                for a in ("FORWARD", "TURN"):
                    agent.learn_pose_move(a, pose, _true(pose, a))


def test_path_integration_is_non_commutative():
    agent = _fresh()
    _teach(agent)
    start = ((30.0, 30.0), 0.0)                 # a NOVEL position, heading East
    agent.set_pose(*start)
    agent.path_integrate("FORWARD")
    agent.path_integrate("TURN")
    fwd_then_turn = agent.pose()                # E:(30,30)→(31,30), then TURN→90°
    agent.set_pose(*start)
    agent.path_integrate("TURN")
    agent.path_integrate("FORWARD")
    turn_then_fwd = agent.pose()                # TURN→90°, then N:(30,30)→(30,31)
    assert _close(fwd_then_turn, ((31.0, 30.0), 90.0)), f"FORWARD;TURN gave {fwd_then_turn}"
    assert _close(turn_then_fwd, ((30.0, 31.0), 90.0)), f"TURN;FORWARD gave {turn_then_fwd}"
    assert not _close(fwd_then_turn, turn_then_fwd), "SE(2) must be NON-commutative (FORWARD;TURN ≠ TURN;FORWARD)"


def test_dead_reckons_a_full_se2_path():
    agent = _fresh()
    _teach(agent)
    pose = ((20.0, 20.0), 0.0)
    agent.set_pose(*pose)
    path = ["FORWARD", "FORWARD", "TURN", "FORWARD", "TURN", "TURN", "FORWARD"]
    ok = 0
    for a in path:
        agent.path_integrate(a)
        pose = _true(pose, a)
        ok += _close(agent.pose(), pose)
    assert ok == len(path), f"SE(2) dead-reckoning drifted ({ok}/{len(path)} steps)"


def test_generalises_to_an_UNOBSERVED_heading():
    """The body-frame delta is heading-invariant, so FORWARD works at a heading never observed during learning — the payoff
    of a continuous state (the old discrete ring could only key on headings it had seen)."""
    agent = _fresh()
    _teach(agent)                                # taught only at 0° and 90°
    agent.set_pose((5.0, 5.0), 37.0)             # a heading never observed, and not on any grid
    agent.path_integrate("FORWARD")
    dx, dy = rotate((STEP, 0.0), 37.0)
    assert _close(agent.pose(), ((5.0 + dx, 5.0 + dy), 37.0)), f"got {agent.pose()}"


if __name__ == "__main__":
    ag = _fresh()
    _teach(ag)
    ag.set_pose((30.0, 30.0), 0.0)
    ag.path_integrate("FORWARD"); ag.path_integrate("TURN")
    a = ag.pose()
    ag.set_pose((30.0, 30.0), 0.0)
    ag.path_integrate("TURN"); ag.path_integrate("FORWARD")
    b = ag.pose()
    print(f"FORWARD;TURN → {a}\nTURN;FORWARD → {b}\nnon-commutative: {not _close(a, b)}")
    ag.set_pose((5.0, 5.0), 37.0); ag.path_integrate("FORWARD")
    print(f"FORWARD at an UNOBSERVED heading 37° → {ag.pose()}")
