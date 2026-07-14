"""End-to-end test of the NON-ABELIAN operator (ARCHITECTURE.md §8, ROADMAP Phase 3a frontier): SE(2) path integration,
where FORWARD's effect DEPENDS on heading, so the group is non-commutative.

SE(2) = R²⋊SO(2) (the semidirect product): TURN rotates the heading ring (abelian); FORWARD translates the location by an
amount that depends on the current heading. Implemented by reusing the abelian `ModularOperator` — the LOCATION operator
keyed by (action, heading) (the heading-conditioned shift), plus a second operator on the heading ring — NOT a tensor code.

The defining test is NON-COMMUTATIVITY: FORWARD then TURN ≠ TURN then FORWARD. Plus: an action's effect learned at some
poses dead-reckons a whole TURN/FORWARD path correctly into NEVER-VISITED positions and every heading (heading- AND
position-invariant, per (action, heading)). RULES #3 acceptance: the agent dead-reckons heading-dependent motion.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent  # noqa: E402

DIR = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}      # heading → unit step (E, N, W, S) — the world's UNKNOWN mapping


def _true(pose, action):
    (x, y), h = pose
    if action == "FORWARD":
        dx, dy = DIR[h]
        return ((x + dx, y + dy), h)
    return ((x, y), (h + 1) % 4)                          # TURN (left)


def _fresh() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _teach(agent: Agent) -> None:
    """Observe both actions at every heading over a small training region — so FORWARD is learned at each heading."""
    for h in range(4):
        for x in range(9, 13):
            for y in range(9, 13):
                pose = ((x, y), h)
                for a in ("FORWARD", "TURN"):
                    agent.learn_pose_move(a, pose, _true(pose, a))


def test_path_integration_is_non_commutative():
    agent = _fresh()
    _teach(agent)
    start = ((30, 30), 0)                                 # a NOVEL position (outside training), heading E
    agent.set_pose(*start)
    agent.path_integrate_pose("FORWARD")
    agent.path_integrate_pose("TURN")
    fwd_then_turn = agent.pose()                          # E:(30,30)→(31,30), then TURN→N  ⇒ ((31,30), 1)
    agent.set_pose(*start)
    agent.path_integrate_pose("TURN")
    agent.path_integrate_pose("FORWARD")
    turn_then_fwd = agent.pose()                          # TURN→N, then N:(30,30)→(30,31)  ⇒ ((30,31), 1)
    assert fwd_then_turn == ((31, 30), 1), f"FORWARD;TURN gave {fwd_then_turn}"
    assert turn_then_fwd == ((30, 31), 1), f"TURN;FORWARD gave {turn_then_fwd}"
    assert fwd_then_turn != turn_then_fwd, "the SE(2) operator must be NON-commutative (FORWARD;TURN ≠ TURN;FORWARD)"


def test_dead_reckons_a_full_se2_path():
    agent = _fresh()
    _teach(agent)
    pose = ((20, 20), 0)                                  # novel region; visits all 4 headings + heading-dependent moves
    agent.set_pose(*pose)
    path = ["FORWARD", "FORWARD", "TURN", "FORWARD", "TURN", "TURN", "FORWARD"]
    ok = 0
    for a in path:
        agent.path_integrate_pose(a)
        pose = _true(pose, a)
        ok += (agent.pose() == pose)
    assert ok == len(path), f"SE(2) dead-reckoning drifted ({ok}/{len(path)} steps) — heading-conditioned integration failed"


if __name__ == "__main__":
    ag = _fresh()
    _teach(ag)
    ag.set_pose((30, 30), 0)
    ag.path_integrate_pose("FORWARD"); ag.path_integrate_pose("TURN")
    a = ag.pose()
    ag.set_pose((30, 30), 0)
    ag.path_integrate_pose("TURN"); ag.path_integrate_pose("FORWARD")
    b = ag.pose()
    print(f"FORWARD;TURN → {a}   vs   TURN;FORWARD → {b}   (non-commutative: {a != b})")
