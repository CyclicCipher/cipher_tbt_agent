"""L6_NONABELIAN Stage 1c -- a NON-ABELIAN test environment (the prerequisite for testing the redesign / the cross-layer
unification). A heading-carrying agent (pose = x, y, theta) with BODY-FRAME actions FORWARD / TURN_L / TURN_R = the SE(2)
group. FORWARD moves in the CURRENT heading, so its effect DEPENDS on theta: the actions do NOT commute, and the abelian
`move_delta` (ONE translation per action) CANNOT represent FORWARD. This is the concrete inconsistency the refactor fixes:
L6 tracks POSITION but must track the POSE (the group element)."""

from __future__ import annotations

import math
import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


class OrientationWorld:
    """A heading-carrying agent on the plane -- the smallest SE(2) (non-abelian) dynamics. Pose = (x, y, theta); actions:
    FORWARD (step in the current heading), TURN_L / TURN_R (rotate by +/- 90 deg, so headings stay axis-aligned/clean)."""

    STEP = 1.0
    DTHETA = math.pi / 2

    def __init__(self):
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

    def pose(self):
        return (round(self.x, 6), round(self.y, 6), round(self.theta % (2 * math.pi), 6))

    def pos(self):
        return (round(self.x, 6), round(self.y, 6))

    def step(self, action):
        if action == "FORWARD":
            self.x += self.STEP * math.cos(self.theta)
            self.y += self.STEP * math.sin(self.theta)
        elif action == "TURN_L":
            self.theta += self.DTHETA
        elif action == "TURN_R":
            self.theta -= self.DTHETA
        return self.pose()


def _drive(actions):
    w = OrientationWorld()
    for a in actions:
        w.step(a)
    return w


def test_env_is_non_abelian_forward_and_turn_do_not_commute():
    """The env is genuinely NON-ABELIAN: FORWARD then TURN lands in a different place than TURN then FORWARD (because
    FORWARD's direction depends on the heading TURN changes). This is the order-dependence the abelian grid cannot hold."""
    fwd_then_turn = _drive(["FORWARD", "TURN_L"]).pos()
    turn_then_fwd = _drive(["TURN_L", "FORWARD"]).pos()
    assert fwd_then_turn != turn_then_fwd                                    # order matters -> SE(2) is non-abelian
    assert fwd_then_turn == (1.0, 0.0) and turn_then_fwd == (0.0, 1.0)       # concretely: +x vs +y


def test_abelian_move_delta_CANNOT_represent_forward():
    """The concrete INCONSISTENCY the refactor fixes: over POSITION-only, the SAME action FORWARD has FOUR different
    displacements (one per heading), so `move_delta` (which learns ONE Δ per action) is ill-defined -- the position-only
    dynamics are NON-DETERMINISTIC. The state must be the full POSE for FORWARD to be a well-defined operator."""
    deltas = set()
    for turns in range(4):                                                  # start at each of the 4 headings
        w = OrientationWorld()
        for _ in range(turns):
            w.step("TURN_L")
        x0, y0 = w.pos()
        w.step("FORWARD")
        x1, y1 = w.pos()
        deltas.add((round(x1 - x0, 6), round(y1 - y0, 6)))
    assert len(deltas) == 4                                                  # 4 distinct FORWARD displacements -> no single move_delta
    # ... whereas over the full POSE, FORWARD IS deterministic (pose -> pose'): a well-defined operator, once L6 tracks pose
    seen = {}
    for turns in range(4):
        w = OrientationWorld()
        for _ in range(turns):
            w.step("TURN_L")
        before = w.pose()
        after = w.step("FORWARD")
        seen[before] = after
    assert len(seen) == 4 and all(a != b for b, a in seen.items())          # deterministic pose->pose (distinct per heading)
