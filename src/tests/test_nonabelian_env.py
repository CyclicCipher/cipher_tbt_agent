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


def test_S1e_operator_pose_path_integration_makes_forward_deterministic():
    """L6_NONABELIAN S1e (the ENGINE): the column path-integrates a POSE by COMPOSING learned body-frame operators
    (`track_pose`: P ← P·G), so the belief tracks HEADING, matches OrientationWorld, and makes FORWARD DETERMINISTIC over
    the pose state -- which the additive POSITION cannot. The operator DRIVING a non-abelian state. (Live wiring into the
    agent awaits HEADING PERCEPTION -- the agent perceives position, not orientation.)"""
    import numpy as np

    from tbt.column import CorticalColumn
    from tbt.operator import Operator

    def se2(x, y, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    # LEARN the body-frame operator per action: G = pose_before^-1 · pose_after -- CONSTANT per action (so learnable).
    # (Use the RAW pose attributes, not pose() which rounds to 6 decimals.)
    Gs = {}
    for a in ("FORWARD", "TURN_L", "TURN_R"):
        incs = []
        for turns in range(4):
            w = OrientationWorld()
            for _ in range(turns):
                w.step("TURN_L")
            before = se2(w.x, w.y, w.theta)
            w.step(a)
            incs.append(np.linalg.inv(before) @ se2(w.x, w.y, w.theta))
        assert all(np.allclose(incs[0], g, atol=1e-9) for g in incs)         # the body-frame op is CONSTANT -> learnable
        Gs[a] = Operator(incs[0])

    # path-integrate a sequence through the column -> matches OrientationWorld's pose
    col = CorticalColumn(n_entities=8, seed=0)
    col.track_pose_reset()
    world = OrientationWorld()
    pose = (0.0, 0.0, 0.0)
    for a in ["FORWARD", "TURN_L", "FORWARD", "FORWARD", "TURN_R", "FORWARD", "TURN_L", "FORWARD"]:
        pose = col.track_pose(Gs[a])
        world.step(a)
    assert np.allclose(pose[:2], (world.x, world.y), atol=1e-6)              # position matches the env
    assert abs(((pose[2] - world.theta + np.pi) % (2 * np.pi)) - np.pi) < 1e-6   # heading matches (dead-reckoned from turns)

    # FORWARD is DETERMINISTIC over the POSE state (heading is in the key) -- 4 headings -> 4 distinct outcomes
    fwd = {}
    for turns in range(4):
        col.track_pose_reset()
        for _ in range(turns):
            col.track_pose(Gs["TURN_L"])
        before = col.pose_state()
        col.track_pose(Gs["FORWARD"])
        fwd[before] = col.pose_state()
    assert len(fwd) == 4 and len(set(fwd.values())) == 4                     # distinct, deterministic (vs 4-valued over position)

    # non-abelian in the BELIEF: FORWARD∘TURN ≠ TURN∘FORWARD
    col.track_pose_reset(); col.track_pose(Gs["FORWARD"]); col.track_pose(Gs["TURN_L"])
    p1 = col.pose_state()
    col.track_pose_reset(); col.track_pose(Gs["TURN_L"]); col.track_pose(Gs["FORWARD"])
    p2 = col.pose_state()
    assert p1 != p2
