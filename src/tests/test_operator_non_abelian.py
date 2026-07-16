"""End-to-end test of NON-ABELIAN path integration (ARCHITECTURE.md §8): an action's effect DEPENDS on the body's
orientation, so the group is non-commutative — in SE(2) = R²⋊SO(2) and, after the 2026-07-15 SO(3) cut-over, in SE(3) too.

THE MECHANISM. The operator learns each action's **body-frame** displacement + **body-frame rotation** from observation, and
maps them to the world through the CURRENT orientation. That makes non-commutativity structural (FORWARD;TURN ≠ TURN;FORWARD)
with **no keying, no ring, and no discretisation** — and one observation generalises to every position AND every orientation,
including orientations never observed.

WHY THE ORIENTATION IS A MATRIX, not an angle. `reference_tbt_pose_invariant_recognition` (Monty's pose is "three orthonormal
vectors") and `reference_operator_as_group_representation` (Gao 2021: "a learned group-representation matrix") both say matrix
outright — a scalar heading is an SO(2)-only encoding, and SO(3) is 3-DOF, so no scalar can name it. Degrees are now a 2-D
READ-OUT (`from_angle`/`to_angle`), exactly as the grid SDR is a read-out of the continuous pose.

THE 3-D CASE IS NOT A PORT — it exhibits something SE(2) CANNOT. In 2-D, rotations commute, so the non-abelian claim rests
entirely on translation-vs-rotation. In 3-D the ROTATIONS THEMSELVES stop commuting (yaw∘pitch ≠ pitch∘yaw), which the same
`R' = R·ΔR` delivers with no new code.

RULES #3 acceptance: the agent dead-reckons orientation-dependent motion, in 2-D and in 3-D.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                                             # noqa: E402
from tbt.operator import compose, eye, from_angle, rotate, to_angle     # noqa: E402

TOL = 1e-6
STEP = 1.0                       # FORWARD advances one unit along the body's facing axis
TURN = 90.0                      # TURN rotates 90° left (any angle would do — orientation is continuous)

# 3-D rotations about the z axis (YAW) and the y axis (PITCH) — chosen because they do NOT commute.
YAW = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))          # +90° about z
PITCH = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))        # +90° about y


def _true(pose, action):
    """The world's (unknown-to-the-agent) 2-D dynamics."""
    p, R = pose
    if action == "FORWARD":
        return tuple(a + b for a, b in zip(p, rotate(R, (STEP, 0.0)))), R
    return p, compose(R, from_angle(TURN))


def _true3(pose, action):
    """The world's 3-D dynamics: FORWARD along the body's x axis; YAW/PITCH turn in the BODY's own frame."""
    p, R = pose
    if action == "FORWARD":
        return tuple(a + b for a, b in zip(p, rotate(R, (STEP, 0.0, 0.0)))), R
    return p, compose(R, YAW if action == "YAW" else PITCH)


def _close_vec(a, b):
    return all(abs(x - y) < TOL for x, y in zip(a, b))


def _close_rot(A, B):
    return all(abs(x - y) < TOL for ra, rb in zip(A, B) for x, y in zip(ra, rb))


def _close(a, b):
    return _close_vec(a[0], b[0]) and _close_rot(a[1], b[1])


def _fresh(dims: int = 2) -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0, dims=dims)


def _teach(agent: Agent) -> None:
    """Observe both actions at a FEW poses. Because the operator stores the BODY-frame delta, this generalises to every
    position and every orientation — including orientations never observed."""
    for h in (0.0, 90.0):                       # deliberately only two orientations — the rest must generalise
        for x in (9.0, 11.0):
            for y in (9.0, 11.0):
                pose = ((x, y), from_angle(h))
                for a in ("FORWARD", "TURN"):
                    agent.learn_pose_move(a, pose, _true(pose, a))


def _teach3(agent: Agent) -> None:
    """The 3-D analogue — a handful of poses, all three actions."""
    for R in (eye(3), YAW, PITCH):
        for p in ((9.0, 9.0, 9.0), (11.0, 10.0, 12.0)):
            pose = (p, R)
            for a in ("FORWARD", "YAW", "PITCH"):
                agent.learn_pose_move(a, pose, _true3(pose, a))


def test_path_integration_is_non_commutative():
    agent = _fresh()
    _teach(agent)
    start = ((30.0, 30.0), from_angle(0.0))     # a NOVEL position, facing +x
    agent.set_pose(*start)
    agent.path_integrate("FORWARD")
    agent.path_integrate("TURN")
    fwd_then_turn = agent.pose()                # (30,30)→(31,30), then TURN→90°
    agent.set_pose(*start)
    agent.path_integrate("TURN")
    agent.path_integrate("FORWARD")
    turn_then_fwd = agent.pose()                # TURN→90°, then (30,30)→(30,31)
    assert _close(fwd_then_turn, ((31.0, 30.0), from_angle(90.0))), f"FORWARD;TURN gave {fwd_then_turn}"
    assert _close(turn_then_fwd, ((30.0, 31.0), from_angle(90.0))), f"TURN;FORWARD gave {turn_then_fwd}"
    assert not _close(fwd_then_turn, turn_then_fwd), "SE(2) must be NON-commutative (FORWARD;TURN ≠ TURN;FORWARD)"


def test_dead_reckons_a_full_se2_path():
    agent = _fresh()
    _teach(agent)
    pose = ((20.0, 20.0), from_angle(0.0))
    agent.set_pose(*pose)
    path = ["FORWARD", "FORWARD", "TURN", "FORWARD", "TURN", "TURN", "FORWARD"]
    ok = 0
    for a in path:
        agent.path_integrate(a)
        pose = _true(pose, a)
        ok += _close(agent.pose(), pose)
    assert ok == len(path), f"SE(2) dead-reckoning drifted ({ok}/{len(path)} steps)"


def test_generalises_to_an_UNOBSERVED_orientation():
    """The body-frame delta is orientation-invariant, so FORWARD works at an orientation never observed during learning — the
    payoff of a continuous state (the retired discrete ring could only key on headings it had seen)."""
    agent = _fresh()
    _teach(agent)                                 # taught only at 0° and 90°
    agent.set_pose((5.0, 5.0), from_angle(37.0))  # an orientation never observed, and not on any grid
    agent.path_integrate("FORWARD")
    dx, dy = rotate(from_angle(37.0), (STEP, 0.0))
    assert _close(agent.pose(), ((5.0 + dx, 5.0 + dy), from_angle(37.0))), f"got {agent.pose()}"


def test_se3_ROTATIONS_do_not_commute():
    """THE 3-D PROPERTY SE(2) CANNOT SHOW. In 2-D, rotations commute — so the non-commutativity test above really tests
    translation-vs-rotation. In 3-D, SO(3) is genuinely non-abelian: YAW;PITCH and PITCH;YAW leave the body facing DIFFERENT
    directions, from the same start, with no translation involved at all. The operator gets this from `R' = R·ΔR` — the same
    line that serves 2-D. A scalar heading could not even express these poses."""
    agent = _fresh(dims=3)
    _teach3(agent)
    start = ((30.0, 30.0, 30.0), eye(3))
    agent.set_pose(*start)
    agent.path_integrate("YAW")
    agent.path_integrate("PITCH")
    yaw_pitch = agent.pose()
    agent.set_pose(*start)
    agent.path_integrate("PITCH")
    agent.path_integrate("YAW")
    pitch_yaw = agent.pose()
    assert _close_rot(yaw_pitch[1], compose(YAW, PITCH)), f"YAW;PITCH gave {yaw_pitch[1]}"
    assert _close_rot(pitch_yaw[1], compose(PITCH, YAW)), f"PITCH;YAW gave {pitch_yaw[1]}"
    assert not _close_rot(yaw_pitch[1], pitch_yaw[1]), "SO(3) rotations must NOT commute — YAW;PITCH ≠ PITCH;YAW"


def test_dead_reckons_a_full_se3_path():
    """SE(3) dead-reckoning through orientations never observed: FORWARD's world effect depends on the whole 3-D attitude,
    which the operator learned from a handful of poses as ONE body-frame delta."""
    agent = _fresh(dims=3)
    _teach3(agent)
    pose = ((20.0, 20.0, 20.0), eye(3))
    agent.set_pose(*pose)
    path = ["FORWARD", "YAW", "FORWARD", "PITCH", "FORWARD", "YAW", "PITCH", "FORWARD", "FORWARD"]
    ok = 0
    for a in path:
        agent.path_integrate(a)
        pose = _true3(pose, a)
        ok += _close(agent.pose(), pose)
    assert ok == len(path), f"SE(3) dead-reckoning drifted ({ok}/{len(path)} steps)"


if __name__ == "__main__":
    ag = _fresh()
    _teach(ag)
    ag.set_pose((30.0, 30.0), from_angle(0.0))
    ag.path_integrate("FORWARD"); ag.path_integrate("TURN")
    a = ag.pose()
    ag.set_pose((30.0, 30.0), from_angle(0.0))
    ag.path_integrate("TURN"); ag.path_integrate("FORWARD")
    b = ag.pose()
    print(f"SE(2)  FORWARD;TURN → {a[0]} facing {to_angle(a[1]):.0f}°   TURN;FORWARD → {b[0]} facing {to_angle(b[1]):.0f}°")
    ag.set_pose((5.0, 5.0), from_angle(37.0)); ag.path_integrate("FORWARD")
    print(f"SE(2)  FORWARD at an UNOBSERVED 37° → {tuple(round(c, 3) for c in ag.pose()[0])}")
    ag3 = _fresh(dims=3)
    _teach3(ag3)
    ag3.set_pose((0.0, 0.0, 0.0), eye(3)); ag3.path_integrate("YAW"); ag3.path_integrate("PITCH")
    yp = ag3.pose()[1]
    ag3.set_pose((0.0, 0.0, 0.0), eye(3)); ag3.path_integrate("PITCH"); ag3.path_integrate("YAW")
    py = ag3.pose()[1]
    print(f"SE(3)  YAW;PITCH faces {tuple(round(x, 2) for x in rotate(yp, (1, 0, 0)))}   "
          f"PITCH;YAW faces {tuple(round(x, 2) for x in rotate(py, (1, 0, 0)))}   → non-commutative: {yp != py}")
