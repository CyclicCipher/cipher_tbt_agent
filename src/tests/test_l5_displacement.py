"""L5's live surface after the symbolic-path retirement: the EFFERENCE COPY (the self's body-frame per-action delta),
the MOTOR command, and the continuous-pose GEOMETRY (the displacement-cell group action recognition reads through). The
config-state/tabular operator (edges/disp/recolor/driver) was deleted with `perceive.py` — only the SDR/bump/recognition
loop remains."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import numpy as np                                                       # noqa: E402
from tbt.l5_displacement import (  # noqa: E402
    L5_Displacement, apply_pose, local_disps, pose_between, rot)


def test_efference_copy_is_the_body_frame_delta_and_move_turn_kind_emerges():
    """L5's core: `learn_efference` stores the action's BODY-FRAME (dx, dy, dtheta) as an EWMA, `efference` reads it, and
    the move-vs-turn KIND emerges from the effect (a large translation vs a large heading increment) — never assigned."""
    op = L5_Displacement()
    assert op.efference(0) == ((0.0, 0.0), 0.0) and not op.learned(0)   # unlearned -> stay
    op.learn_efference(0, (2.0, 0.0), 0.0)
    op.learn_efference(0, (2.0, 0.0), 0.0)
    (dx, dy), dth = op.efference(0)
    assert abs(dx - 2.0) < 1e-6 and abs(dy) < 1e-6 and abs(dth) < 1e-6 and op.learned(0)   # a MOVE
    op.learn_efference(1, (0.0, 0.0), np.pi / 2)
    assert op.learned(1) and abs(op.efference(1)[1] - np.pi / 2) < 1e-6   # a TURN (heading-only)
    assert op.motor(3) == 3                                              # the motor command is the enacted action


# ---- the pose operators (the displacement-cell geometry seated in L5; recognition reads them) ------------
_L = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]                     # an L-tromino+ (asymmetric, one pose only)

_L = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]                     # an L-tromino+ (asymmetric, one pose only)


def test_local_disps_are_the_neighbour_vectors_within_radius():
    """local_disps = the patch's displacement vectors to cells within radius (the 'feature pose' a pose acts on)."""
    locs = [np.asarray(c, float) for c in _L]
    d = local_disps(locs, 0, radius=1.5)                                  # from (0,0): only (1,0) is within 1.5
    assert {tuple(v) for v in d} == {(1.0, 0.0)}


def test_pose_between_solves_the_group_element_and_apply_pose_reproduces_the_cloud():
    """The group-action contract recognition relies on: pose_between SOLVES the rotation aligning model->sensed off
    the local geometry (continuous, no angle search), and apply_pose with that pose reproduces the rotated cloud."""
    locs = [np.asarray(c, float) for c in _L]
    for theta in (0.3, 1.0, np.pi / 2, 2.7):
        sensed = apply_pose(_L, theta, (3.0, -2.0))                       # the object at an unseen continuous pose
        sd = local_disps([np.asarray(p, float) for p in sensed], 0, 1.5)
        solved = pose_between(local_disps(locs, 0, 1.5), sd)
        assert any(abs((s - theta + np.pi) % (2 * np.pi) - np.pi) < 1e-6 for s in solved), (theta, solved)


def test_apply_pose_is_exact_on_the_grid_at_ninety_degrees():
    """apply_pose IS the rotation -- continuous, and exact on the integer grid at 90 degrees (no lookup table)."""
    got = {tuple(np.round(p, 6)) for p in apply_pose([(0, 0), (1, 0), (2, 0), (3, 0)], np.pi / 2, (0.0, 0.0))}
    assert got == {(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)}, got


def test_pose_api_is_reachable_on_the_layer():
    """L5 the LAYER exposes the pose operators (the column coordinates recognition through them, not a side library)."""
    assert L5_Displacement.apply_pose is apply_pose
    assert L5_Displacement.pose_between is pose_between
    assert L5_Displacement.local_disps is local_disps
