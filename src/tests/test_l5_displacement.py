"""The L5 reseat (finish): the per-action OPERATOR generalizes via a POSITION-INVARIANT displacement to state-actions
it has NEVER visited (the discrete graph cannot), observed edges / blocked moves stay as EXCEPTIONS that override it,
and L5 emits the motor command + the thalamus driver -- the four uses of the one displacement object."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.column import CorticalColumn          # noqa: E402
from tbt.l5_displacement import L5_Displacement  # noqa: E402
from tbt.perceive import canonicalize          # noqa: E402
from tbt.sensor import config_state            # noqa: E402

RIGHT = 3                                       # the action that moves the mover +1 in x


def _scene(mover, landmark=(2, 2)):
    """A config-state: a 1-cell mover + a 2x2 landmark (size 4, the anchor)."""
    return config_state({0: (mover, 1), 1: (landmark, 4)})


def _teach_move_right(op):
    """Observe the mover stepping right twice (so the displacement (1,0) is learned for the mover, 0 for the anchor)."""
    op.observe(_scene((6, 6)), RIGHT, _scene((7, 6)))
    op.observe(_scene((7, 6)), RIGHT, _scene((8, 6)))


def test_displacement_generalizes_to_an_unvisited_state():
    """After two observations, L5 predicts the action's effect at a position it has NEVER seen -- the generalization
    the bare graph cannot do (and the static landmark/anchor correctly stays put)."""
    op = L5_Displacement()
    _teach_move_right(op)
    novel = _scene((3, 3))                       # never observed
    assert novel not in op.edges                 # truly unvisited -> only the displacement can answer
    assert op.predict(novel, RIGHT) == _scene((4, 3))   # mover stepped right; landmark unchanged


def test_observed_edge_and_blocked_move_override_the_displacement():
    """An observed transition is the exception: a BLOCKED move (s2 == s) is a self-edge that overrides the base
    displacement (the wall/door), while a DIFFERENT unvisited state still generalizes."""
    op = L5_Displacement()
    _teach_move_right(op)
    wall = _scene((9, 9))
    op.observe(wall, RIGHT, wall)                # blocked here (hit a wall) -> recorded as the exception
    assert op.predict(wall, RIGHT) == wall       # exception wins: it stays, it does NOT generalize a step right
    assert op.predict(_scene((3, 3)), RIGHT) == _scene((4, 3))   # elsewhere the displacement still generalizes


def test_motor_output_and_thalamus_driver():
    """The other two uses of the one displacement: the motor command (the enacted action) and the driver message
    (which feature moved by how much) a higher-order thalamus would relay to another column."""
    op = L5_Displacement()
    _teach_move_right(op)
    assert op.motor(RIGHT) == RIGHT
    msg = dict(op.driver(_scene((3, 3)), RIGHT))
    assert msg[(1,)] == (("move", (1, 0)),)      # the mover (size-1 shape) moves +1 in x
    assert (4,) not in msg                       # the size-4 anchor has a zero delta -> not in the message


def test_opaque_states_are_unchanged_no_generalization():
    """Over opaque symbols (not config-states) L5 is the plain graph: edges only, no displacement, no self-edges,
    no crash -- so the online column tests keep their exact behaviour."""
    op = L5_Displacement()
    op.observe(1, 0, 2)
    op.observe(2, 0, 2)                          # blocked over an opaque symbol -> NOT recorded (no self-edge)
    assert op.predict(1, 0) == 2                 # the observed edge
    assert op.predict(2, 0) == 2                 # no edge, no generalization -> stay
    assert op.predict(99, 0) == 99               # unknown -> stay
    assert 2 not in op.edges                      # the blocked opaque move left no self-edge


def test_canonicalize_is_translation_and_order_invariant():
    """The ONE shared encoding (sensor + L5): the same relative arrangement, shifted and reordered, is one state."""
    a = canonicalize([((6, 6), (1,)), ((2, 2), (4,))])
    b = canonicalize([((7, 7), (4,)), ((11, 11), (1,))])    # the whole scene +5, reversed order
    assert a == b


def test_column_predict_generalizes_through_L5():
    """End to end through the column: observe a few moves, then col.predict answers an unvisited (s, a) by the
    position-invariant displacement -- the operator the planner now rolls forward over."""
    col = CorticalColumn(n_entities=64, seed=0)
    col.observe(_scene((6, 6)), RIGHT, _scene((7, 6)))
    col.observe(_scene((7, 6)), RIGHT, _scene((8, 6)))
    assert col.predict(_scene((3, 3)), RIGHT) == _scene((4, 3))


# ---- the pose operators (the displacement-cell geometry now seated in L5; recognition reads them) ------------
import numpy as np                                                       # noqa: E402
from tbt.l5_displacement import apply_pose, local_disps, pose_between, rot  # noqa: E402

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
