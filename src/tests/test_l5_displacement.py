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
    assert msg[(1,)] == (1, 0)                   # the mover (size-1 feature) moves +1 in x
    assert (4,) not in msg                       # the size-4 anchor has a zero displacement -> not in the message


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
