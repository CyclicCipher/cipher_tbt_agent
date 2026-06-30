"""Interaction-learned object behaviour: a wall is learned to BLOCK by bumping it once, that learning GENERALISES to
another instance (and a rotated instance) via L2/3 recognition WITHOUT re-bumping, a walk-through object stays
passable, and a 'barrier' later found passable is REVISED. Nothing is assumed (an un-probed object is not a barrier).
Pure stdlib + L2/3's object recognition -- the corrected wall-learning the engine-column makes possible."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.behavior import ObjectBehaviour, contact_outcome  # noqa: E402
from tbt.l23_object import L23_Object                       # noqa: E402  (object recognition lives in L2/3 now)


def _ident(rec, cells):
    return rec.recognize([(float(x), float(y)) for (x, y) in cells])[0]


def test_contact_outcome_attributes_the_bump():
    """The move attribution the column owns: hit the object in the attempted direction, advanced iff it actually got
    there; empty space is no interaction."""
    wall = {(5, 5), (6, 5), (7, 5)}
    assert contact_outcome((4, 5), (4, 5), (1, 0), {"w": wall}) == ("w", False)   # blocked at (5,5)
    assert contact_outcome((4, 5), (5, 5), (1, 0), {"w": {(5, 5)}}) == ("w", True)  # passed onto it
    assert contact_outcome((4, 5), (5, 5), (1, 0), {"w": {(9, 9)}}) is None        # moved into empty space


def test_wall_learned_and_generalised_across_instances():
    """Bump ONE wall once -> it blocks; a DIFFERENT instance of the same recognised object is predicted blocked WITHOUT
    bumping (and an un-probed object is not assumed a barrier)."""
    rec = L23_Object()
    wall1 = {(10, 5), (11, 5), (12, 5)}                      # a 1x3 wall
    wall2 = {(30, 20), (31, 20), (32, 20)}                   # another 1x3 wall elsewhere -> the SAME object
    id1, id2 = _ident(rec, wall1), _ident(rec, wall2)
    assert id1 == id2                                        # recognised as one object across position

    beh = ObjectBehaviour()
    assert not beh.is_barrier(id1)                          # no assumption: un-probed -> not a barrier

    out = contact_outcome((9, 5), (9, 5), (1, 0), {id1: wall1})   # bump wall1 (blocked)
    beh.observe_move(*out)
    assert beh.is_barrier(id1)
    assert beh.is_barrier(id2)                              # GENERALISED to the other instance, never bumped


def test_wall_learning_generalises_across_rotation():
    """The Recognizer earns its place: a 1x3 wall learned to block, encountered as a 3x1 wall (the same object rotated
    90 deg), is still predicted blocked -- a translation-only shape key could not do this."""
    rec = L23_Object()
    horiz = {(10, 5), (11, 5), (12, 5)}                      # 1x3
    vert = {(20, 8), (20, 9), (20, 10)}                     # 3x1 = the same wall rotated 90 deg
    idh, idv = _ident(rec, horiz), _ident(rec, vert)
    assert idh == idv, f"rotated wall not recognised as the same object ({idh} vs {idv})"

    beh = ObjectBehaviour()
    beh.observe_move(idh, advanced=False)                   # learned from the horizontal instance
    assert beh.is_barrier(idv)                              # predicted for the rotated instance


def test_walk_through_object_stays_passable_and_barrier_is_revised():
    """A different object passed through stays passable; and a 'barrier' that later passes is REVISED below threshold
    (the painting / a door that opened)."""
    rec = L23_Object()
    wall = {(10, 5), (11, 5), (12, 5)}
    paint = {(5, 5), (6, 5), (5, 6), (6, 6)}                 # a 2x2 -> a different object
    idw, idp = _ident(rec, wall), _ident(rec, paint)
    assert idw != idp

    beh = ObjectBehaviour()
    beh.observe_move(idp, advanced=True)                    # walked through the painting
    assert not beh.is_barrier(idp)

    beh.observe_move(idw, advanced=False)                   # first the wall blocks
    assert beh.is_barrier(idw)
    for _ in range(3):                                      # then it starts letting the agent through (revision)
        beh.observe_move(idw, advanced=True)
    assert not beh.is_barrier(idw)
