"""Perception bridge (tbt.perceive): segment a frame into objects, and identify the controllable SELF as the object
that moves under an action -- reporting true positions for the playing loop. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.perceive import ScenePerceiver, background, segment  # noqa: E402


def _render(objs, n=20):
    """objs: list of (colour, [cells]). Background 0."""
    g = [[0] * n for _ in range(n)]
    for colour, cells in objs:
        for (x, y) in cells:
            g[y][x] = colour
    return g


def test_segment_finds_objects_with_colour_and_centroid():
    g = _render([(7, [(5, 5), (5, 6)]), (3, [(10, 10)])])
    assert background(g) == 0
    objs = {col: (cells, cen) for col, cells, cen in segment(g)}
    assert set(objs) == {7, 3}
    assert len(objs[7][0]) == 2 and objs[7][1] == (5.0, 5.5)
    assert objs[3][1] == (10.0, 10.0)


def test_identifies_the_moving_object_as_the_self():
    sp = ScenePerceiver()
    prev = _render([(7, [(5, 5)]), (3, [(10, 10)])])
    cur = _render([(7, [(6, 5)]), (3, [(10, 10)])])          # the 7 moved right; the 3 is a static landmark
    self_pose, others = sp.perceive(prev, 3, cur)
    assert self_pose == (6.0, 5.0)                            # the mover is the self, at its TRUE current position
    assert others == [(3, (10.0, 10.0))]                     # the static object is an "other"
    assert sp.self_colour == 7


def test_self_identity_persists_when_momentarily_still():
    sp = ScenePerceiver()
    sp.perceive(_render([(7, [(5, 5)]), (3, [(10, 10)])]), 3, _render([(7, [(6, 5)]), (3, [(10, 10)])]))
    # next frame: the self did not move -- it must still be located by its learned identity, not lost
    self_pose, others = sp.perceive(_render([(7, [(6, 5)]), (3, [(10, 10)])]), 0,
                                    _render([(7, [(6, 5)]), (3, [(10, 10)])]))
    assert self_pose == (6.0, 5.0) and others == [(3, (10.0, 10.0))]


def test_self_unlocated_before_any_motion_is_seen():
    sp = ScenePerceiver()
    self_pose, others = sp.perceive(None, None, _render([(7, [(5, 5)]), (3, [(10, 10)])]))
    assert self_pose is None                                  # no motion yet -> caller babbles to make something move
    assert (7, (5.0, 5.0)) in others and (3, (10.0, 10.0)) in others
