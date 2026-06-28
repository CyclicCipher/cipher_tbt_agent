"""Perception bridge (tbt.perceive): segment a frame into objects and track them across frames by pose-continuity --
no self, no colour. A multi-cell object stays ONE coherent object (the cn04 property: clean tracking, not fragmented).
Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.perceive import ObjectField, segment  # noqa: E402


def _render(by_value, n=24):
    g = [[0] * n for _ in range(n)]
    for v, cells in by_value.items():
        for (x, y) in cells:
            g[y][x] = v
    return g


def _id_of_size(result, size):
    return next(oid for oid, (_p, s) in result.items() if s == size)


def test_segment_finds_objects_without_colour():
    objs = segment(_render({7: [(5, 5)], 3: [(10, 10), (11, 10), (10, 11), (11, 11)]}))
    assert len(objs) == 2
    for cells, _centroid in objs:                              # each is (cells, centroid) -- no colour element
        assert isinstance(cells, set)


def test_tracks_objects_across_frames_by_id():
    field = ObjectField()
    static = [(15, 15), (16, 15), (15, 16), (16, 16)]         # a 4-cell static object
    r0 = field.perceive(_render({7: [(5, 5)], 3: static}))    # + a 1-cell mover
    mover, fixed = _id_of_size(r0, 1), _id_of_size(r0, 4)
    r1 = field.perceive(_render({7: [(6, 5)], 3: static}))    # the mover stepped right; the blob did not move
    assert set(r1) == set(r0)                                 # same objects, same ids (permanence)
    assert r1[mover][0] == (6.0, 5.0)                         # the id followed the moving object
    assert r1[fixed][0] == r0[fixed][0]                       # the static object kept its pose and id


def test_a_multicell_object_stays_one_coherent_object():
    """A 3x3 object is ONE tracked object across a move -- not fragmented into pieces (the cn04 coherence property)."""
    field = ObjectField()
    def blob(ox, oy):
        return [(ox + dx, oy + dy) for dx in range(3) for dy in range(3)]
    r0 = field.perceive(_render({7: blob(5, 5)}))
    assert len(r0) == 1 and next(iter(r0.values()))[1] == 9
    oid = next(iter(r0))
    r1 = field.perceive(_render({7: blob(7, 5)}))             # the whole object shifted +2 in x
    assert set(r1) == {oid}                                   # still one object, same id
    assert round(r1[oid][0][0] - r0[oid][0][0]) == 2          # its centroid moved by exactly the shift (clean operator)


def test_self_does_not_fuse_with_a_wall_on_contact():
    """Object permanence through CONTACT: when a moving object touches a static one (one connected blob), the tracker
    keeps them as TWO objects at their own poses -- it does NOT fuse them into a new object and lose the mover. This is
    the bug that blocked obstacle learning: without permanence, touching a wall erased the self, so its blocked move was
    never recorded. (The self has not fused to the wall just because it is stuck to it.)"""
    field = ObjectField()
    wall = [(7, y) for y in range(3, 8)]                       # a 5-cell static wall at x=7
    r0 = field.perceive(_render({7: [(5, 5)], 6: wall}))       # the mover, apart from the wall
    assert len(r0) == 2
    mover, wall_id = _id_of_size(r0, 1), _id_of_size(r0, 5)
    moved = lambda oid, p: (p[0] + 1, p[1]) if oid == mover else p   # dynamics: the mover stepped +x; the wall is static
    r1 = field.perceive(_render({7: [(6, 5)], 6: wall}), moved)      # the mover steps right and now TOUCHES the wall (one blob)
    assert set(r1) == {mover, wall_id}                         # still two objects, same ids -- NOT fused
    assert r1[mover][0] == (6.0, 5.0)                          # the mover kept its own pose
    assert r1[mover][1] == 1 and r1[wall_id][1] == 5           # and its own size -- the wall did not absorb it


def test_no_self_or_colour_anywhere_in_the_api():
    field = ObjectField()
    assert not hasattr(field, "self_colour")
    assert not any("self" in name for name in vars(field))
