"""Perception bridge (tbt.perceive): read moving objects from the MOTION residual (so a mover embedded in / sharing
colour with the static structure is still isolated -- the ls20 case), track them by pose-continuity, and lump the rest
into one static anchor. No self, no colour. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.perceive import ObjectField  # noqa: E402


def _render(by_value, n=24):
    g = [[0] * n for _ in range(n)]
    for v, cells in by_value.items():
        for (x, y) in cells:
            g[y][x] = v
    return g


def _movers(result):
    return {oid: v for oid, v in result.items() if oid != ObjectField.STATIC}


def test_static_anchor_then_mover_from_the_residual():
    field = ObjectField()
    L = [(3, 3), (4, 3), (3, 4), (4, 4)]
    r0 = field.perceive(None, _render({7: [(10, 10)], 5: L}))
    assert set(r0) == {ObjectField.STATIC}                     # no prev -> no motion -> just the static anchor
    r1 = field.perceive(_render({7: [(10, 10)], 5: L}), _render({7: [(11, 10)], 5: L}))
    movers = _movers(r1)
    assert len(movers) == 1
    (pose, size), = movers.values()
    assert pose == (11.0, 10.0) and size == 1                  # the mover, isolated at its true new position


def test_tracks_a_mover_across_frames_by_id():
    field = ObjectField()
    L = [(3, 3), (4, 3), (3, 4), (4, 4)]
    field.perceive(None, _render({7: [(10, 10)], 5: L}))
    r1 = field.perceive(_render({7: [(10, 10)], 5: L}), _render({7: [(11, 10)], 5: L}))
    mid = next(iter(_movers(r1)))
    r2 = field.perceive(_render({7: [(11, 10)], 5: L}), _render({7: [(12, 10)], 5: L}))
    assert mid in r2 and r2[mid][0] == (12.0, 10.0)            # the same id followed the moving object


def test_isolates_a_mover_embedded_in_the_static_structure():
    """The ls20 case: the mover shares colour AND is adjacent to a big blob, so full-frame segmentation would merge
    them into one component -- but reading movers from the MOTION residual isolates the mover anyway."""
    field = ObjectField()
    blob = [(x, y) for x in range(10, 13) for y in range(10, 13)]   # a 3x3 blob, colour 5
    field.perceive(None, _render({5: blob + [(13, 11)]}))           # a 1-cell mover of the SAME colour, ADJACENT
    r1 = field.perceive(_render({5: blob + [(13, 11)]}), _render({5: blob + [(14, 11)]}))
    movers = _movers(r1)
    assert len(movers) == 1
    (pose, size), = movers.values()
    assert pose == (14.0, 11.0) and size == 1                  # isolated despite the colour/adjacency merge
    assert r1[ObjectField.STATIC][1] == 9                      # the static blob, with the mover excluded


def test_no_self_or_colour_anywhere_in_the_api():
    field = ObjectField()
    assert not hasattr(field, "self_colour")
    assert not any("self" in name for name in vars(field))
