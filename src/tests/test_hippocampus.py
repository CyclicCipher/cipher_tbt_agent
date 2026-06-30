"""The hippocampus (tbt.hippocampus): the allocentric cognitive map (H1). Binds proto-objects to world SLOTS,
producing a FACTORED (slot, colour, size) scene state. The point of H1: this object-level state RECURS where the
per-pixel content signature CHURNS (the 7b failure) -- the same objects reshaping in place is ONE state, and a
real transformation (colour toggle / growth) changes it. Plus the allocentric WHERE and translation-invariance."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.hippocampus import Hippocampus  # noqa: E402
from tbt.perceive import content_sig  # noqa: E402

BG = 0


def board(cells, H=12, W=12):
    """A frame from `{(x, y): colour}` on background 0."""
    g = [[BG] * W for _ in range(H)]
    for (x, y), c in cells.items():
        g[y][x] = c
    return g


def test_scene_binds_objects_to_slots():
    """The scene is a sorted set of (slot, colour, size) -- place + content. Two objects, distinct colours/places."""
    h = Hippocampus(pos_bin=4)
    state, objs = h.scene(board({(1, 1): 5, (2, 1): 5, (9, 9): 7}))   # a size-2 colour-5 + a size-1 colour-7
    assert len(state) == 2
    feats = {(c, z) for (_slot, c, z) in state}
    assert feats == {(5, 2), (7, 1)}                                  # colour + size captured
    assert len(objs) == 2


def test_scene_recurs_where_pixels_churn():
    """THE H1 WIN: an object that RESHAPES in place (same colour, same size, same corner) is the SAME scene state,
    while the per-pixel content signature CHURNS. So the loop's states recur where 7b's per-pixel state never did."""
    h = Hippocampus(pos_bin=4)
    # a size-3 colour-5 blob, two different shapes sharing the same bounding-box corner (3,3)
    shape_a = board({(3, 3): 5, (4, 3): 5, (3, 4): 5})
    shape_b = board({(3, 3): 5, (4, 3): 5, (4, 4): 5})
    sa, _ = h.scene(shape_a)
    sb, _ = h.scene(shape_b)
    assert sa == sb                                                   # SAME object-level state (slot, colour, size)
    cells_a = {(3, 3), (4, 3), (3, 4)}
    cells_b = {(3, 3), (4, 3), (4, 4)}
    assert content_sig(shape_a, cells_a) != content_sig(shape_b, cells_b)   # the per-pixel signature DID churn


def test_scene_captures_a_transformation():
    """A real transformation changes the state: a colour TOGGLE and a GROWTH both move the (colour, size) feature."""
    h = Hippocampus(pos_bin=4)
    base, _ = h.scene(board({(3, 3): 5, (4, 3): 5}))                  # colour-5, size-2
    toggled, _ = h.scene(board({(3, 3): 6, (4, 3): 6}))              # same place, colour 5->6
    grown, _ = h.scene(board({(3, 3): 5, (4, 3): 5, (5, 3): 5}))     # colour-5, size 2->3
    assert base != toggled
    assert base != grown
    assert toggled != grown


def test_slot_is_growth_stable():
    """The slot is anchored to the bounding-box CORNER, not the centroid -- a structure growing on one side keeps its
    allocentric place (only its size changes), so the (slot) half of the state stays put under reshaping."""
    h = Hippocampus(pos_bin=4)
    _s0, o0 = h.scene(board({(2, 2): 5, (3, 2): 5}))
    _s1, o1 = h.scene(board({(2, 2): 5, (3, 2): 5, (4, 2): 5, (5, 2): 5}))   # grew rightward from the same corner
    assert o0[0][0] == o1[0][0]                                       # same slot
    assert o0[0][2] != o1[0][2]                                       # different size (the growth is in WHAT)


def test_translation_invariant_mode():
    """With translation_invariant, a rigid shift of the whole scene is ONE state (position-free); absolute mode
    distinguishes them (the board is fixed -> position is real)."""
    ti = Hippocampus(pos_bin=1, translation_invariant=True)
    ab = Hippocampus(pos_bin=1)
    scene1 = board({(1, 1): 5, (2, 1): 5, (5, 5): 7})
    scene2 = board({(3, 3): 5, (4, 3): 5, (7, 7): 7})                 # the whole arrangement shifted by (+2, +2)
    assert ti.scene(scene1)[0] == ti.scene(scene2)[0]                # translation-invariant: same
    assert ab.scene(scene1)[0] != ab.scene(scene2)[0]               # allocentric: different places
