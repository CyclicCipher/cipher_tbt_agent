"""The column's L4 feature-at-location CONTENT MAP (MICROCIRCUIT.md Stage 4), wired live in `perceive`. As the column
tracks a moving object it binds each cell's COLOUR at its OBJECT-FRAME location (world cell mapped through the recognised
pose); it can then IMAGINE the colour at a location without sensing it (object permanence), and BURSTS when a colour
appears where a different one was learned (content surprise). Additive: the perceive/forward pose contracts are unchanged
(covered by test_column_online) — this only exercises the new content path."""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.column import CorticalColumn  # noqa: E402

_OBJ = {(0, 0): 7, (1, 0): 3, (2, 0): 4, (2, 1): 5}     # an asymmetric tetromino, a DISTINCT colour per cell (unique pose)


def _track(col, colours=_OBJ, steps=6):
    """Drive the column across a rightward TRANSLATION of the object so L4 learns its content map; return the last offset."""
    def cloud(ox):
        return [(x + ox, y, colours[(x, y)]) for (x, y) in colours]
    col.perceive(None, cloud(0))
    for k in range(1, steps):
        col.perceive(0, cloud(2 * k))
    return 2 * (steps - 1)


def test_column_l4_content_map_imagines_the_colour_at_a_location():
    """IMAGINATION / object permanence: after tracking the object, the column predicts the colour at each of its current
    world cells from the LEARNED map (a pure query through the recognised pose) — content recalled, not re-sensed."""
    col = CorticalColumn(n_entities=16, seed=0)
    ox = _track(col)
    got = {(x, y): col.predict_content_at_world(x + ox, y) for (x, y) in _OBJ}
    hits = sum(got[(x, y)] == _OBJ[(x, y)] for (x, y) in _OBJ)
    assert hits >= 3, got                                # the map recalls the object's colours at their object-frame cells


def test_column_l4_content_burst_signals_surprise():
    """BURST = content surprise (§3.4): re-sensing the tracked object with its LEARNED colours barely bursts, but a cell
    whose colour CHANGED (unpredicted at that location) drives the content burst up — a dense, self-supervised error."""
    col = CorticalColumn(n_entities=16, seed=0)
    ox = _track(col)

    def cloud(off, recolour=None):
        return [(x + off, y, (recolour if (x, y) == (0, 0) and recolour is not None else _OBJ[(x, y)]))
                for (x, y) in _OBJ]

    col.perceive(0, cloud(ox + 2))                       # same colours, one step on → mostly predicted
    settled = col._content_burst
    col.perceive(0, cloud(ox + 4, recolour=9))           # (0,0) is now colour 9 where 7 was learned → surprise
    assert col._content_burst > settled                  # the changed cell raised the content burst
