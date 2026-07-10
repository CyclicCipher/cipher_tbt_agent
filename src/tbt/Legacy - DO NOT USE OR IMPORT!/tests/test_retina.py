"""The sensorimotor retina (tbt.retina): the OVERLAP-BEARING content SDR (`view_sdr`) + the exogenous-attention salience
channel (`salient_cells`). Transduction only — no segmentation, no exact-match patch codebook (the `Retina` RF-sweep class
retired with the classical cleanup); the column owns grouping + recognition. Pure stdlib; no live API."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.retina import salient_cells, view_sdr  # noqa: E402


def test_view_sdr_is_pose_invariant_colour_aware_and_overlap_bearing():
    """The retina's CONTENT ENCODER (SDR_MIGRATION.md M3): the whole-view descriptor is an OVERLAP-BEARING SDR —
    rotation+translation-invariant (same shape+colours at any pose -> the SAME SDR), colour- + shape-aware (a different
    colouring / shape differs), and — the win the exact-match `view_signature` key lacked — SIMILAR views OVERLAP while
    dissimilar ones do not. The peripheral's modality-specific extraction; the column stays content-opaque."""
    import numpy as np
    from tbt.l5_displacement import apply_pose

    shape = [(0, 0), (1, 0), (2, 0), (2, 1)]                        # an L-shaped tetromino stub
    sig = view_sdr([(x, y, 7) for (x, y) in shape])                # colour 7
    for theta in (np.pi / 2, np.pi, 2.0):                          # rotate (incl. a non-axis angle) + translate far away
        moved = [(px, py, 7) for (px, py) in apply_pose(shape, theta, (5.0, -4.0))]
        assert view_sdr(moved) == sig                              # POSE-INVARIANT (identical) + deterministic
    other_colour = view_sdr([(x, y, 3) for (x, y) in shape])       # same shape, different colour
    block = view_sdr([(0, 0, 7), (1, 0, 7), (0, 1, 7), (1, 1, 7)])  # a 2x2 block: same colour, different shape
    assert other_colour != sig and block != sig                    # colour- AND shape-aware (distinct codes)
    assert sig.overlap(other_colour) > 0                           # ...yet a different colouring still SHARES the geometry bits
    assert sig.overlap(block) > 0                                  # ...and a different shape still shares the colour bits
    # OVERLAP = SIMILARITY: a near-identical shape (one cell added) shares more bits than a wholly different one
    similar = view_sdr([(x, y, 7) for (x, y) in shape + [(3, 1)]])  # the L with one extra cell
    line = view_sdr([(0, 0, 7), (5, 0, 7), (10, 0, 7), (15, 0, 7)])  # a spread-out line (same colour, different geometry)
    assert sig.overlap(similar) > sig.overlap(line)                # the graded similarity the exact-match key could not give


def _frame(n=8, fill=0):
    return [[fill] * n for _ in range(n)]


def test_salience_reports_the_changed_cells():
    """Exogenous attention picks out the cells that CHANGED between two frames (the bottom-up 'what moved' channel)."""
    prev = _frame(10)
    cur = _frame(10)
    for (x, y) in [(3, 3), (4, 3), (3, 4), (4, 4)]:          # a 2x2 object appears
        cur[y][x] = 7
    assert salient_cells(prev, cur) == {(3, 3), (4, 3), (3, 4), (4, 4)}


def test_no_change_no_salience():
    g = _frame(6, fill=2)
    assert salient_cells(g, g) == set()
