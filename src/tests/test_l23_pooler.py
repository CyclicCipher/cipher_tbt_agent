"""Layer 2/3 column pooler (MICROCIRCUIT.md §7b, Stage 3) — the Numenta ColumnPooler: objects as ASSIGNED stable SDRs,
proximal from L4 feature-at-location, UNION-then-NARROW recognition. Driven by the REAL `L4Layer` (over `GridEncoder` L6
locations): L4 binds each feature-at-location to a distinct CONJUNCTIVE cell set, which is exactly what the pooler must
union — feeding the pooler the raw FACTORED grid bits instead exposes the binding problem L4 exists to solve (an object
that has cells in column 1 AND row 0 would falsely "cover" location (1,0)). The single implicit feature = one colour (the
tetromino case: shapes discriminated by WHICH locations they occupy). Subsumes M4's IDENTITY/recognition; `test_l23_object`
stays the live POSE recogniser until Stage 5, and pose-via-L6-anchor is Stage 4."""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import GridEncoder  # noqa: E402
from tbt.l4 import L4Layer  # noqa: E402
from tbt.l23_pooler import L23Pooler  # noqa: E402

_SHAPES = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],   # a bar
    "L": [(0, 0), (0, 1), (0, 2), (1, 2)],   # an L
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],   # a square
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],   # a T
}
_FEATURE = set(range(8))                     # one colour — 8 feature minicolumns (all shapes share it; shape is the signal)


def _library():
    """The real front-end: `GridEncoder` (L6) → `L4Layer` (binds each feature-at-location to a distinct conjunctive cell
    set) → an `L23Pooler` that has LEARNED the shape library (one `learn` per feature-at-location, over L4's cells)."""
    g = GridEncoder(scales=(5, 7, 11), dims=2, mw=1, bounds=[(0, 15), (0, 15)])
    l4 = L4Layer(cells_per_column=8, activation_threshold=5, min_threshold=4, init_perm=0.55)

    def l4_cells(x, y):
        l4.observe(_FEATURE, g.encode((x, y)).active)    # L4 binds the feature at this location → its conjunctive cells
        return set(l4._active)

    all_locs = sorted({c for cells in _SHAPES.values() for c in cells})
    for _ in range(2):                                   # bind L4's map so each location fires a stable, distinct cell set
        for (x, y) in all_locs:
            l4_cells(x, y)

    pool = L23Pooler(n_cells=1024, sdr_size=24, activation_threshold=4, seed=0)
    for name, cells in _SHAPES.items():
        for (x, y) in cells:
            pool.learn(name, l4_cells(x, y))
    return pool, l4_cells


def test_column_pooler_recognises_each_object_in_any_order():
    """RECOGNITION + LOCATION/ORDER-INVARIANCE: sensing an object's feature-at-locations (in ANY order) settles on its
    identity — the pooled object SDR is the same regardless of WHICH locations drive it or their order (the point of the
    output layer: a stable, location-invariant object code)."""
    pool, l4_cells = _library()
    for name, cells in _SHAPES.items():
        for order in (cells, list(reversed(cells))):
            pool.reset()
            for (x, y) in order:
                pool.sense(l4_cells(x, y))
            assert pool.best() == name, (name, order)
            assert pool.confident(), (name, order)


def test_column_pooler_unions_then_narrows():
    """UNION-then-NARROW: I and T share the bar (0,0)-(1,0)-(2,0) and differ only at I:(3,0) vs T:(1,1). Sensing the shared
    bar keeps BOTH fully consistent (a union — a tie) while dropping the objects that lack part of it; the distinguishing
    cell then NARROWS to the one object. Recognition is EVIDENCE over the sequence, not a single-glance guess."""
    pool, l4_cells = _library()
    pool.reset()
    for p in [(0, 0), (1, 0), (2, 0)]:
        pool.sense(l4_cells(*p))
    s = pool.scores()
    assert s["I"] == s["T"] == pool.sdr_size             # both still FULLY consistent with the bar (the union)
    assert s["I"] > s["O"] and s["I"] > s["L"]           # O (no (2,0)) and L (no (1,0)) already narrowed out
    assert not pool.confident()                          # genuinely ambiguous between I and T
    pool.sense(l4_cells(3, 0))                           # only I has (3,0)
    assert pool.best() == "I" and pool.confident()
    pool.reset()                                         # the T branch from the same ambiguous prefix
    for p in [(0, 0), (1, 0), (2, 0), (1, 1)]:
        pool.sense(l4_cells(*p))
    assert pool.best() == "T" and pool.confident()


def test_column_pooler_assigns_separated_identities():
    """SEPARATION: distinct objects get well-separated assigned SDRs (no false merge) — the location-invariant identities
    are distinct codes, so overlap-readout recognition is unambiguous."""
    pool, _ = _library()
    names = list(_SHAPES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert len(pool.objects[names[i]] & pool.objects[names[j]]) <= 3, (names[i], names[j])
