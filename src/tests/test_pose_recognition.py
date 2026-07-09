"""Stage 5 (MICROCIRCUIT.md): SDR-native POSE-INVARIANT recognition = the L6-ANCHORING search (Lewis 2022 virtual
rotation) over L4 + the `L23Pooler`. The column learns each object at its canonical orientation, then recognises it at any
90° rotation (and translation) by searching for the orientation that makes L4's cells pool onto a known identity — the
SDR-native replacement for M4's displacement-residual pose solve. Isolated: calls the recogniser methods directly, NOT via
`perceive` (the live swap + M4 retirement is Stage 5b)."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.column import CorticalColumn  # noqa: E402
from tbt.hippocampus import _rot  # noqa: E402

_SHAPES = {                                       # four tetrominoes whose rotation-orbits are disjoint (no one is a rotation of another;
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],        # NB L and J are 180° rotations of each other — not rotation-distinct — so use I, not J)
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
}
_C = 7
_ANGLES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


def _present(cells, theta, offset):
    """A world presentation of `cells` rotated by `theta` (a 90° multiple → integer) and translated by `offset`."""
    R = _rot(theta)
    out = []
    for (x, y) in cells:
        p = R @ np.array([float(x), float(y)])
        out.append((int(round(float(p[0]))) + offset[0], int(round(float(p[1]))) + offset[1], _C))
    return out


def _library():
    col = CorticalColumn(n_entities=16, seed=0)
    for name, cells in _SHAPES.items():
        col.learn_pose_object([(x, y, _C) for (x, y) in cells], name)
    return col


def test_recognises_each_object_at_its_canonical_pose():
    """Sanity: at the learned (canonical) orientation each object is recognised — the pooler over L4 works through the
    column, translation-invariant."""
    col = _library()
    for name, cells in _SHAPES.items():
        rec = col.recognize_pose([(x + 9, y + 4, _C) for (x, y) in cells])   # translated, canonical orientation
        assert rec is not None and rec[0] == name, (name, rec)


def test_recognises_each_object_at_any_90deg_rotation():
    """POSE-INVARIANCE (the point): every object is recognised at every 90° rotation (and translation) — the anchoring
    search finds the orientation that un-rotates the presentation onto the learned canonical frame, where the pooler
    recognises it. This is what M4's virtual rotation did, now on the L4/L2/3 SDR substrate."""
    col = _library()
    for name, cells in _SHAPES.items():
        for k, theta in enumerate(_ANGLES):
            cloud = _present(cells, theta, offset=(5 * k + 2, 3))
            rec = col.recognize_pose(cloud)
            assert rec is not None and rec[0] == name, (name, k, rec)


def test_recognised_orientation_matches_the_presented_rotation():
    """The anchoring search recovers the ORIENTATION, not just the identity: for an ASYMMETRIC object the reported θ is
    the presented rotation (mod its symmetry). L is asymmetric (its four rotations are distinct), so each presented angle
    is recovered exactly."""
    col = _library()
    cells = _SHAPES["L"]
    for theta in _ANGLES:
        name, th = col.recognize_pose(_present(cells, theta, offset=(3, 3)))
        assert name == "L"
        assert abs(((th - theta + np.pi) % (2 * np.pi)) - np.pi) < 1e-6, (theta, th)
