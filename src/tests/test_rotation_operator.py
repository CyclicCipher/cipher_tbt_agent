"""R2 of the rotation plan (`notes/rotation_invariance_plan.md`): the ROTATION OPERATOR — a circular-buffer shift of the
orientation-module index — and its EQUIVARIANCE.

THE R2 PROPERTY:  `apply(encode(loc), k) == encode(R_ω · loc)`  for ω = k·(360/N).

Rotating the location code is a PERMUTATION (move each module's phase k steps around its scale's orientation ring, cell
unchanged) — the same TRANSFORM primitive as translation (`ModularOperator`), shifting ACROSS modules rather than WITHIN one.
Being a genuine group action it is exact, composes (k1 then k2 == k1+k2), and inverts (k then −k == identity). That is what
lets R3 test a rotation hypothesis by APPLYING it, instead of searching a free pose space.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import GridEncoder      # noqa: E402
from tbt.operator import RotationOperator  # noqa: E402

SCALES = (7, 11, 13, 17)
N = 8                                      # orientations over 360° → Δ = 45°
DELTA = 360.0 / N
LOCS = [(5.0, 3.0), (2.0, 7.0), (0.0, 0.0), (6.0, 6.0)]


def _grid() -> GridEncoder:
    return GridEncoder(scales=SCALES, dims=2, mw=1, bounds=[(0, 15), (0, 15)], orientations=N)


def _rotate(p, deg: float):
    r = math.radians(deg)
    x, y = p
    return (x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r))


def test_rotation_operator_is_equivariant():
    """THE R2 property: applying the operator to the code == encoding the rotated location."""
    g, op = _grid(), RotationOperator(_grid())
    for loc in LOCS:
        for k in range(N):
            assert op.apply(g.encode(loc), k) == g.encode(_rotate(loc, k * DELTA)), (
                f"equivariance broken at loc={loc}, k={k} (ω={k * DELTA}°)")


def test_identity_composition_and_inverse():
    """It is a GROUP ACTION, not an approximation."""
    g, op = _grid(), RotationOperator(_grid())
    code = g.encode((5.0, 3.0))
    assert op.apply(code, 0) == code, "k=0 must be the identity"
    for k1 in (1, 3, 6):
        for k2 in (2, 5):
            assert op.apply(op.apply(code, k1), k2) == op.apply(code, k1 + k2), f"composition failed ({k1} then {k2})"
        assert op.apply(op.apply(code, k1), -k1) == code, f"rotating by {k1} then −{k1} must return the original"


def test_a_full_turn_is_the_identity():
    g, op = _grid(), RotationOperator(_grid())
    code = g.encode((2.0, 7.0))
    assert op.apply(code, N) == code, "N steps = 360° = the identity"


def test_angle_reporting():
    op = RotationOperator(_grid())
    assert op.steps == N
    assert op.angle(0) == 0.0 and op.angle(1) == DELTA and op.angle(N) == 0.0


def test_rejects_an_axis_aligned_grid():
    """An axis-aligned (0°,90°) set is not closed under rotation — the shift would NOT equal the rotation, so refuse it."""
    with pytest.raises(AssertionError):
        RotationOperator(GridEncoder(scales=SCALES, dims=2, mw=1))


if __name__ == "__main__":
    g, op = _grid(), RotationOperator(_grid())
    loc = (5.0, 3.0)
    ok = all(op.apply(g.encode(loc), k) == g.encode(_rotate(loc, k * DELTA)) for k in range(N))
    print(f"equivariance over all {N} steps at loc={loc}: {ok}")
    print(f"steps={op.steps}, angle(3)={op.angle(3)}°, inverse(3 then -3) == original: "
          f"{op.apply(op.apply(g.encode(loc), 3), -3) == g.encode(loc)}")
