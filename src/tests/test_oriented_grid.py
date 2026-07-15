"""R1 of the rotation plan (`notes/rotation_invariance_plan.md`): the MULTI-ORIENTATION GridEncoder.

Per Numenta 2021 ("Orientation Invariant Sensorimotor Object Recognition Using Cortical Grid Cells"), grid modules are
pre-tuned to orientations spread over 360° and ORDERED, so a rotation becomes a CIRCULAR-BUFFER shift of the module index.
That is the whole point: it turns rotation into a PERMUTATION (the TRANSFORM primitive) instead of a search.

THE R1 PROPERTY pinned here: a module at θ reads the location PROJECTED onto θ, and `proj_i(R_ω·loc) == proj_(i−k)(loc)` for
ω = k·(360/N) — so rotating a location moves module j's phase to module j+k, cell unchanged. R2 builds the operator that
performs that shift; R3 scans orientations to recognise a rotated object. The axis-aligned default is unchanged (the rest of
the suite exercises it).
"""

from __future__ import annotations

import math
import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import GridEncoder  # noqa: E402

SCALES = (7, 11, 13, 17)
N = 8                                          # orientations spread over 360° → Δ = 45°
B = [(0, 15), (0, 15)]                         # tight bounds — the oriented decode is a joint scan


def _grid(n_orient: int = N) -> GridEncoder:
    return GridEncoder(scales=SCALES, dims=2, mw=1, bounds=B, orientations=n_orient)


def _rotate(p, deg: float):
    r = math.radians(deg)
    x, y = p
    return (x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r))


def _module_cells(g: GridEncoder, sdr):
    """Per module index → the set of active cell offsets within that module."""
    out = []
    for m in g.modules():
        base = m[0]
        out.append({b - base for b in sdr.active if base <= b < base + len(m)})
    return out


def test_oriented_grid_encodes_and_decodes():
    g = _grid()
    for p in [(0, 0), (3, 5), (12, 2), (9, 14)]:
        assert g.decode(g.encode(p)) == p, f"round-trip failed at {p}"


def test_nearby_locations_overlap_more_than_far():
    g = _grid()
    a = g.encode((7, 7))
    assert a.overlap(g.encode((8, 7))) > a.overlap(g.encode((1, 14))), \
        "graded metric overlap: nearby locations must share more bits than distant ones"


def test_module_structure_is_orientation_ordered():
    g = _grid()
    assert len(g.modules()) == len(SCALES) * N, "one module per (scale, orientation)"
    buf = g.orientation_buffer()
    assert len(buf) == len(SCALES) and all(len(ring) == N for ring in buf), "per scale, a ring of N orientation-modules"
    assert buf[0] == list(range(N)) and buf[1] == list(range(N, 2 * N)), "scale-major, orientation-minor ordering"


def test_rotation_is_a_cyclic_module_shift():
    """THE R1 property — rotating the location by k·Δ moves module j's phase to module j+k (same cell). This circular buffer
    is what lets R2 implement rotation as a permutation rather than a search."""
    g = _grid()
    delta = 360.0 / N
    loc = (5.0, 3.0)
    base = _module_cells(g, g.encode(loc))
    for k in (1, 2, 3, 5):
        rot = _module_cells(g, g.encode(_rotate(loc, k * delta)))
        for s in range(len(SCALES)):
            for i in range(N):
                assert rot[s * N + i] == base[s * N + (i - k) % N], (
                    f"rotation by {k}·Δ must shift the module index by {k} (scale {s}, module {i}): "
                    f"{rot[s * N + i]} != {base[s * N + (i - k) % N]}")


def test_axis_aligned_default_is_unchanged():
    """Backward compatibility: the default grid still reads each axis directly (no trig, exact)."""
    g = GridEncoder(scales=SCALES, dims=2, mw=1, bounds=[(0, 63), (0, 63)])
    assert g._axis_aligned and len(g.modules()) == len(SCALES) * 2
    assert g.decode(g.encode((40, 17))) == (40, 17)


if __name__ == "__main__":
    g = _grid()
    d = 360.0 / N
    print(f"oriented grid: {len(g.modules())} modules ({len(SCALES)} scales x {N} orientations), n={g.n}, w={g.encode((5,3)).w}")
    print(f"decode round-trip (9,14): {g.decode(g.encode((9, 14)))}")
    print(f"rotation by {d}° == module shift by 1: "
          f"{_module_cells(g, g.encode(_rotate((5.0, 3.0), d)))[1] == _module_cells(g, g.encode((5.0, 3.0)))[0]}")
