"""The dimension-general operator: a learned linear `M(action)` on a CONJUNCTIVE pose SDR (position ⊗ heading)
path-integrates SE(2) FAITHFULLY — including the heading-dependent FORWARD (the non-abelian action) — with NO center of
rotation, NO SE(2) matrix on (x,y,θ), NO pose decomposition (`reference_operator_as_group_representation`; Gao et al.
2021). De-risks the P4a operator rebuild: motion is a learned group-representation matrix acting on the code."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import GridEncoder, ScalarEncoder, ConjunctiveEncoder   # noqa: E402
from tbt.operator import OnlineOperator                                    # noqa: E402

HEAD = [(1, 0), (0, 1), (-1, 0), (0, -1)]                                  # E N W S


def _fwd(x, y, h):                                                        # heading-DEPENDENT translation (the non-abelian bit)
    dx, dy = HEAD[h]
    return (x + dx, y + dy, h)


def _tl(x, y, h): return (x, y, (h + 1) % 4)
def _tr(x, y, h): return (x, y, (h - 1) % 4)

_FNS = {"FWD": _fwd, "TL": _tl, "TR": _tr}
_CACHE: dict = {}


def _trained():
    """Learn M(action) for each action from broad (position, heading) coverage — cached (trained once)."""
    if "s" not in _CACHE:
        enc = ConjunctiveEncoder([("pos", GridEncoder(scales=(5, 7, 9), dims=2, mw=1, bounds=[(0, 20), (0, 20)])),
                                  ("head", ScalarEncoder(0.0, 2 * np.pi, n=4, w=1, periodic=True))])
        code = lambda x, y, h: enc.encode({"pos": (x, y), "head": h * np.pi / 2}).dense().astype(float)
        learners = {k: OnlineOperator(enc.n) for k in _FNS}
        rng = np.random.default_rng(0)
        for _ in range(3000):                                            # broad coverage = the plan's linchpin
            x, y, h = int(rng.integers(0, 20)), int(rng.integers(0, 20)), int(rng.integers(0, 4))
            v = code(x, y, h)
            for k, f in _FNS.items():
                learners[k].observe(v, code(*f(x, y, h)))
        _CACHE["s"] = (code, {k: learners[k].operator() for k in learners})
    return _CACHE["s"]


def test_learned_operator_path_integrates_se2_faithfully():
    """M(action)·code(pose) == code(action(pose)) for every action, on held-out poses — the heading-dependent FORWARD
    included. A learned linear map on the conjunctive code IS the SE(2) path integrator."""
    code, M = _trained()
    rng = np.random.default_rng(1)
    for k, f in _FNS.items():
        cos = []
        for _ in range(200):
            x, y, h = int(rng.integers(0, 20)), int(rng.integers(0, 20)), int(rng.integers(0, 4))
            pred = np.asarray(M[k].M) @ code(x, y, h)
            tgt = code(*f(x, y, h))
            cos.append(float(pred @ tgt / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-9)))
        assert np.mean(cos) > 0.99, (k, np.mean(cos))


def test_learned_operator_is_non_abelian_and_composes():
    """FORWARD∘TURN ≠ TURN∘FORWARD (non-abelian, for free — matrix product), and TURN_L∘TURN_R = identity ON CODES."""
    code, M = _trained()
    assert not M["FWD"].commutes_with(M["TL"])                            # non-abelian
    Mtl, Mtr = np.asarray(M["TL"].M), np.asarray(M["TR"].M)
    rng, errs = np.random.default_rng(2), []
    for _ in range(100):
        x, y, h = int(rng.integers(0, 20)), int(rng.integers(0, 20)), int(rng.integers(0, 4))
        v = code(x, y, h)
        errs.append(float(np.linalg.norm(Mtl @ (Mtr @ v) - v) / (np.linalg.norm(v) + 1e-9)))
    assert np.mean(errs) < 0.01, np.mean(errs)                           # round-trip = identity on the code subspace
