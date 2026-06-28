"""The online TD successor representation (tbt.l6_sr) — the eigendecomposition-free, online L6 location code. It must
converge to the EXACT analytic SR (no batch eigh), encode topology in its place codes, and discover states online."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.l6_sr import OnlineSR  # noqa: E402


def _ring_T(k):
    T = np.zeros((k, k))
    for i in range(k):
        T[i, (i + 1) % k] = 1.0
    return T


def test_online_sr_converges_to_the_analytic_sr():
    """TD-SR over observed transitions converges to the exact SR M = (I - gamma T)^-1 -- no eigendecomposition. With
    alpha=1 and ordered sweeps this is Gauss-Seidel on the SR Bellman equation, converging geometrically."""
    k, gamma = 6, 0.9
    sr = OnlineSR(gamma=gamma, alpha=1.0)
    for s in range(k):                                        # discover the states (so rows align with the analytic index)
        sr._ensure(s)
    for _ in range(300):                                      # sweep the ring deterministically
        for i in range(k):
            sr.observe(i, (i + 1) % k)
    analytic = np.linalg.inv(np.eye(k) - gamma * _ring_T(k))
    assert np.allclose(sr.sr(), analytic, atol=1e-4), np.abs(sr.sr() - analytic).max()


def test_online_sr_converges_under_stochastic_updates():
    """The realistic online case: small alpha, random update order -- still approaches the analytic SR (looser tol)."""
    import random
    k, gamma = 6, 0.9
    sr = OnlineSR(gamma=gamma, alpha=0.1)
    for s in range(k):
        sr._ensure(s)
    rng = random.Random(0)
    for _ in range(40000):
        i = rng.randrange(k)
        sr.observe(i, (i + 1) % k)
    analytic = np.linalg.inv(np.eye(k) - gamma * _ring_T(k))
    assert np.allclose(sr.sr(), analytic, atol=0.05), np.abs(sr.sr() - analytic).max()


def test_place_codes_encode_topology():
    """The SR rows are place codes: nearer states on the ring have more similar codes than distant ones."""
    k, gamma = 6, 0.9
    sr = OnlineSR(gamma=gamma, alpha=1.0)
    for s in range(k):
        sr._ensure(s)
    for _ in range(300):
        for i in range(k):
            sr.observe(i, (i + 1) % k)
    c0, c1, c3 = sr.code(0), sr.code(1), sr.code(3)
    assert float(c0 @ c1) > float(c0 @ c3)                    # adjacent more similar than the antipode


def test_states_discovered_online():
    """A never-seen symbol gets a fresh row on first observation -- no fixed state set declared up front."""
    sr = OnlineSR()
    assert sr.sr().shape == (0, 0)
    sr.observe("a", "b")
    assert set(sr.idx) == {"a", "b"} and sr.sr().shape == (2, 2)
    sr.observe("b", "c")
    assert "c" in sr.idx and sr.sr().shape == (3, 3)
