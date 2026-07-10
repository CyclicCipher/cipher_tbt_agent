"""R0 — GUARD THE FINDING: a linear value over grid features cannot hold Sokoban's V*.

(`memory/project_linear_value_cannot_hold_sokoban`; `src/tbt/ROLLOUT_PLAN.md` §0.) Pure representability, decoupled from any
learning method: fit V* (from value iteration) with least squares over a feature code φ, then check greedy on the
BEST-POSSIBLE linear fit. A tabular/lookup rep holds V* (= greedy on exact V*, the VI control); NO grid-cell code does —
absolute, relational, or high-res — capping at R²≈0.75 with greedy FAILING, EVEN THOUGH every code is injective over the
552 configs (so it is the linear form, not lost capacity). Relational buys nothing over absolute and 4× the resolution buys
nothing: the residual (the block-on-pad predicate) is orthogonal to the grid basis. ⇒ Sokoban needs ROLLOUT, not a value
read-off. This test standing-guards that constraint so it cannot be re-litigated by accident.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tasks.games.sokoban import Sokoban  # noqa: E402
from tbt.encoders import SDR, GridEncoder  # noqa: E402

_GAMMA = 0.95
_GRID = GridEncoder(scales=(5, 7, 11), dims=2, mw=1, bounds=[(0, 15), (0, 15)])               # absolute position
_DGRID = GridEncoder(scales=(5, 7, 11), dims=2, mw=1, bounds=[(-15, 15), (-15, 15)])          # a difference vector
_DRICH = GridEncoder(scales=(2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16), dims=2, mw=1,               # 4× the resolution
                     bounds=[(-15, 15), (-15, 15)])


def _blocks(snap):
    return frozenset().union(*snap[1]) if snap[1] else frozenset()


def _block_pos(snap):
    b = _blocks(snap)
    return (sum(c[0] for c in b) / len(b), sum(c[1] for c in b) / len(b)) if b else (0.0, 0.0)


def _reachable(game):
    """BFS the reachable config space → (start, succ), succ[s] = [(s', reward)]; a WIN config is an absorbing self-loop r=1."""
    start = game.snapshot()
    seen, q, succ = {start}, deque([start]), {}
    while q:
        s = q.popleft()
        game.restore(s)
        if game.level_complete():
            succ[s] = [(s, 1.0)]
            continue
        outs = []
        for a in game.available_actions():
            game.restore(s)
            game.apply(a, None)
            s2 = game.snapshot()
            outs.append((s2, 0.0))
            if s2 not in seen:
                seen.add(s2)
                q.append(s2)
        succ[s] = outs
    return start, succ


def _value_iteration(succ, iters=500):
    V = {s: 0.0 for s in succ}
    for _ in range(iters):
        for s, outs in succ.items():
            V[s] = max(r + _GAMMA * V[s2] for (s2, r) in outs)
    return V


def _greedy_solves(game, start, value, budget=80):
    game.restore(start)
    for _ in range(budget):
        if game.level_complete():
            return True
        s = game.snapshot()
        best_a, best_v = None, float("-inf")
        for a in game.available_actions():
            game.restore(s)
            game.apply(a, None)
            v = value(game.snapshot())
            if v > best_v:
                best_v, best_a = v, a
        game.restore(s)
        game.apply(best_a, None)
    return game.level_complete()


def _phi_additive(snap):
    """Absolute object-config: agent SDR ⊕ block-footprint SDR, disjoint fields (additive: no agent×block conjunction)."""
    a = _GRID.encode(snap[0])
    bset = set().union(*(_GRID.encode(c).active for c in _blocks(snap))) if _blocks(snap) else set()
    return SDR.concat([a, SDR(_GRID.n, bset)]).dense()


def _make_relational(game, enc):
    """Relational code: (agent−block, block−pad, agent−goal) difference vectors, each grid-encoded by `enc`, concatenated."""
    pad, goal = next(iter(game.pads)), game.goal

    def phi(snap):
        a, b = snap[0], _block_pos(snap)
        rels = [(a[0] - b[0], a[1] - b[1]), (b[0] - pad[0], b[1] - pad[1]), (a[0] - goal[0], a[1] - goal[1])]
        return SDR.concat([enc.encode(r) for r in rels]).dense()

    return phi


def _fit(succ, V, phi):
    """Best linear fit of V* over φ → (value_fn, R², injective). Injective = φ maps all configs to distinct active-bit sets."""
    states = list(succ)
    Phi = np.array([np.asarray(phi(s), float) for s in states])
    Vs = np.array([V[s] for s in states])
    w, *_ = np.linalg.lstsq(Phi, Vs, rcond=None)
    r2 = 1.0 - ((Phi @ w - Vs) ** 2).sum() / max(((Vs - Vs.mean()) ** 2).sum(), 1e-12)
    injective = len({tuple(np.nonzero(np.asarray(phi(s)))[0].tolist()) for s in states}) == len(states)
    return (lambda s: float(np.asarray(phi(s), float) @ w)), float(r2), injective


def _l0():
    game = Sokoban()
    game.load_level(0)
    start, succ = _reachable(game)
    return game, start, succ


def test_value_iteration_solves_l0():
    """Control: V* exists and greedy-on-exact-V* solves — a tabular/lookup rep DOES hold the value (the harness is valid)."""
    game, start, succ = _l0()
    V = _value_iteration(succ)
    assert _greedy_solves(game, start, lambda s: V.get(s, -1e9))


def test_no_grid_code_linearly_holds_the_value():
    """THE FINDING: the BEST linear fit of V* over any grid code fails greedy at R²≈0.75, though each code is INJECTIVE over
    all 552 configs (so it is the linear form, not capacity). Relational does not beat absolute; 4× resolution does not help."""
    game, start, succ = _l0()
    V = _value_iteration(succ)
    fa, r2a, inja = _fit(succ, V, _phi_additive)
    fr, r2r, injr = _fit(succ, V, _make_relational(game, _DGRID))
    fh, r2h, injh = _fit(succ, V, _make_relational(game, _DRICH))
    assert inja and injr and injh, (inja, injr, injh)          # every code distinguishes all configs (not a capacity limit)
    assert r2a < 0.8 and not _greedy_solves(game, start, fa), r2a   # absolute grid cannot hold V*
    assert r2r < 0.8 and not _greedy_solves(game, start, fr), r2r   # relational grid cannot hold V*
    assert r2h < 0.8 and not _greedy_solves(game, start, fh), r2h   # high-res grid cannot hold V*
    assert abs(r2r - r2a) < 0.03, (r2a, r2r)                    # relational buys ~nothing over absolute
    assert abs(r2h - r2r) < 0.03, (r2r, r2h)                    # 4× resolution buys ~nothing
