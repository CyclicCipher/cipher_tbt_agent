"""The HIPPOCAMPUS module (HIPPOCAMPUS.md) — the allocentric map + the head-direction GAIN FIELD, in isolation.

Proves the one new mechanism before wiring: (1) ONE FORWARD observation generalises to EVERY heading (the coverage sweep
is dissolved); (2) a non-abelian body (FORWARD + turns) reaches a goal by reorient-then-advance (the inverse gain field);
(3) a translation-only (abelian) body reaches a goal by the aligned move. Reuses the column's `GridEncoder` + periodic
`ScalarEncoder`; the gain field is the only new part."""

from __future__ import annotations

import os
import sys

import numpy as np

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.hippocampus import Hippocampus, _rot  # noqa: E402


class _Body:
    """A heading-carrying body (SE(2)): action 0 = FORWARD (2 cells in the current heading), 1 = TURN_L, 2 = TURN_R."""

    STEP = 2.0

    def __init__(self, x=2.0, y=2.0, bound=22):
        self.x, self.y, self.th, self.bound = x, y, 0.0, bound

    def pos(self):
        return (self.x, self.y)

    def head(self):
        return self.th

    def step(self, a):
        if a == 0:
            self.x = min(max(self.x + self.STEP * np.cos(self.th), 0), self.bound)
            self.y = min(max(self.y + self.STEP * np.sin(self.th), 0), self.bound)
        elif a == 1:
            self.th += np.pi / 2
        elif a == 2:
            self.th -= np.pi / 2


class _GridBody:
    """A translation-only (abelian) body: 4 WORLD-frame moves, no heading (ARC-style ACTION1-4)."""

    MOVES = {0: (0, -2), 1: (0, 2), 2: (-2, 0), 3: (2, 0)}

    def __init__(self, bound=22):
        self.x, self.y, self.bound = 0.0, 0.0, bound

    def pos(self):
        return (self.x, self.y)

    def head(self):
        return 0.0

    def step(self, a):
        dx, dy = self.MOVES[a]
        self.x = min(max(self.x + dx, 0), self.bound)
        self.y = min(max(self.y + dy, 0), self.bound)


def test_gain_field_one_observation_generalizes_to_all_headings():
    """The crux: ONE FORWARD sighting (at heading 0) pins the egocentric displacement, and rotating it by head-direction
    gives the correct world move for EVERY heading — no per-heading learning (the coverage sweep is dissolved)."""
    hip, w = Hippocampus(board=32, n_head=4), _Body()
    hip.observe(None, w.pos(), w.head())                       # cold: set the belief
    w.step(0); hip.observe(0, w.pos(), w.head())               # a SINGLE forward at heading 0
    assert np.allclose(hip.ego[0], [2.0, 0.0], atol=0.3), hip.ego[0]
    truth = {0: (2, 0), 1: (0, 2), 2: (-2, 0), 3: (0, -2)}     # E/N/W/S
    for h, t in truth.items():
        pred = _rot(h * np.pi / 2) @ hip.ego[0]                # world Δ = R(head)·ego — derived, not re-learned
        assert np.allclose(pred, t, atol=0.3), (h, pred, t)


def test_reorient_then_advance_reaches_goal_nonabelian():
    """A non-abelian body (FORWARD + turns) reaches an interior goal: after ONE observation of each action, the inverse
    gain field turns to face the goal then advances (SE(2), the order-dependent case)."""
    hip, w, goal = Hippocampus(board=32, n_head=4), _Body(), (12.0, 12.0)
    hip.observe(None, w.pos(), w.head())
    for a in (1, 0, 2, 0):                                     # warm-up: one of each action to learn its effect
        w.step(a); hip.observe(a, w.pos(), w.head())
    assert hip.moves([0, 1, 2]) == [0] and set(hip.turns([0, 1, 2])) == {1, 2}   # move vs turn EMERGED
    for _ in range(80):
        a = hip.navigate(goal, [0, 1, 2])
        if a is None:
            break
        w.step(a); hip.observe(a, w.pos(), w.head())
    assert abs(w.x - 12) <= 1 and abs(w.y - 12) <= 1, (w.x, w.y)


def test_abelian_navigation_reaches_goal_translations():
    """A translation-only (abelian) body reaches the goal by the aligned move — the degenerate one-heading case of the
    same navigator (NavGame's ACTION1-4)."""
    hip, w, goal = Hippocampus(board=32, n_head=4), _GridBody(), (12.0, 12.0)
    hip.observe(None, w.pos(), w.head())
    for a in (0, 1, 2, 3):                                     # learn each move once
        w.step(a); hip.observe(a, w.pos(), w.head())
    for _ in range(80):
        a = hip.navigate(goal, [0, 1, 2, 3])
        if a is None:
            break
        w.step(a); hip.observe(a, w.pos(), w.head())
    assert abs(w.x - 12) <= 1 and abs(w.y - 12) <= 1, (w.x, w.y)
