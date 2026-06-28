"""Agency (tbt.agency): recover the controllable SELF from controllability alone — the part of the frame the
agent's actions move — with no assumption about action direction, single-cell body, colour, or goal. The body-id
the hand-coded efference copy could not do on the live games. Pure stdlib; no live API."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.agency import Agency  # noqa: E402


def _render(cells, n=10, color=7):
    g = [[0] * n for _ in range(n)]
    for (x, y) in cells:
        g[y][x] = color
    return g


def test_agency_finds_the_controllable_self():
    """A coherent object the agent moves -> controllability 1.0, one coherent local self, no static cell changes."""
    obj = {(3, 3), (4, 3), (3, 4), (4, 4)}
    ag = Agency()
    cur, f = obj, _render(obj)
    for i in range(3):                                        # the action moves the object right (agency isn't told)
        nxt = {(x + 1, y) for (x, y) in cur}
        f2 = _render(nxt)
        ag.observe(f, i, f2)
        f, cur = f2, nxt
    assert ag.controllability() == 1.0
    sr, cen = ag.self_region()
    assert sr and ag.coherence() == 1.0                      # one coherent controllable object
    assert len(ag.dynamic_cells()) < 30                      # local: a small part of the 100-cell frame


def test_no_agency_when_nothing_moves():
    """If the agent's actions change nothing, there is no controllable self (controllability 0)."""
    g = _render({(2, 2)})
    ag = Agency()
    for i in range(4):
        ag.observe(g, i, g)
    assert ag.controllability() == 0.0
    assert ag.self_region() == (set(), None)
    assert ag.coherence() == 0.0


def test_min_count_isolates_the_persistent_mover():
    """Raising min_count keeps only cells that change repeatedly — the persistent self, not a one-off flicker."""
    ag = Agency()
    base = {(5, 5), (6, 5)}
    cur, f = base, _render(base)
    for i in range(4):                                        # the persistent object oscillates around (5,5)
        nxt = {(5, 5), (6, 5)} if i % 2 else {(6, 5), (7, 5)}
        f2 = _render(nxt)
        ag.observe(f, i, f2)
        f, cur = f2, nxt
    flicker = _render({(0, 0)})                               # a single one-off change far away
    ag.observe(_render(set()), 99, flicker)
    persistent = ag.dynamic_cells(min_count=2)
    assert (0, 0) not in persistent                          # the one-off flicker is filtered out
    assert persistent                                        # the repeatedly-moving self remains
