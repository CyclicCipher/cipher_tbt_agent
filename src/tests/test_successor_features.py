"""Successor FEATURES over the SDR encoding (tbt.l6_sr.SuccessorFeatures) — the SDR-native replacement for the
localist-symbol `OnlineSR` (ARCHITECTURE §2 L6, §10 P4a/P5). The payoff: value learned over an SDR (the GridEncoder
location code) GENERALISES across nearby states via feature overlap — a navigable gradient that reaches UNVISITED
locations, which the localist SR (an unvisited symbol has a zero row) cannot give."""

from __future__ import annotations

import os
import sys

import pytest

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.encoders import GridEncoder            # noqa: E402
from tbt.l6_sr import SuccessorFeatures         # noqa: E402

_HIPPO = ("Re-seated on the HIPPOCAMPUS module (HIPPOCAMPUS.md H3): the navigator becomes the gain-field inverse transform "
          "(rotate the goal vector by −head-direction) and reads the pose belief, not L5.observe_move deltas.")


def _corridor():
    """A 1-D corridor 0..40, goal at 30, VISITING ONLY EVEN positions 20..30 (odd near-goal stay UNVISITED). Returns
    the encoder + a trained SuccessorFeatures (goal self-loop makes the goal a proper reward SOURCE)."""
    g = GridEncoder(scales=(7, 11, 13), dims=1, mw=3, bounds=[(0, 40)])
    sf = SuccessorFeatures(d=g.n, gamma=0.9, alpha=0.2, beta=0.5)
    phi = lambda x: g.encode([x]).dense()
    goal = 30
    for _ in range(400):
        for x in range(20, 30, 2):
            sf.observe(phi(x), phi(x + 2), 1.0 if x + 2 == goal else 0.0)
        sf.observe(phi(goal), phi(goal), 1.0)               # the goal is an SR source (self-loop) -> value propagates back
    return g, sf


def test_sf_value_climbs_toward_the_goal():
    g, sf = _corridor()
    phi = lambda x: g.encode([x]).dense()
    assert sf.value(phi(20)) < sf.value(phi(24)) < sf.value(phi(28))    # a monotone gradient up the corridor
    assert sf.value(phi(30)) > sf.value(phi(20))                        # the goal is the peak
    assert sf.value(phi(30)) > 5.0                                      # ~ 1/(1-gamma)=10: a genuine reward source


def test_sf_generalises_to_unvisited_states_via_sdr_overlap():
    """THE payoff: states never visited in training still get a value, graded by proximity to the goal — because their
    SDR overlaps the visited ones. The localist OnlineSR gives an unvisited symbol a flat 0."""
    g, sf = _corridor()
    phi = lambda x: g.encode([x]).dense()
    v_near, v_far = sf.value(phi(29)), sf.value(phi(21))                # 29 and 21 were NEVER visited (odd)
    assert v_far > 0.0                                                  # generalises at all (a localist SR gives 0 for an unvisited symbol)
    assert v_near > v_far                                               # graded: nearer the goal -> higher value


@pytest.mark.xfail(reason=_HIPPO, strict=False)
def test_navigate_vector_beelines_and_avoids_learned_cost():
    """The corrected navigator (ARCHITECTURE §8): the grid-cell GOAL VECTOR (attraction) modulated by the SF value
    (learned cost repulsion) — not greedy value-ascent. Beelines toward a goal, and routes around a learned-costly
    cell (the SF value there is negative)."""
    from tbt.column import CorticalColumn
    col = CorticalColumn(n_entities=8, board=32, seed=0)
    for a, d in {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}.items():   # the 4 moves' displacements (as P1 learns them)
        col.L5.observe_move(a, d)
    assert col.navigate_vector((0, 0), (10, 0), [0, 1, 2, 3]) == 3        # goal to +x -> move +x (action 3)
    assert col.navigate_vector((0, 0), (0, 10), [0, 1, 2, 3]) == 1        # goal to +y -> move +y (action 1)
    assert col.navigate_vector((5, 5), (5, 5), [0, 1, 2, 3]) is None      # at the goal -> no move

    for _ in range(200):                                                 # LEARN a strong aversive value (a wall/hazard) at (1,0)
        col.learn_location_value((1, 0), (1, 0), -5.0)
    assert col.navigate_vector((0, 0), (10, 0), [0, 1, 2, 3]) != 3        # +x now leads into the cost -> routed around
