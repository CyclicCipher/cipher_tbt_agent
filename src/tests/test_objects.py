"""Object tracking with permanence (tbt.objects): follow a moving object across frames by pose, separate a mover
from a fixed churner, and break the track at an event boundary (re-localisation). Cell-level analysis could not
find a moving self (0 action-selective cells on ls20); the object's POSE is the stable, trackable signature. Pure
stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.objects import ObjectTracker, components  # noqa: E402


def _render(cells, n=24):
    g = [[0] * n for _ in range(n)]
    for (x, y) in cells:
        g[y][x] = 7
    return g


def test_components_finds_blobs_and_centroids():
    comps = components({(2, 2), (3, 2), (10, 10)})
    assert len(comps) == 2
    assert sorted(len(c) for c, _ in comps) == [1, 2]


def test_tracker_links_a_moving_object_into_one_track():
    tr = ObjectTracker(max_jump=5, min_size=1)
    for x in range(3, 9):                                    # a single cell moves right; salient = old+new each step
        tr.observe(_render({(x, 5)}), 1, _render({(x + 1, 5)}))
    mt = tr.moving_tracks(min_steps=3)
    assert len(mt) == 1                                      # one persistent object, not a new id every step
    poses = [p for _s, p, _a in next(iter(mt.values()))]
    assert poses[-1][0] > poses[0][0]                        # its pose moved right
    assert ObjectTracker.pose_spread(next(iter(mt.values()))) >= 4


def test_tracker_separates_a_mover_from_a_fixed_churner():
    """A roaming object (high pose-spread = self) vs an object that toggles in place (≈0 spread = autonomous)."""
    tr = ObjectTracker(max_jump=4, min_size=1)
    for i in range(8):
        mover_prev, mover = {(3 + i, 4)}, {(4 + i, 4)}       # roams right, far from the churner
        churn_prev = {(18, 18)} if i % 2 else {(19, 18)}
        churn = {(19, 18)} if i % 2 else {(18, 18)}          # toggles between two adjacent fixed cells
        tr.observe(_render(mover_prev | churn_prev), 1, _render(mover | churn))
    spreads = sorted(ObjectTracker.pose_spread(t) for t in tr.moving_tracks(min_steps=3).values())
    assert len(spreads) >= 2
    assert spreads[-1] >= 4 and spreads[0] <= 1             # one clearly roams (self), one stays put (autonomous)


def test_event_boundary_breaks_the_track():
    """Across an event boundary the linkage resets — the post-boundary object is a NEW track (re-localisation)."""
    tr = ObjectTracker(max_jump=30, min_size=1)             # large jump so only the boundary prevents linking
    for x in range(3, 6):
        tr.observe(_render({(x, 5)}), 1, _render({(x + 1, 5)}))
    pre = tr._next
    tr.observe(_render({(10, 5)}), 1, _render({(11, 5)}), boundary=True)
    tr.observe(_render({(11, 5)}), 1, _render({(12, 5)}))   # near the old track, but must not continue it
    assert tr._next > pre
