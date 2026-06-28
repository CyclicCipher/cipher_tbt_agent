"""The forward model (tbt.forward): learn the per-action operator on an object's pose from observed transitions
(never the hand-coded "ACTION1 = up"), roll it forward over a footprint, and score an outcome by prediction error.
The modal operator reads through the minority of blocked/direction-switch steps -- the property that recovered the
live game cn04's (0,-3)/(0,3). The end-to-end test drives raw frames through retina -> ObjectTracker -> ForwardModel,
the same on-frames validation the other front-end modules use. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.forward import ForwardModel              # noqa: E402
from tbt.objects import ObjectTracker             # noqa: E402


def _render(cells, n=24):
    g = [[0] * n for _ in range(n)]
    for (x, y) in cells:
        g[y][x] = 7
    return g


def test_learns_a_distinct_operator_per_action():
    fm = ForwardModel()
    for i in range(5):
        fm.observe((i, 0), "up", (i, -3 + 0))                # "up" always shifts by (0, -3)
        fm.observe((i, 10), "down", (i, 13))                 # "down" always shifts by (0, +3)
    assert fm.delta("up") == (0, -3)
    assert fm.delta("down") == (0, 3)
    assert fm.confidence("up") == 1.0
    assert set(fm.actions()) == {"up", "down"}
    assert fm.delta("never_seen") is None                    # no assumption about an action never taken


def test_mode_reads_through_blocked_moves():
    """A mostly-(3,0) action that occasionally gets blocked (0,0) still yields the clean operator, with the
    conditional misfire showing up as reduced confidence (the signal a residual layer is needed)."""
    fm = ForwardModel()
    for _ in range(7):
        fm.observe((0, 0), "go", (3, 0))                     # the usual effect
    fm.observe((0, 0), "go", (0, 0))                         # blocked twice (against a wall)
    fm.observe((0, 0), "go", (0, 0))
    assert fm.delta("go") == (3, 0)                          # the mode, not the (biased) mean
    assert 0.7 < fm.confidence("go") < 1.0                   # 7/9 -- deterministic-ish but not perfectly


def test_predict_rolls_pose_and_footprint_forward():
    fm = ForwardModel()
    fm.observe((5, 5), "right", (7, 5))                      # operator (2, 0)
    assert fm.predict((10, 10), "right") == (12, 10)
    assert fm.predict_cells({(0, 0), (0, 1)}, "right") == {(2, 0), (2, 1)}   # the whole body shifts rigidly
    assert fm.predict((10, 10), "unknown") == (10, 10)       # no modelled effect -> pose unchanged


def test_prediction_error_is_zero_when_explained_else_large():
    fm = ForwardModel()
    for _ in range(4):
        fm.observe((0, 0), "a", (1, 0))                      # operator (1, 0)
    assert fm.prediction_error((5, 5), "a", (6, 5)) == 0     # the action explained the change (reafferent)
    assert fm.prediction_error((5, 5), "a", (5, 12)) >= 7    # an unexplained jump (exafferent -> boundary cue)


def test_end_to_end_recovers_operators_from_raw_frames():
    """Drive raw frames through the front-end: a single object moves right under action 1 then left under action 2;
    the tracker follows it as one object, and the forward model reads back the two opposite operators."""
    tr = ObjectTracker(max_jump=4, min_size=1)
    x = 5
    for _ in range(6):                                       # action 1 drives it right
        tr.observe(_render({(x, 8)}), 1, _render({(x + 1, 8)}))
        x += 1
    for _ in range(6):                                       # action 2 drives it left
        tr.observe(_render({(x, 8)}), 2, _render({(x - 1, 8)}))
        x -= 1
    track = max(tr.moving_tracks(min_steps=3).values(), key=ObjectTracker.pose_spread)
    fm = ForwardModel()
    fm.learn_track(track)
    assert fm.delta(1) == (1, 0)                             # learned, not assumed: action 1 = +x
    assert fm.delta(2) == (-1, 0)                            # action 2 = -x
