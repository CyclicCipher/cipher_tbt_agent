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
    assert fm.is_action_sensitive()                          # distinct per-action POSE effects -> a movement self


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


def test_content_operator_learns_an_in_place_toggle_and_emerges_as_controllable():
    """The behavior operator: an action that changes the object's CONTENT in place (no movement at all) is learned the
    SAME way as a displacement, and the object is recognized as controllable by CONTENT sensitivity -- so a state-change
    game (ls20: colour toggles in place) is no longer invisible to a pose-only self. The KIND is never declared."""
    fm = ForwardModel()
    X, Y = ("a",), ("b",)
    for _ in range(3):                                        # action 0 -> state X, action 1 -> state Y, from EITHER state
        fm.observe((5, 5), 0, (5, 5)); fm.observe_content(X, 0, X)
        fm.observe((5, 5), 1, (5, 5)); fm.observe_content(X, 1, Y)
        fm.observe((5, 5), 0, (5, 5)); fm.observe_content(Y, 0, X)
        fm.observe((5, 5), 1, (5, 5)); fm.observe_content(Y, 1, Y)
    assert fm.delta(0) == (0, 0) and fm.delta(1) == (0, 0)    # nothing MOVES -- a pose-only model would see no self
    assert fm.next_content(X, 1) == Y and fm.next_content(Y, 0) == X     # the content operator learned the toggle
    assert fm.next_content(("c",), 0) == ("c",)              # unseen content -> unchanged (no-op default)
    assert fm.is_action_sensitive()                          # controllable via CONTENT, though pose never changes


def test_operator_is_state_dependent_and_generalizes():
    """A wall is a context-gated effect, not a binary blocked-cell set: the SAME action moves in an OPEN context and is
    stopped in a WALL context, learned per context. Because the conditional keys on the SENSED context (not the absolute
    pose), one bump teaches the agent to expect a stop at EVERY wall-context destination -- it GENERALISES (the A∘B win a
    per-pose blocked set cannot give). The same shape carries viscosity/risk -- a continuum, not a free/blocked flag."""
    fm = ForwardModel()
    for i in range(6):
        fm.observe((i, 0), "go", (i + 1, 0), context="open")     # in the open, "go" moves +x
    fm.observe((3, 0), "go", (3, 0), context="wall")             # bump a wall ONCE (a stall in the wall context)
    fm.observe((3, 0), "go", (3, 0), context="wall")
    assert fm.delta("go", "open") == (1, 0)                      # the base operator
    assert fm.delta("go", "wall") == (0, 0)                      # stopped in the wall context
    assert fm.predict((9, 9), "go", "open") == (10, 9)
    assert fm.predict((9, 9), "go", "wall") == (9, 9)           # GENERALISES to a NEW pose: any wall context -> stop


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


def test_curiosity_is_maximal_untried_and_falls_to_zero_when_learned():
    fm = ForwardModel()
    assert fm.curiosity("up") == 1.0                         # never tried -> maximal (R-MAX optimism)
    for _ in range(8):
        fm.observe((0, 0), "up", (0, -1))                    # a consistent, learnable operator
    assert fm.curiosity("up") < 0.2                          # learned -> little learning progress left
    assert fm.curiosity("down") == 1.0                       # a still-untried action stays maximal


def test_curiosity_does_not_chase_unlearnable_noise():
    """Learning progress, NOT raw error: a noisy action whose error never falls must not stay maximally curious -- the
    'noisy-TV' problem that count-based novelty falls for. After sampling, its curiosity is well below an untried 1.0."""
    import random
    rng = random.Random(1)
    fm = ForwardModel()
    for _ in range(12):
        fm.observe((0, 0), "noise", (rng.randint(-3, 3), rng.randint(-3, 3)))   # no consistent operator
    assert fm.curiosity("noise") < 0.5                       # sampled, no learning progress -> not worth chasing
