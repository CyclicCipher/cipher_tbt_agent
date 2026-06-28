"""Event segmentation (tbt.events): a game-over/reset/level-change is an event boundary (the reafference principle:
a change the agent's action cannot explain), excluded from operator learning. The lifecycle handling the live games
need; the thing whose absence corrupted a dynamics probe. Pure stdlib."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.events import EventSegmenter  # noqa: E402


def test_normal_steps_are_not_boundaries():
    seg = EventSegmenter()
    assert not any(seg.is_boundary(50) for _ in range(10))   # a steady ~50-cell per-action change is normal


def test_anomalous_full_frame_change_is_a_boundary():
    seg = EventSegmenter()
    for _ in range(8):
        seg.is_boundary(50)                                  # learn the normal
    assert seg.is_boundary(4000)                             # a full-frame jump cannot be self-caused -> boundary


def test_lifecycle_forces_a_boundary_regardless_of_size():
    seg = EventSegmenter()
    for _ in range(8):
        seg.is_boundary(50)
    assert seg.is_boundary(50, lifecycle=True)               # the API said GAME_OVER/WIN/level -> boundary


def test_a_boundary_does_not_pollute_the_running_normal():
    """The reafference normal must not be inflated by the boundary itself, or it would mask the next one."""
    seg = EventSegmenter()
    for _ in range(8):
        seg.is_boundary(50)
    assert seg.is_boundary(4000)                             # a boundary (not folded into the normal)
    assert not seg.is_boundary(70)                           # normal still ~50, so 70 is within-event
    assert seg.is_boundary(4000)                             # and the next full-frame jump is still caught


def test_warmup_avoids_false_boundaries_before_a_normal_exists():
    seg = EventSegmenter(warmup=4)
    assert not seg.is_boundary(10)                           # no normal yet -> not flagged
    assert not seg.is_boundary(1000)                         # still within warmup -> not flagged on magnitude alone
