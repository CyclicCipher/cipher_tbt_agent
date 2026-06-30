"""FM1 -- the generative forward model as a COLUMN capability (FORWARD_MODEL_PLAN). The column reads L4's
feature-at-location field and L5's per-location operator predicts the next field from the local feature-context +
action -- the TEM objective ('predict the next observation at each location') at cell grain, NOT a raw-pixel harness.

The test: a HIDDEN deterministic cellular automaton (a frontier that spreads under the action). The column learns it
from a few transitions and predicts held-out steps EXACTLY -- and the prediction is ACTION-CONDITIONED (spread right
vs left). Plus: an unseen context defaults to no-change; the field is L4 feature ids (the seating contract)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.column import CorticalColumn  # noqa: E402

H, W = 5, 14


def grow(frame, action):
    """The HIDDEN rule the column must learn: a background cell turns 1 if its neighbour in the action's direction is
    1 (action 0 spreads a 1-region RIGHTWARD, action 1 LEFTWARD). Deterministic + local."""
    out = [row[:] for row in frame]
    for y in range(H):
        for x in range(W):
            if frame[y][x] == 0:
                if action == 0 and x > 0 and frame[y][x - 1] == 1:
                    out[y][x] = 1
                if action == 1 and x < W - 1 and frame[y][x + 1] == 1:
                    out[y][x] = 1
    return out


def _bar(x):
    g = [[0] * W for _ in range(H)]
    for y in (1, 2, 3):
        g[y][x] = 1
    return g


def _rollout(start, action, steps):
    frames = [start]
    for _ in range(steps):
        frames.append(grow(frames[-1], action))
    return frames


def test_field_is_l4_feature_ids():
    """The seating contract: the field the model predicts in is L4's feature-at-location (encoded ids), not raw
    pixels -- so the SAME operator handles richer L4 features later without change."""
    col = CorticalColumn(n_entities=64)
    field = col.feature_field(_bar(3))
    flat = {v for row in field for v in row}
    assert flat == {col.L4.encode((0,)), col.L4.encode((1,))}     # exactly the two colours, as L4 ids


def test_column_learns_a_hidden_ca_and_predicts_exactly():
    """Train L5's per-location operator on a few transitions; predict held-out steps EXACTLY (deterministic local rule
    -> single-entry rule; coverage from the repeating frontier pattern)."""
    col = CorticalColumn(n_entities=64)
    frames = _rollout(_bar(3), action=0, steps=7)
    fields = [col.feature_field(f) for f in frames]
    for i in range(4):                                            # learn transitions 0..3
        col.observe_field(fields[i], 0, fields[i + 1])
    for i in (4, 5, 6):                                           # predict held-out steps 4..6
        assert col.predict_field(fields[i], 0) == fields[i + 1], f"step {i} mispredicted"


def test_prediction_is_action_conditioned():
    """The same context predicts DIFFERENTLY per action -- the model found the action's control of the dynamics
    (the cn04 lesson: action-conditioning is what the tabular loop could not see)."""
    col = CorticalColumn(n_entities=64)
    right = _rollout(_bar(7), action=0, steps=5)                  # spreads right
    left = _rollout(_bar(7), action=1, steps=5)                   # spreads left
    rf = [col.feature_field(f) for f in right]
    lf = [col.feature_field(f) for f in left]
    for i in range(3):
        col.observe_field(rf[i], 0, rf[i + 1])
        col.observe_field(lf[i], 1, lf[i + 1])
    field = col.feature_field(_bar(7))
    pr = col.predict_field(field, 0)
    pl = col.predict_field(field, 1)
    assert pr == rf[1]                                            # action 0 -> grew right
    assert pl == lf[1]                                            # action 1 -> grew left
    assert pr != pl                                               # the action genuinely steers the prediction


def test_unseen_context_defaults_to_no_change():
    """A location whose (neighbourhood, action) was never observed keeps its current feature -- the safe default that
    makes the rule a GENERALISING base over the discrete edges, not a guess."""
    col = CorticalColumn(n_entities=64)
    field = col.feature_field(_bar(3))
    assert col.predict_field(field, 9) == field                  # action 9 never trained -> identity
