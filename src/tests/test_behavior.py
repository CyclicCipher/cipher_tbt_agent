"""test_behavior.py -- the object BEHAVIOR model: the L5 TRANSFORM (changes@locations, one cortical layer + a read-out).

There is no taxonomy of behaviours to test, because there is no taxonomy: "it moves", "it does not", "it blocks me" are three
learned deltas out of one mechanism. What these pin is the mechanism's properties -- one-shot exactness, conditioning on the
interaction's situation, superposition over cues, cue competition -- and, deliberately, its honest limit.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.behavior import Transform                                                  # noqa: E402
from tbt.encoders import GridEncoder                                                # noqa: E402

RIGHT, DOWN = (1.0, 0.0), (0.0, 1.0)
ZERO = (0.0, 0.0)
_GRID = GridEncoder(scales=(7, 11, 13, 17), dims=2, mw=3)


def test_worldmodel_contact_path_runs_on_the_one_transform():
    """The rollout's contact path (`WorldModel`): pressing something the transform says moves, moves it and lets the body
    advance; pressing something whose learned correction cancels the body's own motion leaves both where they were. There is no
    yield and no resist in the model -- only two learned deltas -- and the world model contains no physics at all."""
    from tbt.hippocampus import WorldMap, WorldModel
    from tbt.operator import MotionOperator, eye
    from tasks.core import GameAction
    R = GameAction.ACTION4
    body = MotionOperator(ego=True)
    body.learn(R, ((0.0, 0.0), eye(2)), ((1.0, 0.0), eye(2)))          # RIGHT = +x
    t = Transform()
    cues = lambda tag, felt, beyond: {(tag, felt)} | ({("beyond", beyond)} if beyond is not None else set())
    param = lambda v: {("g", b) for b in _GRID.encode(v).active}
    t.learn(cues("of", 6, None), param(RIGHT), RIGHT)                  # 6 takes the press and moves with it
    t.learn(cues("into", 6, None), param(RIGHT), ZERO)                 # ...and does not hold the body up
    t.learn(cues("into", 1, None), param(RIGHT), (-1.0, 0.0))          # 1 cancels the body's own motion exactly
    t.learn(cues("of", 1, None), param(RIGHT), ZERO)
    wm = WorldModel(t, param, cues)

    box = wm.step(WorldMap(((1.0, 0.0), eye(2)), {6: ((2.0, 0.0), eye(2))}, bounds=[(0, 9), (0, 9)], body=body), R)
    assert tuple(round(c) for c in box.objects[6][0]) == (3, 0), "the pressed thing advances one cell"
    assert tuple(round(c) for c in box.agent[0]) == (2, 0), "the body advances into the cell it vacated"

    wall = wm.step(WorldMap(((1.0, 0.0), eye(2)), {1: ((2.0, 0.0), eye(2))}, bounds=[(0, 9), (0, 9)], body=body), R)
    assert tuple(round(c) for c in wall.agent[0]) == (1, 0), "the body's motion is cancelled by the learned correction"
    assert tuple(round(c) for c in wall.objects[1][0]) == (2, 0), "and the thing pressed does not move"


# ── the L5 TRANSFORM (notes/l5_unified_transform_design.md) ────────────────────────────────────────────────────────────
# ONE cortical layer read out as a metric quantity — `HTMLayer(proximal=cues, basal=parameter)` → `PopulationReadout`. No
# kinds, no branches, no priors, and nothing written that the substrate does not already do. These pin the properties the
# bespoke machinery it replaces was carrying, plus the two the substrate adds for free.

LEFT, UP = (-1.0, 0.0), (0.0, -1.0)


def _param(v):
    """The interaction parameter as a basal context SDR — the encoder is UPSTREAM of the layer, as it is everywhere."""
    return {("g", b) for b in _GRID.encode(v).active}


def _cues(*names):
    """Each cue drives its own disjoint block of minicolumns (the proximal drive)."""
    return {(nm, i) for nm in names for i in range(8)}


def _close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_transform_learns_a_fixed_delta_per_cue_one_shot():
    """The per-action OPERATOR case: with the parameter held constant the cue IS the whole effect, and ONE observation is
    exact — the property `MotionOperator`'s running mean gave, now falling out of a layer whose grown segment is connected
    immediately. An unlearned cue drives only bursting columns, which are not read, so it contributes nothing."""
    t = Transform()
    t.learn(_cues("act_E"), _param(ZERO), RIGHT)
    assert _close(t.predict(_cues("act_E"), _param(ZERO)), RIGHT), t.predict(_cues("act_E"), _param(ZERO))
    assert _close(t.predict(_cues("never_seen"), _param(ZERO)), ZERO), "no evidence, no effect"


def test_transform_learns_an_interaction_parameterised_delta():
    """The PUSH case: the SAME cue under four different parameters decodes four different deltas, because the basal context
    selects a different cell within the cue's columns each time. That conjunction is the interaction term, formed by the
    substrate rather than by an affine formula."""
    t = Transform()
    for p, d in [(RIGHT, RIGHT), (DOWN, DOWN), (LEFT, LEFT), (UP, UP)]:
        t.learn(_cues("contact"), _param(p), d)
    for p, d in [(RIGHT, RIGHT), (DOWN, DOWN), (LEFT, LEFT), (UP, UP)]:
        assert _close(t.predict(_cues("contact"), _param(p)), d), (p, t.predict(_cues("contact"), _param(p)))


def test_transform_learns_the_magnitude_not_just_the_direction():
    """An 'ice' box that slides two cells per push: the delta is whatever was OBSERVED, so a scaled response needs no extra
    machinery and no notion of 'co-motion'."""
    t = Transform()
    t.learn(_cues("box"), _param(RIGHT), (2.0, 0.0))
    assert _close(t.predict(_cues("box"), _param(RIGHT)), (2.0, 0.0))


def test_transform_superposes_independently_learned_cues():
    """SUPERPOSITION, free: two cues learned apart drive the union of their assemblies when present together, and the
    population vector SUMS them. There is no summation over cues written anywhere — it is a property of the code."""
    t = Transform()
    t.learn(_cues("a"), _param(ZERO), (1.0, 0.0))
    t.learn(_cues("b"), _param(ZERO), (0.0, 2.0))
    assert _close(t.predict(_cues("a", "b"), _param(ZERO)), (1.0, 2.0)), t.predict(_cues("a", "b"), _param(ZERO))


def test_transform_does_not_extrapolate_an_unobserved_parameter():
    """HONEST LIMIT, recorded deliberately: with no prior the transform does not invent a delta for a parameter it has never
    seen. Any such generalisation must come from the parameter's ENCODING or from the frame the caller supplies, never from
    a prior baked into the mechanism."""
    t = Transform()
    t.learn(_cues("contact"), _param(RIGHT), RIGHT)
    assert _close(t.predict(_cues("contact"), _param((5.0, 5.0))), ZERO), "an unseen parameter predicts nothing"


def test_transform_cue_competition_rejects_the_spurious_cue():
    """The KEY-DISCOVERY property survives: the read-out's delta rule shares the error over the ACTIVE CELLS, so a neighbour
    that merely co-occurs is blocked once SUPPORT explains 'stays' (Kamin blocking) — while the neighbour alone still
    predicts the fall it really does explain."""
    t = Transform(lr=0.5)
    for _ in range(30):
        t.learn(_cues("support", "neighbour"), _param(ZERO), ZERO)      # supported ⇒ stays
        t.learn(_cues("neighbour"), _param(ZERO), DOWN)                 # neighbour alone ⇒ falls
    assert _close(t.predict(_cues("support", "neighbour"), _param(ZERO)), ZERO, tol=0.05)
    assert _close(t.predict(_cues("neighbour"), _param(ZERO)), DOWN, tol=0.05)
