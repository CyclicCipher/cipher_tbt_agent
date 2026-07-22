"""test_behavior.py -- the object BEHAVIOR model (changes@locations, touch-conditioned, PE-discriminated).

Yield / resist / pass are DISCOVERED from the prediction error between the body's predicted motion (the efference) and the
actual outcome -- never assumed. The yielded change is direction-GENERAL from ONE observation (it lives in the invariant frame),
which is the whole point (kills the per-direction push #3 and the snap #4). Solidity is LEARNED (kills the hard-coded #5).
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.behavior import PASS, RESIST, UNKNOWN, YIELD, ContactDynamics   # noqa: E402

RIGHT, DOWN = (1.0, 0.0), (0.0, 1.0)
ZERO = (0.0, 0.0)


def test_yield_resist_pass_discriminated_by_prediction_error():
    """Three objects, three outcomes of the SAME press -- told apart only by what actually happened, not by any label."""
    cd = ContactDynamics()
    # box: body advanced by the efference AND the box moved with it -> YIELD
    cd.observe("box", RIGHT, body_disp=RIGHT, obj_disp=RIGHT)
    # wall: pressed, neither body nor wall moved -> RESIST (solidity, learned)
    cd.observe("wall", RIGHT, body_disp=ZERO, obj_disp=ZERO)
    # ghost: body advanced THROUGH it, it did not move -> PASS (non-solid)
    cd.observe("ghost", RIGHT, body_disp=RIGHT, obj_disp=ZERO)
    assert cd.kind_of("box") == YIELD and cd.kind_of("wall") == RESIST and cd.kind_of("ghost") == PASS


def test_yield_change_is_direction_general_from_one_observation():
    """Observing the box yield to ONE rightward push predicts a DOWNward push correctly -- the change is stored in the invariant
    frame, so it generalises across directions with no second observation (the LID win; no snap, no per-direction key)."""
    cd = ContactDynamics()
    cd.observe("box", RIGHT, body_disp=RIGHT, obj_disp=RIGHT)
    obj_disp, blocked = cd.predict("box", DOWN)
    assert obj_disp == DOWN and not blocked, f"a co-moving box must yield DOWN too, got {obj_disp}"


def test_scaled_yield_learns_the_magnitude_and_self_corrects():
    """An 'ice' box that slides 2 cells per push: the observed direction is exact one-shot; the unseen perpendicular keeps the
    identity prior (bold) until its own observation corrects it -- generalise-then-correct."""
    cd = ContactDynamics()
    cd.observe("ice", RIGHT, body_disp=RIGHT, obj_disp=(2.0, 0.0))       # slides 2x right
    assert cd.predict("ice", RIGHT)[0] == (2.0, 0.0), "observed direction exact one-shot"
    assert cd.predict("ice", DOWN)[0] == DOWN, "unseen perpendicular holds the identity prior (revisable)"
    cd.observe("ice", DOWN, body_disp=DOWN, obj_disp=(0.0, 2.0))         # now observe a down slide
    assert cd.predict("ice", DOWN)[0] == (0.0, 2.0), "corrected after the second observation"


def test_resist_blocks_and_unknown_is_the_honest_prior():
    cd = ContactDynamics()
    assert cd.predict("never_felt", RIGHT) == (ZERO, False) and cd.kind_of("never_felt") == UNKNOWN
    cd.observe("wall", RIGHT, body_disp=ZERO, obj_disp=ZERO)
    assert cd.predict("wall", DOWN) == (ZERO, True), "a resisting object blocks the body in any direction"


def test_worldmodel_contact_path_steps_yield_and_resist():
    """The rollout's contact path (`WorldModel(contact=...)`): pressing a YIELD object moves it and advances the body; pressing a
    RESIST object blocks the body. Contact is geometric in imagination; the OUTCOME is the learned behavior."""
    from tbt.behavior import ContactDynamics as CD
    from tbt.hippocampus import WorldMap, WorldModel
    from tbt.operator import MotionOperator, eye
    from tasks.core import GameAction
    R = GameAction.ACTION4
    body = MotionOperator(ego=True); body.learn(R, ((0.0, 0.0), eye(2)), ((1.0, 0.0), eye(2)))   # RIGHT = +x
    cd = CD()
    cd.observe(6, RIGHT, body_disp=RIGHT, obj_disp=RIGHT)     # box (6) yields, co-moves
    cd.observe(1, RIGHT, body_disp=ZERO, obj_disp=ZERO)       # wall (1) resists
    wm = WorldModel(contact=cd)

    yld = wm.step(WorldMap(((1.0, 0.0), eye(2)), {6: ((2.0, 0.0), eye(2))}, bounds=[(0, 9), (0, 9)], body=body), R)
    assert tuple(round(c) for c in yld.objects[6][0]) == (3, 0), "the yielding box advances one cell"
    assert tuple(round(c) for c in yld.agent[0]) == (2, 0), "the body advances into the box's old cell"

    res = wm.step(WorldMap(((1.0, 0.0), eye(2)), {1: ((2.0, 0.0), eye(2))}, bounds=[(0, 9), (0, 9)], body=body), R)
    assert tuple(round(c) for c in res.agent[0]) == (1, 0), "the body is blocked by the resisting object"
    assert tuple(round(c) for c in res.objects[1][0]) == (2, 0), "the resisting object stays put"
