"""operator.py — the TRANSFORM primitive (ARCHITECTURE.md §8) on a CONTINUOUS location state.

The SECOND of the column's two primitives, distinct from ASSOCIATE (`htm.HTMLayer`) and composing with it: the operator moves
the LOCATION (L6a), then the HTMLayer reads the feature at that new location (L4). That composition IS the TBT forward model
— "predict the next feature from the next MOVEMENT, not from the previous feature" (`htm.py`).

WHY THIS IS NOT AN HTMLayer. An HTMLayer MEMORISES (context → next) per instance, with NO weight-sharing across positions, so
a transition learned at one place does not exist at another (`project_place_invariance_needs_factored_state`). Path
integration needs the OPPOSITE: one action's effect, learned once, applied EVERYWHERE. That is a group action, not an
association — hence a second primitive. Do NOT try to unify them.

WHY THE STATE IS CONTINUOUS (cut-over 2026-07-14; full record: `notes/rotation_invariance_plan.md`).
This module previously represented the location as a discrete SDR and each action as an exact PERMUTATION of it (a per-module
cyclic shift). That is elegant but it is a **lattice-restricted special case**: a permutation can only express group elements
that map the code's lattice onto itself (crystallographic restriction), so translation was exact only on an axis-aligned grid
and rotation only at multiples of 360/N. Two experiments settled it:
  * DRIFT: with QUANTISED phases, a unit move on an oriented grid shifts module i by cos θ_i (not an integer) — the rounded
    shift is a systematic, coherent bias, so multi-scale error correction (which fixes INDEPENDENT errors) cannot bound it.
    Decode drifted linearly, failing by step 2.
  * BAKE-OFF: continuous MODULAR phases fix that (translation exact, rotation exact at k·Δ) — but preserving the code under an
    off-grid rotation needs N > 2π·r orientations, i.e. modules growing LINEARLY with object radius (512 modules / ~6k bits at
    r=16), and quadratically in SO(3). No viable operating point for continuous-rotation environments.
So the location STATE is continuous and the SDR is a per-fixation READ-OUT (its quantisation is then bounded and NEVER
accumulates, because nothing is applied repeatedly). This is what `reference_operator_as_group_representation` described all
along — Gao 2021: "a learned group-representation acting on the location CODE", where the code is continuous.

WHAT IS LEARNED vs GIVEN. The operator LEARNS what each ACTION does — a body-frame displacement + heading change, averaged
over observations, position- AND heading-invariant BY CONSTRUCTION (the same body-frame delta everywhere). Nothing about the
action is coded: the agent cannot know what "ACTION1" means (`feedback_bitter_lesson`). ROTATION by a given angle is not an
action to discover but a HYPOTHESIS to test (the pose scan), so `rotate` is a plain group action on the state, exact at any
angle — the geometry is supplied, as the reference frame's structure always was (ARCHITECTURE §10 P3).

NON-ABELIAN SE(2) falls out for free: a body-frame displacement is mapped to the world through the CURRENT heading, so
FORWARD's effect depends on heading and FORWARD;TURN ≠ TURN;FORWARD — the semidirect product R²⋊SO(2), with heading now
CONTINUOUS rather than a discrete ring. Pure translation is the heading-invariant special case, not a separate mechanism.
Pure stdlib.
"""

from __future__ import annotations

import math
from typing import Hashable


def rotate(v, deg: float):
    """Rotate a 2-vector by `deg` degrees — exact at ANY angle. The group action the pose scan applies to the state (and the
    body→world map below). This is geometry, not a learned quantity."""
    r = math.radians(deg)
    x, y = v
    return (x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r))


def wrap(deg: float) -> float:
    """Normalise an angle to [0, 360)."""
    return float(deg) % 360.0


class MotionOperator:
    """The TRANSFORM primitive: learn what each ACTION does to the continuous pose, then apply it.

    A pose is `((x, y), heading_degrees)`. For each action the operator learns the mean **body-frame** displacement and the
    mean heading change, from observed `(before, action, after)` transitions. Body-frame is what makes it general: the same
    learned delta applies at EVERY position and EVERY heading, and mapping it to the world through the current heading is
    exactly the SE(2) semidirect product — so heading-dependent motion is non-commutative by construction, with no keying,
    no ring, and no discretisation. An unlearned action is the identity (predict staying put — the correct prior, and a large
    prediction error until it is learned)."""

    def __init__(self) -> None:
        self._acc: dict = {}                      # action -> [n, Σ dx_body, Σ dy_body, Σ dtheta]

    def learn(self, action: Hashable, before, after) -> None:
        """Observe one transition. `before`/`after` are poses `((x, y), heading)`. The world displacement is un-rotated by the
        heading it was made AT, giving a body-frame delta that is invariant to where and which way the body was facing."""
        (bx, by), bh = before
        (ax, ay), ah = after
        d_body = rotate((ax - bx, ay - by), -bh)
        dth = (ah - bh + 180.0) % 360.0 - 180.0   # signed shortest heading change
        acc = self._acc.setdefault(action, [0, 0.0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += d_body[0]
        acc[2] += d_body[1]
        acc[3] += dth

    def move_of(self, action: Hashable):
        """The learned `((dx_body, dy_body), dtheta)` for an action; None if unlearned."""
        acc = self._acc.get(action)
        if not acc or acc[0] == 0:
            return None
        n = acc[0]
        return ((acc[1] / n, acc[2] / n), acc[3] / n)

    def known(self, action: Hashable) -> bool:
        return action in self._acc

    def apply(self, pose, action: Hashable):
        """Dead-reckon: apply the learned action to a pose. The body-frame displacement is mapped to the world through the
        CURRENT heading (hence non-commutative), then the heading turns. Exact — no rounding, so nothing accumulates."""
        m = self.move_of(action)
        if m is None:
            return pose
        (dbx, dby), dth = m
        (x, y), h = pose
        wx, wy = rotate((dbx, dby), h)
        return ((x + wx, y + wy), wrap(h + dth))
