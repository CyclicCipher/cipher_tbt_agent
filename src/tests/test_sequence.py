"""The ONE temporal sequence memory (ARCHITECTURE.md §5): context-conditioned next-element prediction, high-order via
the phase (the recurrent context). Reused across L4 (features), L2/3 (displacements/behaviors), L5 (actions)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from tbt.sequence import SequenceMemory  # noqa: E402


def test_sequence_memory_is_high_order():
    """The core property (§5): the SAME element predicts a DIFFERENT successor depending on the CONTEXT (the phase) --
    high-order sequences become first-order over the context-specific state. Two sequences share the middle element B
    (A B C and D B E); with enough context, B's successor is disambiguated by what preceded it."""
    sm = SequenceMemory(order=2)
    for _ in range(3):
        sm.reset()
        for e in ("A", "B", "C"):
            sm.observe(e)
        sm.reset()
        for e in ("D", "B", "E"):
            sm.observe(e)
    sm.reset(); sm.observe("A"); sm.observe("B")
    assert sm.predict() == "C" and sm.confident()             # A..B -> C (disambiguated by the A context)
    sm.reset(); sm.observe("D"); sm.observe("B")
    assert sm.predict() == "E" and sm.confident()             # D..B -> E (the SAME B, different context)

    # a FIRST-order memory (context = only the last element) CONFLATES B's successors -- the high-order gain
    sm1 = SequenceMemory(order=1)
    for _ in range(3):
        sm1.reset(); [sm1.observe(e) for e in ("A", "B", "C")]
        sm1.reset(); [sm1.observe(e) for e in ("D", "B", "E")]
    sm1.reset(); sm1.observe("A"); sm1.observe("B")
    assert not sm1.confident() and set(sm1.table[("B",)]) == {"C", "E"}   # (B,) -> {C, E}: ambiguous without context


def test_sequence_memory_predicts_a_cyclic_behavior():
    """A behavior is a repeating sequence of elements (a patrol / toggle): once learned, the memory predicts the next
    element of the cycle from the phase, and a sequence boundary (`reset`) drops the phase without forgetting the table."""
    sm = SequenceMemory(order=2)
    cycle = ("up", "up", "right", "down", "down", "left")     # a patrol behavior
    for _ in range(4):
        for e in cycle:
            sm.observe(e)
    # replay the phase and check the next step is predicted along the cycle
    sm.reset()
    sm.observe("up"); sm.observe("up")
    assert sm.predict() == "right"                            # up,up -> right
    sm.observe("right"); sm.observe("down")
    assert sm.predict() == "down"                            # right,down -> down
    sm.reset()
    assert sm.phase == () and sm.table                       # the boundary CLEARED the phase without forgetting the behavior
    sm.observe("left"); sm.observe("left")                   # a context never seen in the cycle -> a BURST (no prediction)
    assert sm.predict() is None


def test_behavior_forward_models_an_object_toggle():
    """P3b: a Behavior learns an OTHER object's dynamics as a DISPLACEMENT sequence and forward-models its next pose. A
    toggling object (x alternates 0<->1) -> the behavior predicts the toggle-back displacement; applying it to the pose
    (the SAME apply-operator-to-pose as self-motion) predicts where the object goes next -- the self/other unification."""
    import numpy as np
    from tbt.sequence import Behavior

    def pose(x):
        m = np.eye(3); m[0, 2] = float(x); return m

    b = Behavior(order=1)
    for x in (0, 1, 0, 1, 0, 1, 0, 1):                            # the object toggles between x=0 and x=1
        b.observe(pose(x))
    d = b.predict()                                               # from the last pose (x=1), the next displacement
    assert d is not None
    nxt = pose(1) @ d                                             # forward-model: apply the behavior's operator to the pose
    assert abs(float(nxt[0, 2]) - 0.0) < 0.6                     # -> predicts x=0 (toggle back)


def test_behavior_learns_a_high_order_patrol():
    """A patrol (right, right, left, left) repeats the SAME displacement (+x) with a DIFFERENT continuation by phase, so
    it needs the HIGH-ORDER memory (order>=2). The behavior forward-models the turn-back correctly."""
    import numpy as np
    from tbt.sequence import Behavior

    def pose(x):
        m = np.eye(3); m[0, 2] = float(x); return m

    b = Behavior(order=2)
    for x in [0, 1, 2, 1] * 4:                                    # the patrol: 0->1->2->1->0->1->2->1 ...
        b.observe(pose(x))
    b.reset()
    b.observe(pose(0)); b.observe(pose(1)); b.observe(pose(2))   # phase = (+1, +1): two rights
    d = b.predict()                                               # the patrol turns back -> -1
    assert d is not None
    assert abs(float((pose(2) @ d)[0, 2]) - 1.0) < 0.6           # forward-models x=1 (turning back)


def test_backward_modelling_is_inverse_operators():
    """P3c: backward modelling = INVERSE operators (§5). inv(d) UNDOES d (retrodiction -> the prior pose); a behavior's
    displacement sequence run BACKWARD (the inverses, in reverse order) recovers the start -- the stapler: closing IS
    opening reversed. Not a separate mechanism, just the forward sequence with inverse operators."""
    import numpy as np
    from tbt.sequence import inverse

    def se2(x, y, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s, float(x)], [s, c, float(y)], [0.0, 0.0, 1.0]])

    disps = [se2(1, 0, 0), se2(0, 1, np.pi / 6), se2(1, 0, 0), se2(0, 0, np.pi / 6)]   # an "opening" behavior
    start = np.eye(3)
    pose = start.copy()
    for d in disps:                                              # OPEN: apply the forward displacements
        pose = pose @ d
    end = pose.copy()
    for d in reversed(disps):                                    # CLOSE = OPEN reversed: inverses in reverse order
        pose = pose @ inverse(d)
    assert np.allclose(pose, start, atol=1e-9)                  # -> recovered the start pose (closing IS opening reversed)

    prior = end @ inverse(disps[-1])                            # RETRODICTION: undo the last displacement
    check = start.copy()
    for d in disps[:-1]:
        check = check @ d
    assert np.allclose(prior, check, atol=1e-9)                # -> the pose that PRECEDED `end`
