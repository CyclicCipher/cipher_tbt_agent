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
