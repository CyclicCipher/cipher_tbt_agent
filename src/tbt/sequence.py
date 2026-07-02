"""The ONE temporal sequence memory (ARCHITECTURE.md §5) — context-conditioned next-element prediction.

HTM temporal memory (Hawkins & Ahmad 2016): a cell's CONTEXT (its distal/basal input = the recently-active cells) puts
it in a PREDICTIVE state; when input arrives, the predicted cells fire and INHIBIT their siblings, making the
representation CONTEXT-SPECIFIC. So the SAME element in a different context becomes a different (predictive) state, and
HIGH-ORDER sequences (ABCD vs XBCY) collapse to FIRST-ORDER transitions over the context-specific states — the predictive
cells ARE the next-element prediction.

This is ONE mechanism (§5), reused wherever a layer predicts the next element of a sequence — L4 (next feature / content
dynamics), L2/3 (next DISPLACEMENT / an object's BEHAVIOR, indexed by its PHASE), L5 (next action / a motor skill) —
differing ONLY in the element type and the context that drives it. L6 is the exception (its temporal structure is the
SR). Never reimplement it per layer (rule 1); the same difference the neuroscience notes — "the primary difference being
the contextual input sent to the active dendrites."
"""

from __future__ import annotations

from collections import Counter


class SequenceMemory:
    """Predict the next element of a sequence from the recent CONTEXT (the PHASE), learned online. `order` = how much
    context makes the representation context-specific (the high-order depth): the same element in a different last-`order`
    context predicts a different successor. The PHASE is the ONE recurrence — a recurrent state advanced each `observe`;
    a `behavior` is a path traced through the learned transitions, `predict` its next step."""

    def __init__(self, order: int = 2):
        self.order = order
        self.phase: tuple = ()                                   # the recurrent PHASE = the last `order` elements (the context)
        self.table: dict = {}                                    # context -> Counter(next element): the learned high-order transitions

    def observe(self, element) -> None:
        """One element arrived: LEARN (the current phase/context → this element), then ADVANCE the phase (the recurrence)."""
        self.table.setdefault(self.phase, Counter())[element] += 1
        self.phase = (self.phase + (element,))[-self.order:]

    def predict(self):
        """The predicted next element given the current phase (the predictive state) — the most-supported successor of
        this context, or None when the context has not been seen (no prediction, a burst)."""
        c = self.table.get(self.phase)
        return c.most_common(1)[0][0] if c else None

    def confident(self) -> bool:
        """Is the current context UNAMBIGUOUS (one learned successor)? — mastered vs still-branching (a learning signal)."""
        c = self.table.get(self.phase)
        return c is not None and len(c) == 1

    def reset(self) -> None:
        """A sequence boundary: clear the phase (do not carry context across it). The learned `table` persists."""
        self.phase = ()
