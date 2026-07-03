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

import numpy as np


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


def inverse(displacement):
    """BACKWARD MODELLING (§5): the INVERSE of an SE(2) displacement. Because operators are invertible group elements,
    running a behavior BACKWARD is just applying the inverse displacements in REVERSE order (the stapler: closing IS
    opening reversed) — it is NOT a separate mechanism, only the forward sequence memory with inverse operators. Uses:
    RETRODICTION (infer the pose that PRECEDED a state: `prior = pose @ inverse(d)`) and reverse-replay credit assignment
    (walk a trajectory backward to propagate reward to the earlier states/actions that led to it)."""
    return np.linalg.inv(np.asarray(displacement, dtype=float))


class Behavior:
    """An OTHER object's DYNAMICS as a learned temporal sequence of DISPLACEMENTS (§5) — the self/other unification: your
    EFFERENCE drives self-motion (`column.forward`), and this learned behavior drives an OTHER object's next displacement,
    both by the SAME apply-operator-to-pose. `observe(pose)` reads the object's SE(2) pose each step, takes the body-frame
    displacement `pose_before⁻¹·pose_after`, quantizes it to a hashable KEY, and a `SequenceMemory` over the keys learns
    the behavior (a patrol, a toggle-cycle, an opening/closing) indexed by its PHASE. `predict()` returns the next
    displacement (a 3×3 SE(2) matrix) to apply — the OTHER-object driver of the forward model. Because operators are
    invertible, the behavior also runs BACKWARD via `predict().inverse` (P3c)."""

    def __init__(self, order: int = 2, pos_tol: float = 0.5, ang_bins: int = 16):
        self.mem = SequenceMemory(order=order)
        self.disps: dict = {}                                    # displacement KEY -> the representative displacement (3×3)
        self.pos_tol, self.ang_bins = pos_tol, ang_bins
        self._prev = None                                        # the previous observed pose (3×3)

    def _key(self, d):
        """A hashable, tolerance-quantized key for a displacement (so a repeating behavior recurs as the SAME symbol)."""
        return (round(float(d[0, 2]) / self.pos_tol), round(float(d[1, 2]) / self.pos_tol),
                round(float(np.arctan2(d[1, 0], d[0, 0]) / (2 * np.pi) * self.ang_bins)) % self.ang_bins)

    def observe(self, pose) -> None:
        """The object's SE(2) pose this step: learn the DISPLACEMENT since the last pose into the sequence memory."""
        pose = np.asarray(pose, dtype=float)
        if self._prev is not None:
            d = np.linalg.inv(self._prev) @ pose                 # the body-frame displacement (the operator)
            k = self._key(d)
            self.disps.setdefault(k, d)
            self.mem.observe(k)
        self._prev = pose

    def predict(self):
        """The predicted next DISPLACEMENT (3×3), from the behavior's phase — or None (no learned continuation yet)."""
        k = self.mem.predict()
        return self.disps.get(k) if k is not None else None

    def reset(self) -> None:
        """A boundary (the object left / a new episode): drop the phase + the last pose; keep the learned behavior."""
        self.mem.reset()
        self._prev = None
