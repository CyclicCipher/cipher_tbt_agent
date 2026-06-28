"""The forward model -- the per-action operator on an object's pose (Layer 5, learned not assumed).

This is L5 for the object-pose representation: `l5_displacement.py` holds the same thing (one movement operator per
relation) over the column's SR-frame place vectors; here the operator acts on a tracked object's pose, because the
rebuild found the controllable self is an OBJECT followed by pose, not a pattern of cells (`objects.py`). The
operator for an action is LEARNED from observed `(pose, action, next_pose)` transitions -- never the hand-coded
"ACTION1 = up" the old agent assumed, the rule real games break.

The operator is the MODAL integer pose-displacement an action causes, not the mean. `objects.py` follows an object
by the centroid of its change-blob (old cells + new cells), whose step-to-step delta equals the true per-action
displacement during steady motion but is biased on the minority of steps that switch direction or get blocked (a
conditional effect a later residual layer -- `residual.py` -- will model). The mode reads through that minority to
the clean operator; this is exactly how `(0,-3)`/`(0,3)` were recovered on the live game cn04. Because the cells of
a rigidly-translating object all shift by the same integer, the operator applies to a whole footprint (`predict_cells`),
which is what a planner rolls forward.

`prediction_error` measures how far an outcome falls from the operator -- the reafference signal: 0 = the action
explained the change, large = unexplained (exafferent). That is what upgrades `events.py` from "the change was big"
to "the change was unpredicted", the honest definition of a boundary. (When the pose carries orientation, the
rotation operator is `recognize.cells_at`; step 1 is translational, matching the centroid pose `objects.py` tracks.)
Pure stdlib.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def _round(pose, nxt):
    """The integer cell displacement from `pose` to `nxt` (ARC frames live on a cell grid, so the displacement of a
    rigid translate is integer; rounding reads through sub-cell centroid jitter)."""
    return (round(nxt[0] - pose[0]), round(nxt[1] - pose[1]))


class ForwardModel:
    """One object's per-action displacement operator, learned online. Instantiate one per tracked object ("one
    column type, many instances"): feed it that object's transitions (or its `ObjectTracker` track) and read the
    operator with `delta`, roll it forward with `predict`/`predict_cells`, and score an outcome with
    `prediction_error`. The operator for an action is the modal integer displacement that action has caused."""

    def __init__(self):
        self._disp: dict = defaultdict(Counter)               # action -> Counter[ integer displacement -> count ]

    # ---- learning -------------------------------------------------------------------------------------------
    def observe(self, pose, action, next_pose) -> None:
        """Record one transition: `action` carried the object from `pose` to `next_pose`."""
        self._disp[action][_round(pose, next_pose)] += 1

    # ---- curiosity (competence-based intrinsic motivation -- explore until the operator is learned, not forever) ----
    def curiosity(self, action) -> float:
        """How much there is still to LEARN about this action's effect (the intrinsic-motivation drive that replaces
        count-based novelty): 1.0 for a never-tried action (R-MAX optimism), 0.0 once the operator is confidently
        learned, and 0.0 once we have tried enough without it resolving -- an unlearnable / noisy / conditional
        effect, so we do NOT keep chasing it (the noisy-TV problem that raw novelty/error falls for)."""
        c = self._disp.get(action)
        if not c:
            return 1.0                                        # never tried -> maximal curiosity
        n = sum(c.values())
        confidence = c.most_common(1)[0][1] / n
        if confidence >= 0.9:
            return 0.0                                        # a confident operator -> learned, nothing left
        if n >= 5:
            return 0.0                                        # tried enough, not resolving -> noise/conditional, give up
        return 1.0                                            # still confirming -> keep practising

    def learn_track(self, track) -> None:
        """Learn from one `ObjectTracker` track ([(step, pose, action), ...]). Each consecutive pair is a transition
        whose displacement was produced by the LATER entry's action (the action that yielded the arrival pose)."""
        for (_s0, p0, _a0), (_s1, p1, a1) in zip(track, track[1:]):
            self.observe(p0, a1, p1)

    # ---- the operator ---------------------------------------------------------------------------------------
    def actions(self):
        """The actions whose effect has been observed."""
        return list(self._disp.keys())

    def delta(self, action):
        """The learned operator for `action`: its modal integer displacement, or None if the action is unseen."""
        c = self._disp.get(action)
        return c.most_common(1)[0][0] if c else None

    def confidence(self, action) -> float:
        """Fraction of `action`'s transitions that match its modal operator -- how deterministic the effect is.
        1.0 = a clean unconditional operator; below 1.0 = the effect is sometimes conditional (blocked, contact,
        a wall) and a residual layer is needed. Unseen action -> 0.0."""
        c = self._disp.get(action)
        return c.most_common(1)[0][1] / sum(c.values()) if c else 0.0

    def summary(self):
        """{action: (delta, confidence)} -- the whole learned operator set, for inspection and voting."""
        return {a: (self.delta(a), self.confidence(a)) for a in self._disp}

    # ---- prediction -----------------------------------------------------------------------------------------
    def predict(self, pose, action):
        """Where `action` takes the object's pose (pose + operator). An unseen action leaves the pose unchanged --
        an honest "no modelled effect", which `prediction_error` then flags if the action did in fact move it."""
        d = self.delta(action)
        return pose if d is None else (pose[0] + d[0], pose[1] + d[1])

    def predict_cells(self, cells, action):
        """Roll a whole object footprint forward: the operator is a translation, so every cell shifts by the same
        integer delta. This is the next occupied set a planner reasons over. Unseen action -> cells unchanged."""
        d = self.delta(action)
        return set(cells) if d is None else {(x + d[0], y + d[1]) for (x, y) in cells}

    def prediction_error(self, pose, action, next_pose) -> int:
        """Manhattan distance between the predicted and the actual outcome -- the reafference/exafference signal.
        0 = the action explained the change; large = unexplained (an event-boundary / recruit-an-operator cue)."""
        px, py = self.predict(pose, action)
        return abs(next_pose[0] - px) + abs(next_pose[1] - py)
