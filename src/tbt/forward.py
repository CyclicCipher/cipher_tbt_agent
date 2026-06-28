"""The forward model -- the per-action operator on an object's pose (Layer 5, learned not assumed), STATE-DEPENDENT.

This is L5 for the object-pose representation: `l5_displacement.py` holds the same thing (one movement operator per
relation) over the column's SR-frame place vectors; here the operator acts on a tracked object's pose, because the
rebuild found the controllable self is an OBJECT followed by pose, not a pattern of cells (`objects.py`). The
operator for an action is LEARNED from observed `(pose, action, next_pose)` transitions -- never the hand-coded
"ACTION1 = up" the old agent assumed, the rule real games break.

**The operator is conditioned on CONTEXT, so a wall is not a special case.** An action's effect is keyed by
`(action, context)`, where `context` is whatever the agent senses locally about the transition (supplied by
perception; `None` = unconditioned). Most of the time the context is "open" and the effect is the clean displacement.
In a wall-adjacent context the SAME action yields no displacement -- a *context-gated* effect, exactly the
state-dependent operator `residual.py` learns for a door (TARGET_ARCHITECTURE "The unifying operation" and "Obstacles
and cost"). So "blocked" is never a binary obstacle predicate or a labelled wall object (the bitter-lesson trap the
doc forbids); it is one point on a continuum -- the same machinery represents viscosity (a context with a slower
displacement) and risk (a context that ends in death, weighed by SIGNED value). Because the conditional keys on the
SENSED context, not the absolute pose, learning one wall-bump generalises to EVERY occupied destination -- the agent
routes around any wall it can sense, not just the cell it already hit.

Within a context the operator is the MODAL MOVING (non-zero) integer pose-displacement, not the mean: `objects.py`
follows an object by the centroid of its change-blob, whose step-to-step delta equals the true per-action displacement
during steady motion but is biased on the minority of steps that switch direction or momentarily stall. The mode reads
through that minority to the clean operator; this is how `(0,-3)`/`(0,3)` were recovered on the live game cn04. Because
the cells of a rigidly-translating object all shift by the same integer, the operator applies to a whole footprint
(`predict_cells`), which a planner rolls forward.

`prediction_error` measures how far an outcome falls from the operator -- the reafference signal: 0 = the action
explained the change, large = unexplained (exafferent). That is what upgrades `events.py` from "the change was big"
to "the change was unpredicted", the honest definition of a boundary. Pure stdlib.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def _round(pose, nxt):
    """The integer cell displacement from `pose` to `nxt` (ARC frames live on a cell grid, so the displacement of a
    rigid translate is integer; rounding reads through sub-cell centroid jitter)."""
    return (round(nxt[0] - pose[0]), round(nxt[1] - pose[1]))


class ForwardModel:
    """One object's per-action, per-CONTEXT displacement operator, learned online. Instantiate one per tracked object
    ("one column type, many instances"): feed it that object's transitions and read the operator with `delta`, roll it
    forward with `predict`/`predict_cells`, and score an outcome with `prediction_error`. Each effect is the modal
    moving displacement that `(action, context)` has caused; `context=None` is the unconditioned (open-field) effect."""

    def __init__(self):
        self._disp: dict = defaultdict(Counter)               # (action, context) -> Counter[ integer displacement -> count ]

    # ---- learning -------------------------------------------------------------------------------------------
    def observe(self, pose, action, next_pose, context=None) -> None:
        """Record one transition: `action`, in `context`, carried the object from `pose` to `next_pose`. A no-move in
        a wall-adjacent context is logged like any other effect -- it is simply the `(0,0)` displacement THIS context
        produces, not a special blocked flag."""
        self._disp[(action, context)][_round(pose, next_pose)] += 1

    def learn_track(self, track) -> None:
        """Learn from one `ObjectTracker` track ([(step, pose, action), ...]). Each consecutive pair is a transition
        whose displacement was produced by the LATER entry's action (the action that yielded the arrival pose)."""
        for (_s0, p0, _a0), (_s1, p1, a1) in zip(track, track[1:]):
            self.observe(p0, a1, p1)

    # ---- curiosity (competence-based intrinsic motivation -- explore until the operator is learned, not forever) ----
    def curiosity(self, action, context=None) -> float:
        """How much there is still to LEARN about `action` (the intrinsic-motivation drive that replaces count-based
        novelty): 1.0 for a never-tried action, 0.0 once a MOVEMENT operator is established, and 0.0 once we have tried
        enough with no movement (a no-op action -- do NOT keep chasing it; the noisy-TV problem raw novelty/error falls
        for). With an explicit `context` it judges that context; with `None` it aggregates over all contexts seen for
        the action (babbling learns the base operator -- the wall conditional is learned passively, on contact)."""
        if context is not None:
            counters = [c for c in (self._disp.get((action, context)),) if c]
        else:
            counters = [c for (a, _ctx), c in self._disp.items() if a == action and c]
        if not counters:
            return 1.0                                        # never tried -> maximal curiosity
        if sum(v for c in counters for d, v in c.items() if d != (0, 0)) >= 2:
            return 0.0                                        # a movement operator is established -> learned
        if sum(v for c in counters for v in c.values()) >= 5:
            return 0.0                                        # tried enough, no movement -> no-op, give up
        return 1.0                                            # still confirming -> keep practising

    # ---- the operator ---------------------------------------------------------------------------------------
    def actions(self):
        """The actions whose effect has been observed (across all contexts)."""
        return list({a for (a, _ctx) in self._disp})

    def delta(self, action, context=None):
        """The learned operator for `action` in `context`: its modal MOVING (non-zero) integer displacement -- so the
        occasional stall does not corrupt it. A context NOT yet seen falls back to the unconditioned base operator
        (R-MAX optimism: assume the usual effect until a bump proves otherwise -- a wall is a LEARNED exception, not
        the default, so the agent routes TO objects and only routes AROUND ones it has learned block it). None if even
        the base is unseen; `(0,0)` if this context only ever produces no movement (the learned wall operator)."""
        c = self._disp.get((action, context))
        if not c:
            return self.delta(action, None) if context is not None else None
        nz = [(d, n) for d, n in c.items() if d != (0, 0)]
        return max(nz, key=lambda dn: dn[1])[0] if nz else (0, 0)

    def confidence(self, action, context=None) -> float:
        """Fraction of this `(action, context)`'s transitions that match its modal outcome -- how deterministic the
        effect is. 1.0 = a clean operator; below 1.0 = a still-unmodelled conditional (a finer context is needed).
        Unseen -> 0.0."""
        c = self._disp.get((action, context))
        return c.most_common(1)[0][1] / sum(c.values()) if c else 0.0

    def summary(self):
        """{action: (delta, confidence)} for the unconditioned operator -- the whole base operator set, for inspection."""
        return {a: (self.delta(a), self.confidence(a)) for a in self.actions()}

    # ---- prediction -----------------------------------------------------------------------------------------
    def predict(self, pose, action, context=None):
        """Where `action` takes the object's pose IN `context`: pose + the context's operator. A no-op/wall context
        (delta `(0,0)`) or an unseen action leaves the pose unchanged -- so the planner rolling this forward routes
        AROUND high-cost contexts, never asserting a move the model has learned does not happen."""
        d = self.delta(action, context)
        if d is None or d == (0, 0):
            return pose
        return (pose[0] + d[0], pose[1] + d[1])

    def predict_cells(self, cells, action, context=None):
        """Roll a whole object footprint forward: the operator is a translation, so every cell shifts by the same
        integer delta. This is the next occupied set a planner reasons over. No-op/unseen -> cells unchanged."""
        d = self.delta(action, context)
        return set(cells) if (d is None or d == (0, 0)) else {(x + d[0], y + d[1]) for (x, y) in cells}

    def prediction_error(self, pose, action, next_pose, context=None) -> int:
        """Manhattan distance between the predicted and the actual outcome -- the reafference/exafference signal.
        0 = the action explained the change; large = unexplained (an event-boundary / recruit-an-operator cue)."""
        px, py = self.predict(pose, action, context)
        return abs(next_pose[0] - px) + abs(next_pose[1] - py)
