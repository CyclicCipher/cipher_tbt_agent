"""The forward model -- the per-action operator on an object's STATE (Layer 5, learned not assumed), STATE-DEPENDENT.

This is L5 for the object representation: `l5_displacement.py` holds the same thing (one movement operator per
relation) over the column's SR-frame place vectors; here the operator acts on a tracked object, because the
rebuild found the controllable self is an OBJECT, not a pattern of cells (`objects.py`). The operator for an action is
LEARNED from observed transitions -- never the hand-coded "ACTION1 = up" the old agent assumed, the rule real games break.

**An object's state is `(pose, content)` -- where it is AND what it looks like -- and an action can move EITHER.** A
movement game changes the pose (`observe`/`delta`/`predict`); a state-change game (ls20: a block toggles colour in
place) changes the CONTENT (`observe_content`/`next_content`). Both use the SAME modal-transition mechanism over a
factored state -- Monty's "movement and in-place change are one thing, a change at a location" -- so the operator's
KIND (translate vs toggle) FALLS OUT of the transitions, never declared. The controllable self emerges
(`is_action_sensitive`) from a pose- OR content-sensitive operator: "the self is the factor your operators move OR
change." This removed the deadly assumption that the self must translate (which made ls20 invisible).

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
        self._disp: dict = defaultdict(Counter)               # (action, context) -> Counter[ integer pose displacement ]
        self._content: dict = defaultdict(Counter)            # (action, from_content) -> Counter[ to_content ] (behavior)

    # ---- learning -------------------------------------------------------------------------------------------
    def observe(self, pose, action, next_pose, context=None) -> None:
        """Record one transition: `action`, in `context`, carried the object from `pose` to `next_pose`. A no-move in
        a wall-adjacent context is logged like any other effect -- it is simply the `(0,0)` displacement THIS context
        produces, not a special blocked flag."""
        self._disp[(action, context)][_round(pose, next_pose)] += 1

    def observe_content(self, content, action, next_content) -> None:
        """Record how `action` transforms the object's CONTENT (its appearance/state) -- the BEHAVIOR operator. It is
        the SAME modal-transition mechanism as the pose operator, over the feature factor instead of position (Monty:
        movement and in-place change are one thing, "a change at a location"). Keyed by the current content, so a
        toggle (A->B, B->A) or an action-set-state (any->X) is learnable. The KIND (move vs change) is never declared
        -- it falls out of which factor an action actually moves."""
        self._content[(action, content)][next_content] += 1

    def learn_track(self, track) -> None:
        """Learn from one `ObjectTracker` track ([(step, pose, action), ...]). Each consecutive pair is a transition
        whose displacement was produced by the LATER entry's action (the action that yielded the arrival pose)."""
        for (_s0, p0, _a0), (_s1, p1, a1) in zip(track, track[1:]):
            self.observe(p0, a1, p1)

    # ---- curiosity (competence-based intrinsic motivation -- explore until the operator is learned, not forever) ----
    def curiosity(self, action, context=None) -> float:
        """How much there is still to LEARN about `action` (the intrinsic-motivation drive that replaces count-based
        novelty), KIND-agnostic: 1.0 for a never-tried action, 0.0 once an EFFECT is established -- a movement (a pose
        displacement) OR a behavior (a content change) -- and 0.0 once we have tried enough with no effect (a no-op
        action; the noisy-TV problem raw novelty/error falls for). An explicit `context` judges the pose effect in that
        context; `None` aggregates over contexts. So babbling learns the operator whatever KIND it is, not only when
        the object MOVES (the residual pose-bias removed)."""
        if context is not None:
            counters = [c for c in (self._disp.get((action, context)),) if c]
        else:
            counters = [c for (a, _ctx), c in self._disp.items() if a == action and c]
        pose_moves = sum(v for c in counters for d, v in c.items() if d != (0, 0))
        pose_tries = sum(v for c in counters for v in c.values())
        content_items = [(frm, c) for (a, frm), c in self._content.items() if a == action]
        content_changes = sum(n for frm, c in content_items for nxt, n in c.items() if nxt != frm)
        content_tries = sum(n for _frm, c in content_items for n in c.values())
        if pose_tries == 0 and content_tries == 0:
            return 1.0                                        # never tried -> maximal curiosity
        if pose_moves >= 2 or content_changes >= 2:
            return 0.0                                        # an effect (movement OR behavior) is established -> learned
        if max(pose_tries, content_tries) >= 5:
            return 0.0                                        # tried enough, no effect -> a no-op action, give up
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

    # ---- content / behavior operator (the in-place change is the same mechanism as movement) ----------------
    def next_content(self, content, action):
        """The content `action` produces from `content`: its modal learned outcome, or `content` UNCHANGED if unseen
        (a thing keeps its appearance unless an action is learned to change it -- the behavior analogue of the pose
        operator's no-op default)."""
        c = self._content.get((action, content))
        return c.most_common(1)[0][0] if c else content

    def is_action_sensitive(self) -> bool:
        """Is this object CONTROLLABLE? -- the self emerges, KIND-agnostic, as the object some action moves where
        another does not, in POSE (movement) OR in CONTENT (behavior). No declaration of which kind a game is."""
        if len({self.delta(a) for a in self.actions()}) >= 2:    # pose varies by action -> a movement self
            return True
        by_from: dict = defaultdict(dict)                        # from_content -> {action: modal next_content}
        for (a, frm), c in self._content.items():
            by_from[frm][a] = c.most_common(1)[0][0]
        return any(len(set(amap.values())) >= 2 for amap in by_from.values())   # content varies by action -> a behavior self

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
