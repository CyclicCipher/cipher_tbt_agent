"""Dynamics — learn the world's CONDITIONAL effects (precondition → effect) from experience.

The efference copy predicts how the BODY moves; whatever the world does that this does NOT explain is the
EXAFFERENCE (von Holst & Mittelstaedt), the residual — and it is structured: a door opens BECAUSE of a
precondition. This models that residual as feature-conditioned causal rules, with the SAME predicate search
that found carry (tbt/residual.py): under what PRECONDITION (a feature of the state) does each EFFECT occur.
The effect is a discrete world-change (a door opens, a block moves) rather than a coordinate delta, but the
mechanism is identical — so there is NO hand-coded 'key opens door' rule-type. A column's world-modelling
role, the dynamics the §5/§6 control loop will plan THROUGH. Pure stdlib."""

from __future__ import annotations

from tbt.residual import _find_predicate


class DynamicsModel:
    def __init__(self):
        self.obs = []                                        # (feature_tuple, effect)  — effect hashable or None
        self.rules = []                                      # (predicate, description, effect)

    def observe(self, features, effect):
        """One step of experience: the perceived state `features` and the EXAFFERENT `effect` (or None)."""
        self.obs.append((tuple(features), effect))

    def learn(self):
        """For each distinct effect, find the SIMPLEST precondition (a predicate over the state features) that
        selects exactly the states where it occurred — the residual predicate search. An effect with no
        compressing precondition is dropped (refused, not memorised) — the MDL stop."""
        self.rules = []
        for eff in sorted({e for _, e in self.obs if e is not None}, key=repr):
            need = [f for f, e in self.obs if e == eff]
            correct = [f for f, e in self.obs if e != eff]   # states where this effect did NOT occur
            pred, desc = _find_predicate(need, correct)
            if pred is not None:
                self.rules.append((pred, desc, eff))
        return self.rules

    def predict(self, features):
        """The effect the current state triggers (the first matching rule), or None — what the control loop
        reads to plan through the dynamics (e.g. 'reach the precondition and the door opens')."""
        f = tuple(features)
        for pred, _desc, eff in self.rules:
            if pred(f):
                return eff
        return None
