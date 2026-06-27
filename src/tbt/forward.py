"""G — the general learned forward model over factored coordinates (the EZ-V2 dynamics, made general).

`recursive_residual` (tbt/residual.py) learns, per action, the conditional coordinate-DELTAS of the world from
observed transitions — the SAME MDL predicate search that found arithmetic carry. ANY conditional mechanic is
just such a delta: a door is "a state-coordinate flips when its precondition holds", a push is "the block's
position-coordinate takes the move-delta when its precondition holds", pad/toggle/collect are the same shape.
So there is NO per-mechanic code here — push/door/pad/toggle all emerge as learned rules, and `predict` rolls a
state forward for the value-search to plan through.

This stays domain-free: it sees only coordinate TUPLES and action labels, never a grid / colour / object role.
The coordinate ENCODING — which is EGOCENTRIC (objects relative to the body, so relational mechanics become
plain literals) and which uses the learned roles — lives in perception (`egocentric_coords`), so `tbt/` carries
no task knowledge. Validated end-to-end on the real stack: one G predicts LockPath's door (L1) and push (L2) at
100%, with the static-blocked cases left to the map (traversability, not a world-change). Pure stdlib + residual.
"""

from __future__ import annotations

from .residual import recursive_residual, predict as _rr_predict


class ForwardModel:
    """Accumulate (coords, action, next_coords) transitions; `learn` fits the residual rules; `predict` rolls a
    coordinate state one step forward under an action (identity when the action/state is unseen)."""

    def __init__(self):
        self.transitions = []
        self.rules = {}
        self._cache = {}                                        # (coords, action) -> next_coords (G is deterministic)

    def observe(self, coords, action, next_coords):
        """One transition over the factored coordinates (skipped if either side is missing or the dimensionality
        changed — e.g. a level switch, where the object set differs)."""
        if coords is not None and next_coords is not None and len(coords) == len(next_coords):
            self.transitions.append((coords, action, next_coords))

    def learn(self):
        if not self.transitions:
            return self.rules
        actions = sorted({a for _, a, _ in self.transitions})
        states = {s for s, _, _ in self.transitions} | {sn for _, _, sn in self.transitions}
        self.rules = recursive_residual({s: s for s in states}, self.transitions, actions)   # state IS its coords
        self._cache = {}
        return self.rules

    def predict(self, coords, action):
        """Roll `coords` forward one step under `action` (the learned G). Identity if nothing was learned for it.
        Cached: the model is deterministic and the rollout revisits the same (coords, action) constantly."""
        if coords is None or action not in self.rules:
            return coords
        ck = (coords, action)
        v = self._cache.get(ck)
        if v is None:
            v = _rr_predict(self.rules, {coords: coords}, coords, action)
            self._cache[ck] = v
        return v
