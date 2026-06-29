"""Layer 5 — the per-action OPERATOR / displacement layer: the column's dynamics + motor-output seat.

L5 is the cortex's main OUTPUT layer and its displacement-cell layer (see reference_layer5_role). The chosen
DISPLACEMENT is ONE object with four uses: the position-invariant GENERALIZING base operator ("action a shifts feature
f by Δ" -> predicts a's effect at an UNVISITED state), the MOTOR command (the enacted action), the EFFERENCE COPY (the
predicted effect -> the predictive state), and the feed-forward DRIVER of the higher-order thalamus (the inter-column
message). L6 is the location frame (WHERE, the SR); L5 is the OPERATOR (HOW an action moves between locations).

The operator has two parts, and the column reads them together:
  * the position-invariant DISPLACEMENT (`disp`): per (feature, action) the modal pose-shift, learned from observed
    transitions over the translation-invariant config-state. It GENERALIZES -- it predicts an action's effect at a
    state never visited (the discrete graph cannot). The self is not labelled: a non-responsive object simply has a
    zero displacement, so "the object your operators move" emerges.
  * the discrete EDGES (`edges`): the observed per-(state, action) transitions, INCLUDING blocked moves (s2 == s)
    recorded as self-edges -- the state-dependent EXCEPTIONS (a wall/door) that OVERRIDE the base displacement.
`predict` = edges first (highest fidelity / exceptions), else the displacement (generalize), else stay.

The MATRIX associative memory (`learn`/`apply`, M_r = Σ place(t) ⊗ place(s)) is the offline / archived form (it
crosstalks over correlated SR codes; reserved for orthonormal codes).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import torch
import torch.nn as nn

from .perceive import canonicalize


class L5_Displacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops: dict = {}                                          # (domain, relation) -> matrix operator (offline/archived)
        self.edges: dict = {}                                        # state -> {action -> next state}: the observed operator / exceptions
        self.disp: dict = {}                                         # (feature_key, action) -> modal displacement (dx, dy): the base operator
        self._votes: dict = {}                                       # (feature_key, action) -> Counter of observed deltas (-> disp = mode)

    # ---- the online operator: edges (exceptions) + the position-invariant displacement (generalization) -------
    def observe(self, s, a, s2) -> None:
        """Learn one per-action transition. A real move (s2 != s) records its edge AND votes the displacement; a
        blocked move (s2 == s) over a config-state records a self-edge -- the EXCEPTION that overrides the base
        displacement -- but does NOT vote it down (the displacement is the rule; the block is the exception)."""
        if s2 != s or self._is_config(s):                           # config: record blocked self-edges (exceptions); opaque: keep no-self-edge
            self.edges.setdefault(s, {})[a] = s2
        if s2 != s:
            self._learn_disp(s, a, s2)

    def predict(self, s, a):
        """The operator / efference copy: where action `a` takes state `s`. Observed edge (incl. a blocked self-edge)
        first -- the state-dependent exception; else the position-invariant displacement GENERALIZES to this unvisited
        (s, a); else stay (no model yet)."""
        edge = self.edges.get(s, {}).get(a)
        if edge is not None:
            return edge
        gen = self._generalize(s, a)
        return gen if gen is not None else s

    def successors(self, s):
        """{action -> next state} learned from `s` — the operator's outgoing edges."""
        return self.edges.get(s, {})

    # ---- motor output + thalamus driver (the other two uses of the one displacement) ---------------------
    def motor(self, a):
        """The MOTOR command: the enacted action. L5 is the cortex's output layer -- the chosen action is its output
        (the name->GameAction mapping is the motor ORGAN, in arc_sdk). Identity over discrete actions, by design."""
        return a

    def driver(self, s, a):
        """The feed-forward DRIVER message (what a higher-order thalamus would relay to another column): the nonzero
        displacements action `a` causes among the features present in state `s` -- 'this feature moved by that'."""
        if not self._is_config(s):
            return ()
        msg = {}
        for elem in s:
            key = self._key(elem)
            d = self.disp.get((key, a))
            if d and d != (0, 0):
                msg[key] = d
        return tuple(sorted(msg.items()))

    # ---- the config-state structure the displacement reads (CMP: features at poses) ---------------------
    @staticmethod
    def _is_config(s) -> bool:
        """True if `s` is a config-state (a tuple of `(size, pose, *content)` elements) rather than an opaque symbol."""
        if not (isinstance(s, tuple) and s):
            return False
        e = s[0]
        return isinstance(e, tuple) and len(e) >= 2 and isinstance(e[1], tuple) and len(e[1]) == 2

    @staticmethod
    def _key(elem):
        """A config element's FEATURE identity (size + any content) -- the displacement is keyed on it, so same-shape
        objects share one position-invariant displacement and the pose is factored out."""
        return (elem[0],) + tuple(elem[2:])

    def _learn_disp(self, s, a, s2) -> None:
        """Vote the per-(feature, action) displacement from a real transition: align each element of `s` to the
        same-feature element of `s2` nearest in pose, accumulate the pose delta; the MODE is the base displacement."""
        if not (self._is_config(s) and self._is_config(s2)):
            return
        by_key = defaultdict(list)
        for e in s2:
            by_key[self._key(e)].append(e[1])
        for e in s:
            key, pose = self._key(e), e[1]
            cands = by_key.get(key)
            if not cands:
                continue
            tx, ty = min(cands, key=lambda p: abs(p[0] - pose[0]) + abs(p[1] - pose[1]))
            delta = (tx - pose[0], ty - pose[1])
            self._votes.setdefault((key, a), Counter())[delta] += 1
            self.disp[(key, a)] = self._votes[(key, a)].most_common(1)[0][0]

    def _generalize(self, s, a):
        """Predict an UNVISITED (s, a) by applying the position-invariant displacement to each feature's pose, then
        re-encoding to the SAME translation-invariant form. None if `s` is opaque or nothing moves (no generalization)."""
        if not self._is_config(s):
            return None
        elements, changed = [], False
        for elem in s:
            size, pose, rest = elem[0], elem[1], tuple(elem[2:])
            d = self.disp.get((self._key(elem), a))
            if d and d != (0, 0):
                pose = (pose[0] + d[0], pose[1] + d[1])
                changed = True
            elements.append((pose, (size,) + rest))
        return canonicalize(elements) if changed else None

    # ---- the matrix associative-memory operator (offline / archived) -------------------------------------
    def learn(self, key, place: torch.Tensor, edges) -> None:
        M = torch.zeros(place.shape[1], place.shape[1], device=place.device)
        for s, t in edges:
            M = M + torch.outer(place[t], place[s])
        self.ops[key] = M

    def apply(self, key, v: torch.Tensor) -> torch.Tensor:
        return self.ops[key] @ v
