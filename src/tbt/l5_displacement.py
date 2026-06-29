"""Layer 5 — the per-action OPERATOR / displacement layer: the column's dynamics + motor-output seat.

L5 is the cortex's main OUTPUT layer and its displacement-cell layer (see reference_layer5_role): it owns the
per-action operator (how an action moves state -> state), it is the MOTOR output (the enacted action), the EFFERENCE
COPY (the predicted effect -> the predictive state), and the DRIVER of the higher-order thalamus (inter-column
feed-forward). L6 is the location frame (WHERE, the SR); L5 is the OPERATOR (HOW an action moves between locations).

Two forms of the operator live here:
  * the ONLINE DISCRETE operator (`observe`/`predict`/`successors`) — the learned per-action transitions, with blocked
    moves as state-dependent EXCEPTIONS (the wall/door). This is what the column uses online (option 1).
  * the MATRIX associative memory (`learn`/`apply`, M_r = Σ place(t) ⊗ place(s), so M_r·place(s) ≈ place(t)) — the
    offline / archived form (it crosstalks over correlated SR codes; reserved for orthonormal codes).
The position-invariant DISPLACEMENT (a generalizing base operator over poses) + the literal motor / thalamus output
activate at the L5 reseat once the sensor supplies poses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L5_Displacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops: dict = {}                                          # (domain, relation) -> matrix operator (offline/archived)
        self.edges: dict = {}                                        # state -> {action -> next state}: the online operator

    # ---- the online discrete operator (the dynamics seat) -------------------------------------------------
    def observe(self, s, a, s2) -> None:
        """Learn one per-action transition. A blocked move (s2 == s) is a state-dependent EXCEPTION with no edge (a
        wall/door); cross-state generalization is the displacement, added at the reseat once poses exist."""
        if s2 != s:
            self.edges.setdefault(s, {})[a] = s2

    def predict(self, s, a):
        """The operator / efference copy: where action `a` takes state `s` (stay if unobserved or blocked)."""
        return self.edges.get(s, {}).get(a, s)

    def successors(self, s):
        """{action -> next state} learned from `s` — the operator's outgoing edges."""
        return self.edges.get(s, {})

    # ---- the matrix associative-memory operator (offline / archived) -------------------------------------
    def learn(self, key, place: torch.Tensor, edges) -> None:
        M = torch.zeros(place.shape[1], place.shape[1], device=place.device)
        for s, t in edges:
            M = M + torch.outer(place[t], place[s])
        self.ops[key] = M

    def apply(self, key, v: torch.Tensor) -> torch.Tensor:
        return self.ops[key] @ v
