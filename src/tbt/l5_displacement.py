"""Layer 5 — displacement cells: the per-relation movement operators.

A clean grid uses one uniform displacement per relation; a graph/tree needs a learned, location-dependent
operator per relation. L5 holds one operator per (domain, relation) — an edge associative memory
M_r = Σ_{(s→t)∈edges} place(t) ⊗ place(s), so M_r·place(s) ≈ place(t). For a metric relation this is a
uniform translation; for a graph relation the location-dependent edge map. Relational inference / path
integration = COMPOSING these operators (matrix products).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L5_Displacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops: dict = {}                                          # (domain, relation) -> operator matrix

    def learn(self, key, place: torch.Tensor, edges) -> None:
        M = torch.zeros(place.shape[1], place.shape[1], device=place.device)
        for s, t in edges:
            M = M + torch.outer(place[t], place[s])
        self.ops[key] = M

    def apply(self, key, v: torch.Tensor) -> torch.Tensor:
        return self.ops[key] @ v
