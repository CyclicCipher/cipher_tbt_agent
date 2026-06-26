"""Layer 2/3 — object (output) layer: the shared object memory + its updates.

L2/3 holds the ONE object memory S (a superposition of feature-at-location bindings across all domains)
and updates it: `pool` accumulates a binding (learning); `revise` is the delta-rule overwrite that
corrects a stale binding when the world changes (the microwave) without disturbing the rest. Many domains
coexist in S because each occupies its own orthogonal slot (the remap allocated upstream). Lateral
inter-column voting attaches here (multi-column; not used single-column).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L23_Object(nn.Module):
    def __init__(self, feat_dim: int, d_mem: int):
        super().__init__()
        self.register_buffer("S", torch.zeros(feat_dim, d_mem))     # the one shared object memory

    def pool(self, binding: torch.Tensor) -> None:
        self.S = self.S + binding                                   # accumulate a feature-at-location

    def revise(self, place: torch.Tensor, target_value: torch.Tensor) -> None:
        """Delta-rule overwrite: drive the stored value at `place` to target_value, leaving the rest intact."""
        self.S = self.S + torch.outer(target_value - self.S @ place, place)

    def vote(self, neighbours):
        raise NotImplementedError("lateral inter-column voting: multi-column work")
