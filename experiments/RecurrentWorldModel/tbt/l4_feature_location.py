"""Layer 4 — sensory input layer: the content codebook and feature-at-location binding/readout.

L4 holds the entity ("what") codebook E and binds a feature to a location (place code): bind(f, p) =
E[f] ⊗ p. It reads the content stored at a location out of the object memory S by unbinding (S·p) and
matching the recovered value to the codebook → a score per entity. (Near-orthonormal E so distinct
entities are separable.)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L4_FeatureLocation(nn.Module):
    def __init__(self, n_entities: int, feat_dim: int = 96, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        E = torch.linalg.qr(torch.randn(feat_dim, n_entities, generator=gen))[0].t()   # (n_entities, feat_dim)
        self.register_buffer("E", E)                                  # near-orthonormal content codebook

    def bind(self, label: int, place: torch.Tensor) -> torch.Tensor:
        return torch.outer(self.E[label], place)                     # feature ⊗ location (M, d_mem)

    def readout(self, S: torch.Tensor, place: torch.Tensor) -> torch.Tensor:
        return self.E @ (S @ place)                                  # unbind S at place → entity scores
