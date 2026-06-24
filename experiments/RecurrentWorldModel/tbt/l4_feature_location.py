"""Layer 4 — sensory input layer: the content codebook and feature-at-location binding/readout.

L4 holds the entity ("what") codebook E and binds a feature to a location (place code): bind(f, p) =
E[f] ⊗ p. It reads the content stored at a location out of the object memory S by unbinding (S·p) and
matching the recovered value to the codebook → a score per entity.

The codebook uses SPARSE high-dimensional codes — the cortical capacity trick (dentate-gyrus / cerebellar /
mushroom-body expansion + sparsification). DENSE orthonormal codes cap HARD at feat_dim (you cannot fit more
than feat_dim mutually-orthonormal vectors), whereas random k-sparse codes are near-orthogonal in
EXPONENTIALLY many numbers (~ C(feat_dim, k)), so capacity >> feat_dim. Pairwise crosstalk is ~ k/feat_dim,
cleaned up by the argmax readout (Kanerva's Sparse Distributed Memory / hyperdimensional computing).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L4_FeatureLocation(nn.Module):
    def __init__(self, n_entities: int, feat_dim: int = 256, k: int = 12, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        E = torch.zeros(n_entities, feat_dim)
        for i in range(n_entities):                                   # each code: k active units (sparse, expanded)
            E[i, torch.randperm(feat_dim, generator=gen)[:k]] = 1.0
        self.register_buffer("E", torch.nn.functional.normalize(E, dim=1))   # sparse unit-norm content codebook

    def bind(self, label: int, place: torch.Tensor) -> torch.Tensor:
        return torch.outer(self.E[label], place)                     # feature ⊗ location (M, d_mem)

    def readout(self, S: torch.Tensor, place: torch.Tensor) -> torch.Tensor:
        return self.E @ (S @ place)                                  # unbind S at place → entity scores
