"""Layer 4 — sensory input layer: feature-at-location. Inherits L6's location, forms none of its own.

L4 binds a SENSED FEATURE to a LOCATION (place code) and reads it back: bind(f, p) = E[f] ⊗ p,
readout(S, p) = E·(S·p). The feature is a label-free, online-grown code for the local content — L4 owns the
content VOCABULARY (the codebook: a raw patch, or the rotation-invariant shape signature of a local patch,
→ a feature id). Per the neuroscience (Lewis 2019; reference_tbt_layers_4_23) L4 does NOT build locations of
its own: the location comes from L6 (modulatory/predictive), the feature confirms which cell wins — so the
predict-then-compare (`predict_feature`) is seated HERE, given where I am (place) + the active object (S).

Dorsal/ventral, one column: L5's `local_disps` is the EQUIVARIANT local geometry (the 'where/how'); L4's
`invariant_sig` is the rotation-INVARIANT signature of that geometry (the 'what'/content) — the same shape at
any orientation yields the same feature, so content recurs across pose.

The codebook uses SPARSE high-dimensional codes — the cortical capacity trick (dentate-gyrus / cerebellar /
mushroom-body expansion + sparsification). DENSE orthonormal codes cap HARD at feat_dim; random k-sparse codes
are near-orthogonal in EXPONENTIALLY many numbers (~ C(feat_dim, k)), so capacity >> feat_dim and the
vocabulary GROWS online without a hard wall (reference_cortical_capacity). Pairwise crosstalk ~ k/feat_dim,
cleaned up by the argmax readout (Kanerva's Sparse Distributed Memory / hyperdimensional computing).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def invariant_sig(disps):
    """The rotation-INVARIANT feature descriptor of a local patch (sorted neighbour distances) — the 'what'
    (content / ventral), complementary to L5's equivariant `local_disps` (the 'where/how' / dorsal). The same
    shape at any orientation yields the same signature, so content recurs across pose."""
    return tuple(sorted(round(float(np.linalg.norm(v)), 3) for v in disps))


def view_signature(cloud, ndigits: int = 3):
    """The rotation+translation-invariant, COLOUR-AWARE content descriptor of a whole LOCAL VIEW — the ventral 'what' at
    view grain (L4). `cloud` = the view's coloured cells `[(x, y, colour), ...]`. The descriptor pairs the sorted COLOUR
    multiset (the composition; also covers a single cell) with the sorted multiset of `(colour-pair, pairwise distance)`
    over all cell pairs (the invariant GEOMETRY). Pairwise distances are preserved by any rotation/translation, so the
    SAME shape+colours at any pose/position yields the SAME descriptor, while a different colouring or shape differs —
    the invariant-distance idea of `invariant_sig` lifted to view grain and made colour-aware. `encode()` turns it into a
    feature id, so content RECURS across pose (the prerequisite P2's operator needs: content invariant to where/how)."""
    cells = [(float(x), float(y), c) for (x, y, c) in cloud]
    colours = tuple(sorted(c for _x, _y, c in cells))
    pairs = []
    for i in range(len(cells)):
        xi, yi, ci = cells[i]
        for j in range(i + 1, len(cells)):
            xj, yj, cj = cells[j]
            d = round(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5, ndigits)
            pairs.append((min(ci, cj), max(ci, cj), d))
    return colours, tuple(sorted(pairs))


class L4_FeatureLocation(nn.Module):
    def __init__(self, n_entities: int, feat_dim: int = 256, k: int = 12, seed: int = 0):
        super().__init__()
        self.feat_dim, self.k = feat_dim, k
        self.gen = torch.Generator().manual_seed(seed)
        E = torch.zeros(n_entities, feat_dim)
        for i in range(n_entities):                                   # each code: k active units (sparse, expanded)
            E[i, torch.randperm(feat_dim, generator=self.gen)[:k]] = 1.0
        self.register_buffer("E", torch.nn.functional.normalize(E, dim=1))   # sparse unit-norm content codebook
        self.codebook: dict = {}                                      # descriptor -> feature id (label-free, online)

    # ---- the content vocabulary (label-free, online — absorbs the retina codebook) -----------------------
    def encode(self, descriptor) -> int:
        """The feature id for a local descriptor (a raw patch, or an `invariant_sig` shape signature), growing
        the vocabulary if novel — label-free online content discovery (the bitter lesson). The sparse codebook
        E grows with it (cortical capacity is large), so the content vocabulary has no hard wall."""
        fid = self.codebook.get(descriptor)
        if fid is None:
            fid = self.codebook[descriptor] = len(self.codebook)
            if fid >= self.E.shape[0]:                                # grow the sparse codebook on demand
                self.E = torch.cat([self.E, self._new_code()], 0)
        return fid

    def _new_code(self) -> torch.Tensor:
        row = torch.zeros(1, self.feat_dim)
        row[0, torch.randperm(self.feat_dim, generator=self.gen)[:self.k]] = 1.0
        return torch.nn.functional.normalize(row, dim=1)

    invariant_sig = staticmethod(invariant_sig)                       # the rotation-invariant feature descriptor (L4's 'what')
    view_signature = staticmethod(view_signature)                     # the colour-aware invariant descriptor of a whole local view

    # ---- feature ⊗ location: bind, read back, and PREDICT (predict-then-compare seated in L4) -------------
    def bind(self, label: int, place: torch.Tensor) -> torch.Tensor:
        return torch.outer(self.E[label], place)                     # feature ⊗ location (M, d_mem)

    def readout(self, S: torch.Tensor, place: torch.Tensor) -> torch.Tensor:
        return self.E @ (S @ place)                                  # unbind S at place → entity scores

    def predict_feature(self, S: torch.Tensor, place: torch.Tensor) -> int:
        """PREDICT the feature expected at `place` given the active object memory `S` — feature-at-location
        readout, argmax. The predict half of the column's predict-then-compare, seated where the feature lives
        (L4 inherits L6's `place`, forms no location of its own)."""
        return int(self.readout(S, place).argmax())
