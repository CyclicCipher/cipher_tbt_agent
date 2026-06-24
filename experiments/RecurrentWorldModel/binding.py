"""Feature ⊗ location binding (BLUEPRINT gap C) -- VSA/HRR-style "what bound to where".

An object is a feature `f` (what) bound to a location code `z` (where, from the grid machinery). We use
**outer-product binding**, because it makes movement a single linear operator:

    bind(f, z)      = f ⊗ z                         (a dim_f × dim_z matrix)
    scene S         = Σ_i f_i ⊗ z(loc_i)            (superpose objects)
    query_what(S,z) = S · z      ≈ feature at loc   (cleanup: nearest feature)
    query_where(S,f)= Sᵀ · f     ≈ location of f    (cleanup: nearest location)
    move(S, R)      = S · Rᵀ = Σ_i f_i ⊗ (R·z(loc_i))   -- the WHOLE scene shifts by one movement

`move` is the point: because `z(loc+1)=R·z(loc)`, right-multiplying the scene by `Rᵀ` advances every
object's location coherently in one shot -- sensorimotor prediction over a bound scene. Clean unbinding
needs the location keys to be near-orthogonal; the grid code is *metric* (nearby ≈ similar), so binding
capacity is bounded by the code's effective dimensionality (the grid→place-cell orthogonalisation is the
expected refinement).
"""

from __future__ import annotations

import torch


def bind(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.outer(f, z)


def scene(feats, zs) -> torch.Tensor:
    return sum(torch.outer(f, z) for f, z in zip(feats, zs))


def query_what(S: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return S @ z                                     # feature-space readout


def query_where(S: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return S.t() @ f                                 # location-space readout


def move(S: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return S @ R.t()                                 # advance the whole scene by the generator


def nearest(query: torch.Tensor, codebook: torch.Tensor) -> int:
    """Cleanup: index of the codebook row most aligned with `query` (dot-product nearest)."""
    return (codebook @ query).argmax().item()


def place_from_grid(Zgrid: torch.Tensor, k: int):
    """Grid→place orthogonalisation (GRID_PLACE_REFERENCE.md), in closed form. A place field is a
    location's **similarity profile across the grid code** -- the inverse-Fourier synthesis, a circulant
    `g(q−p)` peaked at each location -- then **top-k sparsified** (the dentate-gyrus nonlinearity).
    Returns (P, Shift): `P[p]` = a sparse, near-orthogonal place key for location p; `Shift` = the
    place-space movement, a **cyclic-shift permutation** (the regular representation's action, Fourier-dual
    to the grid generator `R`; `Shift @ P[p] = P[p+1]`). Smaller k = sparser = more orthogonal = higher
    binding capacity (validated: top-k≤3 → 1.00 binding where the raw grid code gave ~0.85). Movement is
    preserved because top-k commutes with the shift."""
    L = Zgrid.shape[0]
    A = Zgrid @ Zgrid.t()                            # place fields = similarity profiles (circulant)
    P = torch.zeros(L, L)
    v, idx = A.topk(k, dim=1)
    P.scatter_(1, idx, v.clamp_min(0))
    P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Shift = torch.zeros(L, L)
    for i in range(L):
        Shift[(i + 1) % L, i] = 1.0
    return P, Shift


def orthonormal_locations(L: int, dim: int, seed: int = 0):
    """Idealised place-cell-style location codes: L near-orthonormal vectors in R^dim, with the generator
    R that cyclically shifts them (R·z(i)=z(i+1)). Isolates the binding mechanism from grid-code overlap."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(dim, L, generator=g))   # dim × L, orthonormal columns
    Z = Q.t()                                                  # L × dim, orthonormal rows
    R = torch.roll(Z, -1, 0).t() @ Z                           # R·z(i) = z(i+1)
    return Z, R
