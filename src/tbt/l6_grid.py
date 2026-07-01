"""Layer 6 — grid cells: the LOCATION signal, full gap-A architecture.

Biology (BRAIN_BIOLOGY.md): L6 is a grid-cell location code — multi-scale, hexagonal, path-integrable,
no origin. It receives the displacement (efference copy) from L5 and updates position by path integration.

Gap-A architecture (the FULL, domain-general grid — see feedback_use_full_gap_a). This grid is the
UNIVERSAL, intrinsic entorhinal map: the SAME grid for every environment, never matched to the world's
size or topology (matching the grid to the world is exactly what SLAM is supposed to *learn*, not assume):
  * HEXAGONAL: each module is 3 plane waves at 0/120/240° (the conformal-isometry lattice — optimal 2-D
    metric, Wei-Fiete), not a separable x/y pair.
  * MULTI-SCALE / CRT: several modules at coprime scales λ_m → unique over lcm(λ) ≫ the environment AND
    redundancy for error correction (Sreenivasan-Fiete). Scales are deliberately INCOMMENSURATE with the
    world (that is what gives the huge range); the world is bounded continuous space the grid floats over.
  * METRIC: nearby positions → similar codes (scales > 1 are smooth). This is what lets `place(z)`'s
    top-k be the spatial neighborhood (locality) — the property the orthonormal DFT destroyed.
  * GRID↔PLACE: `place(z)` = top-k of the similarity to the grid codebook → sparse, near-orthogonal place
    code (dentate-gyrus nonlinearity). Binding is in place space (capacity); the grid carries the metric.
  * ERROR-CORRECTING DECODE: `decode(z)` = nearest valid codeword over the whole codebook.

Path integration is exact within the range: a move (dx,dy) advances each plane-wave's phase by
(2π/λ_m)(dir_m·(dx,dy)). The world's topology (boundaries, wraps) is NOT encoded here — it is discovered
by loop closure (place recognition) in the layers above.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class L6_GridLocation(nn.Module):
    def __init__(self, torus_size: int = 10, scales=(11, 13, 17), lattice: str = "hex", place_k: int = 3):
        super().__init__()
        self.N = torus_size
        self.L = torus_size * torus_size
        self.place_k = place_k
        angles = [0.0, 120.0, 240.0] if lattice == "hex" else [0.0, 90.0]
        dirs = torch.tensor([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles])
        scl = torch.tensor(list(scales), dtype=torch.float32)
        # universal grid frequencies: W[(scale,dir)] = (2π/λ)·dir  (incommensurate with the world)
        W = ((2 * math.pi / scl)[:, None, None] * dirs[None]).reshape(-1, 2)      # (M, 2)
        self.register_buffer("W", W)
        self.M = W.shape[0]
        self.dim = 2 * self.M
        # codebook over the bounded region the agent inhabits (cells 0..N-1; no wrap is assumed by the grid)
        idx = torch.arange(self.L)
        pos = torch.stack([idx // self.N, idx % self.N], dim=-1).float()   # (L, 2)
        ph = pos @ W.t()                                               # (L, M)
        Zgrid = torch.stack([ph.cos(), ph.sin()], dim=-1).reshape(self.L, self.dim)
        self.register_buffer("Zgrid", Zgrid)                          # (L, dim)
        self.register_buffer("Pall", self.place(Zgrid))              # (L, L) place code per cell (cache)

    def initial(self, batch: int, device=None) -> torch.Tensor:
        return self.Zgrid[0].to(device).expand(batch, -1).contiguous()    # code at the relative origin

    def code_at(self, disp: torch.Tensor) -> torch.Tensor:
        """Grid code at a (relative) displacement disp=(...,2) from the origin — the parallel-form analog
        of path_integrate (used by the column's scan form to make all locations at once)."""
        ph = disp @ self.W.t()
        return torch.stack([ph.cos(), ph.sin()], dim=-1).reshape(*disp.shape[:-1], self.dim)

    def path_integrate(self, z: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """Advance the grid code by displacement disp=(dx,dy): rotate each plane-wave by 2π(dir·disp)/λ."""
        ang = disp @ self.W.t()                                       # (..., M) phase increment per wave
        c, s = ang.cos(), ang.sin()
        zr = z.reshape(*z.shape[:-1], self.M, 2)
        x, y = zr[..., 0], zr[..., 1]
        return torch.stack([c * x - s * y, s * x + c * y], dim=-1).reshape(z.shape)

    def operator(self, disp):
        """L6_NONABELIAN Stage 0 -- `path_integrate` RE-EXPRESSED as an `Operator`: a BLOCK-DIAGONAL rotation (one 2×2
        phase rotation per module) acting on the grid code, so `operator(disp).apply(z) == path_integrate(z, disp)`. This
        exhibits the abelian grid AS a (commuting, unitary) GROUP REPRESENTATION of translation -- the special case the
        non-abelian refactor generalises (Stage 1: learned, possibly non-commuting blocks). Returns a `tbt.operator.Operator`."""
        import numpy as np
        from .operator import Operator
        ang = self.W.detach().cpu().numpy() @ np.asarray(disp, dtype=float)   # (M,) phase increment per module
        blocks = np.zeros((self.dim, self.dim))
        for m in range(self.M):                                       # block-diagonal: one 2×2 rotation per module
            c, s = np.cos(ang[m]), np.sin(ang[m])
            blocks[2 * m:2 * m + 2, 2 * m:2 * m + 2] = [[c, -s], [s, c]]
        return Operator(blocks)

    def place(self, z: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Grid→place: sparse, near-orthogonal location code = top-k of the similarity to the grid
        codebook (z·Zgrid). Binding in this space has capacity ≈ #cells. Shapes: (...,dim) -> (...,L)."""
        k = k or self.place_k
        sim = z @ self.Zgrid.t()                                      # (..., L) similarity profile
        v, i = sim.topk(k, dim=-1)
        p = torch.zeros_like(sim)
        p.scatter_(-1, i, v.clamp_min(0))
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Error-correcting decode: nearest valid codeword over the full multi-scale codebook."""
        return (z @ self.Zgrid.t()).argmax(-1)
