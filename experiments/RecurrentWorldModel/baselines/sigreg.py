"""Anti-collapse regularizers for the JEPA world model.

SIGReg -- Sketched Isotropic Gaussian Regularization. Shared by LeJEPA (Balestriero &
LeCun, arXiv:2511.08544) and LeWorldModel (Maes/Le Lidec/LeCun/Balestriero, arXiv:2603.19312).
Pushes an embedding batch toward an isotropic *unit* Gaussian -- proven (LeJEPA) to be the
risk-optimal embedding distribution -- via random 1D slices + an Epps-Pulley
characteristic-function test. No EMA, no stop-grad, no whitening; bounded gradients.

MultiSubspaceSIGReg -- Sub-JEPA's Subspace Gaussian Regularization (Zhao et al., Shanghai U,
arXiv:2605.09241; code intcomp/Sub-JEPA). Projects the embedding into K frozen-orthogonal
d'-dim subspaces (d' = D/K) and runs the same Epps-Pulley 1D-slice test *within each
subspace*. Confining the slice directions to those subspaces (vs the full sphere) RELAXES the
strong "full ambient isotropy" bias of SIGReg -- a better bias-variance point; the paper
reports a consistent win over LeWM. This is the default regularizer for our LeWorldModel.

Per-slice Epps-Pulley statistic (projections compared to N(0,1), NOT standardised, so it
pins mean->0, var->1 AND Gaussianity at once):
    T = N * sum_t  trapz_t * e^{-t^2/2} * [ (mean_i cos(t p_i) - e^{-t^2/2})^2
                                            + (mean_i sin(t p_i))^2 ]
with t in [0, t_max] over n_knots trapezoid nodes.

Faithful to the official sources: galilai-group/lejepa (univariate/epps_pulley.py,
multivariate/slicing.py), lucas-maes/le-wm (module.py), intcomp/Sub-JEPA (subjepa.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, n_slices: int = 1024, n_knots: int = 17, t_max: float = 3.0) -> None:
        super().__init__()
        self.n_slices = n_slices
        t = torch.linspace(0.0, t_max, n_knots, dtype=torch.float32)
        dt = t_max / (n_knots - 1)
        w = torch.full((n_knots,), 2.0 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt                                          # trapezoid, endpoints halved
        phi = torch.exp(-t.square() / 2.0)                  # standard-normal CF (real part)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)            # quadrature weights x Gaussian window

    def forward(self, z: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """z: (..., N, d) embedding batch -> scalar regulariser (lower = closer to N(0,I))."""
        d = z.shape[-1]
        a = torch.randn(d, self.n_slices, device=z.device, dtype=z.dtype, generator=generator)
        a = a / a.norm(p=2, dim=0, keepdim=True)            # unit-sphere slice directions
        proj = z @ a                                        # (..., N, S)
        xt = proj.unsqueeze(-1) * self.t                    # (..., N, S, K)
        err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()  # (..., S, K)
        stat = (err @ self.weights) * proj.shape[-2]        # (..., S)  -- the *N scaling
        return stat.mean()


class MultiSubspaceSIGReg(nn.Module):
    """Sub-JEPA's Subspace Gaussian Regularization. Faithful to intcomp/Sub-JEPA subjepa.py
    (MultiSubspaceSIGReg, default init_mode='orthogonal_frozen'). The trainable-projection
    mode + orthogonality_loss are omitted (theta defaults to 0, i.e. off, in the LeWM config)."""

    def __init__(self, embed_dim: int, num_subspaces: int = 32, subspace_dim: int | None = None,
                 n_knots: int = 17, num_proj: int = 256, t_max: float = 3.0) -> None:
        super().__init__()
        if subspace_dim is None:
            if embed_dim % num_subspaces != 0:
                raise ValueError(f"embed_dim {embed_dim} must be divisible by num_subspaces "
                                 f"{num_subspaces} (or pass subspace_dim)")
            subspace_dim = embed_dim // num_subspaces
        self.num_subspaces, self.subspace_dim, self.num_proj = num_subspaces, subspace_dim, num_proj
        # Epps-Pulley quadrature buffers (same as SIGReg)
        t = torch.linspace(0.0, t_max, n_knots, dtype=torch.float32)
        dt = t_max / (n_knots - 1)
        w = torch.full((n_knots,), 2.0 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w * phi)
        # K frozen orthonormal projections (K, d', D): QR of randn, rows orthonormal
        mats = [torch.linalg.qr(torch.randn(embed_dim, subspace_dim), mode="reduced")[0].transpose(0, 1)
                for _ in range(num_subspaces)]
        self.register_buffer("projection_matrices", torch.stack(mats, dim=0))   # (K, d', D)

    def forward(self, emb: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """emb: (B, T, D) -> scalar. Project to K subspaces, Epps-Pulley 1D-slice each."""
        proj = torch.einsum("btd,ked->btke", emb, self.projection_matrices)
        proj = proj.permute(2, 1, 0, 3).contiguous()        # (K, T, B, d')
        a = torch.randn(proj.shape[0], self.subspace_dim, self.num_proj,
                        device=emb.device, dtype=emb.dtype, generator=generator)
        a = a / a.norm(p=2, dim=1, keepdim=True)             # unit slices within each subspace
        xt = torch.einsum("ktbd,kdn->ktbn", proj, a).unsqueeze(-1) * self.t   # (K,T,B,N,knots)
        err = (xt.cos().mean(2) - self.phi).square() + xt.sin().mean(2).square()  # (K,T,N,knots)
        stat = (err @ self.weights) * proj.shape[2]          # (K, T, N)  -- *B
        return stat.mean()
