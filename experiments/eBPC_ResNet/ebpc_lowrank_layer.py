"""
Low-Rank eBPC Layer — Scalable Bayesian Predictive Coding with Off-Diagonal Recovery

Uses low-rank V + diagonal Ψ approximation for MNW posterior:
- V^{-1} = diag(d) + U·U^T   [in_features] diagonal + [in_features, k] low-rank factor
- η3_diag: diagonal of η3 = Ψ^{-1} + M·V^{-1}·M^T  [out_features]
- η2 (MV^{-1}): out×in matrix (same as full)

The low-rank factor U captures the top-k input correlation modes, recovering
off-diagonal benefits that pure diagonal V loses.

Key design: We store diag(η3) and compute diag(Φ) = diag(η3) - diag(M·η1·M^T)
  via the Schur complement. This maintains MNW conjugacy — all four natural
  parameters (η1, η2, η3, η4) are updated jointly from the same sufficient
  statistics, providing self-correcting negative feedback on M growth.

  Quadratic constraint: The MNW block matrix [[η1, η2^T], [η2, η3]] must be PSD.
  Low-rank truncation of η1 can violate this (η1_approx < η1_true → Φ goes negative).
  Fix: inflate diagonal by spectral norm of dropped component so η1_approx ≥ η1_true.

  Efficient Schur complement diagonal:
    diag(M · η1 · M^T) = (M² @ d) + ((M @ U)²).sum(dim=1)
  Cost: O(out·in + out·k) — trivially fast.

Cost: O(k·in²) for Woodbury instead of O(in³) for full inverse.
Parameters: O(k·in + in + out·in + out) vs O(in² + out² + out·in) for full.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LowRankeBPCLayer(nn.Module):
    """eBPC layer with low-rank V + diagonal Ψ MNW posterior.

    Stored parameters:
      eta1_d:    diag part of V^{-1}           [in_features]
      eta1_U:    low-rank part of V^{-1}       [in_features, rank_k]
      eta2:      M · V^{-1}                    [out_features, in_features]
      eta3_diag: diag(Ψ^{-1} + M·V^{-1}·M^T)  [out_features]
      eta4:      ν - d_y + d_x - 1             [scalar]

    Standard parameters recovered via Woodbury + Schur complement:
      V = (diag(d) + U·U^T)^{-1}  [Woodbury]
      M = η2 @ V
      diag(Φ) = diag(η3) - diag(M·η1·M^T)  [Schur complement]
      Ψ_diag = 1 / clamp(diag(Φ), min=ε)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank_k: int = 20,
        prior_V_scale: float = 10.0,
        prior_Psi_iw_scale: float = 1000.0,
        prior_nu: Optional[int] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank_k = rank_k

        if prior_nu is None:
            prior_nu = out_features + 2
        self.prior_nu = prior_nu

        # Convert Ψ from IW convention: Ψ_wishart = 1/(ν * Ψ_iw)
        prior_Psi_w_scale = 1.0 / (prior_nu * prior_Psi_iw_scale)

        # Prior natural parameters
        # η1_prior = (1/V_scale) · I  → d_prior = 1/V_scale, U_prior = 0
        d_prior = torch.ones(in_features) / prior_V_scale
        U_prior = torch.zeros(in_features, rank_k)

        self.register_buffer('eta1_d_prior', d_prior)
        self.register_buffer('eta1_U_prior', U_prior)

        # η2_prior = M_prior · V^{-1}_prior = 0 (M_prior = 0)
        eta2_prior = torch.zeros(out_features, in_features)
        self.register_buffer('eta2_prior', eta2_prior)

        # η3_prior = Ψ^{-1}_prior + M_prior · V^{-1}_prior · M_prior^T
        # With M_prior = 0: η3_prior = Ψ^{-1}_prior = diag(1/Ψ_w_scale)
        psi_inv_prior_val = 1.0 / prior_Psi_w_scale
        eta3_diag_prior = torch.ones(out_features) * psi_inv_prior_val
        self.register_buffer('eta3_diag_prior', eta3_diag_prior)

        # η4_prior = ν - d_y + d_x - 1
        eta4_prior = prior_nu - out_features + in_features - 1
        self.register_buffer('eta4_prior', torch.tensor(eta4_prior, dtype=torch.float32))

        # Initialize posterior: M from uniform, V and Ψ from prior (Appendix F.1)
        k_init = 1.0 / in_features
        M_init = torch.zeros(out_features, in_features).uniform_(
            -torch.sqrt(torch.tensor(k_init)), torch.sqrt(torch.tensor(k_init))
        )

        # Posterior η1: start at prior (no off-diagonal correlations yet)
        self.register_buffer('eta1_d', d_prior.clone())
        self.register_buffer('eta1_U', U_prior.clone())

        # Posterior η2 = M_init · V^{-1}_prior = M_init · diag(d_prior)
        # (Since U_prior = 0, V^{-1}_prior = diag(d_prior))
        self.register_buffer('eta2', M_init * d_prior.unsqueeze(0))

        # Posterior η3_diag = Ψ^{-1}_prior + diag(M_init · diag(d_prior) · M_init^T)
        # = Ψ^{-1}_prior + (M_init^2 @ d_prior)
        eta3_diag_init = eta3_diag_prior + (M_init ** 2) @ d_prior
        self.register_buffer('eta3_diag', eta3_diag_init)

        # Posterior η4 = prior η4
        self.register_buffer('eta4', torch.tensor(eta4_prior, dtype=torch.float32))

        # Cache M for fast predict() — recomputed after each Hebbian update
        self.register_buffer('_M_cache', M_init.clone())
        self._cache_valid = True

    def _woodbury_V(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute V = (diag(d) + U·U^T)^{-1} via Woodbury identity.

        V = diag(1/d) - diag(1/d)·U·C^{-1}·U^T·diag(1/d)
        where C = I_k + U^T·diag(1/d)·U  [k×k, cheap to invert]

        Returns:
            d_inv: 1/d vector [in_features]
            C_inv: (I_k + U^T·diag(1/d)·U)^{-1}  [k, k]
            d_inv_U: diag(1/d) @ U  [in_features, k]
        """
        d_inv = 1.0 / self.eta1_d  # [in]

        # If U is all zeros (e.g., initialization), skip Woodbury — V = diag(1/d)
        if self.eta1_U.abs().max() < 1e-10:
            C_inv = torch.eye(self.rank_k, device=self.eta1_d.device, dtype=self.eta1_d.dtype)
            d_inv_U = torch.zeros_like(self.eta1_U)
            return d_inv, C_inv, d_inv_U

        d_inv_U = d_inv.unsqueeze(1) * self.eta1_U  # [in, k]
        C = torch.eye(self.rank_k, device=self.eta1_d.device, dtype=self.eta1_d.dtype)
        C = C + self.eta1_U.T @ d_inv_U  # [k, k]

        # Handle NaN in C
        if torch.isnan(C).any() or torch.isinf(C).any():
            self.eta1_U.data.zero_()
            C_inv = torch.eye(self.rank_k, device=self.eta1_d.device, dtype=self.eta1_d.dtype)
            d_inv_U = torch.zeros_like(self.eta1_U)
            return d_inv, C_inv, d_inv_U

        try:
            L = torch.linalg.cholesky(C)
            C_inv = torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            C = C + 1e-6 * torch.eye(self.rank_k, device=C.device, dtype=C.dtype)
            try:
                C_inv = torch.inverse(C)
            except torch.linalg.LinAlgError:
                self.eta1_U.data.zero_()
                C_inv = torch.eye(self.rank_k, device=self.eta1_d.device, dtype=self.eta1_d.dtype)
                d_inv_U = torch.zeros_like(self.eta1_U)
        return d_inv, C_inv, d_inv_U

    def _compute_M(self) -> torch.Tensor:
        """Compute M = η2 @ V via Woodbury.

        M = η2 @ [diag(1/d) - diag(1/d)·U·C^{-1}·U^T·diag(1/d)]
          = η2·diag(1/d) - (η2·diag(1/d)·U)·C^{-1}·(U^T·diag(1/d))^T
        """
        d_inv, C_inv, d_inv_U = self._woodbury_V()

        eta2_dinv = self.eta2 * d_inv.unsqueeze(0)  # [out, in]
        eta2_dinv_U = eta2_dinv @ self.eta1_U  # [out, k]
        correction = (eta2_dinv_U @ C_inv) @ d_inv_U.T  # [out, in]
        M = eta2_dinv - correction

        return M

    def _compute_schur_diag(self, M: torch.Tensor) -> torch.Tensor:
        """Compute diag(Φ) = diag(η3) - diag(M · η1 · M^T).

        Efficient formula using η1 = diag(d) + U·U^T:
          diag(M · η1 · M^T) = (M² @ d) + ((M @ U)²).sum(dim=1)
        Cost: O(out·in + out·k)
        """
        # diag(M · diag(d) · M^T) = (M^2) @ d
        M_eta1_diag = (M ** 2) @ self.eta1_d  # [out]

        # diag(M · U·U^T · M^T) = ((M @ U)^2).sum(dim=1)
        MU = M @ self.eta1_U  # [out, k]
        M_eta1_lowrank = (MU ** 2).sum(dim=1)  # [out]

        phi_diag = self.eta3_diag - M_eta1_diag - M_eta1_lowrank
        return phi_diag

    def _update_M_cache(self):
        """Recompute and cache M after Hebbian update."""
        self._M_cache.data = self._compute_M()
        self._cache_valid = True

    def natural_to_standard(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Convert to standard parameters (M, Psi_diag, nu).

        Psi_diag computed via Schur complement: diag(Φ) = diag(η3) - diag(M·η1·M^T).
        Clamped to a positive floor for numerical safety.
        """
        if not self._cache_valid:
            self._update_M_cache()
        M = self._M_cache

        phi_diag = self._compute_schur_diag(M)
        # With spectral norm inflation, η1_approx ≥ η1_true guarantees Phi > 0.
        # Tiny safety floor for numerical noise only (not load-bearing).
        phi_diag = torch.clamp(phi_diag, min=1e-30)
        Psi_diag = 1.0 / phi_diag  # [out_features]

        nu = self.eta4.item() + self.out_features - self.in_features + 1

        return M, Psi_diag, nu

    def get_expected_precision_diag(self) -> torch.Tensor:
        """E[Σ^{-1}] diagonal = ν · Ψ_diag."""
        _, Psi_diag, nu = self.natural_to_standard()
        return nu * Psi_diag  # [out_features]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward prediction using E[W] = M."""
        if not self._cache_valid:
            self._update_M_cache()
        return F.linear(x, self._M_cache, bias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval-mode forward."""
        return self.predict(x)


class LowRankeBPCNetwork(nn.Module):
    """eBPC network with low-rank V + diagonal Ψ posterior approximation."""

    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'relu',
        rank_k: int = 20,
        prior_V_scale: float = 10.0,
        prior_Psi_iw_scale: float = 1000.0,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.rank_k = rank_k

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = LowRankeBPCLayer(
                in_features=layer_sizes[i] + 1,  # +1 for bias augmentation
                out_features=layer_sizes[i + 1],
                rank_k=rank_k,
                prior_V_scale=prior_V_scale,
                prior_Psi_iw_scale=prior_Psi_iw_scale,
            )
            self.layers.append(layer)

    @staticmethod
    def _augment_with_bias(x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval-mode feedforward using expected weights."""
        h = self.activation(x)
        h = self._augment_with_bias(h)
        for layer in self.layers[:-1]:
            h = layer(h)
            h = self.activation(h)
            h = self._augment_with_bias(h)
        return self.layers[-1](h)

    def init_errors(self, x: torch.Tensor) -> list:
        """Initialize zero error tensors."""
        errors = []
        h = self.activation(x)
        h = self._augment_with_bias(h)
        with torch.no_grad():
            for layer in self.layers[:-1]:
                pred = layer.predict(h)
                errors.append(torch.zeros_like(pred, requires_grad=True))
                h = self.activation(pred)
                h = self._augment_with_bias(h)
        return errors

    def epc_forward(self, x: torch.Tensor, errors: list):
        """Training forward with error reparameterization."""
        h = self.activation(x)
        h = self._augment_with_bias(h)
        states = []

        for i, layer in enumerate(self.layers[:-1]):
            pred = layer.predict(h)
            s_i = pred + errors[i]
            states.append(s_i)
            h = self.activation(s_i)
            h = self._augment_with_bias(h)

        last_input = h
        output = self.layers[-1].predict(h)
        return output, states, last_input

    def get_natural_parameters(self):
        params = []
        for layer in self.layers:
            params.extend([layer.eta1_d, layer.eta1_U, layer.eta2,
                          layer.eta3_diag, layer.eta4])
        return params

    def get_uncertainties(self):
        """Sum of diag(V) per layer as uncertainty measure."""
        uncertainties = []
        for layer in self.layers:
            d_inv, C_inv, d_inv_U = layer._woodbury_V()
            correction = (d_inv_U @ C_inv) * d_inv_U  # [in, k]
            V_diag = d_inv - correction.sum(dim=1)  # [in]
            uncertainties.append(V_diag.sum().item())
        return uncertainties

    def get_rank_info(self):
        """Diagnostic: effective rank and singular values of U per layer."""
        info = []
        for layer in self.layers:
            U = layer.eta1_U
            if U.abs().max() < 1e-10:
                info.append({'effective_rank': 0, 'sv_ratio': 0.0})
            else:
                sv = torch.linalg.svdvals(U)
                effective_rank = (sv > sv[0] * 0.01).sum().item()
                sv_ratio = (sv[0] / sv[-1]).item() if sv[-1] > 0 else float('inf')
                info.append({'effective_rank': effective_rank, 'sv_ratio': sv_ratio})
        return info
