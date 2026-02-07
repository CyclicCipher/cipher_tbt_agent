"""
Diagonal eBPC Layer — Scalable Bayesian Predictive Coding

Uses diagonal approximation for V and Ψ:
- V^{-1}: in_features vector (not in×in matrix)
- Ψ^{-1}: out_features vector (not out×out matrix)
- η2 (MV^{-1}): out×in matrix (same as weight matrix, unavoidable)

This reduces parameters from O(in² + out² + out·in) to O(in + out + out·in),
making ResNet-scale networks feasible.

What we lose: correlations between weight columns (V) and output dimensions (Ψ).
What we keep: per-weight uncertainty, per-output precision, closed-form Hebbian updates.

Supports both Linear and Conv2d operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiagonaleBPCLayer(nn.Module):
    """eBPC layer with diagonal MNW posterior.

    Natural parameters (diagonal):
      η1 = diag(V^{-1})          [in_features]
      η2 = M · diag(V^{-1})      [out_features, in_features]
      η3 = diag(Ψ^{-1} + M·diag(V^{-1})·M^T)  [out_features]
      η4 = ν - d_y + d_x - 1     [scalar]

    Standard parameters recovered:
      V = 1/η1  (element-wise)
      M = η2 / η1  (broadcast division)
      Ψ = 1/(η3 - sum(η2² / η1, dim=1))  (derived)
      ν = η4 + d_y - d_x + 1
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_V_scale: float = 10.0,
        prior_Psi_iw_scale: float = 1000.0,
        prior_nu: Optional[int] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if prior_nu is None:
            prior_nu = out_features + 2
        self.prior_nu = prior_nu

        # Convert Ψ from IW convention: Ψ_wishart = 1/(ν * Ψ_iw)
        prior_Psi_w_scale = 1.0 / (prior_nu * prior_Psi_iw_scale)

        # Prior natural parameters (diagonal)
        v_inv_prior = torch.ones(in_features) / prior_V_scale  # diag(V^{-1})
        psi_inv_prior = torch.ones(out_features) / prior_Psi_w_scale  # diag(Ψ^{-1})

        # η1_prior = diag(V^{-1}_prior)
        self.register_buffer('eta1_prior', v_inv_prior)

        # η2_prior = M_prior · diag(V^{-1}_prior) = 0 (M_prior = 0)
        eta2_prior = torch.zeros(out_features, in_features)
        self.register_buffer('eta2_prior', eta2_prior)

        # η3_prior = diag(Ψ^{-1}_prior + M_prior · diag(V^{-1}) · M_prior^T)
        # With M_prior = 0: η3_prior = diag(Ψ^{-1}_prior)
        self.register_buffer('eta3_prior', psi_inv_prior)

        # η4_prior = ν - d_y + d_x - 1
        eta4_prior = prior_nu - out_features + in_features - 1
        self.register_buffer('eta4_prior', torch.tensor(eta4_prior, dtype=torch.float32))

        # Initialize posterior (M from uniform, rest from prior — Appendix F.1)
        k = 1.0 / in_features
        M_init = torch.zeros(out_features, in_features).uniform_(
            -torch.sqrt(torch.tensor(k)), torch.sqrt(torch.tensor(k))
        )

        # Posterior η1 = prior η1 (V starts at prior)
        self.register_buffer('eta1', v_inv_prior.clone())

        # Posterior η2 = M_init · diag(V^{-1}_prior)
        self.register_buffer('eta2', M_init * v_inv_prior.unsqueeze(0))  # broadcast [out, in]

        # Posterior η3 = diag(Ψ^{-1}_prior) + diag(M_init · diag(V^{-1}) · M_init^T)
        # = diag(Ψ^{-1}) + sum(M_init² · V^{-1}, dim=1)
        eta3_init = psi_inv_prior + (M_init ** 2 * v_inv_prior.unsqueeze(0)).sum(dim=1)
        self.register_buffer('eta3', eta3_init)

        # Posterior η4 = prior η4
        self.register_buffer('eta4', torch.tensor(eta4_prior, dtype=torch.float32))

    def natural_to_standard(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Convert natural parameters to standard (M, V_diag, Psi_diag, nu).

        Returns diagonal V and Ψ as vectors, not matrices.
        """
        # V_diag = 1 / η1  (element-wise)
        V_diag = 1.0 / self.eta1  # [in_features]

        # M = η2 · diag(V) = η2 / η1  (broadcast)
        M = self.eta2 * V_diag.unsqueeze(0)  # [out, in]

        # Ψ_diag = 1 / (η3 - sum(η2² / η1, dim=1))
        # = 1 / (η3 - sum(M · η2, dim=1))
        Phi_diag = self.eta3 - (M * self.eta2).sum(dim=1)  # [out_features]
        Psi_diag = 1.0 / Phi_diag  # [out_features]

        # ν = η4 + d_y - d_x + 1
        nu = self.eta4.item() + self.out_features - self.in_features + 1

        return M, V_diag, Psi_diag, nu

    def get_expected_precision_diag(self) -> torch.Tensor:
        """E[Σ^{-1}] diagonal = ν · Ψ_diag."""
        _, _, Psi_diag, nu = self.natural_to_standard()
        return nu * Psi_diag  # [out_features]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward prediction using E[W] = M."""
        M, _, _, _ = self.natural_to_standard()
        return F.linear(x, M, bias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval-mode forward."""
        return self.predict(x)


class DiagonaleBPCNetwork(nn.Module):
    """eBPC network with diagonal posterior approximation.

    Supports both MLP and ResNet architectures.
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'relu',
        prior_V_scale: float = 10.0,
        prior_Psi_iw_scale: float = 1000.0,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        if activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = DiagonaleBPCLayer(
                in_features=layer_sizes[i] + 1,  # +1 for bias augmentation
                out_features=layer_sizes[i + 1],
                prior_V_scale=prior_V_scale,
                prior_Psi_iw_scale=prior_Psi_iw_scale,
            )
            self.layers.append(layer)

    @staticmethod
    def _augment_with_bias(x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Eval-mode feedforward."""
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
            params.extend([layer.eta1, layer.eta2, layer.eta3, layer.eta4])
        return params

    def get_uncertainties(self):
        """Sum of V_diag per layer (analogous to Tr(V) for full matrices)."""
        uncertainties = []
        for layer in self.layers:
            _, V_diag, _, _ = layer.natural_to_standard()
            uncertainties.append(V_diag.sum().item())
        return uncertainties
