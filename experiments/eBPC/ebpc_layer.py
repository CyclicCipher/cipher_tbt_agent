"""
eBPC Layer — Error-based Bayesian Predictive Coding

Combines:
- BPC (Tschantz et al. 2025): Matrix Normal Wishart weight posteriors, Hebbian updates
- ePC (Goemaere et al. 2025): Error reparameterization for fast inference

The layer holds the Bayesian weight posterior (MNW natural parameters).
Inference (error optimization) is handled externally by the trainer.

Architecture: z_l = W_l · f(z_{l-1}) + ε_l  [weights OUTSIDE activation, errors added]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class eBPCLayer(nn.Module):
    """eBPC layer with Matrix Normal Wishart weight posterior.

    Stores natural parameters η = [V^{-1}, MV^{-1}, Ψ^{-1}+MV^{-1}M^T, ν-d_y+d_x-1]
    for Bayesian weight posterior q(W, Σ | M, V, Ψ, ν).

    During training, the trainer manages error tensors ε externally.
    This layer provides:
    - predict(x): Forward using expected weights E[W] = M
    - get_expected_precision(): E[Σ^{-1}] = ν·Ψ for precision-weighted errors
    - natural_to_standard(): Convert η → (M, V, Ψ, ν)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_M_scale: float = 0.0,
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

        # Convert Ψ from Inverse Wishart convention to Wishart-on-precision
        prior_Psi_w_scale = 1.0 / (prior_nu * prior_Psi_iw_scale)

        # Prior parameters
        M_prior = torch.zeros(out_features, in_features) + prior_M_scale
        V_inv_prior = torch.eye(in_features) / prior_V_scale
        Psi_inv_prior = torch.eye(out_features) / prior_Psi_w_scale

        # Natural parameters (Equation 27)
        eta1_prior = V_inv_prior
        eta2_prior = M_prior @ V_inv_prior
        eta3_prior = Psi_inv_prior + M_prior @ V_inv_prior @ M_prior.T
        eta4_prior = prior_nu - out_features + in_features - 1

        self.register_buffer('eta1_prior', eta1_prior)
        self.register_buffer('eta2_prior', eta2_prior)
        self.register_buffer('eta3_prior', eta3_prior)
        self.register_buffer('eta4_prior', torch.tensor(eta4_prior, dtype=torch.float32))

        # Initialize posterior: M from uniform, rest from prior (Appendix F.1)
        k = 1.0 / in_features
        M_init = torch.zeros(out_features, in_features).uniform_(
            -torch.sqrt(torch.tensor(k)), torch.sqrt(torch.tensor(k))
        )

        eta1_init = V_inv_prior
        eta2_init = M_init @ V_inv_prior
        eta3_init = Psi_inv_prior + M_init @ V_inv_prior @ M_init.T
        eta4_init = prior_nu - out_features + in_features - 1

        self.register_buffer('eta1', eta1_init)
        self.register_buffer('eta2', eta2_init)
        self.register_buffer('eta3', eta3_init)
        self.register_buffer('eta4', torch.tensor(eta4_init, dtype=torch.float32))

    def natural_to_standard(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Convert natural parameters η to standard (M, V, Ψ, ν)."""
        V_inv = self.eta1
        V = torch.inverse(V_inv)
        M = self.eta2 @ V
        Phi = self.eta3 - M @ V_inv @ M.T
        Psi = torch.inverse(Phi)
        nu = self.eta4.item() + self.out_features - self.in_features + 1
        return M, V, Psi, nu

    def get_expected_precision(self) -> torch.Tensor:
        """E[Σ^{-1}] = ν·Ψ for Wishart."""
        _, _, Psi, nu = self.natural_to_standard()
        return nu * Psi

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute prediction E[W] @ x."""
        M, _, _, _ = self.natural_to_standard()
        return F.linear(x, M, bias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (eval mode): just predict."""
        return self.predict(x)


class eBPCNetwork(nn.Module):
    """eBPC network: MNW weight posteriors + ePC error reparameterization.

    Eval mode: standard feedforward using E[W].
    Training: trainer calls epc_forward() to build global computational graph.
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'relu',
        prior_M_scale: float = 0.0,
        prior_V_scale: float = 10.0,
        prior_Psi_iw_scale: float = 1000.0,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = eBPCLayer(
                in_features=layer_sizes[i] + 1,  # +1 for bias augmentation
                out_features=layer_sizes[i + 1],
                prior_M_scale=prior_M_scale,
                prior_V_scale=prior_V_scale,
                prior_Psi_iw_scale=prior_Psi_iw_scale,
            )
            self.layers.append(layer)

    @staticmethod
    def _augment_with_bias(x: torch.Tensor) -> torch.Tensor:
        """Append column of 1s for bias."""
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
        """Initialize zero error tensors with correct shapes.

        Runs a feedforward pass (no grad) to discover output shapes,
        then creates zero tensors with requires_grad=True.

        Returns list of L-1 error tensors (no error on output layer).
        """
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

    def epc_forward(self, x: torch.Tensor, errors: list) -> Tuple[torch.Tensor, list, torch.Tensor]:
        """Training forward with error reparameterization.

        s_i = M_i @ f(s_{i-1}) + ε_i for hidden layers
        output = M_L @ f(s_{L-1}) (no error on output)

        The output depends on ALL errors → global computational graph.

        Returns:
            output: Network prediction [batch, out_dim]
            states: List of hidden states s_i (for Hebbian update)
            last_input: Input to output layer f(s_{L-1}) augmented (for logging)
        """
        h = self.activation(x)
        h = self._augment_with_bias(h)
        states = []

        for i, layer in enumerate(self.layers[:-1]):
            pred = layer.predict(h)       # ŝ_i = M_i @ f(s_{i-1})
            s_i = pred + errors[i]         # s_i = ŝ_i + ε_i
            states.append(s_i)
            h = self.activation(s_i)
            h = self._augment_with_bias(h)

        last_input = h
        output = self.layers[-1].predict(h)
        return output, states, last_input

    def get_natural_parameters(self):
        """Get all natural parameters."""
        params = []
        for layer in self.layers:
            params.extend([layer.eta1, layer.eta2, layer.eta3, layer.eta4])
        return params

    def get_uncertainties(self):
        """Get Tr(V) per layer as uncertainty measure."""
        uncertainties = []
        for layer in self.layers:
            _, V, _, _ = layer.natural_to_standard()
            uncertainties.append(torch.trace(V).item())
        return uncertainties
