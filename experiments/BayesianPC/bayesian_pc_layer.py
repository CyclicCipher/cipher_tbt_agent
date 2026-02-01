"""
Bayesian Predictive Coding Layer

Implements BPC following Algorithm 1 from:
"Bayesian Predictive Coding" (Tschantz et al., 2025, arXiv:2503.24016)

Key differences from standard PC:
- Value nodes: MAP estimates (scalars), optimized via gradient descent
- Weights: Matrix Normal Wishart posterior q(W, Σ | M, V, Ψ, ν)
- Learning: Closed-form Bayesian updates (Equation 7)
- Architecture: z = W·f(z_{l-1}) [weights OUTSIDE activation]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BayesianPCLayer(nn.Module):
    """Bayesian PC layer with Matrix Normal Wishart weight posterior.

    Architecture: z_l = W_l · f(z_{l-1}) + noise
    Note: Weights are OUTSIDE activation function (required for conjugacy)

    Weight posterior: q(W, Σ) = MatrixNormalWishart(W, Σ | M, V, Ψ, ν)
    Natural parameters: η = [V^{-1}, MV^{-1}, Ψ^{-1} + MV^{-1}M^T, ν - d_y + d_x - 1]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_M_scale: float = 0.0,      # Prior mean scale
        prior_V_scale: float = 10.0,     # Prior column covariance scale
        prior_Psi_scale: float = 1.0,    # Prior Wishart scale (paper uses 1000, but causes 258k precision!)
        prior_nu: Optional[int] = None,  # Prior degrees of freedom
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Set prior degrees of freedom
        if prior_nu is None:
            prior_nu = out_features + 2
        self.prior_nu = prior_nu

        # Initialize prior natural parameters η^(0)
        # Following paper: M^(0) = zeros, V^(0) = scale*I, Ψ^(0) = scale*I, ν^(0) = d_y + 2

        # Prior mean matrix M^(0) (out_features x in_features)
        M_prior = torch.zeros(out_features, in_features) + prior_M_scale

        # Prior column covariance V^(0) (in_features x in_features)
        V_prior = torch.eye(in_features) * prior_V_scale
        V_inv_prior = torch.eye(in_features) / prior_V_scale

        # Prior Wishart scale Ψ^(0) (out_features x out_features)
        Psi_prior = torch.eye(out_features) * prior_Psi_scale
        Psi_inv_prior = torch.eye(out_features) / prior_Psi_scale

        # Convert to natural parameters (Equation 27)
        # η = [V^{-1}, MV^{-1}, Φ + MV^{-1}M^T, ν - d_y + d_x - 1]
        # where Φ = Ψ^{-1}
        eta1_prior = V_inv_prior  # V^{-1}
        eta2_prior = M_prior @ V_inv_prior  # MV^{-1}
        eta3_prior = Psi_inv_prior + M_prior @ V_inv_prior @ M_prior.T  # Φ + MV^{-1}M^T
        eta4_prior = prior_nu - out_features + in_features - 1  # ν - d_y + d_x - 1

        # Store prior natural parameters (not learnable)
        self.register_buffer('eta1_prior', eta1_prior)
        self.register_buffer('eta2_prior', eta2_prior)
        self.register_buffer('eta3_prior', eta3_prior)
        self.register_buffer('eta4_prior', torch.tensor(eta4_prior, dtype=torch.float32))

        # Initialize posterior natural parameters (BUFFERS not Parameters!)
        # These are updated via closed-form Bayesian updates, NOT gradient descent
        # CRITICAL: Must be buffers to avoid gradient accumulation during inference
        self.register_buffer('eta1', eta1_prior.clone())
        self.register_buffer('eta2', eta2_prior.clone())
        self.register_buffer('eta3', eta3_prior.clone())
        self.register_buffer('eta4', self.eta4_prior.clone())

        # Bias term (standard point estimate for simplicity)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Value nodes (created during forward pass)
        self._x = None  # Will be nn.Parameter during training
        self._energy = None

    def natural_to_standard(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Convert natural parameters η to standard parameters (M, V, Ψ, ν).

        From Equation 27:
        η = [V^{-1}, MV^{-1}, Φ + MV^{-1}M^T, ν - d_y + d_x - 1]

        Returns:
            M: Mean matrix (out_features x in_features)
            V: Column covariance (in_features x in_features)
            Ψ: Wishart scale (out_features x out_features)
            ν: Degrees of freedom (scalar)
        """
        # V^{-1} = eta1
        V_inv = self.eta1
        V = torch.inverse(V_inv)

        # MV^{-1} = eta2
        M = self.eta2 @ V

        # Φ + MV^{-1}M^T = eta3
        # Φ = Ψ^{-1}, so Ψ = (eta3 - MV^{-1}M^T)^{-1}
        Phi = self.eta3 - M @ V_inv @ M.T
        Psi = torch.inverse(Phi)

        # ν - d_y + d_x - 1 = eta4
        nu = self.eta4.item() + self.out_features - self.in_features + 1

        return M, V, Psi, nu

    def get_expected_W_and_Sigma(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get expected weight matrix and covariance under posterior.

        Returns:
            E[W]: Expected weight matrix (out_features x in_features)
            E[Σ]: Expected observation covariance (out_features x out_features)
        """
        M, V, Psi, nu = self.natural_to_standard()

        # E[W] = M
        W_mean = M

        # E[Σ] = Ψ / (ν - d_y - 1) for Wishart distribution
        Sigma_mean = Psi / (nu - self.out_features - 1)

        return W_mean, Sigma_mean

    def get_expected_precision(self) -> torch.Tensor:
        """Get expected precision matrix E[Σ^{-1}].

        For Wishart distribution W(Ψ, ν): E[Σ^{-1}] = ν·Ψ
        """
        _, _, Psi, nu = self.natural_to_standard()
        return nu * Psi

    def forward(self, x: torch.Tensor, sample_x: bool = True) -> torch.Tensor:
        """Forward pass with value node optimization.

        Args:
            x: Input activations f(z_{l-1}) [batch_size, in_features]
               NOTE: Input is AFTER activation function (weights outside activation)
            sample_x: If True, initialize/use value nodes (training)
                     If False, use expected weights (testing)

        Returns:
            Output (value nodes during training, deterministic during testing)
        """
        batch_size = x.size(0)

        if self.training and sample_x:
            # Training: Initialize value nodes if needed
            if self._x is None or self._x.size(0) != batch_size:
                # Initialize at E[W] @ x + bias
                W_mean, _ = self.get_expected_W_and_Sigma()
                mu = F.linear(x, W_mean, self.bias)
                self._x = nn.Parameter(mu.clone().detach(), requires_grad=True)

            # Compute energy
            self._compute_energy(x)

            return self._x
        else:
            # Testing: Use expected weights
            W_mean, _ = self.get_expected_W_and_Sigma()
            return F.linear(x, W_mean, self.bias)

    def _compute_energy(self, x: torch.Tensor):
        """Compute precision-weighted prediction error (Equation 5).

        E_l = 0.5 * <(z - Wf(z_{l-1}))^T Σ^{-1} (z - Wf(z_{l-1}))>_{q(W,Σ)}

        Using Equations 17-18 for expectations under Matrix Normal Wishart.
        """
        batch_size = x.size(0)

        # Get posterior parameters
        M, V, Psi, nu = self.natural_to_standard()

        # Expected precision: E[Σ^{-1}] = ν·Ψ  (from Wishart W(Ψ, ν))
        Sigma_inv_mean = nu * Psi

        # Prediction error: z - E[W]·x
        # Using E[W] = M for the mean prediction
        error = self._x - F.linear(x, M, self.bias)  # [batch_size, out_features]

        # Precision-weighted error (dominant term)
        # (z - Mx)^T E[Σ^{-1}] (z - Mx)
        energy = 0.5 * torch.sum(
            error @ Sigma_inv_mean * error,  # Precision-weighted squared error
            dim=1  # Sum over output dimensions
        ).sum()  # Sum over batch

        # Weight uncertainty term from E[W^T Σ^{-1} W] = M^T ν Ψ M + d_y V
        # This adds: 0.5 * Tr(d_y V) * ||x||^2 = 0.5 * d_y * Tr(V) * ||x||^2
        # (encourages posterior concentration)
        x_norm_sq = (x ** 2).sum()  # Sum over batch and features
        uncertainty_term = 0.5 * self.out_features * torch.trace(V) * x_norm_sq

        self._energy = energy + uncertainty_term

    def energy(self) -> Optional[torch.Tensor]:
        """Get current energy value."""
        return self._energy

    def get_value_nodes(self):
        """Get value node parameters for optimizer."""
        if self._x is not None:
            return [self._x]
        return []

    def set_sample_x(self, mode: bool):
        """Enable/disable value node sampling."""
        if not mode:
            self._x = None
            self._energy = None


class BayesianPCNetwork(nn.Module):
    """Bayesian PC network with multiple layers.

    Architecture: z_l = W_l · f(z_{l-1})
    Weights are OUTSIDE activation function for conjugacy.
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'relu',
        prior_M_scale: float = 0.0,
        prior_V_scale: float = 10.0,
        prior_Psi_scale: float = 1000.0,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Create Bayesian PC layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BayesianPCLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                prior_M_scale=prior_M_scale,
                prior_V_scale=prior_V_scale,
                prior_Psi_scale=prior_Psi_scale,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, sample_x: bool = True) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input [batch_size, input_dim]
            sample_x: Whether to use value nodes (training) or expected weights (testing)

        Returns:
            Output [batch_size, output_dim]
        """
        # Apply activation to input (weights are outside activation)
        h = self.activation(x)

        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, sample_x=sample_x)
            h = self.activation(h)  # Apply activation after layer

        # Final layer (no activation after)
        output = self.layers[-1](h, sample_x=sample_x)

        return output

    def get_energies(self):
        """Get energy values from all layers."""
        return [layer.energy() for layer in self.layers if layer.energy() is not None]

    def get_value_nodes(self):
        """Get all value node parameters."""
        nodes = []
        for layer in self.layers:
            nodes.extend(layer.get_value_nodes())
        return nodes

    def get_natural_parameters(self):
        """Get all natural parameters (for learning)."""
        params = []
        for layer in self.layers:
            params.extend([layer.eta1, layer.eta2, layer.eta3, layer.eta4])
        return params

    def set_sample_x(self, mode: bool):
        """Enable/disable value node sampling for all layers."""
        for layer in self.layers:
            layer.set_sample_x(mode)

    def get_uncertainties(self):
        """Get weight uncertainties from all layers."""
        uncertainties = []
        for layer in self.layers:
            _, V, _, _ = layer.natural_to_standard()
            # Use trace of V as uncertainty measure
            uncertainties.append(torch.trace(V).item())
        return uncertainties
