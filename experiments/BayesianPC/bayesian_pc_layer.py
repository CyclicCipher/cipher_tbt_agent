"""
Bayesian Predictive Coding Layer

Implements BPC following Algorithm 1 from:
"Bayesian Predictive Coding" (Tschantz et al., 2025, arXiv:2503.24016)

Key differences from standard PC:
- Value nodes: MAP estimates (scalars), optimized via gradient descent
- Weights: Matrix Normal Wishart posterior q(W, Σ | M, V, Ψ, ν)
- Learning: Closed-form Bayesian updates (Equation 7)
- Architecture: z = W·f(z_{l-1}) [weights OUTSIDE activation]

Psi convention note:
    The paper specifies Ψ=1000 (Appendix F.1), but this uses Inverse Wishart
    semantics where large Ψ = vague prior on covariance Σ. The MNW formulation
    actually places a Wishart on the PRECISION Σ^{-1}, where E[Σ^{-1}] = νΨ.
    With Ψ=1000 and ν=130 this gives E[Σ^{-1}]=130,000 (catastrophically stiff).

    We accept Ψ in the paper's IW convention (large = vague) and convert
    internally: Ψ_wishart = 1/(ν * Ψ_iw) so that E[Σ^{-1}] = 1/Ψ_iw.
    With the paper's Ψ_iw=1000, this gives E[Σ^{-1}] = 0.001*I (very vague).

    We MUST keep Wishart on precision internally because it is conjugate to the
    Gaussian likelihood — this is what enables the closed-form Hebbian updates.
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
        prior_M_scale: float = 0.0,        # Prior mean scale
        prior_V_scale: float = 10.0,       # Prior column covariance scale
        prior_Psi_iw_scale: float = 1000.0, # Inverse Wishart convention (paper default)
        prior_nu: Optional[int] = None,    # Prior degrees of freedom
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Set prior degrees of freedom: ν = d_y + 2 (paper default)
        if prior_nu is None:
            prior_nu = out_features + 2
        self.prior_nu = prior_nu

        # Convert Ψ from Inverse Wishart convention to Wishart-on-precision
        # Paper says Ψ_iw=1000 meaning "vague prior on covariance Σ"
        # Internally we need Ψ_w for Wishart on Σ^{-1} where E[Σ^{-1}] = ν·Ψ_w
        # We want E[Σ^{-1}] ≈ 1/Ψ_iw (large Ψ_iw → low precision → vague)
        # So Ψ_w = 1/(ν * Ψ_iw)
        prior_Psi_w_scale = 1.0 / (prior_nu * prior_Psi_iw_scale)

        # Prior mean matrix M^(0) (out_features x in_features)
        M_prior = torch.zeros(out_features, in_features) + prior_M_scale

        # Prior column covariance V^(0) (in_features x in_features)
        V_prior = torch.eye(in_features) * prior_V_scale
        V_inv_prior = torch.eye(in_features) / prior_V_scale

        # Prior Wishart scale Ψ^(0) (out_features x out_features)
        # Using converted Wishart scale internally
        Psi_prior = torch.eye(out_features) * prior_Psi_w_scale
        Psi_inv_prior = torch.eye(out_features) / prior_Psi_w_scale

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
        #
        # From Appendix F.1: "All initial estimates of the posterior natural parameters η
        # were set to the same as the prior, besides M which uses the initialisation described for W."
        #
        # W initialization: "We initialized the linear weights W of shape (out_features, in_features)
        # from a uniform distribution U(−√k, √k), where k = 1/in_features"

        # Initialize posterior M from uniform distribution (NOT zeros like prior)
        k = 1.0 / in_features
        M_init = torch.zeros(out_features, in_features).uniform_(-torch.sqrt(torch.tensor(k)),
                                                                   torch.sqrt(torch.tensor(k)))

        # Convert initialized (M_init, V_prior, Psi_prior, prior_nu) to natural parameters
        # η = [V^{-1}, MV^{-1}, Φ + MV^{-1}M^T, ν - d_y + d_x - 1]
        eta1_init = V_inv_prior  # Same as prior (V^{-1})
        eta2_init = M_init @ V_inv_prior  # MV^{-1} with initialized M
        eta3_init = Psi_inv_prior + M_init @ V_inv_prior @ M_init.T  # Φ + MV^{-1}M^T with initialized M
        eta4_init = prior_nu - out_features + in_features - 1  # Same as prior

        self.register_buffer('eta1', eta1_init)
        self.register_buffer('eta2', eta2_init)
        self.register_buffer('eta3', eta3_init)
        self.register_buffer('eta4', torch.tensor(eta4_init, dtype=torch.float32))

        # NOTE: No separate bias parameter
        # Bias is handled by augmenting input with a column of 1s
        # This makes bias part of the weight matrix W, enabling Bayesian treatment
        # The last column of W acts as the bias vector

        # Value nodes (created during forward pass)
        self._x = None  # Will be nn.Parameter during training
        self._energy = None
        self._uncertainty_term = None

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
            # Training: Initialize value nodes at prediction E[W] @ x
            # Re-initialize every forward call so the optimizer in the trainer
            # creates a fresh optimizer each batch (see _create_optimizer_x at t==0).
            if self._x is None:
                W_mean, _ = self.get_expected_W_and_Sigma()
                mu = F.linear(x, W_mean, bias=None)
                self._x = nn.Parameter(mu.clone().detach(), requires_grad=True)

            # Compute energy
            self._compute_energy(x)

            return self._x
        else:
            # Testing: Use expected weights
            W_mean, _ = self.get_expected_W_and_Sigma()
            return F.linear(x, W_mean, bias=None)

    def _compute_energy(self, x: torch.Tensor):
        """Compute precision-weighted prediction error (Equation 5).

        E_l = (1/N) * Σ_n [0.5 * (z_n - Mf_n)^T (νΨ) (z_n - Mf_n) + 0.5 * d_y * f_n^T V f_n]

        Normalized to per-sample mean so it's on the same scale as F.cross_entropy(reduction='mean').

        The uncertainty term (0.5 * d_y * f^T V f) depends on the INPUT x, not the
        value node self._x. Its gradient w.r.t. self._x is zero, so it does not
        contribute to inference. We store it separately for diagnostics but exclude
        it from the energy used for inference gradient computation.
        """
        batch_size = x.size(0)

        # Get posterior parameters
        M, V, Psi, nu = self.natural_to_standard()

        # Expected precision: E[Σ^{-1}] = ν·Ψ  (from Wishart W(Ψ, ν))
        Sigma_inv_mean = nu * Psi

        # Prediction error: z - E[W]·x
        error = self._x - F.linear(x, M, bias=None)  # [batch_size, out_features]

        # Precision-weighted error: (1/N) Σ_n (z_n - Mx_n)^T E[Σ^{-1}] (z_n - Mx_n)
        # Mean over batch to match F.cross_entropy(reduction='mean')
        energy = 0.5 * torch.sum(
            error @ Sigma_inv_mean * error,
            dim=1  # Sum over output dimensions
        ).mean()  # MEAN over batch (not sum!)

        # Weight uncertainty term: (1/N) Σ_n 0.5 * d_y * f_n^T V f_n
        # This has ZERO gradient w.r.t. self._x (depends on input x only).
        # Stored separately — not included in energy for inference.
        Vx = x @ V
        self._uncertainty_term = 0.5 * self.out_features * (Vx * x).sum() / batch_size

        # Energy for inference only includes the term with gradient w.r.t. z
        self._energy = energy

    def energy(self) -> Optional[torch.Tensor]:
        """Get current energy value."""
        return self._energy

    def get_value_nodes(self):
        """Get value node parameters for optimizer."""
        if self._x is not None:
            return [self._x]
        return []

    def set_sample_x(self, mode: bool):
        """Enable/disable value node sampling.

        When enabling (True): Reset value nodes so they get re-initialized
        on the next forward pass. This prevents stale values from a previous batch.
        When disabling (False): Clear value nodes for eval mode.
        """
        self._x = None
        self._energy = None
        self._uncertainty_term = None


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
        prior_Psi_iw_scale: float = 1000.0,  # Inverse Wishart convention (paper default)
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
        # NOTE: Input is augmented with +1 for bias, so in_features is +1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BayesianPCLayer(
                in_features=layer_sizes[i] + 1,  # +1 for bias augmentation
                out_features=layer_sizes[i+1],
                prior_M_scale=prior_M_scale,
                prior_V_scale=prior_V_scale,
                prior_Psi_iw_scale=prior_Psi_iw_scale,
            )
            self.layers.append(layer)

    @staticmethod
    def _augment_with_bias(x: torch.Tensor) -> torch.Tensor:
        """Augment input with column of 1s for bias.

        Args:
            x: Input [batch_size, features]

        Returns:
            Augmented input [batch_size, features+1]
        """
        batch_size = x.size(0)
        ones = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=1)

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

        # Augment with 1 for bias (first layer)
        h = self._augment_with_bias(h)

        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h, sample_x=sample_x)
            h = self.activation(h)  # Apply activation after layer
            h = self._augment_with_bias(h)  # Augment for next layer

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
