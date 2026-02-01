"""
Bayesian Predictive Coding Layer Implementation

Extends standard PC with uncertainty quantification and KL divergence.

Key additions:
- Value nodes are distributions: (mean, variance)
- Energy = Accuracy (precision-weighted error) + Complexity (KL divergence)
- Implements variational Bayesian inference
- Approximates full Bayesian posterior

Based on:
- Friston's Free Energy Principle
- Variational inference in neural networks
- Rao & Ballard (1999) predictive coding
"""

import torch
import torch.nn as nn
import math


class BayesianPCLayer(nn.Module):
    """Bayesian Predictive Coding Layer - value nodes with uncertainty."""

    def __init__(self, prior_variance=1.0, min_variance=0.01, max_variance=10.0):
        """Initialize Bayesian PCLayer.

        Args:
            prior_variance: Prior variance for value nodes (regularization strength)
            min_variance: Minimum variance to prevent numerical instability (0.01, not 1e-6!)
            max_variance: Maximum variance to prevent precision collapse
        """
        super().__init__()

        # Value node mean (what we're uncertain about)
        self._x_mean = None

        # Value node variance (how uncertain we are)
        # Using log variance for numerical stability (always positive)
        self._x_log_var = None

        # Prior parameters (for KL divergence)
        self.prior_mean = 0.0  # Centered prior
        self.prior_variance = prior_variance
        self.min_variance = min_variance
        self.max_variance = max_variance

        # Energy accumulator
        self._energy = None

        # Flag to re-initialize at start of batch
        self._is_sample_x = False

    def energy(self):
        """Get the free energy held by this layer."""
        return self._energy

    def clear_energy(self):
        """Clear accumulated energy."""
        self._energy = None

    def get_x(self):
        """Get value node mean (for compatibility with optimizer)."""
        return self._x_mean

    def get_x_mean(self):
        """Get value node mean."""
        return self._x_mean

    def get_x_variance(self):
        """Get value node variance."""
        if self._x_log_var is None:
            return None
        return torch.exp(self._x_log_var).clamp(min=self.min_variance, max=self.max_variance)

    def get_statistics(self):
        """Get both mean and variance."""
        return self._x_mean, self.get_x_variance()

    def set_is_sample_x(self, value: bool):
        """Set flag to re-sample x on next forward pass."""
        self._is_sample_x = value

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Forward pass with uncertainty.

        Args:
            mu: Prediction from layer below (point estimate for now)

        Returns:
            During training: value node mean
            During eval: pass-through (mu)
        """
        if not self.training:
            # Eval mode: just pass through
            return mu

        # Training mode: Bayesian predictive coding

        # Initialize or re-initialize if needed
        if self._x_mean is None or self._is_sample_x or mu.shape != self._x_mean.shape:
            # Initialize mean from prediction
            self._x_mean = nn.Parameter(mu.clone(), requires_grad=True)

            # Initialize variance (log scale)
            # Start with moderate uncertainty
            initial_log_var = math.log(0.1)
            self._x_log_var = nn.Parameter(
                torch.full_like(mu, initial_log_var),
                requires_grad=True
            )

            self._is_sample_x = False

        # Get variance (clamped for stability)
        x_var = self.get_x_variance()

        # Compute precision-weighted prediction error (ACCURACY term)
        # E_accuracy = 0.5 * precision * (mu - x_mean)^2 + 0.5 * log(variance)
        error = mu - self._x_mean
        precision = 1.0 / x_var  # x_var already clamped to [min_variance, max_variance]

        accuracy_term = 0.5 * (precision * error ** 2).sum()
        entropy_term = 0.5 * torch.log(x_var).sum()

        # Compute KL divergence from prior (COMPLEXITY term)
        # KL[q(x) || p(x)] for Gaussians:
        # 0.5 * (log(var_prior/var_post) + var_post/var_prior + (mean_post - mean_prior)^2/var_prior - 1)

        var_ratio = x_var / self.prior_variance
        mean_diff = (self._x_mean - self.prior_mean) ** 2

        kl_divergence = 0.5 * (
            math.log(self.prior_variance) - torch.log(x_var) +  # x_var already clamped
            var_ratio +
            mean_diff / self.prior_variance -
            1.0
        ).sum()

        # Total variational free energy = Accuracy + Complexity
        self._energy = accuracy_term + entropy_term + kl_divergence

        # Return mean estimate
        return self._x_mean


class BayesianPCNetwork(nn.Module):
    """Bayesian Predictive Coding Network with uncertainty quantification."""

    def __init__(self, layer_sizes, activation='relu', prior_variance=1.0):
        """Initialize Bayesian PC Network.

        Args:
            layer_sizes: List of layer dimensions
            activation: Activation function
            prior_variance: Prior variance for regularization
        """
        super().__init__()

        self.prior_variance = prior_variance

        # Create layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add linear transformation
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            # Add Bayesian PC layer
            layers.append(BayesianPCLayer(prior_variance=prior_variance))

            # Add activation (except after last layer)
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights(activation)

    def _initialize_weights(self, activation):
        """Initialize weights properly for deep networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if activation == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif activation == 'tanh':
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through network."""
        return self.model(x)

    def get_pc_layers(self):
        """Get all Bayesian PCLayers in the network."""
        pc_layers = []
        for module in self.modules():
            if isinstance(module, BayesianPCLayer):
                pc_layers.append(module)
        return pc_layers

    def get_energies(self):
        """Get free energies from all PC layers."""
        energies = []
        for pc_layer in self.get_pc_layers():
            energy = pc_layer.energy()
            if energy is not None:
                energies.append(energy)
        return energies

    def get_value_nodes(self):
        """Get value node means from all PC layers."""
        value_nodes = []
        for pc_layer in self.get_pc_layers():
            x = pc_layer.get_x()
            if x is not None:
                value_nodes.append(x)

        # Also get variance parameters
        for pc_layer in self.get_pc_layers():
            if pc_layer._x_log_var is not None:
                value_nodes.append(pc_layer._x_log_var)

        return value_nodes

    def get_uncertainties(self):
        """Get uncertainty (variance) from each layer.

        Returns:
            List of average variance per layer
        """
        uncertainties = []
        for pc_layer in self.get_pc_layers():
            var = pc_layer.get_x_variance()
            if var is not None:
                uncertainties.append(var.mean().item())
        return uncertainties

    def set_sample_x(self, value: bool):
        """Set all PC layers to re-sample x on next forward."""
        for pc_layer in self.get_pc_layers():
            pc_layer.set_is_sample_x(value)

    def clear_energies(self):
        """Clear all accumulated energies."""
        for pc_layer in self.get_pc_layers():
            pc_layer.clear_energy()

    def get_network_parameters(self):
        """Get only network parameters (weights/biases), excluding value nodes.

        Returns:
            Generator of parameters (excludes value node means and variances)
        """
        # Get all value nodes to exclude
        value_node_set = set(self.get_value_nodes())

        # Yield only parameters that are NOT value nodes
        for param in self.parameters():
            if not any(param is x for x in value_node_set):
                yield param
