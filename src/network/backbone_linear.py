"""
Linear backbone network for prospective learning.

Uses linear activations to enable direct equilibrium solving.
"""

import torch
import torch.nn as nn
from typing import List

from .layer_linear import LinearPredictiveCodingLayer


class LinearBackboneNetwork(nn.Module):
    """
    Linear predictive coding backbone for prospective learning.

    With linear activations, equilibrium can be solved directly
    using block tridiagonal matrix operations.
    """

    def __init__(
        self,
        num_layers: int = 5,
        neurons_per_layer: int = 100,
        input_size: int = 1000,
        initial_gate: float = 0.5,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu"
    ):
        """
        Initialize linear backbone network.

        Args:
            num_layers: Total number of layers (including hidden layers)
            neurons_per_layer: Neurons in each hidden layer
            input_size: Dimension of input layer
            initial_gate: Initial gate value for all layers
            dtype: Data type for computations
            device: Device for computations
        """
        super().__init__()

        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_size = input_size
        self.dtype = dtype
        self.device = device

        # Build layers
        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            # Determine input sizes
            if i == 0:
                input_below = input_size  # Sensory input
            else:
                input_below = neurons_per_layer

            # All layers have same hidden size
            if i == num_layers - 2:
                input_above = neurons_per_layer  # Self-predict
            else:
                input_above = neurons_per_layer

            layer = LinearPredictiveCodingLayer(
                layer_index=i,
                num_neurons=neurons_per_layer,
                input_size_below=input_below,
                input_size_above=input_above,
                initial_gate=initial_gate,
                dtype=dtype
            )

            self.layers.append(layer)

        # Move to device
        self.to(device)

        # Input buffer (layer 0)
        self.register_buffer('input_buffer', torch.zeros(input_size, dtype=dtype))

    def forward(self, sensory_input: torch.Tensor, num_iterations: int = 50) -> torch.Tensor:
        """
        Forward pass with iterative inference to equilibrium.

        Linear networks converge faster, so we use more iterations.

        Args:
            sensory_input: Sensory input tensor (input_size,)
            num_iterations: Number of inference iterations (default 50 for linear)

        Returns:
            Final state of top layer
        """
        # Set input buffer
        self.input_buffer.copy_(sensory_input)

        # Iterative inference to reach equilibrium
        # Linear systems converge faster than nonlinear
        for _ in range(num_iterations):
            self._inference_step()

        return self.layers[-1].get_state()

    def _inference_step(self) -> None:
        """
        Single step of inference: update all layer states.

        Linear version - no nonlinearities means faster convergence.
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer receives sensory input
                input_below = self.input_buffer
            else:
                # Receive RAW state from layer below
                input_below = self.layers[i - 1].get_state()

            if i == len(self.layers) - 1:
                # Top layer predicts itself (self-prediction)
                input_above = layer.get_state()
            else:
                # Receive RAW state from layer above
                input_above = self.layers[i + 1].get_state()

            # Update layer state (linear activation)
            layer(input_below, input_above)

    def compute_reconstruction(self) -> torch.Tensor:
        """
        Reconstruct input from current network state.

        Returns:
            Reconstructed input (input_size,)
        """
        # First layer predicts the input
        return self.layers[0].compute_prediction_for_below()

    def compute_total_error(self) -> float:
        """
        Compute total prediction error across all layers.

        Returns:
            Sum of squared errors across all layers
        """
        # Error at input layer (sensory prediction error)
        reconstruction = self.compute_reconstruction()
        input_error = ((self.input_buffer - reconstruction) ** 2).sum()

        # Errors at hidden layers
        layer_errors = sum(layer.get_total_error() for layer in self.layers)

        return (input_error + layer_errors).item()

    def update_weights(self, lr: float) -> None:
        """
        Update all weights using local learning rules from prospective learning.

        In prospective learning/predictive coding:
        1. Each layer's error = its state - prediction from layer above (value error)
        2. Weights updated using local Hebbian rules: ΔW = lr * error * input
        3. No backpropagation or error projection through weights

        Args:
            lr: Learning rate
        """
        # Update each layer using value error (state - prediction_from_above)
        for i in range(len(self.layers)):
            layer = self.layers[i]

            # Get prediction from layer above
            if i == len(self.layers) - 1:
                # Top layer: no layer above, so use 0 as prediction
                prediction_from_above = torch.zeros_like(layer.get_state())
            else:
                prediction_from_above = self.layers[i + 1].compute_prediction_for_below()

            # Value error: difference between actual state and prediction from above
            layer_error = layer.get_state() - prediction_from_above
            layer.error = layer_error

            # Get inputs for weight update (use RAW states)
            if i == 0:
                # First layer receives sensory input from below
                input_from_below = self.input_buffer
            else:
                # Higher layers: use RAW state from layer below
                input_from_below = self.layers[i - 1].get_state()

            if i == len(self.layers) - 1:
                # Top layer: use own state
                input_from_above = layer.get_state()
            else:
                # Other layers: use RAW state from layer above
                input_from_above = self.layers[i + 1].get_state()

            # Update weights using local learning rule
            layer.update_weights(
                input_from_below=input_from_below,
                input_from_above=input_from_above,
                lr=lr
            )

    def reset_states(self) -> None:
        """Reset all layer states to zero."""
        for layer in self.layers:
            layer.reset_state()

    def get_layer_states(self) -> List[torch.Tensor]:
        """Get states of all layers."""
        return [layer.get_state() for layer in self.layers]

    def get_layer_errors(self) -> List[float]:
        """Get total error for each layer."""
        return [layer.get_total_error() for layer in self.layers]

    def __repr__(self) -> str:
        return (f"LinearBackboneNetwork(num_layers={self.num_layers}, "
                f"neurons_per_layer={self.neurons_per_layer}, "
                f"input_size={self.input_size}, "
                f"device={self.device})")
