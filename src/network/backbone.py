"""
Backbone network for predictive coding.

Implements a stack of predictive coding layers with prospective learning.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .layer import PredictiveCodingLayer


class BackboneNetwork(nn.Module):
    """
    Predictive coding backbone network.

    A stack of layers that learn to predict sensory inputs through
    bidirectional prediction error minimization.

    Attributes:
        num_layers: Number of layers (including input layer)
        neurons_per_layer: Neurons in each hidden layer
        input_size: Size of sensory input
        layers: List of PredictiveCodingLayer modules
    """

    def __init__(
        self,
        num_layers: int,
        neurons_per_layer: int,
        input_size: int,
        initial_gate: float = 0.5,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        """
        Initialize backbone network.

        Args:
            num_layers: Number of layers (5 for MVP)
            neurons_per_layer: Neurons per hidden layer (1500 for full, can reduce for testing)
            input_size: Dimension of sensory input (320*320*3 = 307200 for foveal)
            initial_gate: Initial gate parameter for all neurons
            dtype: Data type for computations
            device: Device to run on ("cuda" or "cpu")
        """
        super().__init__()

        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_size = input_size
        self.initial_gate = initial_gate
        self.dtype = dtype
        self.device = device

        # Build layer stack
        self.layers = nn.ModuleList()

        for i in range(1, num_layers):  # Skip layer 0 (input is not a learnable layer)
            # Determine input sizes
            if i == 1:
                # First hidden layer receives input from sensory layer
                input_below = input_size
            else:
                input_below = neurons_per_layer

            if i == num_layers - 1:
                # Top layer (no layer above)
                input_above = neurons_per_layer  # Self-predict
            else:
                input_above = neurons_per_layer

            layer = PredictiveCodingLayer(
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

    def forward(self, sensory_input: torch.Tensor, num_iterations: int = 5) -> torch.Tensor:
        """
        Forward pass with iterative inference.

        Args:
            sensory_input: Sensory input tensor (input_size,)
            num_iterations: Number of inference iterations to reach equilibrium

        Returns:
            Final state of top layer
        """
        # Set input buffer
        self.input_buffer.copy_(sensory_input)

        # Iterative inference to reach equilibrium
        for _ in range(num_iterations):
            self._inference_step()

        return self.layers[-1].get_state()

    def _inference_step(self) -> None:
        """
        Single step of inference: update all layer states.

        Updates proceed in both directions:
        - Bottom-up: signals flow from input to top
        - Top-down: predictions flow from top to input
        """
        # Bottom-up pass
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer receives input
                input_below = self.input_buffer
            else:
                # Receive signal from layer below
                input_below = self.layers[i - 1].compute_signal_for_above()

            if i == len(self.layers) - 1:
                # Top layer predicts itself (self-prediction)
                input_above = layer.get_state()
            else:
                # Receive prediction from layer above
                input_above = self.layers[i + 1].compute_prediction_for_below()

            # Update layer state
            layer(input_below, input_above)

    def compute_reconstruction(self) -> torch.Tensor:
        """
        Reconstruct input from current network state.

        The reconstruction is the network's prediction of the input.

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
        Update all weights using current layer states and errors.

        In predictive coding, each layer predicts its input from below.
        Errors are computed in the input space, then used to update weights.

        Args:
            lr: Learning rate
        """
        # First, compute prediction error at the input level
        reconstruction = self.compute_reconstruction()
        input_error = self.input_buffer - reconstruction

        # Update first layer using input prediction error
        # The error is in input space (input_size), but we need to map it to affect layer state
        # For simplicity in MVP, use the basal weights to project error to layer space
        first_layer = self.layers[0]

        if len(self.layers) == 1:
            input_above = first_layer.get_state()  # Self-predict for single layer
        else:
            input_above = self.layers[1].compute_prediction_for_below()

        # Project input error to layer space for gradient computation
        # error_in_layer_space = W_basal @ input_error
        layer_error = first_layer.neurons.W_basal @ input_error
        first_layer.error = layer_error

        first_layer.update_weights(
            apical_input=input_above,
            basal_input=self.input_buffer,
            lr=lr
        )

        # Update higher layers using state prediction errors
        for i in range(1, len(self.layers)):
            layer = self.layers[i]

            # Input from below
            input_below = self.layers[i - 1].compute_signal_for_above()

            # Input from above
            if i == len(self.layers) - 1:
                input_above = layer.get_state()  # Top layer self-predicts
            else:
                input_above = self.layers[i + 1].compute_prediction_for_below()

            # Error: difference between actual and predicted state of layer below
            actual_state_below = self.layers[i - 1].get_state()
            predicted_state_below = layer.compute_prediction_for_below()

            # Error in the space of layer below (neurons_per_layer)
            error_below = actual_state_below - predicted_state_below

            # Project error to current layer space
            layer_error = layer.neurons.W_basal @ error_below
            layer.error = layer_error

            layer.update_weights(input_below, input_above, lr)

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
        return (f"BackboneNetwork(num_layers={self.num_layers}, "
                f"neurons_per_layer={self.neurons_per_layer}, "
                f"input_size={self.input_size}, "
                f"device={self.device})")
