"""
Predictive coding layer with two-compartment neurons.

Each layer maintains its own state and computes prediction errors relative
to signals from adjacent layers.
"""

import torch
import torch.nn as nn
from typing import Optional

from .neuron import TwoCompartmentNeuron


class PredictiveCodingLayer(nn.Module):
    """
    A layer in the predictive coding network.

    Each layer:
    - Receives predictions from the layer above (apical)
    - Receives signals from the layer below (basal)
    - Computes prediction errors
    - Updates its state to minimize error

    Attributes:
        layer_index: Index of this layer in the network (0 = input)
        num_neurons: Number of neurons in this layer
        neurons: TwoCompartmentNeuron module
        state: Current layer state
        error: Current prediction error
    """

    def __init__(
        self,
        layer_index: int,
        num_neurons: int,
        input_size_below: int,
        input_size_above: int,
        initial_gate: float = 0.5,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize predictive coding layer.

        Args:
            layer_index: Index in network (0 = input layer)
            num_neurons: Number of neurons in this layer
            input_size_below: Size of input from layer below
            input_size_above: Size of input from layer above
            initial_gate: Initial gate parameter value
            dtype: Data type for computations
        """
        super().__init__()

        self.layer_index = layer_index
        self.num_neurons = num_neurons
        self.input_size_below = input_size_below
        self.input_size_above = input_size_above
        self.dtype = dtype

        # Two-compartment neurons
        self.neurons = TwoCompartmentNeuron(
            num_neurons=num_neurons,
            apical_size=input_size_above,
            basal_size=input_size_below,
            initial_gate=initial_gate,
            dtype=dtype
        )

        # Layer state and error (buffers, not parameters)
        self.register_buffer('state', torch.zeros(num_neurons, dtype=dtype))
        self.register_buffer('error', torch.zeros(num_neurons, dtype=dtype))

    def forward(
        self,
        input_from_below: torch.Tensor,
        input_from_above: torch.Tensor
    ) -> torch.Tensor:
        """
        Update layer state based on inputs from adjacent layers.

        Args:
            input_from_below: Signal from layer below (basal input)
            input_from_above: Prediction from layer above (apical input)

        Returns:
            Updated layer state
        """
        # Compute new state via neuron integration
        self.state = self.neurons(input_from_above, input_from_below)
        return self.state

    def compute_prediction_for_below(self) -> torch.Tensor:
        """
        Generate prediction for the layer below.

        This is the top-down prediction sent to the layer below's apical input.

        Returns:
            Prediction tensor (size matches layer below)
        """
        # Simple linear projection (can be enhanced later)
        # For MVP, just use the basal weights transposed
        return self.neurons.W_basal.T @ self.state

    def compute_signal_for_above(self) -> torch.Tensor:
        """
        Generate signal for the layer above.

        This is the bottom-up signal sent to the layer above's basal input.

        Returns:
            Signal tensor (size matches layer above)
        """
        # Simple linear projection
        # For MVP, just use the apical weights transposed
        return self.neurons.W_apical.T @ self.state

    def compute_error(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction error for this layer.

        Args:
            target: Target state (what this layer should represent)

        Returns:
            Prediction error
        """
        self.error = target - self.state
        return self.error

    def update_weights(
        self,
        input_from_below: torch.Tensor,
        input_from_above: torch.Tensor,
        lr: float,
        weight_decay: float = 0.01
    ) -> None:
        """
        Update weights using current error with L2 regularization.

        Args:
            input_from_below: Basal input used in forward pass
            input_from_above: Apical input used in forward pass
            lr: Learning rate
            weight_decay: L2 regularization coefficient (default 0.01)
        """
        self.neurons.update_weights(
            apical_input=input_from_above,
            basal_input=input_from_below,
            error=self.error,
            lr=lr,
            weight_decay=weight_decay
        )

    def reset_state(self) -> None:
        """Reset layer state and error to zero."""
        self.state.zero_()
        self.error.zero_()
        self.neurons.reset_state()

    def get_state(self) -> torch.Tensor:
        """Get current layer state."""
        return self.state

    def get_error(self) -> torch.Tensor:
        """Get current prediction error."""
        return self.error

    def get_total_error(self) -> float:
        """Get total prediction error (sum of squared errors)."""
        return (self.error ** 2).sum().item()

    def __repr__(self) -> str:
        return (f"PredictiveCodingLayer(index={self.layer_index}, "
                f"neurons={self.num_neurons}, "
                f"input_below={self.input_size_below}, "
                f"input_above={self.input_size_above})")
