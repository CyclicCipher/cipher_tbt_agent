"""
Linear predictive coding layer.

Uses linear neurons to enable prospective learning via direct equilibrium solving.
"""

import torch
import torch.nn as nn
from typing import Optional

from .neuron_linear import LinearTwoCompartmentNeuron


class LinearPredictiveCodingLayer(nn.Module):
    """
    A linear layer in the predictive coding network.

    Linear activations enable analytical equilibrium solving.
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
        Initialize linear predictive coding layer.

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

        # Linear two-compartment neurons
        self.neurons = LinearTwoCompartmentNeuron(
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
        # Compute new state via neuron integration (linear)
        self.state = self.neurons(input_from_above, input_from_below)
        return self.state

    def compute_prediction_for_below(self) -> torch.Tensor:
        """
        Generate prediction for the layer below.

        Returns:
            Prediction tensor (size matches layer below)
        """
        # Linear projection
        return self.neurons.W_basal.T @ self.state

    def compute_signal_for_above(self) -> torch.Tensor:
        """
        Generate signal for the layer above.

        Returns:
            Signal tensor (size matches layer above)
        """
        # Linear projection
        return self.neurons.W_apical.T @ self.state

    def compute_error(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction error for this layer.

        Args:
            target: Target state

        Returns:
            Prediction error
        """
        self.error = target - self.state
        return self.error

    def update_weights(
        self,
        input_from_below: torch.Tensor,
        input_from_above: torch.Tensor,
        lr: float
    ) -> None:
        """
        Update weights using current error.

        Args:
            input_from_below: Basal input used in forward pass
            input_from_above: Apical input used in forward pass
            lr: Learning rate
        """
        self.neurons.update_weights(
            apical_input=input_from_above,
            basal_input=input_from_below,
            error=self.error,
            lr=lr
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
        return (f"LinearPredictiveCodingLayer(index={self.layer_index}, "
                f"neurons={self.num_neurons}, "
                f"input_below={self.input_size_below}, "
                f"input_above={self.input_size_above})")
