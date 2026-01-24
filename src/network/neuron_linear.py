"""
Linear two-compartment neuron for prospective learning.

Removes nonlinearities to enable direct solving of equilibrium states.
"""

import torch
import torch.nn as nn


class LinearTwoCompartmentNeuron(nn.Module):
    """
    Linear two-compartment neuron (no activation functions).

    This enables analytical solution of equilibrium states via
    block tridiagonal solving (true prospective learning).

    Attributes:
        num_neurons: Number of neurons in this layer
        apical_size: Input size from layer above
        basal_size: Input size from layer below
        gate: Learnable parameter controlling apical vs basal influence
    """

    def __init__(
        self,
        num_neurons: int,
        apical_size: int,
        basal_size: int,
        initial_gate: float = 0.5,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize linear two-compartment neuron.

        Args:
            num_neurons: Number of neurons in this layer
            apical_size: Dimension of input from layer above
            basal_size: Dimension of input from layer below
            initial_gate: Initial value for gate parameter (0-1)
            dtype: Data type for computations (float16 for efficiency)
        """
        super().__init__()

        self.num_neurons = num_neurons
        self.apical_size = apical_size
        self.basal_size = basal_size
        self.dtype = dtype

        # Apical weights (top-down predictions)
        self.W_apical = nn.Parameter(
            torch.randn(num_neurons, apical_size, dtype=dtype) * 0.01
        )

        # Basal weights (bottom-up signals)
        self.W_basal = nn.Parameter(
            torch.randn(num_neurons, basal_size, dtype=dtype) * 0.01
        )

        # Gate parameter (learnable)
        self.gate = nn.Parameter(
            torch.full((num_neurons,), initial_gate, dtype=dtype)
        )

        # Neuron state (maintained across inference iterations)
        self.register_buffer('state', torch.zeros(num_neurons, dtype=dtype))

    def forward(
        self,
        apical_input: torch.Tensor,
        basal_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute neuron state from apical and basal inputs (LINEAR).

        Args:
            apical_input: Top-down prediction from layer above (apical_size,)
            basal_input: Bottom-up signal from layer below (basal_size,)

        Returns:
            Neuron state (num_neurons,)
        """
        # Linear compartment activities (NO tanh)
        apical_activity = self.W_apical @ apical_input
        basal_activity = self.W_basal @ basal_input

        # Gate parameter (clamp to [0, 1])
        gate = torch.clamp(self.gate, 0.0, 1.0)

        # Integrate: state = gate * apical + (1 - gate) * basal
        self.state = gate * apical_activity + (1 - gate) * basal_activity

        return self.state

    def update_weights(
        self,
        apical_input: torch.Tensor,
        basal_input: torch.Tensor,
        error: torch.Tensor,
        lr: float
    ) -> None:
        """
        Update weights using local learning rules.

        Args:
            apical_input: Apical input used in forward pass
            basal_input: Basal input used in forward pass
            error: Prediction error
            lr: Learning rate
        """
        # Update weights via gradient descent on energy function
        # Energy minimization: ΔW = -η * ∂E/∂W
        with torch.no_grad():
            # Reshape error to (num_neurons, 1) for broadcasting
            error_col = error.unsqueeze(1)

            # Outer product: error (N,1) @ input (1,M) -> (N,M)
            # Using SUBTRACTION for gradient descent (not addition)
            self.W_apical -= lr * error_col * apical_input.unsqueeze(0)
            self.W_basal -= lr * error_col * basal_input.unsqueeze(0)

            # Update gate: LINEAR version (no recomputation of activities)
            # Gate moves toward compartment with smaller error
            apical_activity = self.W_apical @ apical_input
            basal_activity = self.W_basal @ basal_input

            gate_update = error * (apical_activity - basal_activity)
            self.gate -= lr * gate_update  # Note: subtract for gradient descent
            self.gate.clamp_(0.0, 1.0)

    def reset_state(self) -> None:
        """Reset neuron state to zero."""
        self.state.zero_()

    def get_state(self) -> torch.Tensor:
        """Get current neuron state."""
        return self.state

    def __repr__(self) -> str:
        return (f"LinearTwoCompartmentNeuron(num_neurons={self.num_neurons}, "
                f"apical_size={self.apical_size}, "
                f"basal_size={self.basal_size}, "
                f"dtype={self.dtype})")
