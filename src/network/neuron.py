"""
Two-compartment neuron for predictive coding.

Implements a simplified version of the two-compartment neuron architecture
described in the planning document. Temporal convolution deferred to Phase 3.
"""

import torch
import torch.nn as nn


class TwoCompartmentNeuron(nn.Module):
    """
    Two-compartment neuron with apical (top-down) and basal (bottom-up) inputs.

    The neuron integrates predictions from above (apical) with signals from
    below (basal) using a learnable gating parameter.

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
        Initialize two-compartment neuron.

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
        # Initialized to initial_gate, clamped to [0, 1] during forward pass
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
        Compute neuron state from apical and basal inputs.

        Args:
            apical_input: Top-down prediction from layer above (apical_size,)
            basal_input: Bottom-up signal from layer below (basal_size,)

        Returns:
            Neuron state (num_neurons,)
        """
        # Compute compartment activities
        apical_activity = torch.tanh(self.W_apical @ apical_input)
        basal_activity = torch.tanh(self.W_basal @ basal_input)

        # Gate parameter (clamp to [0, 1])
        gate = torch.clamp(self.gate, 0.0, 1.0)

        # Integrate: state = gate * apical + (1 - gate) * basal
        self.state = gate * apical_activity + (1 - gate) * basal_activity

        return self.state

    def compute_error(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction error.

        Args:
            target: Target signal for this layer

        Returns:
            Prediction error (num_neurons,)
        """
        return target - self.state

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
            error: Prediction error from compute_error()
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

            # Update gate: move toward compartment that was more accurate
            apical_activity = torch.tanh(self.W_apical @ apical_input)
            basal_activity = torch.tanh(self.W_basal @ basal_input)

            # Increase gate if apical was closer to target, decrease if basal was
            gate_update = error * (apical_activity - basal_activity)
            self.gate += lr * gate_update
            self.gate.clamp_(0.0, 1.0)

    def reset_state(self) -> None:
        """Reset neuron state to zero."""
        self.state.zero_()

    def get_state(self) -> torch.Tensor:
        """Get current neuron state."""
        return self.state

    def __repr__(self) -> str:
        return (f"TwoCompartmentNeuron(num_neurons={self.num_neurons}, "
                f"apical_size={self.apical_size}, "
                f"basal_size={self.basal_size}, "
                f"dtype={self.dtype})")
