"""
Backbone network for predictive coding.

Implements a stack of predictive coding layers with prospective learning.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .layer import PredictiveCodingLayer
from ..optimizers import Muon


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
        device: str = "cuda",
        inference_lr: float = 0.1,
        temperature: float = 0.0,
        use_muon: bool = False,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        saturation_penalty: float = 0.01,
        activity_target: float = 0.3
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
            inference_lr: Learning rate for inference phase (state updates)
            temperature: Noise level for Langevin dynamics (0.0 = no noise, >0 = simulated annealing)
            use_muon: Whether to use Muon optimizer (default: False, uses manual updates)
            muon_lr: Muon optimizer learning rate (default: 0.02)
            muon_momentum: Muon optimizer momentum (default: 0.95)
            saturation_penalty: Penalty for saturated activations (default: 0.01)
            activity_target: Target mean activation level (default: 0.3)
        """
        super().__init__()

        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_size = input_size
        self.initial_gate = initial_gate
        self.dtype = dtype
        self.device = device
        self.inference_lr = inference_lr
        self.temperature = temperature
        self.use_muon = use_muon
        self.saturation_penalty = saturation_penalty
        self.activity_target = activity_target

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

        # Initialize Muon optimizer if requested
        self.optimizer = None
        if use_muon:
            self.optimizer = Muon(
                self.parameters(),
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=0.01  # Keep consistent with manual updates
            )

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

        # Initialize states with feedforward pass (critical for higher layers to activate)
        # Without this, states start at zero and higher layers receive zero input
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_below = self.input_buffer
            else:
                input_below = self.layers[i - 1].get_state()

            # Initialize state to bottom-up prediction
            layer.state.copy_(torch.tanh(layer.neurons.W_basal @ input_below))

        # Iterative inference to reach equilibrium
        for _ in range(num_iterations):
            self._inference_step()

        return self.layers[-1].get_state()

    def _inference_step(self) -> None:
        """
        Single step of inference: update states via gradient descent on free energy.

        Implements proper predictive coding dynamics from Millidge et al. (2022), Eq. 1:
        ẋ_l = -∂F/∂x_l = -ε_l + ε_{l+1} · f'(W_{l+1}x_l)W^T_{l+1}

        Where ε_l = x_l - f(W_l x_{l-1}) is the bottom-up prediction error.

        Note: W_l in paper corresponds to layer[l].neurons.W_basal in our code
        (both connect layer l-1 to layer l).
        """
        # Compute prediction errors for all layers
        # ε_l = x_l - f(W_basal @ x_{l-1})
        errors = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_below = self.input_buffer
            else:
                input_below = self.layers[i - 1].get_state()

            # Bottom-up prediction: f(W_basal @ x_{l-1})
            bottom_up_prediction = torch.tanh(layer.neurons.W_basal @ input_below)

            # Prediction error: ε_l = x_l - bottom_up_prediction
            error = layer.get_state() - bottom_up_prediction
            errors.append(error)

        # Update states via gradient descent: ẋ_l = -∂F/∂x_l
        for i, layer in enumerate(self.layers):
            # Gradient term 1: -ε_l (local error)
            gradient = -errors[i]

            # Gradient term 2: +ε_{l+1} · f'(W_{l+1}x_l) · W^T_{l+1} (feedback from above)
            if i < len(self.layers) - 1:
                current_state = layer.get_state()

                # W_{l+1} @ x_l (forward connection from this layer to next)
                weighted_input = self.layers[i + 1].neurons.W_basal @ current_state

                # f'(W_{l+1} @ x_l) where f = tanh
                tanh_derivative = 1 - torch.tanh(weighted_input) ** 2

                # ε_{l+1} · f'(...) (element-wise)
                error_times_deriv = errors[i + 1] * tanh_derivative

                # Project back through W^T_{l+1}
                # Shape: (num_neurons_l, num_neurons_l+1) @ (num_neurons_l+1,) = (num_neurons_l,)
                feedback = self.layers[i + 1].neurons.W_basal.T @ error_times_deriv
                gradient += feedback

            # Update: x_l += lr * (gradient of -F) = x_l -= lr * (gradient of F)
            # Since gradient = -∂F/∂x_l, we do: x_l += lr * gradient
            new_state = layer.get_state() + self.inference_lr * gradient

            # Add Langevin noise for simulated annealing / escape from local minima
            # Analogous to "heating" the system like protein denaturation
            if self.temperature > 0:
                noise = torch.randn_like(new_state) * (self.temperature ** 0.5)
                new_state = new_state + noise

            layer.state.copy_(new_state)

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

    def update_weights(self, lr: float = None, weight_decay: float = 0.01) -> None:
        """
        Update all weights using local learning rules from prospective learning.

        In prospective learning/predictive coding:
        1. Each layer's error = its state - prediction from layer above (value error)
        2. Weights updated using local Hebbian rules: ΔW = lr * error * input - decay * W
        3. No backpropagation or error projection through weights

        If use_muon=True, uses Muon optimizer with momentum and adaptive learning rates.
        Otherwise uses manual updates with fixed learning rate.

        Args:
            lr: Learning rate (only used if not using Muon)
            weight_decay: L2 regularization coefficient (default 0.01, critical for stability)
        """
        # Compute gradients for all layers
        activations = []  # Track for activity regularization

        for i in range(len(self.layers)):
            layer = self.layers[i]

            # Track activation for pathology detection
            activations.append(layer.get_state())

            # Get prediction from layer above
            if i == len(self.layers) - 1:
                # Top layer: no layer above, so use state as prediction (gives error=0)
                # OR use 0 as prediction (gives error=state)
                # For learning to occur at top layer, use 0 as prediction
                prediction_from_above = torch.zeros_like(layer.get_state())
            else:
                prediction_from_above = self.layers[i + 1].compute_prediction_for_below()

            # Value error: difference between actual state and prediction from above
            # This is the LOCAL error signal at this layer
            layer_error = layer.get_state() - prediction_from_above
            layer.error = layer_error

            # Get inputs for weight update
            # CRITICAL: Use RAW STATES for Hebbian learning (per architecture doc)
            # ΔW_apical = lr * error * layer_above_state (RAW, not transformed)
            # ΔW_basal = lr * error * layer_below_state (RAW, not transformed)
            if i == 0:
                # First layer receives sensory input from below
                input_from_below = self.input_buffer
            else:
                # Higher layers: use RAW state from layer below
                input_from_below = self.layers[i - 1].get_state()

            if i == len(self.layers) - 1:
                # Top layer: use own state (no layer above)
                input_from_above = layer.get_state()
            else:
                # Other layers: use RAW state from layer above
                input_from_above = self.layers[i + 1].get_state()

            if self.use_muon and self.optimizer is not None:
                # Compute gradients manually for Muon
                # Gradient is negative because we want to INCREASE weights when error is positive
                error_col = layer_error.unsqueeze(1)

                # Set gradients (Muon will apply momentum and adaptive learning)
                layer.neurons.W_basal.grad = -(error_col @ input_from_below.unsqueeze(0))
                layer.neurons.W_apical.grad = -(error_col @ input_from_above.unsqueeze(0))
            else:
                # Manual update with fixed learning rate
                layer.update_weights(
                    input_from_below=input_from_below,
                    input_from_above=input_from_above,
                    lr=lr,
                    weight_decay=weight_decay
                )

        # Apply Muon optimizer step if using it
        if self.use_muon and self.optimizer is not None:
            # Add activity regularization (prevent saturation/coma)
            self._add_activity_regularization(activations)

            # Muon step (handles momentum, weight decay internally)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _add_activity_regularization(self, activations: List[torch.Tensor]) -> None:
        """
        Add activity regularization to prevent pathologies.

        Penalizes:
        - Saturation (neurons stuck at ±1)
        - Too-high mean activation (seizure-like)

        Args:
            activations: List of activation tensors from each layer
        """
        for i, act in enumerate(activations):
            # Saturation penalty (especially important for layer 0)
            saturation_mask = (act.abs() > 0.9).float()
            saturation_rate = saturation_mask.mean()

            if saturation_rate > 0.1:  # If >10% saturated, penalize
                # Add small regularization to all weights in saturated layers
                # This gently discourages saturation without complex gradient surgery
                if self.layers[i].neurons.W_basal.grad is not None:
                    # Add small L1-like penalty to weights (push toward sparsity)
                    self.layers[i].neurons.W_basal.grad += self.saturation_penalty * self.layers[i].neurons.W_basal.sign()
                if self.layers[i].neurons.W_apical.grad is not None:
                    self.layers[i].neurons.W_apical.grad += self.saturation_penalty * self.layers[i].neurons.W_apical.sign()

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
