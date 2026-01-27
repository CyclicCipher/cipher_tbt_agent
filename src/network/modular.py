"""
Modular predictive coding architecture.

Supports multiple sub-networks organized in pipeline positions:
- Position 0: Parallel sensory/motor sub-networks (vision, audio, keyboard, mouse)
- Position 1: Association network (multi-modal integration)
- Position 2: Abstract reasoning network

Each sub-network is a stack of layers with variable sizes.
Sub-networks at same position run in parallel.
Outputs concatenated at position boundaries.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from src.network.layer import PredictiveCodingLayer


class SubNetwork(nn.Module):
    """
    A sub-network: stack of layers processing one modality or function.

    Args:
        name: Identifier (e.g., "vision", "keyboard", "association")
        layer_sizes: Number of neurons in each layer [layer0_size, layer1_size, ...]
        input_size: Size of input to layer 0
        position: Pipeline position (0=sensory/motor, 1=association, 2=reasoning)
        dtype: Data type for computations
        device: Device to run on
    """

    def __init__(
        self,
        name: str,
        layer_sizes: List[int],
        input_size: int,
        position: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ):
        super().__init__()

        self.name = name
        self.position = position
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.input_size = input_size
        self.dtype = dtype
        self.device = device

        # Create layers
        self.layers = nn.ModuleList()
        current_input_size = input_size

        for i, layer_size in enumerate(layer_sizes):
            # Determine input from above (0 for top layer)
            input_size_above = layer_sizes[i + 1] if i < len(layer_sizes) - 1 else 0

            layer = PredictiveCodingLayer(
                layer_index=i,
                num_neurons=layer_size,
                input_size_below=current_input_size,
                input_size_above=input_size_above,
                dtype=dtype
            )
            self.layers.append(layer)
            current_input_size = layer_size  # Next layer's input is this layer's output

        # Input buffer for layer 0
        self.input_buffer = torch.zeros(input_size, dtype=dtype, device=device)

        # Move all layers to device
        self.to(device)

        # Parameters for inference
        self.inference_lr = 0.1
        self.temperature = 0.0

    def get_output_size(self) -> int:
        """Return size of top layer (output of this sub-network)."""
        return self.layer_sizes[-1]

    def get_top_state(self) -> torch.Tensor:
        """Return state of top layer."""
        return self.layers[-1].get_state()

    def set_input(self, input_data: torch.Tensor) -> None:
        """Set input buffer for this sub-network."""
        self.input_buffer.copy_(input_data)

    def reset_temporal_state(self) -> None:
        """Reset temporal state for all layers."""
        for layer in self.layers:
            layer.reset_temporal_state()

    def update_temporal_state(self) -> None:
        """Update temporal state for all layers."""
        for layer in self.layers:
            layer.update_temporal_state()


class ModularNetwork(nn.Module):
    """
    Modular predictive coding network with multiple sub-networks.

    Sub-networks organized by pipeline position:
    - Position 0: Parallel processing (vision, audio, keyboard, mouse)
    - Position 1: Integration (association)
    - Position 2+: Higher-level processing (reasoning)

    Information flows:
    1. Each position's sub-networks process in parallel
    2. Outputs concatenated and fed to next position
    3. Prediction errors computed locally within each sub-network
    """

    def __init__(
        self,
        subnetworks: List[SubNetwork],
        inference_lr: float = 0.1,
        temperature: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
        use_stable: bool = False,
        stable_lr: float = 0.001,
        stable_max_iterations: int = 400,
        stable_lr_schedule: str = "cosine",
        stable_decay_strong: float = 0.01,
        stable_decay_weak: float = 0.001,
        saturation_penalty: float = 0.01,
        activity_target: float = 0.3
    ):
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.inference_lr = inference_lr
        self.temperature = temperature
        self.use_stable = use_stable
        self.saturation_penalty = saturation_penalty
        self.activity_target = activity_target

        # Organize sub-networks by position
        self.subnetworks_by_position: Dict[int, List[SubNetwork]] = {}
        for subnet in subnetworks:
            pos = subnet.position
            if pos not in self.subnetworks_by_position:
                self.subnetworks_by_position[pos] = []
            self.subnetworks_by_position[pos].append(subnet)

            # Set inference parameters
            subnet.inference_lr = inference_lr
            subnet.temperature = temperature

        self.max_position = max(self.subnetworks_by_position.keys())

        # Store all sub-networks as ModuleList for proper registration
        self.all_subnetworks = nn.ModuleList(subnetworks)

        # Cross-position prediction weights (position N → position N-1)
        # This allows higher positions to send top-down predictions to lower positions
        self.cross_position_predictions = nn.ParameterDict()

        # Create prediction weights from position 1 to position 0
        # Position 1 BOTTOM layer predicts Position 0 TOP layer (adjacent layers)
        if 1 in self.subnetworks_by_position and 0 in self.subnetworks_by_position:
            for subnet_pos1 in self.subnetworks_by_position[1]:
                for subnet_pos0 in self.subnetworks_by_position[0]:
                    # Adjacent layer connection: pos1 bottom → pos0 top
                    key = f"{subnet_pos1.name}_to_{subnet_pos0.name}"
                    pred_size = subnet_pos0.layers[-1].num_neurons  # Target: TOP layer of pos 0
                    source_size = subnet_pos1.layers[0].num_neurons  # Source: BOTTOM layer of pos 1

                    # Initialize with small random weights
                    self.cross_position_predictions[key] = nn.Parameter(
                        torch.randn(pred_size, source_size, dtype=dtype, device=device) * 0.01
                    )

        # Validate architecture
        self._validate_architecture()

        # Initialize optimizer if requested
        self.optimizer = None
        if use_stable:
            from ..optimizers import StableProspectiveLearning
            self.optimizer = StableProspectiveLearning(
                self.parameters(),
                lr=stable_lr,
                max_iterations=stable_max_iterations,
                lr_schedule=stable_lr_schedule,
                weight_decay_strong=stable_decay_strong,
                weight_decay_weak=stable_decay_weak,
                stability_threshold=1.2,
                early_stopping=False,
                patience=50
            )

    def _validate_architecture(self) -> None:
        """Validate that architecture is properly configured."""
        # Check position 0 exists
        if 0 not in self.subnetworks_by_position:
            raise ValueError("Must have at least one sub-network at position 0")

        # Check positions are contiguous
        for pos in range(self.max_position):
            if pos not in self.subnetworks_by_position:
                raise ValueError(f"Missing sub-networks at position {pos}")

        # Validate input sizes for position > 0
        for pos in range(1, self.max_position + 1):
            # Compute expected input size (concatenation of previous position outputs)
            prev_output_size = sum(
                subnet.get_output_size()
                for subnet in self.subnetworks_by_position[pos - 1]
            )

            # Check each sub-network at this position
            for subnet in self.subnetworks_by_position[pos]:
                if subnet.input_size != prev_output_size:
                    raise ValueError(
                        f"Sub-network '{subnet.name}' at position {pos} has "
                        f"input_size={subnet.input_size}, but should be {prev_output_size} "
                        f"(concatenation of position {pos-1} outputs)"
                    )

    def set_position0_inputs(self, inputs: Dict[str, torch.Tensor]) -> None:
        """
        Set inputs for position 0 sub-networks.

        Args:
            inputs: Dict mapping sub-network name to input tensor
                   e.g., {"vision": vision_tensor, "keyboard": keyboard_tensor}
        """
        for subnet in self.subnetworks_by_position[0]:
            if subnet.name not in inputs:
                raise ValueError(f"Missing input for sub-network '{subnet.name}'")
            subnet.set_input(inputs[subnet.name])

    def _get_concatenated_output(self, position: int) -> torch.Tensor:
        """Get concatenated output from all sub-networks at given position."""
        outputs = [
            subnet.get_top_state()
            for subnet in self.subnetworks_by_position[position]
        ]
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        position0_inputs: Dict[str, torch.Tensor],
        num_iterations: int = 20,
        clamp_layers: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with iterative inference and optional clamping.

        Args:
            position0_inputs: Dict of inputs for position 0 sub-networks
            num_iterations: Number of inference iterations
            clamp_layers: Optional dict mapping subnet names to tensors that should be
                         clamped during ALL inference iterations. Used for:
                         - Sensory clamping (always): Force sensory input to observed values
                         - Motor clamping (training): Force motor output to target for supervised learning
                         - Motor free (testing): Let motor emerge from sensory input

        Returns:
            Output from highest position (typically reasoning output)
        """
        # Set position 0 inputs
        self.set_position0_inputs(position0_inputs)

        # Initialize all sub-network states with feedforward pass
        self._initialize_states()

        # Iterative inference to equilibrium with clamping
        for _ in range(num_iterations):
            self._inference_step()

            # CLAMP: Re-apply clamped values after each inference step
            # This prevents trivial zero solution
            if clamp_layers is not None:
                for subnet_name, clamped_value in clamp_layers.items():
                    subnet = self.get_subnet(subnet_name)
                    if subnet is not None:
                        # Clamp input buffer (raw input to subnet)
                        subnet.input_buffer.copy_(clamped_value)

        # Return output from highest position
        return self._get_concatenated_output(self.max_position)

    def _initialize_states(self) -> None:
        """Initialize all layer states with feedforward pass."""
        for pos in range(self.max_position + 1):
            # Get input for this position
            if pos == 0:
                # Position 0: use input_buffer for each sub-network
                for subnet in self.subnetworks_by_position[pos]:
                    self._initialize_subnet_states(subnet, subnet.input_buffer)
            else:
                # Position > 0: use concatenated output from previous position
                concat_input = self._get_concatenated_output(pos - 1)
                for subnet in self.subnetworks_by_position[pos]:
                    self._initialize_subnet_states(subnet, concat_input)

    def _initialize_subnet_states(self, subnet: SubNetwork, input_data: torch.Tensor) -> None:
        """Initialize states for a single sub-network."""
        for i, layer in enumerate(subnet.layers):
            if i == 0:
                input_below = input_data
            else:
                input_below = subnet.layers[i - 1].get_state()

            # Initialize to bottom-up prediction
            layer.state.copy_(torch.tanh(layer.neurons.W_basal @ input_below))

    def _inference_step(self) -> None:
        """Single inference step across all sub-networks."""
        for pos in range(self.max_position + 1):
            if pos == 0:
                # Position 0: each sub-network uses its input_buffer
                for subnet in self.subnetworks_by_position[pos]:
                    self._inference_step_subnet(subnet, subnet.input_buffer)
            else:
                # Position > 0: use concatenated output from previous position
                concat_input = self._get_concatenated_output(pos - 1)
                for subnet in self.subnetworks_by_position[pos]:
                    self._inference_step_subnet(subnet, concat_input)

    def _inference_step_subnet(self, subnet: SubNetwork, subnet_input: torch.Tensor) -> None:
        """Inference step for a single sub-network."""
        # Compute prediction errors for all layers
        errors = []
        for i, layer in enumerate(subnet.layers):
            if i == 0:
                input_below = subnet_input
            else:
                input_below = subnet.layers[i - 1].get_state()

            # Bottom-up prediction
            bottom_up_prediction = torch.tanh(layer.neurons.W_basal @ input_below)

            # Prediction error
            error = layer.get_state() - bottom_up_prediction
            errors.append(error)

        # Update states via gradient descent
        for i, layer in enumerate(subnet.layers):
            # Gradient term 1: -error_l (local error)
            gradient = -errors[i]

            # Gradient term 2: feedback from layer above (if exists)
            if i < len(subnet.layers) - 1:
                current_state = layer.get_state()
                weighted_input = subnet.layers[i + 1].neurons.W_basal @ current_state
                tanh_derivative = 1 - torch.tanh(weighted_input) ** 2
                error_times_deriv = errors[i + 1] * tanh_derivative
                feedback = subnet.layers[i + 1].neurons.W_basal.T @ error_times_deriv
                gradient += feedback

            # Gradient term 3: cross-position feedback from higher position
            # For TOP layer of position 0 subnets, add feedback from position 1 BOTTOM layer
            if i == len(subnet.layers) - 1 and subnet.position == 0 and 1 in self.subnetworks_by_position:
                for subnet_pos1 in self.subnetworks_by_position[1]:
                    key = f"{subnet_pos1.name}_to_{subnet.name}"
                    if key in self.cross_position_predictions:
                        # Get top-down prediction from position 1 bottom layer
                        assoc_bottom_state = subnet_pos1.layers[0].get_state()
                        top_down_pred = self.cross_position_predictions[key] @ assoc_bottom_state

                        # Error between current state and top-down prediction
                        cross_pos_error = layer.get_state() - top_down_pred

                        # Add feedback (push toward top-down prediction)
                        gradient += -cross_pos_error
                        break

            # Update state
            new_state = layer.get_state() + self.inference_lr * gradient

            # Add Langevin noise
            if self.temperature > 0:
                noise = torch.randn_like(new_state) * (self.temperature ** 0.5)
                new_state = new_state + noise

            # Prevent saturation
            new_state = new_state.clamp(-0.85, 0.85)

            layer.state.copy_(new_state)

    def update_weights(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        motor_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Update weights for all sub-networks using proper predictive coding.

        Implements FULL predictive coding learning:
        1. Each layer's error = state - prediction from above (value error)
        2. Updates BOTH W_basal (bottom-up) and W_apical (top-down) weights
        3. Uses local Hebbian rules: ΔW = lr * error * input - decay * W

        If use_stable=True, uses StableProspectiveLearning optimizer.
        Otherwise uses manual updates with fixed learning rate.

        Args:
            lr: Learning rate (only used if not using optimizer)
            weight_decay: L2 regularization coefficient
            motor_targets: Optional dict of motor targets for supervised learning
                          Keys are sub-network names at position 0
        """
        # Handle motor clamping for position 0 sub-networks
        if motor_targets is not None:
            for subnet in self.subnetworks_by_position[0]:
                if subnet.name in motor_targets:
                    # Clamp layer 0 of this sub-network
                    subnet.layers[0].state.copy_(motor_targets[subnet.name])

        # Compute gradients for all sub-networks
        activations = []  # Track for activity regularization

        for pos in range(self.max_position + 1):
            if pos == 0:
                for subnet in self.subnetworks_by_position[pos]:
                    subnet_activations = self._compute_subnet_gradients(
                        subnet, subnet.input_buffer, lr, weight_decay
                    )
                    activations.extend(subnet_activations)
            else:
                concat_input = self._get_concatenated_output(pos - 1)
                for subnet in self.subnetworks_by_position[pos]:
                    subnet_activations = self._compute_subnet_gradients(
                        subnet, concat_input, lr, weight_decay
                    )
                    activations.extend(subnet_activations)

        # Update cross-position prediction weights (position 1 → position 0)
        # Position 1 bottom layer predicts position 0 top layer (adjacent layers)
        if 1 in self.subnetworks_by_position and 0 in self.subnetworks_by_position:
            for subnet_pos1 in self.subnetworks_by_position[1]:
                for subnet_pos0 in self.subnetworks_by_position[0]:
                    key = f"{subnet_pos1.name}_to_{subnet_pos0.name}"
                    if key in self.cross_position_predictions:
                        # Get states from adjacent layers
                        target_layer = subnet_pos0.layers[-1]  # TOP layer of position 0
                        source_layer = subnet_pos1.layers[0]  # BOTTOM layer of position 1

                        # Compute prediction error
                        prediction = self.cross_position_predictions[key] @ source_layer.get_state()
                        error = target_layer.get_state() - prediction

                        # Hebbian weight update: ΔW = lr * error * source
                        error_col = error.unsqueeze(1)
                        source_row = source_layer.get_state().unsqueeze(0)

                        if self.optimizer is not None:
                            # Set gradient for optimizer
                            self.cross_position_predictions[key].grad = -(error_col @ source_row)
                        else:
                            # Manual update
                            self.cross_position_predictions[key].data += (
                                lr * (error_col @ source_row) -
                                weight_decay * self.cross_position_predictions[key].data
                            )

        # Apply optimizer step if using one
        if self.optimizer is not None:
            # Add activity regularization (prevent saturation/coma)
            self._add_activity_regularization(activations)

            # Optimizer step (handles momentum, adaptive LR, weight decay internally)
            # For StableProspectiveLearning, pass current error for adaptive decay
            if self.use_stable:
                # Compute total reconstruction error
                total_error = sum(
                    ((subnet.input_buffer - self._compute_subnet_reconstruction(subnet)) ** 2).sum().item()
                    for subnet in self.subnetworks_by_position[0]
                )
                self.optimizer.step(current_error=total_error)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

    def _compute_subnet_reconstruction(self, subnet: SubNetwork) -> torch.Tensor:
        """Compute reconstruction (bottom-up prediction) for subnet input."""
        if len(subnet.layers) > 0:
            return subnet.layers[0].compute_prediction_for_below()
        return torch.zeros_like(subnet.input_buffer)

    def _compute_subnet_gradients(
        self,
        subnet: SubNetwork,
        subnet_input: torch.Tensor,
        lr: float,
        weight_decay: float
    ) -> List[torch.Tensor]:
        """
        Compute gradients for a single sub-network.

        Implements proper predictive coding weight updates:
        - Updates BOTH W_basal and W_apical
        - Uses value error (state - prediction from above)
        - Local Hebbian learning rules

        Returns:
            List of activation tensors (for activity regularization)
        """
        activations = []

        for i, layer in enumerate(subnet.layers):
            # Track activation for pathology detection
            activations.append(layer.get_state())

            # Get inputs from below and above
            if i == 0:
                input_from_below = subnet_input
            else:
                input_from_below = subnet.layers[i - 1].get_state()

            if i == len(subnet.layers) - 1:
                # Top layer: no layer above, use zeros or self-prediction
                prediction_from_above = torch.zeros_like(layer.get_state())
                input_from_above = layer.get_state()
            else:
                # Get prediction from layer above (top-down)
                prediction_from_above = subnet.layers[i + 1].compute_prediction_for_below()
                input_from_above = subnet.layers[i + 1].get_state()

            # Value error: difference between actual state and prediction from above
            # This is the LOCAL error signal at this layer (proper PC learning)
            layer_error = layer.get_state() - prediction_from_above
            layer.error = layer_error

            if self.optimizer is not None:
                # Compute gradients manually for optimizer
                # Gradient is negative because we want to INCREASE weights when error is positive
                error_col = layer_error.unsqueeze(1)

                # Set gradients (optimizer will apply its update rule)
                layer.neurons.W_basal.grad = -(error_col @ input_from_below.unsqueeze(0))

                # Update W_apical only if it exists (not None) and has correct size
                if (hasattr(layer.neurons, 'W_apical') and
                    layer.neurons.W_apical is not None and
                    input_from_above.size(0) == layer.neurons.W_apical.size(1)):
                    layer.neurons.W_apical.grad = -(error_col @ input_from_above.unsqueeze(0))
            else:
                # Manual update with fixed learning rate
                # Update both W_basal and W_apical
                error_col = layer_error.unsqueeze(1)

                # Update W_basal (bottom-up weights)
                input_row = input_from_below.unsqueeze(0)
                layer.neurons.W_basal.data += lr * (error_col @ input_row) - weight_decay * layer.neurons.W_basal.data

                # Update W_apical (top-down weights) if it exists
                if (hasattr(layer.neurons, 'W_apical') and
                    layer.neurons.W_apical is not None and
                    input_from_above.size(0) == layer.neurons.W_apical.size(1)):
                    input_above_row = input_from_above.unsqueeze(0)
                    layer.neurons.W_apical.data += lr * (error_col @ input_above_row) - weight_decay * layer.neurons.W_apical.data

        return activations

    def _add_activity_regularization(self, activations: List[torch.Tensor]) -> None:
        """
        Add activity regularization to prevent pathologies.

        Penalizes:
        - Saturation (neurons stuck at ±1)
        - Too-high mean activation (seizure-like)

        Args:
            activations: List of activation tensors from all layers
        """
        for act in activations:
            # Saturation penalty
            saturation_mask = (act.abs() > 0.9).float()
            saturation_rate = saturation_mask.mean()

            if saturation_rate > 0.1:  # If >10% saturated, penalize
                # Find corresponding layer and add penalty to gradients
                for subnet in self.all_subnetworks:
                    for layer in subnet.layers:
                        if layer.get_state() is act:
                            # Add small L1-like penalty to weights (push toward sparsity)
                            if layer.neurons.W_basal.grad is not None:
                                layer.neurons.W_basal.grad += self.saturation_penalty * layer.neurons.W_basal.sign()
                            if (hasattr(layer.neurons, 'W_apical') and
                                layer.neurons.W_apical is not None and
                                layer.neurons.W_apical.grad is not None):
                                layer.neurons.W_apical.grad += self.saturation_penalty * layer.neurons.W_apical.sign()

    def reset_temporal_state(self) -> None:
        """Reset temporal state for all sub-networks."""
        for subnet in self.all_subnetworks:
            subnet.reset_temporal_state()

    def update_temporal_state(self) -> None:
        """Update temporal state for all sub-networks."""
        for subnet in self.all_subnetworks:
            subnet.update_temporal_state()

    def get_subnet(self, name: str) -> Optional[SubNetwork]:
        """Get sub-network by name."""
        for subnet in self.all_subnetworks:
            if subnet.name == name:
                return subnet
        return None

    def print_architecture(self) -> None:
        """Print architecture summary."""
        print("\nMODULAR ARCHITECTURE:")
        print("=" * 70)

        total_neurons = 0
        total_params = 0

        for pos in range(self.max_position + 1):
            print(f"\nPosition {pos}:")
            for subnet in self.subnetworks_by_position[pos]:
                subnet_neurons = sum(subnet.layer_sizes)
                subnet_params = sum(
                    layer.neurons.W_basal.numel()
                    for layer in subnet.layers
                )
                total_neurons += subnet_neurons
                total_params += subnet_params

                print(f"  {subnet.name}:")
                print(f"    Input size: {subnet.input_size}")
                print(f"    Layers: {len(subnet.layers)}")
                print(f"    Layer sizes: {subnet.layer_sizes}")
                print(f"    Total neurons: {subnet_neurons:,}")
                print(f"    Parameters: {subnet_params:,}")

        print(f"\nTOTAL:")
        print(f"  Neurons: {total_neurons:,}")
        print(f"  Parameters: {total_params:,}")
        print("=" * 70)
