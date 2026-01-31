"""
Categorical Predictive Coding Network - Complete Implementation

Contains all network logic for PC-based learning:
- PCConvLayer: Convolutional layers with predictive coding dynamics
- PCConvVisionPreprocessor: Hierarchical PC vision processing
- CanonicalPCLayer: Dense PC layers with dendrite structure
- CanonicalMicrocircuit: 3-layer canonical microcircuit

All components use local learning rules (no backpropagation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ============================================================================
# PC CONVOLUTIONAL LAYER
# ============================================================================

class PCConvLayer(nn.Module):
    """
    Predictive Coding Convolutional Layer with local learning.

    Features:
    - State neurons updated through inference
    - Precision-weighted errors (prevents decay in deep networks)
    - Bi-directional predictions (predict both up and down)
    - Lateral/temporal connections
    - Local Hebbian weight updates
    """

    def __init__(
        self,
        layer_index: int,
        in_channels: int,
        out_channels: int,
        channels_above: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        precision: float = 1.0,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.layer_index = layer_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_above = channels_above
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        # Precision weight (higher = stronger error signal)
        self.precision = nn.Parameter(torch.tensor(precision, dtype=dtype))

        # Bottom-up prediction weights
        self.W_bottom_up = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=True, dtype=dtype
        )

        # Top-down prediction weights
        if channels_above > 0:
            if stride > 1:
                self.W_top_down = nn.ConvTranspose2d(
                    channels_above, out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    output_padding=stride - 1, bias=True, dtype=dtype
                )
            else:
                self.W_top_down = nn.Conv2d(
                    channels_above, out_channels,
                    kernel_size=kernel_size, stride=1, padding=padding,
                    bias=True, dtype=dtype
                )
        else:
            self.W_top_down = None

        # Backward prediction (predict layer below)
        if stride > 1:
            self.W_predict_below = nn.ConvTranspose2d(
                out_channels, in_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                output_padding=stride - 1, bias=True, dtype=dtype
            )
        else:
            self.W_predict_below = nn.Conv2d(
                out_channels, in_channels,
                kernel_size=kernel_size, stride=1, padding=padding,
                bias=True, dtype=dtype
            )

        # Forward prediction (predict layer above)
        if channels_above > 0:
            if stride > 1:
                self.W_predict_above = nn.Conv2d(
                    out_channels, channels_above,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    bias=True, dtype=dtype
                )
            else:
                self.W_predict_above = nn.Conv2d(
                    out_channels, channels_above,
                    kernel_size=kernel_size, stride=1, padding=padding,
                    bias=True, dtype=dtype
                )
        else:
            self.W_predict_above = None

        # Lateral connections
        self.W_lateral = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False, dtype=dtype
        )
        nn.init.normal_(self.W_lateral.weight, mean=0.0, std=0.01)

        # State buffers
        self.state = None
        self.error = None
        self.prev_state = None

    def init_state(self, batch_size: int, height: int, width: int, device: torch.device):
        """Initialize state buffers."""
        self.state = torch.zeros(
            batch_size, self.out_channels, height, width,
            dtype=self.dtype, device=device
        )
        self.error = torch.zeros_like(self.state)
        self.prev_state = torch.zeros_like(self.state)

    def compute_prediction(
        self,
        input_below: Optional[torch.Tensor] = None,
        input_above: Optional[torch.Tensor] = None,
        use_lateral: bool = True
    ) -> torch.Tensor:
        """Compute prediction from adjacent layers."""
        prediction = 0.0
        count = 0

        if input_below is not None:
            prediction = prediction + self.W_bottom_up(input_below)
            count += 1

        if input_above is not None and self.W_top_down is not None:
            prediction = prediction + self.W_top_down(input_above)
            count += 1

        if use_lateral and self.prev_state is not None:
            prediction = prediction + 0.5 * self.W_lateral(self.prev_state)
            count += 0.5

        if count > 0:
            prediction = prediction / count

        return prediction

    def compute_error(
        self,
        input_below: Optional[torch.Tensor] = None,
        input_above: Optional[torch.Tensor] = None,
        use_lateral: bool = True
    ) -> torch.Tensor:
        """Compute precision-weighted prediction error."""
        prediction = self.compute_prediction(input_below, input_above, use_lateral)
        raw_error = self.state - prediction
        precision = torch.clamp(self.precision, min=0.1, max=100.0)
        self.error = precision * raw_error
        return self.error

    def update_state(
        self,
        input_below: Optional[torch.Tensor] = None,
        input_above: Optional[torch.Tensor] = None,
        inference_lr: float = 0.1,
        use_lateral: bool = True
    ) -> torch.Tensor:
        """Update state to minimize prediction error."""
        error = self.compute_error(input_below, input_above, use_lateral)
        self.state = self.state - inference_lr * error
        return self.state

    def update_weights(
        self,
        input_below: Optional[torch.Tensor] = None,
        input_above: Optional[torch.Tensor] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> None:
        """Update weights using local Hebbian learning."""
        with torch.no_grad():
            if input_below is not None:
                grad_bottom_up = F.conv2d(
                    input_below.transpose(0, 1),
                    self.error.transpose(0, 1),
                    stride=self.stride,
                    padding=self.padding
                ).transpose(0, 1)

                grad_bottom_up = grad_bottom_up.mean(dim=0, keepdim=True)
                self.W_bottom_up.weight.data += learning_rate * (
                    grad_bottom_up.squeeze(0) - weight_decay * self.W_bottom_up.weight.data
                )

            if self.prev_state is not None:
                error_flat = self.error.mean(dim=(0, 2, 3))
                state_flat = self.prev_state.mean(dim=(0, 2, 3))
                delta_lateral = learning_rate * torch.outer(error_flat, state_flat)
                self.W_lateral.weight.data[:, :, 0, 0] += (
                    delta_lateral - weight_decay * self.W_lateral.weight.data[:, :, 0, 0]
                )

    def update_temporal_state(self) -> None:
        """Update temporal state buffer."""
        if self.state is not None:
            self.prev_state = self.state.clone()

    def reset_state(self) -> None:
        """Reset state and error to zero."""
        if self.state is not None:
            self.state.zero_()
            self.error.zero_()
            self.prev_state.zero_()

    def get_state(self) -> torch.Tensor:
        return self.state

    def get_error(self) -> torch.Tensor:
        return self.error


# ============================================================================
# PC CONVOLUTIONAL VISION PREPROCESSOR
# ============================================================================

class PCConvVisionPreprocessor(nn.Module):
    """
    Hierarchical PC vision preprocessor.

    3-layer hierarchy:
    - Layer 0: V1 simple cells (edge detection) - 3→64 channels
    - Layer 1: V1 complex cells (texture) - 64→128 channels
    - Layer 2: V2/V4 features (object parts) - 128→256 channels

    All layers learn through local PC dynamics (no backprop).
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        precisions: List[float] = [1.0, 10.0, 100.0]
    ):
        super().__init__()

        self.dtype = dtype

        # Layer 0: V1 simple cells (3→64, 100×100→50×50)
        self.pc_conv0 = PCConvLayer(
            layer_index=0, in_channels=3, out_channels=64,
            channels_above=128, kernel_size=7, stride=2, padding=3,
            precision=precisions[0], dtype=dtype
        )
        self.pool0 = nn.AdaptiveAvgPool2d((16, 16))

        # Layer 1: V1 complex cells (64→128, 16×16→8×8)
        self.pc_conv1 = PCConvLayer(
            layer_index=1, in_channels=64, out_channels=128,
            channels_above=256, kernel_size=3, stride=2, padding=1,
            precision=precisions[1], dtype=dtype
        )
        self.pool1 = nn.AdaptiveAvgPool2d((4, 4))

        # Layer 2: V2/V4 features (128→256, 4×4→2×2)
        self.pc_conv2 = PCConvLayer(
            layer_index=2, in_channels=128, out_channels=256,
            channels_above=0, kernel_size=3, stride=2, padding=1,
            precision=precisions[2], dtype=dtype
        )
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))

        self.pc_layers = [self.pc_conv0, self.pc_conv1, self.pc_conv2]

    def init_states(self, batch_size: int, device: torch.device):
        """Initialize state buffers for all layers."""
        self.pc_conv0.init_state(batch_size, 50, 50, device)
        self.pc_conv1.init_state(batch_size, 8, 8, device)
        self.pc_conv2.init_state(batch_size, 2, 2, device)

    def forward(
        self,
        x: torch.Tensor,
        num_iterations: int = 20,
        inference_lr: float = 0.1,
        use_lateral: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through PC conv hierarchy.

        Args:
            x: Input image (30000,) flattened or (3, 100, 100)
            num_iterations: Number of inference iterations
            inference_lr: Learning rate for state updates
            use_lateral: Whether to use lateral connections

        Returns:
            (1024,) flattened features
        """
        # Reshape input
        if x.dim() == 1:
            if x.size(0) == 30000:
                x = x.view(3, 100, 100)
            else:
                raise ValueError(f"Expected 30000 dims, got {x.size(0)}")

        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, 3, 100, 100)

        device = x.device
        batch_size = x.size(0)

        # Initialize states if needed
        if self.pc_conv0.state is None:
            self.init_states(batch_size, device)

        # PC inference iterations
        for iteration in range(num_iterations):
            # Update from top to bottom
            self.pc_conv2.update_state(
                input_below=self.pc_conv1.state,
                input_above=None,
                inference_lr=inference_lr,
                use_lateral=use_lateral
            )

            self.pc_conv1.update_state(
                input_below=self.pc_conv0.state,
                input_above=self.pc_conv2.state,
                inference_lr=inference_lr,
                use_lateral=use_lateral
            )

            self.pc_conv0.update_state(
                input_below=x,
                input_above=self.pc_conv1.state,
                inference_lr=inference_lr,
                use_lateral=use_lateral
            )

        # Extract features
        features = self.pool2(self.pc_conv2.state)  # (B, 256, 2, 2)
        features = features.flatten(1)  # (B, 1024)

        if batch_size == 1:
            features = features.squeeze(0)  # (1024,)

        return features

    def get_weight_stats(self) -> dict:
        """Get weight statistics for debugging."""
        stats = {}
        for i, layer in enumerate(self.pc_layers):
            w = layer.W_bottom_up.weight.data
            stats[f'pc_conv{i}'] = {
                'mean': w.mean().item(),
                'std': w.std().item(),
                'abs_mean': w.abs().mean().item(),
                'precision': layer.precision.item()
            }
        return stats

    def reset_states(self) -> None:
        """Reset states for all layers."""
        for layer in self.pc_layers:
            layer.reset_state()

    def update_temporal_states(self) -> None:
        """Update temporal states for all layers."""
        for layer in self.pc_layers:
            layer.update_temporal_state()


# ============================================================================
# CANONICAL PC LAYER (DENSE)
# ============================================================================

class CanonicalPCLayer(nn.Module):
    """
    Dense PC layer with proper dendrite structure.

    Dendrites:
    - Feedforward (bottom-up from layer below)
    - Lateral (within same layer)
    - Feedback (top-down from layer above)
    """

    def __init__(
        self,
        num_neurons: int,
        input_size_below: int,
        input_size_above: int = 0,
        has_lateral: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.num_neurons = num_neurons
        self.dtype = dtype

        # Feedforward connections
        self.W_feedforward = nn.Linear(input_size_below, num_neurons, bias=False)

        # Lateral connections
        if has_lateral:
            self.W_lateral = nn.Linear(num_neurons, num_neurons, bias=False)
        else:
            self.W_lateral = None

        # Feedback connections
        if input_size_above > 0:
            self.W_feedback = nn.Linear(input_size_above, num_neurons, bias=False)
        else:
            self.W_feedback = None

        # State
        self.register_buffer('state', torch.zeros(num_neurons, dtype=dtype))

    def get_state(self) -> torch.Tensor:
        return self.state

    def compute_feedforward(self, input_below: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_feedforward(input_below))

    def compute_lateral(self) -> torch.Tensor:
        if self.W_lateral is not None:
            return torch.tanh(self.W_lateral(self.state))
        return torch.zeros_like(self.state)

    def compute_feedback(self, input_above: torch.Tensor) -> torch.Tensor:
        if self.W_feedback is not None:
            return torch.tanh(self.W_feedback(input_above))
        return torch.zeros_like(self.state)

    def update_weights_local(
        self,
        input_below: torch.Tensor,
        error: torch.Tensor,
        state_above: torch.Tensor = None,
        learning_rate: float = 0.01
    ):
        """Update weights using local Hebbian-like PC learning."""
        with torch.no_grad():
            if input_below.dim() == 1:
                input_below = input_below.unsqueeze(0)
            if error.dim() == 1:
                error = error.unsqueeze(1)

            delta_ff = learning_rate * (error @ input_below)
            self.W_feedforward.weight.data += delta_ff

            if self.W_lateral is not None:
                state_for_lateral = self.state.unsqueeze(0)
                delta_lat = learning_rate * (error @ state_for_lateral)
                self.W_lateral.weight.data += delta_lat

            if self.W_feedback is not None and state_above is not None:
                if state_above.dim() == 1:
                    state_above = state_above.unsqueeze(0)
                delta_fb = learning_rate * (error @ state_above)
                self.W_feedback.weight.data += delta_fb


# ============================================================================
# CANONICAL MICROCIRCUIT
# ============================================================================

class CanonicalMicrocircuit(nn.Module):
    """
    3-layer canonical microcircuit for PC inference.

    Layer 0 (Superficial): Input processing, lateral connections
    Layer 1 (Middle): Integration
    Layer 2 (Deep): Output generation
    """

    def __init__(
        self,
        num_classes: int,
        input_features: int,
        layer0_size: int,
        layer1_size: int,
        layer2_size: int,
        use_4bit: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.num_classes = num_classes

        # Layer 0: Superficial (has lateral)
        self.layer0 = CanonicalPCLayer(
            num_neurons=layer0_size,
            input_size_below=input_features,
            input_size_above=layer1_size,
            has_lateral=True,
            dtype=dtype
        )

        # Layer 1: Middle
        self.layer1 = CanonicalPCLayer(
            num_neurons=layer1_size,
            input_size_below=layer0_size,
            input_size_above=layer2_size,
            has_lateral=False,
            dtype=dtype
        )

        # Layer 2: Deep (output)
        self.layer2 = CanonicalPCLayer(
            num_neurons=layer2_size,
            input_size_below=layer1_size,
            input_size_above=0,
            has_lateral=False,
            dtype=dtype
        )

        self.inference_lr = 0.1

    def forward(self, input_data: torch.Tensor, num_iterations: int = 20) -> torch.Tensor:
        """Run PC inference to minimize prediction errors."""
        for _ in range(num_iterations):
            # Layer 0
            ff_0 = self.layer0.compute_feedforward(input_data)
            lat_0 = self.layer0.compute_lateral()
            fb_0 = self.layer0.compute_feedback(self.layer1.get_state())

            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = self.layer0.state - target_0
            self.layer0.state.data -= self.inference_lr * error_0.data

            # Layer 1
            ff_1 = self.layer1.compute_feedforward(self.layer0.get_state())
            fb_1 = self.layer1.compute_feedback(self.layer2.get_state())

            target_1 = ff_1 + fb_1
            error_1 = self.layer1.state - target_1
            self.layer1.state.data -= self.inference_lr * error_1.data

            # Layer 2
            ff_2 = self.layer2.compute_feedforward(self.layer1.get_state())

            target_2 = ff_2
            error_2 = self.layer2.state - target_2
            self.layer2.state.data -= self.inference_lr * error_2.data

        return self.layer2.get_state()

    def update_weights(
        self,
        input_data: torch.Tensor,
        learning_rate: float = 0.01,
        weight_decay: float = 0.01
    ):
        """Update all weights using local PC learning."""
        # Compute final errors
        ff_0 = self.layer0.compute_feedforward(input_data)
        lat_0 = self.layer0.compute_lateral()
        fb_0 = self.layer0.compute_feedback(self.layer1.get_state())
        error_0 = self.layer0.state - (ff_0 + 0.5 * lat_0 + fb_0)

        ff_1 = self.layer1.compute_feedforward(self.layer0.get_state())
        fb_1 = self.layer1.compute_feedback(self.layer2.get_state())
        error_1 = self.layer1.state - (ff_1 + fb_1)

        ff_2 = self.layer2.compute_feedforward(self.layer1.get_state())
        error_2 = self.layer2.state - ff_2

        # Update weights
        self.layer0.update_weights_local(
            input_below=input_data,
            error=error_0,
            state_above=self.layer1.get_state(),
            learning_rate=learning_rate
        )

        self.layer1.update_weights_local(
            input_below=self.layer0.get_state(),
            error=error_1,
            state_above=self.layer2.get_state(),
            learning_rate=learning_rate
        )

        self.layer2.update_weights_local(
            input_below=self.layer1.get_state(),
            error=error_2,
            state_above=None,
            learning_rate=learning_rate
        )

    def reset_states(self):
        """Reset all layer states."""
        self.layer0.state.zero_()
        self.layer1.state.zero_()
        self.layer2.state.zero_()
