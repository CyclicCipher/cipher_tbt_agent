"""
Predictive Coding Convolutional Layer with local learning.

Implements Option 4: Full PC Conv Hierarchy
Based on VERSES AI research (2025):
- Each conv layer has state neurons and error neurons
- Precision-weighted optimization (Option A)
- Bi-directional error propagation (Option B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PCConvLayer(nn.Module):
    """
    Predictive Coding Convolutional Layer.

    Each layer:
    - Maintains state (feature maps) updated through inference
    - Computes prediction errors relative to adjacent layers
    - Updates weights based on LOCAL error signals
    - Supports precision weighting to prevent error decay in deep networks
    - Implements bi-directional predictions (predict layer above AND below)

    Attributes:
        layer_index: Index of this layer in the network
        in_channels: Number of input channels (from layer below)
        out_channels: Number of output channels (this layer's state)
        kernel_size: Convolutional kernel size
        stride: Convolutional stride
        padding: Convolutional padding
        precision: Precision weight for error signal (inverse variance)
        state: Current layer state (feature maps)
        error: Current prediction error
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
        """
        Initialize PC convolutional layer.

        Args:
            layer_index: Index in network (0 = first conv layer after input)
            in_channels: Number of channels from layer below
            out_channels: Number of channels in this layer's state
            channels_above: Number of channels in layer above (for top-down)
            kernel_size: Conv kernel size
            stride: Conv stride
            padding: Conv padding
            precision: Precision weight for error (higher = stronger signal)
            dtype: Data type for computations
        """
        super().__init__()

        self.layer_index = layer_index
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels_above = channels_above
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        # Precision weight (inverse variance) for error signal
        # Higher precision = stronger error signal
        # Prevents error decay in deep networks (VERSES Option A)
        self.precision = nn.Parameter(torch.tensor(precision, dtype=dtype))

        # === BOTTOM-UP PREDICTION (from layer below) ===
        # Convolutional weights: predict this layer's state from layer below
        self.W_bottom_up = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            dtype=dtype
        )

        # === TOP-DOWN PREDICTION (from layer above) ===
        # Predict this layer's state from layer above
        # Use ConvTranspose2d if stride > 1, otherwise regular Conv2d
        if channels_above > 0:
            if stride > 1:
                # Need to upsample from layer above
                self.W_top_down = nn.ConvTranspose2d(
                    channels_above,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=stride - 1,
                    bias=True,
                    dtype=dtype
                )
            else:
                # Same spatial size
                self.W_top_down = nn.Conv2d(
                    channels_above,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=True,
                    dtype=dtype
                )
        else:
            # Top layer: no layer above
            self.W_top_down = None

        # === BACKWARD PREDICTION (for bi-directional propagation) ===
        # Predict layer BELOW from this layer's state (VERSES Option B)
        # This creates error signals that flow backward
        if stride > 1:
            self.W_predict_below = nn.ConvTranspose2d(
                out_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride - 1,
                bias=True,
                dtype=dtype
            )
        else:
            self.W_predict_below = nn.Conv2d(
                out_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=True,
                dtype=dtype
            )

        # === FORWARD PREDICTION (for bi-directional propagation) ===
        # Predict layer ABOVE from this layer's state (VERSES Option B)
        if channels_above > 0:
            if stride > 1:
                # Layer above has smaller spatial size
                self.W_predict_above = nn.Conv2d(
                    out_channels,
                    channels_above,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=True,
                    dtype=dtype
                )
            else:
                self.W_predict_above = nn.Conv2d(
                    out_channels,
                    channels_above,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=True,
                    dtype=dtype
                )
        else:
            self.W_predict_above = None

        # === LATERAL (RECURRENT) CONNECTIONS ===
        # Predict this layer's state from its own previous state
        self.W_lateral = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,  # Pointwise for simplicity
            stride=1,
            padding=0,
            bias=False,
            dtype=dtype
        )
        # Initialize lateral to small values
        nn.init.normal_(self.W_lateral.weight, mean=0.0, std=0.01)

        # State and error buffers (will be initialized on first forward pass)
        self.state = None
        self.error = None
        self.prev_state = None  # For temporal/lateral connections

    def init_state(self, batch_size: int, height: int, width: int, device: torch.device):
        """Initialize state and error buffers."""
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
        """
        Compute prediction for this layer's state.

        Combines:
        - Bottom-up prediction from layer below
        - Top-down prediction from layer above
        - Lateral prediction from previous state

        Args:
            input_below: State of layer below (B, C_below, H_below, W_below)
            input_above: State of layer above (B, C_above, H_above, W_above)
            use_lateral: Whether to include lateral/temporal prediction

        Returns:
            Prediction for this layer's state (B, C, H, W)
        """
        prediction = 0.0
        count = 0

        # Bottom-up prediction
        if input_below is not None:
            pred_bottom_up = self.W_bottom_up(input_below)
            prediction = prediction + pred_bottom_up
            count += 1

        # Top-down prediction
        if input_above is not None and self.W_top_down is not None:
            pred_top_down = self.W_top_down(input_above)
            prediction = prediction + pred_top_down
            count += 1

        # Lateral prediction (from previous state)
        if use_lateral and self.prev_state is not None:
            pred_lateral = self.W_lateral(self.prev_state)
            prediction = prediction + 0.5 * pred_lateral  # Weaker weight
            count += 0.5

        # Average predictions
        if count > 0:
            prediction = prediction / count

        return prediction

    def compute_error(
        self,
        input_below: Optional[torch.Tensor] = None,
        input_above: Optional[torch.Tensor] = None,
        use_lateral: bool = True
    ) -> torch.Tensor:
        """
        Compute prediction error for this layer.

        Error = state - prediction (precision-weighted)

        Args:
            input_below: State of layer below
            input_above: State of layer above
            use_lateral: Whether to include lateral prediction

        Returns:
            Precision-weighted prediction error
        """
        # Compute prediction from adjacent layers
        prediction = self.compute_prediction(input_below, input_above, use_lateral)

        # Compute raw error
        raw_error = self.state - prediction

        # Apply precision weighting (VERSES Option A)
        # Clamp precision to prevent numerical instability
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
        """
        Update state to minimize prediction error (inference step).

        Args:
            input_below: State of layer below
            input_above: State of layer above
            inference_lr: Learning rate for state update
            use_lateral: Whether to use lateral connections

        Returns:
            Updated state
        """
        # Compute error
        error = self.compute_error(input_below, input_above, use_lateral)

        # Update state to minimize error: state <- state - lr * error
        self.state = self.state - inference_lr * error

        return self.state

    def predict_below(self) -> torch.Tensor:
        """
        Predict the layer below's state (for bi-directional propagation).

        Returns:
            Prediction for layer below (B, C_below, H_below, W_below)
        """
        return self.W_predict_below(self.state)

    def predict_above(self) -> torch.Tensor:
        """
        Predict the layer above's state (for bi-directional propagation).

        Returns:
            Prediction for layer above (B, C_above, H_above, W_above)
        """
        if self.W_predict_above is not None:
            return self.W_predict_above(self.state)
        else:
            return None

    def update_weights(
        self,
        input_below: Optional[torch.Tensor] = None,
        input_above: Optional[torch.Tensor] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> None:
        """
        Update weights using local Hebbian-style learning.

        Weight update rule: Δ�� = lr * (error ⊗ input)
        Where ⊗ represents appropriate convolution/correlation operation.

        Args:
            input_below: State of layer below
            input_above: State of layer above
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        with torch.no_grad():
            # Update bottom-up weights
            if input_below is not None:
                # Compute gradient via convolution: error * input_below
                # This is local Hebbian learning
                grad_bottom_up = F.conv2d(
                    input_below.transpose(0, 1),  # (C_below, B, H, W)
                    self.error.transpose(0, 1),    # (C_out, B, H, W)
                    stride=self.stride,
                    padding=self.padding
                ).transpose(0, 1)  # Back to (B, ...) format

                # Average over batch
                grad_bottom_up = grad_bottom_up.mean(dim=0, keepdim=True)

                # Apply learning rate and weight decay
                self.W_bottom_up.weight.data += learning_rate * (
                    grad_bottom_up.squeeze(0) - weight_decay * self.W_bottom_up.weight.data
                )

            # Update top-down weights
            if input_above is not None and self.W_top_down is not None:
                # Similar Hebbian update for top-down weights
                # Implementation depends on stride (Conv2d vs ConvTranspose2d)
                # For simplicity, use gradient-free update
                # Δ�� = lr * error @ input_above.T
                pass  # TODO: Implement top-down weight updates

            # Update lateral weights
            if self.prev_state is not None:
                # Δ W_lateral = lr * error @ prev_state.T
                # For pointwise conv (1x1), this is straightforward
                error_flat = self.error.mean(dim=(0, 2, 3))  # (C_out,)
                state_flat = self.prev_state.mean(dim=(0, 2, 3))  # (C_out,)

                delta_lateral = learning_rate * torch.outer(error_flat, state_flat)
                self.W_lateral.weight.data[:, :, 0, 0] += (
                    delta_lateral - weight_decay * self.W_lateral.weight.data[:, :, 0, 0]
                )

    def update_temporal_state(self) -> None:
        """Update temporal state buffer (call after each timestep)."""
        if self.state is not None:
            self.prev_state = self.state.clone()

    def reset_state(self) -> None:
        """Reset state and error to zero."""
        if self.state is not None:
            self.state.zero_()
            self.error.zero_()
            self.prev_state.zero_()

    def get_state(self) -> torch.Tensor:
        """Get current layer state."""
        return self.state

    def get_error(self) -> torch.Tensor:
        """Get current prediction error."""
        return self.error

    def __repr__(self) -> str:
        return (f"PCConvLayer(index={self.layer_index}, "
                f"in={self.in_channels}, out={self.out_channels}, "
                f"above={self.channels_above}, precision={self.precision.item():.2f})")
