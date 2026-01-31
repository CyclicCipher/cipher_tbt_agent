"""
Predictive Coding Convolutional Vision Preprocessor.

Replaces standard conv layers with full PC conv hierarchy (Option 4).

Each convolutional layer:
- Maintains state (feature maps) updated through inference
- Computes prediction errors
- Updates weights through local learning (no backprop!)
- Includes precision weighting (VERSES Option A)
- Implements bi-directional propagation (VERSES Option B)
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.network.conv_layer import PCConvLayer


class PCConvVisionPreprocessor(nn.Module):
    """
    PC-based convolutional vision preprocessor.

    Implements hierarchical predictive coding for vision:
    - Layer 0: V1 simple cells (edge detection)
    - Layer 1: V1 complex cells (texture)
    - Layer 2: V2/V4 features (object parts)

    Unlike standard conv nets, ALL layers learn through local PC dynamics.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        # Precision weights for each layer (higher = stronger error signal)
        precisions: List[float] = [1.0, 10.0, 100.0]
    ):
        super().__init__()

        self.dtype = dtype

        # === LAYER 0: V1 Simple Cells ===
        # Input: 3 channels (RGB) → 64 channels
        # Spatial: 100×100 → 50×50 (stride=2)
        self.pc_conv0 = PCConvLayer(
            layer_index=0,
            in_channels=3,
            out_channels=64,
            channels_above=128,  # Layer 1 has 128 channels
            kernel_size=7,
            stride=2,
            padding=3,
            precision=precisions[0],
            dtype=dtype
        )
        # Pool to 16×16 for efficiency
        self.pool0 = nn.AdaptiveAvgPool2d((16, 16))

        # === LAYER 1: V1 Complex Cells ===
        # Input: 64 channels → 128 channels
        # Spatial: 16×16 → 8×8 (stride=2)
        self.pc_conv1 = PCConvLayer(
            layer_index=1,
            in_channels=64,
            out_channels=128,
            channels_above=256,  # Layer 2 has 256 channels
            kernel_size=3,
            stride=2,
            padding=1,
            precision=precisions[1],
            dtype=dtype
        )
        # Pool to 4×4 for efficiency
        self.pool1 = nn.AdaptiveAvgPool2d((4, 4))

        # === LAYER 2: V2/V4 Features ===
        # Input: 128 channels → 256 channels
        # Spatial: 4×4 → 2×2 (stride=2)
        self.pc_conv2 = PCConvLayer(
            layer_index=2,
            in_channels=128,
            out_channels=256,
            channels_above=0,  # No layer above (top of conv hierarchy)
            kernel_size=3,
            stride=2,
            padding=1,
            precision=precisions[2],
            dtype=dtype
        )
        # Pool to 2×2
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))

        # Store layers for easy access
        self.pc_layers = [self.pc_conv0, self.pc_conv1, self.pc_conv2]

    def init_states(self, batch_size: int, device: torch.device):
        """Initialize state buffers for all layers."""
        # Layer 0: (B, 64, 50, 50) before pooling → (B, 64, 16, 16) after
        self.pc_conv0.init_state(batch_size, 50, 50, device)

        # Layer 1: (B, 128, 8, 8) before pooling → (B, 128, 4, 4) after
        self.pc_conv1.init_state(batch_size, 8, 8, device)

        # Layer 2: (B, 256, 2, 2)
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
            x: Input image (30000,) flattened OR (3, 100, 100) tensor
            num_iterations: Number of inference iterations
            inference_lr: Learning rate for state updates
            use_lateral: Whether to use lateral connections

        Returns:
            (1024,) flattened features from final layer
        """
        # Reshape input to image format
        if x.dim() == 1:
            if x.size(0) == 30000:
                x = x.view(3, 100, 100)
            else:
                raise ValueError(f"Expected 30000 dims, got {x.size(0)}")

        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, 3, 100, 100)

        device = x.device
        batch_size = x.size(0)

        # Initialize states if not done yet
        if self.pc_conv0.state is None:
            self.init_states(batch_size, device)

        # === PC INFERENCE ===
        # Run iterative inference to minimize prediction errors
        for iteration in range(num_iterations):
            # LAYER 2 (top layer): Only bottom-up prediction
            self.pc_conv2.update_state(
                input_below=self.pc_conv1.state,
                input_above=None,  # No layer above
                inference_lr=inference_lr,
                use_lateral=use_lateral
            )

            # LAYER 1 (middle): Bottom-up and top-down predictions
            self.pc_conv1.update_state(
                input_below=self.pc_conv0.state,
                input_above=self.pc_conv2.state,
                inference_lr=inference_lr,
                use_lateral=use_lateral
            )

            # LAYER 0 (bottom): Input from image + top-down from layer 1
            self.pc_conv0.update_state(
                input_below=x,  # Actual sensory input
                input_above=self.pc_conv1.state,
                inference_lr=inference_lr,
                use_lateral=use_lateral
            )

        # === EXTRACT FEATURES ===
        # Pool and flatten final layer
        features = self.pool2(self.pc_conv2.state)  # (B, 256, 2, 2)
        features = features.flatten(1)  # (B, 1024)

        # Remove batch dimension if single sample
        if batch_size == 1:
            features = features.squeeze(0)  # (1024,)

        return features

    def inject_output_error(
        self,
        target_features: torch.Tensor,
        error_strength: float = 1.0
    ) -> None:
        """
        Inject error at output layer (for supervised learning).

        Instead of teaching signal, directly inject error at top layer.
        This is the proper PC way to provide supervision!

        Args:
            target_features: Desired output features (B, 1024) or (1024,)
            error_strength: Strength of injected error
        """
        # Reshape target to match layer 2 state
        if target_features.dim() == 1:
            target_features = target_features.unsqueeze(0)  # (1, 1024)

        # Reshape to feature maps
        batch_size = target_features.size(0)
        target_maps = target_features.view(batch_size, 256, 2, 2)

        # Inject error directly into layer 2
        # Error = target - current_state
        injected_error = error_strength * (target_maps - self.pc_conv2.state)
        self.pc_conv2.error = injected_error

        # Let error propagate backward through bi-directional predictions
        # This happens naturally in the next inference iteration

    def update_weights(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> None:
        """
        Update weights for all PC conv layers using local learning.

        Args:
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        # Update layer 0 (needs input image, which we'll get from state)
        # NOTE: For simplicity, we'll update weights after inference
        # In the training script, we'll pass the correct inputs

        # For now, just update layers with their current states
        # The training script will handle proper weight updates with correct inputs
        pass

    def get_weight_stats(self) -> dict:
        """Get statistics about weight changes (for debugging)."""
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
        """Reset states for all layers (call between samples)."""
        for layer in self.pc_layers:
            layer.reset_state()

    def update_temporal_states(self) -> None:
        """Update temporal states for all layers (call after each timestep)."""
        for layer in self.pc_layers:
            layer.update_temporal_state()

    def __repr__(self) -> str:
        return f"PCConvVisionPreprocessor(layers={len(self.pc_layers)})"
