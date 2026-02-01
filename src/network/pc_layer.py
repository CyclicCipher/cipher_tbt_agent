"""
Minimal Predictive Coding Layer Implementation

Based on standard PC architecture from:
- Bogacz-Group/PredictiveCoding
- Whittington & Bogacz (2017)

Key Concepts:
- PCLayer sits BETWEEN regular layers (Linear, etc.)
- Holds "value nodes" (_x) that are optimized during inference
- Computes energy: E = 0.5 * (mu - x)^2 where mu=prediction, x=value
- During training: returns x (optimized values)
- During eval: returns mu (pass-through)
"""

import torch
import torch.nn as nn


class PCLayer(nn.Module):
    """Predictive Coding Layer - holds value nodes and computes prediction error energy."""

    def __init__(self):
        """Initialize PCLayer.

        PCLayer behavior:
        - Training mode: Holds value nodes (_x), computes energy, returns _x
        - Eval mode: Pass-through, returns input (mu)
        """
        super().__init__()

        # Value nodes - will be initialized on first forward pass
        self._x = None

        # Energy accumulator
        self._energy = None

        # Flag to re-initialize x at start of each batch
        self._is_sample_x = False

    def energy(self):
        """Get the energy held by this layer."""
        return self._energy

    def clear_energy(self):
        """Clear accumulated energy."""
        self._energy = None

    def get_x(self):
        """Get value nodes."""
        return self._x

    def set_is_sample_x(self, value: bool):
        """Set flag to re-sample x on next forward pass."""
        self._is_sample_x = value

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            mu: Prediction from layer below (bottom-up signal)

        Returns:
            During training: value nodes (_x)
            During eval: pass-through (mu)
        """
        if not self.training:
            # Eval mode: just pass through
            return mu

        # Training mode: predictive coding

        # Initialize or re-initialize x if needed
        if self._x is None or self._is_sample_x or mu.shape != self._x.shape:
            # Initialize x as detached copy of mu
            self._x = nn.Parameter(mu.detach().clone(), requires_grad=True)
            self._is_sample_x = False

        # Compute prediction error energy: E = 0.5 * (mu - x)^2
        # Note: mu.detach() prevents gradients flowing through bottom-up pathway
        error = mu.detach() - self._x
        self._energy = 0.5 * (error ** 2).sum()

        # Return value nodes (not prediction)
        return self._x


class PCNetwork(nn.Module):
    """Simple Predictive Coding Network.

    Architecture:
        Input -> Linear -> PCLayer -> ... -> Linear -> PCLayer -> Output

    PCLayers are inserted between every Linear layer to enable PC learning.
    """

    def __init__(self, layer_sizes, activation='relu'):
        """Initialize PC Network.

        Args:
            layer_sizes: List of layer dimensions, e.g. [784, 256, 256, 10]
            activation: Activation function ('relu', 'tanh')
        """
        super().__init__()

        # Create layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add linear transformation
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            # Add PC layer after each linear layer
            layers.append(PCLayer())

            # Add activation (except after last layer for classification)
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

        self.model = nn.Sequential(*layers)

        # Initialize weights (He initialization for ReLU, Xavier for tanh)
        self._initialize_weights(activation)

    def _initialize_weights(self, activation):
        """Initialize weights properly for deep networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if activation == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif activation == 'tanh':
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through network."""
        return self.model(x)

    def get_pc_layers(self):
        """Get all PCLayers in the network."""
        pc_layers = []
        for module in self.modules():
            if isinstance(module, PCLayer):
                pc_layers.append(module)
        return pc_layers

    def get_energies(self):
        """Get energies from all PC layers."""
        energies = []
        for pc_layer in self.get_pc_layers():
            energy = pc_layer.energy()
            if energy is not None:
                energies.append(energy)
        return energies

    def get_value_nodes(self):
        """Get value nodes (x) from all PC layers for inference optimization."""
        value_nodes = []
        for pc_layer in self.get_pc_layers():
            x = pc_layer.get_x()
            if x is not None:
                value_nodes.append(x)
        return value_nodes

    def set_sample_x(self, value: bool):
        """Set all PC layers to re-sample x on next forward."""
        for pc_layer in self.get_pc_layers():
            pc_layer.set_is_sample_x(value)

    def clear_energies(self):
        """Clear all accumulated energies."""
        for pc_layer in self.get_pc_layers():
            pc_layer.clear_energy()

    def get_network_parameters(self):
        """Get only the actual network parameters (weights/biases), excluding value nodes.

        This is critical for the weight optimizer - it should NOT optimize value nodes,
        only the Linear layer parameters.

        Returns:
            Generator of parameters (excludes PCLayer value nodes)
        """
        # Get all value nodes to exclude
        value_node_set = set(self.get_value_nodes())

        # Yield only parameters that are NOT value nodes
        for param in self.parameters():
            if not any(param is x for x in value_node_set):
                yield param
