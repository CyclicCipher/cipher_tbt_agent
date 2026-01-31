"""
Lightweight diagnostics for PC networks.

Quick checks and visualizations without requiring full training runs:
- Weight statistics
- State dynamics visualization
- Error propagation analysis
- Single-sample inference testing
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from categorical_network import (
    PCConvLayer,
    PCConvVisionPreprocessor,
    CanonicalPCLayer,
    CanonicalMicrocircuit
)


def check_weight_stats(model) -> Dict:
    """Get weight statistics for all layers."""
    stats = {}

    # PC conv layers
    if hasattr(model, 'pc_conv_preprocessor'):
        conv_stats = model.pc_conv_preprocessor.get_weight_stats()
        stats['conv_layers'] = conv_stats

    # PC inference layers
    if hasattr(model, 'pc_inference'):
        pc_stats = {}
        for i, layer in enumerate([model.pc_inference.layer0,
                                   model.pc_inference.layer1,
                                   model.pc_inference.layer2]):
            w = layer.W_feedforward.weight.data
            pc_stats[f'layer{i}'] = {
                'mean': w.mean().item(),
                'std': w.std().item(),
                'abs_mean': w.abs().mean().item(),
                'has_lateral': layer.W_lateral is not None,
                'has_feedback': layer.W_feedback is not None
            }
        stats['pc_inference_layers'] = pc_stats

    return stats


def check_state_dynamics(model, input_data: torch.Tensor, num_iterations: int = 20):
    """Trace state evolution during inference."""
    states_history = {
        'layer0': [],
        'layer1': [],
        'layer2': []
    }

    errors_history = {
        'layer0': [],
        'layer1': [],
        'layer2': []
    }

    # Run inference and track states
    model.reset_states()

    for iteration in range(num_iterations):
        # Layer 0
        ff_0 = model.pc_inference.layer0.compute_feedforward(input_data)
        lat_0 = model.pc_inference.layer0.compute_lateral()
        fb_0 = model.pc_inference.layer0.compute_feedback(
            model.pc_inference.layer1.get_state()
        )

        target_0 = ff_0 + 0.5 * lat_0 + fb_0
        error_0 = model.pc_inference.layer0.state - target_0
        model.pc_inference.layer0.state.data -= 0.1 * error_0.data

        states_history['layer0'].append(model.pc_inference.layer0.state.norm().item())
        errors_history['layer0'].append(error_0.norm().item())

        # Layer 1
        ff_1 = model.pc_inference.layer1.compute_feedforward(
            model.pc_inference.layer0.get_state()
        )
        fb_1 = model.pc_inference.layer1.compute_feedback(
            model.pc_inference.layer2.get_state()
        )

        target_1 = ff_1 + fb_1
        error_1 = model.pc_inference.layer1.state - target_1
        model.pc_inference.layer1.state.data -= 0.1 * error_1.data

        states_history['layer1'].append(model.pc_inference.layer1.state.norm().item())
        errors_history['layer1'].append(error_1.norm().item())

        # Layer 2
        ff_2 = model.pc_inference.layer2.compute_feedforward(
            model.pc_inference.layer1.get_state()
        )

        target_2 = ff_2
        error_2 = model.pc_inference.layer2.state - target_2
        model.pc_inference.layer2.state.data -= 0.1 * error_2.data

        states_history['layer2'].append(model.pc_inference.layer2.state.norm().item())
        errors_history['layer2'].append(error_2.norm().item())

    return states_history, errors_history


def test_single_sample(model, image: torch.Tensor, label: int = None):
    """Test model on a single sample and return detailed info."""
    model.reset_states()

    # Get conv features
    conv_features = model.pc_conv_preprocessor.forward(
        image,
        num_iterations=20,
        inference_lr=0.1
    )

    # Run inference
    output = model(image, target=None)

    pred_class = output.squeeze().argmax().item()
    confidence = torch.softmax(output.squeeze(), dim=0).max().item()

    result = {
        'predicted_class': pred_class,
        'confidence': confidence,
        'output_vector': output.squeeze().detach().cpu().numpy(),
        'correct': pred_class == label if label is not None else None
    }

    return result


def print_network_summary(model):
    """Print a summary of the network architecture."""
    print("=" * 60)
    print("NETWORK ARCHITECTURE SUMMARY")
    print("=" * 60)

    if hasattr(model, 'pc_conv_preprocessor'):
        print("\nPC Convolutional Layers:")
        for i, layer in enumerate(model.pc_conv_preprocessor.pc_layers):
            print(f"  Layer {i}: {layer.in_channels}→{layer.out_channels} channels")
            print(f"    Precision: {layer.precision.item():.2f}")
            print(f"    Kernel size: {layer.kernel_size}, Stride: {layer.stride}")

    if hasattr(model, 'pc_inference'):
        print("\nPC Inference Layers:")
        for i, layer in enumerate([model.pc_inference.layer0,
                                   model.pc_inference.layer1,
                                   model.pc_inference.layer2]):
            print(f"  Layer {i}: {layer.num_neurons} neurons")
            print(f"    Lateral: {layer.W_lateral is not None}")
            print(f"    Feedback: {layer.W_feedback is not None}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("=" * 60)


def diagnose_model(model):
    """Run comprehensive diagnostics on a model."""
    print("\nRunning diagnostics...")

    # Print summary
    print_network_summary(model)

    # Check weight stats
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)
    weight_stats = check_weight_stats(model)

    if 'conv_layers' in weight_stats:
        print("\nConvolutional Layers:")
        for layer_name, stats in weight_stats['conv_layers'].items():
            print(f"  {layer_name}:")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Std: {stats['std']:.6f}")
            print(f"    Precision: {stats['precision']:.2f}")

    if 'pc_inference_layers' in weight_stats:
        print("\nInference Layers:")
        for layer_name, stats in weight_stats['pc_inference_layers'].items():
            print(f"  {layer_name}:")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Std: {stats['std']:.6f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    print("PC Network Diagnostics Tool")
    print("=" * 60)

    # Create a dummy model for testing
    from train_mnist import PCConvClassifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PCConvClassifier(num_classes=10, dtype=torch.float32).to(device)

    # Run diagnostics
    diagnose_model(model)

    # Test with random input
    print("\nTesting with random input...")
    random_image = torch.randn(3, 100, 100, device=device)
    result = test_single_sample(model, random_image)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")

    print("\n" + "=" * 60)
    print("Diagnostics complete")
    print("=" * 60)
