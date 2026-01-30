"""
Diagnostic tools for analyzing PC network training.

Usage:
    python diagnostics.py --check inference  # Check inference convergence
    python diagnostics.py --check features   # Check feature quality
    python diagnostics.py --check weights    # Check weight magnitudes
    python diagnostics.py --compare          # Compare PC vs backprop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
import argparse

from train_vision_mnist_pc import VisionPCClassifier


def diagnose_inference_convergence(model, image, num_iterations=50, device='cuda'):
    """
    Check if PC inference is converging properly.

    Good convergence: errors decrease exponentially to near-zero
    Bad convergence: errors stay high or oscillate
    """
    print("=" * 60)
    print("DIAGNOSTIC: Inference Convergence")
    print("=" * 60)

    model.eval()
    image = image.to(device)

    # Reset state
    model.pc_inference.layer0.state.data.zero_()
    model.pc_inference.layer1.state.data.zero_()
    model.pc_inference.layer2.state.data.zero_()

    # Conv features
    conv_features = model.conv_preprocess(image).squeeze(0)

    # Track errors over iterations
    errors_0 = []
    errors_1 = []
    errors_2 = []

    with torch.no_grad():
        for i in range(num_iterations):
            # Single inference step
            ff_0 = model.pc_inference.layer0.compute_feedforward(conv_features)
            lat_0 = model.pc_inference.layer0.compute_lateral()
            fb_0 = model.pc_inference.layer0.compute_feedback(
                model.pc_inference.layer1.get_state()
            )

            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = model.pc_inference.layer0.state - target_0
            model.pc_inference.layer0.state.data -= 0.1 * error_0.data

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

            # Layer 2
            ff_2 = model.pc_inference.layer2.compute_feedforward(
                model.pc_inference.layer1.get_state()
            )
            target_2 = ff_2
            error_2 = model.pc_inference.layer2.state - target_2
            model.pc_inference.layer2.state.data -= 0.1 * error_2.data

            # Record error magnitudes
            errors_0.append(error_0.norm().item())
            errors_1.append(error_1.norm().item())
            errors_2.append(error_2.norm().item())

    # Plot convergence
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(errors_0)
    plt.xlabel('Iteration')
    plt.ylabel('Error Magnitude')
    plt.title('Layer 0 Error Convergence')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(errors_1)
    plt.xlabel('Iteration')
    plt.ylabel('Error Magnitude')
    plt.title('Layer 1 Error Convergence')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(errors_2)
    plt.xlabel('Iteration')
    plt.ylabel('Error Magnitude')
    plt.title('Layer 2 Error Convergence')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('diagnostics_inference_convergence.png')
    print("✓ Saved convergence plot to diagnostics_inference_convergence.png")

    # Analysis
    final_error_0 = errors_0[-1]
    final_error_1 = errors_1[-1]
    final_error_2 = errors_2[-1]

    print(f"\nFinal errors after {num_iterations} iterations:")
    print(f"  Layer 0: {final_error_0:.6f}")
    print(f"  Layer 1: {final_error_1:.6f}")
    print(f"  Layer 2: {final_error_2:.6f}")

    # Check convergence
    converged = all([
        final_error_0 < 0.1,
        final_error_1 < 0.1,
        final_error_2 < 0.1
    ])

    if converged:
        print("\n✓ GOOD: Inference converged (errors < 0.1)")
    else:
        print("\n✗ BAD: Inference did not converge")
        print("  → Try increasing num_iterations or learning rate")

    return {
        'errors_0': errors_0,
        'errors_1': errors_1,
        'errors_2': errors_2,
        'converged': converged
    }


def diagnose_feature_quality(model, train_loader, test_loader, device='cuda'):
    """
    Check if conv features are discriminative.

    Good features: within-class distance < between-class distance
    Bad features: random/collapsed
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Feature Quality")
    print("=" * 60)

    model.eval()

    # Extract features for each class
    features_by_class = {i: [] for i in range(10)}

    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            feat = model.conv_preprocess(data).squeeze(0).cpu().numpy()
            features_by_class[target.item()].append(feat)

            # Limit to 100 samples per class
            if all(len(v) >= 100 for v in features_by_class.values()):
                break

    # Convert to arrays
    for k in features_by_class:
        features_by_class[k] = np.stack(features_by_class[k])

    # Compute within-class and between-class distances
    within_class_dists = []
    between_class_dists = []

    for class_idx in range(10):
        features = features_by_class[class_idx]

        # Within-class: pairwise distances within same class
        if len(features) > 1:
            dist_matrix = pairwise_distances(features)
            # Take upper triangle (exclude diagonal)
            within = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            within_class_dists.extend(within)

        # Between-class: distances to other classes
        for other_class in range(10):
            if other_class != class_idx:
                other_features = features_by_class[other_class]
                between = pairwise_distances(features, other_features).flatten()
                between_class_dists.extend(between)

    within_mean = np.mean(within_class_dists)
    between_mean = np.mean(between_class_dists)
    ratio = between_mean / within_mean if within_mean > 0 else 0

    print(f"\nFeature separation:")
    print(f"  Within-class distance:  {within_mean:.4f}")
    print(f"  Between-class distance: {between_mean:.4f}")
    print(f"  Ratio (between/within): {ratio:.4f}")

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(within_class_dists, bins=50, alpha=0.5, label='Within-class', density=True)
    plt.hist(between_class_dists, bins=50, alpha=0.5, label='Between-class', density=True)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Feature Distance Distributions')
    plt.legend()
    plt.grid(True)
    plt.savefig('diagnostics_feature_quality.png')
    print("✓ Saved feature quality plot to diagnostics_feature_quality.png")

    # Analysis
    if ratio > 1.5:
        print("\n✓ GOOD: Features are discriminative (ratio > 1.5)")
    elif ratio > 1.0:
        print("\n⚠ OK: Features have some separation (ratio > 1.0)")
    else:
        print("\n✗ BAD: Features are not discriminative (ratio < 1.0)")
        print("  → Conv layers may not be learning useful representations")

    return {
        'within_mean': within_mean,
        'between_mean': between_mean,
        'ratio': ratio
    }


def diagnose_weight_magnitudes(model):
    """
    Check weight statistics to detect vanishing/exploding weights.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Weight Magnitudes")
    print("=" * 60)

    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_norm = param.data.norm().item()
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()

            print(f"\n{name}:")
            print(f"  Norm: {weight_norm:.6f}")
            print(f"  Mean: {weight_mean:.6f}")
            print(f"  Std:  {weight_std:.6f}")

            if weight_norm < 1e-6:
                print(f"  ⚠ WARNING: Very small weights (vanishing)")
            if weight_norm > 1000:
                print(f"  ⚠ WARNING: Very large weights (exploding)")


def compare_pc_vs_backprop():
    """
    Compare PC learning vs backprop on same data.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: PC vs Backprop Comparison")
    print("=" * 60)
    print("\nThis would train two models side-by-side:")
    print("  1. PC learning (train_vision_mnist_pc.py)")
    print("  2. Backprop (train_vision_mnist.py)")
    print("\nThen compare final accuracy and generalization gap.")
    print("\nTo run: execute both scripts and compare results manually.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Diagnostic tools for PC network')
    parser.add_argument('--check', choices=['inference', 'features', 'weights', 'all'],
                        default='all', help='Which diagnostic to run')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        '../data', train=False, transform=transform
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, range(1000)),
        batch_size=1
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(test_dataset, range(100)),
        batch_size=1
    )

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = VisionPCClassifier(num_classes=10, use_4bit=False)
    model = model.to(device)

    if args.model:
        model.load_state_dict(torch.load(args.model))
        print(f"Loaded model from {args.model}")

    # Run diagnostics
    if args.check in ['inference', 'all']:
        # Get a sample image
        sample_image, _ = next(iter(train_loader))
        diagnose_inference_convergence(model, sample_image, device=device)

    if args.check in ['features', 'all']:
        diagnose_feature_quality(model, train_loader, test_loader, device=device)

    if args.check in ['weights', 'all']:
        diagnose_weight_magnitudes(model)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
