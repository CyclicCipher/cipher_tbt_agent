"""
Visualize what the model actually "sees" - answer the question:
Can the model see the digits? Is it looking at the right thing?

This will show:
1. Original MNIST digits
2. What conv layer 0 detects (edge filters)
3. What conv layer 1 detects (texture patterns)
4. What conv layer 2 detects (higher-level features)
5. Final features fed to PC network

If the model can't see digits, we'll know immediately.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from train_vision_mnist import VisionPCClassifier


def visualize_what_model_sees(model, num_samples=10, device='cuda'):
    """
    Visualize the visual processing pipeline.

    Shows what each conv layer detects for sample digits.
    """
    print("=" * 60)
    print("VISUALIZING: What Does The Model See?")
    print("=" * 60)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        '../data', train=False, download=True, transform=transform
    )

    model.eval()

    # Get sample images (one per digit)
    samples = []
    labels_seen = set()

    for img, label in test_dataset:
        if label not in labels_seen:
            samples.append((img, label))
            labels_seen.add(label)
        if len(samples) >= 10:
            break

    # Sort by label for clean visualization
    samples = sorted(samples, key=lambda x: x[1])

    # Create large figure
    fig = plt.figure(figsize=(20, 12))

    for idx, (img, label) in enumerate(samples):
        img_batch = img.unsqueeze(0).to(device)

        with torch.no_grad():
            # Original image (denormalized for visualization)
            original = img.squeeze().cpu().numpy()
            original = (original * 0.3081) + 0.1307  # Denormalize
            original = np.clip(original, 0, 1)

            # Pass through conv layers and capture activations
            conv_preprocess = model.conv_preprocess

            # Upsample
            x = F.interpolate(img_batch, size=(32, 32), mode='bilinear', align_corners=False)

            # Layer 0: Conv 1→64 (edge detection)
            x = conv_preprocess[1](x)  # Conv2d
            conv0_out = x.clone()
            x = conv_preprocess[2](x)  # Tanh

            # Layer 1: Conv 64→128 (texture patterns)
            x = conv_preprocess[3](x)  # Conv2d
            conv1_out = x.clone()
            x = conv_preprocess[4](x)  # Tanh

            # Layer 2: Conv 128→256 (higher features)
            x = conv_preprocess[5](x)  # Conv2d
            conv2_out = x.clone()
            x = conv_preprocess[6](x)  # Tanh

            # Flatten and final dense
            x = conv_preprocess[7](x)  # Flatten
            x = conv_preprocess[8](x)  # Linear
            final_features = conv_preprocess[9](x)  # Tanh

            # Visualize each stage
            # Column 1: Original
            plt.subplot(10, 6, idx * 6 + 1)
            plt.imshow(original, cmap='gray')
            plt.title(f'Digit {label}')
            plt.axis('off')

            # Column 2: Conv0 (show first 4 channels as 2x2 grid)
            plt.subplot(10, 6, idx * 6 + 2)
            conv0_viz = conv0_out[0, :4].cpu().numpy()  # First 4 channels
            conv0_grid = np.concatenate([
                np.concatenate([conv0_viz[0], conv0_viz[1]], axis=1),
                np.concatenate([conv0_viz[2], conv0_viz[3]], axis=1)
            ], axis=0)
            plt.imshow(conv0_grid, cmap='viridis')
            plt.title('Conv0 (edges)')
            plt.axis('off')

            # Column 3: Conv1 (show first 4 channels)
            plt.subplot(10, 6, idx * 6 + 3)
            conv1_viz = conv1_out[0, :4].cpu().numpy()
            conv1_grid = np.concatenate([
                np.concatenate([conv1_viz[0], conv1_viz[1]], axis=1),
                np.concatenate([conv1_viz[2], conv1_viz[3]], axis=1)
            ], axis=0)
            plt.imshow(conv1_grid, cmap='viridis')
            plt.title('Conv1 (textures)')
            plt.axis('off')

            # Column 4: Conv2 (show first 4 channels)
            plt.subplot(10, 6, idx * 6 + 4)
            conv2_viz = conv2_out[0, :4].cpu().numpy()
            conv2_grid = np.concatenate([
                np.concatenate([conv2_viz[0], conv2_viz[1]], axis=1),
                np.concatenate([conv2_viz[2], conv2_viz[3]], axis=1)
            ], axis=0)
            plt.imshow(conv2_grid, cmap='viridis')
            plt.title('Conv2 (features)')
            plt.axis('off')

            # Column 5: Feature activity histogram
            plt.subplot(10, 6, idx * 6 + 5)
            features = final_features.cpu().numpy().flatten()
            plt.hist(features, bins=50, alpha=0.7)
            plt.title('Feature distribution')
            plt.xlim(-1, 1)
            plt.ylim(0, 100)

            # Column 6: Feature statistics
            plt.subplot(10, 6, idx * 6 + 6)
            plt.text(0.1, 0.8, f'Mean: {features.mean():.3f}', fontsize=10)
            plt.text(0.1, 0.6, f'Std: {features.std():.3f}', fontsize=10)
            plt.text(0.1, 0.4, f'Min: {features.min():.3f}', fontsize=10)
            plt.text(0.1, 0.2, f'Max: {features.max():.3f}', fontsize=10)
            plt.title('Feature stats')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('model_vision_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to: model_vision_visualization.png")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: Can The Model See?")
    print("=" * 60)

    # Check if features are discriminative
    print("\nChecking if features respond to different digits...")

    feature_responses = []
    for img, label in samples:
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.conv_preprocess(img_batch).squeeze(0).cpu().numpy()
            feature_responses.append((label, features))

    # Compute mean features per digit
    digit_features = {}
    for label, features in feature_responses:
        if label not in digit_features:
            digit_features[label] = []
        digit_features[label].append(features)

    for label in sorted(digit_features.keys()):
        mean_feat = np.mean(digit_features[label], axis=0)
        print(f"  Digit {label}: mean={mean_feat.mean():.3f}, std={mean_feat.std():.3f}")

    # Check if conv filters are doing anything
    print("\nChecking conv filter activity...")
    sample_img = samples[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        x = F.interpolate(sample_img, size=(32, 32), mode='bilinear', align_corners=False)

        # Conv0
        conv0_weight = model.conv_preprocess[1].weight
        print(f"  Conv0 weights: mean={conv0_weight.mean():.6f}, std={conv0_weight.std():.6f}")

        x = model.conv_preprocess[1](x)
        print(f"  Conv0 output: mean={x.mean():.3f}, std={x.std():.3f}, max={x.max():.3f}")

        # Conv1
        x = model.conv_preprocess[2](x)
        conv1_weight = model.conv_preprocess[3].weight
        print(f"  Conv1 weights: mean={conv1_weight.mean():.6f}, std={conv1_weight.std():.6f}")

        x = model.conv_preprocess[3](x)
        print(f"  Conv1 output: mean={x.mean():.3f}, std={x.std():.3f}, max={x.max():.3f}")

        # Conv2
        x = model.conv_preprocess[4](x)
        conv2_weight = model.conv_preprocess[5].weight
        print(f"  Conv2 weights: mean={conv2_weight.mean():.6f}, std={conv2_weight.std():.6f}")

        x = model.conv_preprocess[5](x)
        print(f"  Conv2 output: mean={x.mean():.3f}, std={x.std():.3f}, max={x.max():.3f}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT:")
    print("=" * 60)

    # Check if weights are too small (not learning)
    conv_weights_ok = conv0_weight.std() > 0.01 and conv1_weight.std() > 0.01

    # Check if activations are reasonable
    activations_ok = x.std() > 0.1

    if not conv_weights_ok:
        print("✗ PROBLEM: Conv weights have very small std")
        print("  → Weights haven't changed from initialization")
        print("  → Conv layers are NOT learning!")
        print("\n  ROOT CAUSE: We're using backprop on PC network")
        print("  but gradients aren't flowing to conv layers!")

    if not activations_ok:
        print("✗ PROBLEM: Conv activations are very small/uniform")
        print("  → Features are not discriminative")
        print("  → Model is effectively blind")

    if conv_weights_ok and activations_ok:
        print("✓ Conv layers appear to be functioning")
        print("  → Check visualization to see what they detect")

    print("\nOpen 'model_vision_visualization.png' to see what model sees!")
    print("=" * 60)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = VisionPCClassifier(num_classes=10, use_4bit=False)
    model = model.to(device)

    print("Analyzing untrained (random) model...")
    print("This shows what a model with frozen conv layers sees.\n")

    visualize_what_model_sees(model, device=device)


if __name__ == "__main__":
    main()
