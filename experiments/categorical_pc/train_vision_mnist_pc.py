"""
MNIST training with PROPER predictive coding learning.

Key differences from train_vision_mnist.py:
- NO backprop (.backward())
- NO optimizer (Adam/SGD)
- Local Hebbian-like learning rules
- Supervised via output clamping

This should generalize much better!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Import our categorical network
from categorical_network_impl import (
    ConvolutionalVisionPreprocessor,
    CanonicalMicrocircuit,
    HAS_4BIT
)


class VisionPCClassifier(nn.Module):
    """
    Vision classifier using convolutional preprocessor + PC inference + PC learning.
    """

    def __init__(self, num_classes: int = 10, use_4bit: bool = False):  # Disable 4-bit for now
        super().__init__()

        self.use_4bit = use_4bit and HAS_4BIT
        dtype = torch.float32  # Use FP32 for stable learning

        # Convolutional preprocessor: 784 (28×28) → 1024
        self.conv_preprocess = nn.Sequential(
            # Upsample 28×28 to 32×32 for cleaner conv math
            nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),

            # Layer 0: Simple cells (32×32×1 → 16×16×64)
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.Tanh(),

            # Layer 1: Complex cells (16×16×64 → 8×8×128)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),

            # Layer 2: Higher features (8×8×128 → 4×4×256)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),

            # Flatten to 4×4×256 = 4096 dims
            nn.Flatten(),

            # Dense projection to 1024 (aligned to 64)
            nn.Linear(4096, 1024),
            nn.Tanh()
        )

        # Predictive coding inference over features
        self.pc_inference = CanonicalMicrocircuit(
            input_size=1024,   # From conv preprocessor
            layer0_size=512,   # 8 * 64
            layer1_size=256,   # 4 * 64
            layer2_size=num_classes,  # Output layer = num classes
            is_granular=True,
            use_4bit=self.use_4bit,
            dtype=dtype
        )

        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor = None,
        num_iterations: int = 10
    ) -> torch.Tensor:
        """
        Args:
            x: (1, 1, 28, 28) MNIST image
            target: (optional) class label for supervised learning
            num_iterations: PC inference iterations

        Returns:
            (1, num_classes) logits
        """
        # Reset PC network state
        self.pc_inference.layer0.state.data.zero_()
        self.pc_inference.layer1.state.data.zero_()
        self.pc_inference.layer2.state.data.zero_()

        # Conv preprocessing
        conv_features = self.conv_preprocess(x)  # (1, 1024)
        conv_features = conv_features.squeeze(0)  # (1024,)

        # Create target representation if supervised
        target_output = None
        if target is not None:
            # One-hot encoding
            target_output = torch.zeros(self.num_classes, device=x.device)
            target_output[target] = 1.0

        # PC inference (with optional clamping)
        output, error_0, error_1, error_2 = self.pc_inference.forward_with_errors(
            conv_features,
            num_iterations=num_iterations,
            target_output=target_output
        )

        # Store errors for learning
        self.last_conv_features = conv_features
        self.last_errors = (error_0, error_1, error_2)

        # Return with batch dimension
        return output.unsqueeze(0)  # (1, num_classes)

    def update_weights_pc(self, learning_rate: float = 0.01):
        """
        Update weights using local predictive coding learning rules.

        No backprop - pure local Hebbian learning!
        """
        error_0, error_1, error_2 = self.last_errors

        # Update PC network weights
        self.pc_inference.update_weights_pc(
            input_data=self.last_conv_features,
            error_0=error_0,
            error_1=error_1,
            error_2=error_2,
            learning_rate=learning_rate
        )


def train_epoch_pc(model, train_loader, device, learning_rate=0.01, num_iterations=10):
    """Train for one epoch using predictive coding learning."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass (includes inference)
        output = model(data, target=target.item(), num_iterations=num_iterations)

        # Update weights using local PC learning
        model.update_weights_pc(learning_rate=learning_rate)

        # Compute loss for monitoring (not used for learning!)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test(model, test_loader, device, num_iterations=10):
    """Test the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward without clamping (unsupervised inference)
            output = model(data, target=None, num_iterations=num_iterations)
            loss = F.cross_entropy(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("VISION ENCODER VALIDATION - MNIST")
    print("Using PROPER Predictive Coding Learning")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 1  # Process one sample at a time
    num_epochs = 5
    learning_rate = 0.01  # PC learning rate
    num_iterations = 20  # More iterations for better convergence

    # Data
    print("\nLoading MNIST...")
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

    # Use smaller subset for faster testing
    train_subset = torch.utils.data.Subset(train_dataset, range(6000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")

    # Model
    print("\nCreating model...")
    model = VisionPCClassifier(num_classes=10, use_4bit=False)  # No 4-bit for now
    model = model.to(device)

    # NO OPTIMIZER - using local PC learning!
    print("Using local predictive coding learning (no backprop)")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING WITH PREDICTIVE CODING")
    print("=" * 60)

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch_pc(
            model, train_loader, device, learning_rate, num_iterations
        )
        test_loss, test_acc = test(
            model, test_loader, device, num_iterations
        )

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print(f"  Generalization gap: {train_acc - test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final train accuracy: {train_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final generalization gap: {train_acc - test_acc:.2f}%")

    if test_acc > 50:
        print("\n✓ Vision encoder WORKING - test accuracy > 50%")
        if train_acc - test_acc < 20:
            print("✓ Good generalization - gap < 20%")
        else:
            print("⚠ Large generalization gap - may need regularization")
    else:
        print("\n✗ Vision encoder still BROKEN - test accuracy < 50%")

    print("=" * 60)

    # Return metrics for diagnostics
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc,
        'best_test_acc': best_test_acc
    }


if __name__ == "__main__":
    results = main()
