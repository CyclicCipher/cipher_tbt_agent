"""
Test vision encoder generalization on MNIST.

This validates that the convolutional vision encoder fixes the broken
dense encoder (which achieved 100% train, 10% test).

We test:
1. Convolutional preprocessor + PC network
2. Train/test accuracy to measure generalization
3. Compare to baseline if needed
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
    Vision classifier using convolutional preprocessor + PC inference.
    """

    def __init__(self, num_classes: int = 10, use_4bit: bool = True):
        super().__init__()

        self.use_4bit = use_4bit and HAS_4BIT
        dtype = torch.float16 if self.use_4bit else torch.float32

        # Convolutional preprocessor: 784 (28×28) → 1024
        # Note: MNIST is 28×28, not 100×100, so we need to adapt
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
            layer2_size=256,   # 4 * 64
            is_granular=True,
            use_4bit=self.use_4bit,
            dtype=dtype
        )

        # Classifier head (keep in FP32/FP16, not 4-bit)
        # Only 2560 params (256*10), so quantization not needed
        # 4-bit can have gradient issues with bias in small layers
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, num_iterations: int = 10) -> torch.Tensor:
        """
        Args:
            x: (1, 1, 28, 28) MNIST image (batch_size must be 1)
            num_iterations: PC inference iterations

        Returns:
            (1, num_classes) logits
        """
        # State is already reset to zero after each training step
        # Conv preprocessing
        conv_features = self.conv_preprocess(x)  # (1, 1024)
        conv_features = conv_features.squeeze(0)  # (1024,)

        # PC inference
        pc_features = self.pc_inference(conv_features, num_iterations)  # (256,)

        # Classification (cast to FP32 if needed for classifier)
        if pc_features.dtype == torch.float16:
            pc_features = pc_features.float()
        logits = self.classifier(pc_features)  # (10,)

        # Return with batch dimension
        return logits.unsqueeze(0)  # (1, 10)


def train_epoch(model, train_loader, optimizer, device, num_iterations=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Clear any lingering gradients and computation graph
        optimizer.zero_grad(set_to_none=True)

        output = model(data, num_iterations=num_iterations)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Explicitly detach state buffers after backward to prevent graph accumulation
        with torch.no_grad():
            model.pc_inference.layer0.state.data.zero_()
            model.pc_inference.layer1.state.data.zero_()
            model.pc_inference.layer2.state.data.zero_()

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
            output = model(data, num_iterations=num_iterations)
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
    print("=" * 60)
    print(f"4-bit available: {HAS_4BIT}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    batch_size = 1  # Must be 1 due to PC network state reuse issues
    num_epochs = 5
    learning_rate = 0.001
    num_iterations = 10  # PC inference iterations

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

    # Use smaller subset for faster testing (can increase later)
    train_subset = torch.utils.data.Subset(train_dataset, range(6000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")

    # Model
    print("\nCreating model...")
    model = VisionPCClassifier(num_classes=10, use_4bit=True)

    if HAS_4BIT and torch.cuda.is_available():
        model = model.to(device)
        print("Model moved to CUDA for 4-bit quantization")
    else:
        model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, num_iterations
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

    if test_acc > 50:  # Baseline check
        print("\n✓ Vision encoder WORKING - test accuracy > 50%")
        if train_acc - test_acc < 20:
            print("✓ Good generalization - gap < 20%")
        else:
            print("⚠ Large generalization gap - may need regularization")
    else:
        print("\n✗ Vision encoder still BROKEN - test accuracy < 50%")

    print("=" * 60)


if __name__ == "__main__":
    main()
