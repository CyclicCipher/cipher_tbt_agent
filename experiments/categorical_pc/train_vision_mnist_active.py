"""
MNIST training with Active Inference & Curiosity-Driven Learning

Key Features:
- Predictive coding learning (local Hebbian rules)
- Active curriculum manager (learning progress-based sampling)
- Curiosity-driven sample selection (Zone of Proximal Development)
- Replaces random shuffling with intelligent sample prioritization

Based on:
- Oudeyer et al. (2007): Learning Progress framework
- Friston et al. (2017): Active Inference
- Schmidhuber (2010): Compression Progress

Expected Result:
- Faster convergence than random sampling
- Better data efficiency (fewer samples needed to reach accuracy)
- Clear developmental stages (easy digits mastered first, then harder ones)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our categorical network
from categorical_network_impl import (
    ConvolutionalVisionPreprocessor,
    CanonicalMicrocircuit,
    HAS_4BIT
)

# Import diagnostics
from diagnostics_training import TrainingDiagnostics

# Import active inference components
from src.active_inference import ActiveCurriculumManager, LearningProgressTracker


class VisionPCClassifier(nn.Module):
    """
    Vision classifier using convolutional preprocessor + PC inference + PC learning.
    """

    def __init__(self, num_classes: int = 10, use_4bit: bool = False):
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

        # PURE PC LEARNING: Disable gradients on conv layers
        # Conv layers will learn via their own local rules (or stay fixed)
        for param in self.conv_preprocess.parameters():
            param.requires_grad = False

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

    def get_total_error(self) -> float:
        """
        Get total prediction error across all PC layers.

        This is used by the curriculum manager to track learning progress.
        """
        error_0, error_1, error_2 = self.last_errors
        total_error = (
            torch.mean(error_0 ** 2).item() +
            torch.mean(error_1 ** 2).item() +
            torch.mean(error_2 ** 2).item()
        ) / 3.0
        return total_error


def train_epoch_active(
    model,
    train_dataset,
    curriculum_manager,
    device,
    learning_rate=0.01,
    num_iterations=10,
    diagnostics=None,
    epoch=None,
    use_active_sampling=True,
):
    """
    Train for one epoch using active inference curriculum.

    Instead of random sampling, the curriculum manager selects samples
    based on learning progress.

    PURE PC LEARNING - No backprop, only local learning rules!

    Args:
        model: The PC classifier
        train_dataset: Full training dataset (not DataLoader)
        curriculum_manager: Active curriculum manager
        device: torch device
        learning_rate: PC learning rate
        num_iterations: PC inference iterations
        diagnostics: Diagnostics tracker
        epoch: Current epoch number
        use_active_sampling: If False, fall back to random sampling

    Returns:
        avg_loss, accuracy, curriculum_stats
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Get epoch indices from curriculum manager
    curriculum_manager.start_epoch()

    if use_active_sampling:
        epoch_indices = curriculum_manager.get_epoch_indices(prioritize_learnable=True)
        print(f"\n[Active Curriculum] Using learning progress-based sampling")
    else:
        epoch_indices = list(range(len(train_dataset)))
        np.random.shuffle(epoch_indices)
        print(f"\n[Baseline] Using random sampling")

    # Training loop
    for step_idx, sample_idx in enumerate(epoch_indices):
        # Get sample from dataset
        data, target = train_dataset[sample_idx]
        data = data.unsqueeze(0).to(device)  # Add batch dimension
        target = torch.tensor([target]).to(device)

        # Forward pass (includes PC inference with clamping)
        output = model(data, target=target.item(), num_iterations=num_iterations)

        # Update weights using ONLY PC local learning rules (NO backprop!)
        model.update_weights_pc(learning_rate=learning_rate)

        # Compute loss for monitoring only (no backward pass)
        with torch.no_grad():
            loss = F.cross_entropy(output, target)

        # Get prediction error for curriculum manager
        prediction_error = model.get_total_error()

        # Update curriculum manager with learning progress
        if use_active_sampling:
            sample_stats = curriculum_manager.update(
                sample_idx=sample_idx,
                error=prediction_error,
                logits=output.squeeze(0),  # Pass logits for EFE computation
                target=target.item(),      # Pass true label
                additional_stats={
                    'loss': loss.item(),
                    'correct': (output.argmax(dim=1) == target).item(),
                }
            )

        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Print progress
        if step_idx % 500 == 0:
            if use_active_sampling and step_idx > 0:
                print(f'  Step {step_idx}/{len(epoch_indices)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%, '
                      f'Sample {sample_idx} - Category: {sample_stats["category"]}, '
                      f'LP: {sample_stats["learning_progress"]:.4f}')
            else:
                print(f'  Step {step_idx}/{len(epoch_indices)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

    avg_loss = total_loss / len(epoch_indices)
    accuracy = 100. * correct / total

    # Get curriculum statistics
    curriculum_stats = curriculum_manager.get_statistics() if use_active_sampling else {}

    return avg_loss, accuracy, curriculum_stats


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
    print("ACTIVE INFERENCE MNIST TRAINING")
    print("Curiosity-Driven Learning with Predictive Coding")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.01  # PC learning rate
    num_iterations = 20  # PC inference iterations
    use_active_sampling = True  # Set to False to compare with baseline

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

    # Create DataLoader only for test set (train will be managed by curriculum)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    print(f"Train samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")

    # Model
    print("\nCreating model...")
    model = VisionPCClassifier(num_classes=10, use_4bit=False)
    model = model.to(device)

    print("Using PURE predictive coding learning:")
    print("  - PC layers: Local Hebbian learning rules")
    print("  - Conv layers: Frozen (no learning)")
    print("  - NO backprop anywhere!")

    # Initialize Active Curriculum Manager
    print("\nInitializing Active Curriculum Manager...")
    curriculum_manager = ActiveCurriculumManager(
        num_samples=len(train_subset),
        num_classes=10,  # MNIST has 10 classes
        sampling_strategy='learning_progress',  # Options: 'random', 'learning_progress', 'pure_epistemic', 'balanced'
        temperature=1.0,
        exploration_rate=0.1,
        batch_size=1,
        epistemic_weight=1.0,    # Weight for exploration (uncertainty reduction)
        pragmatic_weight=1.0,    # Weight for exploitation (goal achievement)
        window_size=20,
        mastery_threshold=0.05,  # Low error = mastered
        noise_threshold=1.5,     # High persistent error = noise
        noise_patience=50,       # Attempts before declaring unlearnable
    )

    # Initialize diagnostics
    print("\nInitializing diagnostics...")
    diagnostics = TrainingDiagnostics(model)

    # Pre-training diagnostics
    print("\n" + "=" * 60)
    print("PRE-TRAINING DIAGNOSTICS")
    print("=" * 60)

    # Get a sample for diagnostics
    sample_data, _ = train_subset[0]
    sample_data = sample_data.unsqueeze(0).to(device)

    # Check device placement
    diagnostics.check_device_placement(sample_data)

    # Check PC inference quality
    diagnostics.check_pc_inference_quality(model, sample_data, device)

    # Check for learning conflicts
    diagnostics.check_learning_conflicts()

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING WITH ACTIVE INFERENCE")
    print("=" * 60)

    best_test_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # Train with active curriculum
        train_loss, train_acc, curriculum_stats = train_epoch_active(
            model=model,
            train_dataset=train_subset,
            curriculum_manager=curriculum_manager,
            device=device,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            diagnostics=diagnostics,
            epoch=epoch,
            use_active_sampling=use_active_sampling,
        )

        # Test
        test_loss, test_acc = test(model, test_loader, device, num_iterations)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"{'='*60}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print(f"  Generalization gap: {train_acc - test_acc:.2f}%")

        if use_active_sampling:
            print(f"\n  Curriculum Statistics:")
            print(f"    Visited: {curriculum_stats['visited_samples']}/{curriculum_stats['total_samples']} "
                  f"({curriculum_stats['visited_samples']/curriculum_stats['total_samples']:.1%})")
            print(f"    Learnable: {curriculum_stats['learnable_count']} (Zone of Proximal Development)")
            print(f"    Mastered: {curriculum_stats['mastered_count']} (Already learned)")
            print(f"    Noise: {curriculum_stats['noise_count']} (Unlearnable)")
            print(f"    Avg visits per sample: {curriculum_stats['avg_visit_count']:.1f}")

            # Print curriculum status
            curriculum_manager.print_status()

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Post-epoch diagnostics
        diagnostics.check_weight_changes(epoch)
        diagnostics.check_feature_quality(model, test_loader, device)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final train accuracy: {train_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Final generalization gap: {train_acc - test_acc:.2f}%")

    if use_active_sampling:
        print("\n" + "=" * 60)
        print("CURRICULUM ANALYSIS")
        print("=" * 60)

        # Show which samples are in which category
        breakdown = curriculum_manager.get_category_breakdown()
        print(f"Sample Distribution:")
        print(f"  Learnable: {len(breakdown['learnable'])} samples")
        print(f"  Mastered: {len(breakdown['mastered'])} samples")
        print(f"  Noise: {len(breakdown['noise'])} samples")
        print(f"  Unvisited: {len(breakdown['unvisited'])} samples")

        # Show some example samples from each category
        if breakdown['mastered']:
            print(f"\nExample mastered samples (first 10): {breakdown['mastered'][:10]}")
        if breakdown['learnable']:
            print(f"Example learnable samples (first 10): {breakdown['learnable'][:10]}")
        if breakdown['noise']:
            print(f"Example noise samples (first 10): {breakdown['noise'][:10]}")

    if test_acc > 50:
        print("\n✓ Vision encoder WORKING - test accuracy > 50%")
        if train_acc - test_acc < 20:
            print("✓ Good generalization - gap < 20%")
        else:
            print("⚠ Large generalization gap - may need regularization")
    else:
        print("\n✗ Vision encoder still BROKEN - test accuracy < 50%")

    print("=" * 60)

    # Final diagnostic summary
    diagnostics.summary_report()

    # Return metrics for diagnostics
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc,
        'best_test_acc': best_test_acc,
        'curriculum_stats': curriculum_stats if use_active_sampling else None,
    }


if __name__ == "__main__":
    results = main()
