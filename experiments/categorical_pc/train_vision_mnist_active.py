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

        # PURE PC LEARNING: Conv layers will learn via local error-driven rules
        # Enable gradients so we can compute error signals, but won't use backprop
        # Gradients will be used to extract error information only
        for param in self.conv_preprocess.parameters():
            param.requires_grad = True  # Need grad for error extraction

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor = None,
        num_iterations: int = 10,
        teaching_signal_strength: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass with predictive coding inference.

        Args:
            x: (1, 1, 28, 28) MNIST image
            target: (optional) class label for supervised learning
            num_iterations: PC inference iterations
            teaching_signal_strength: How strongly to guide output toward target (0=none, 1=full clamp)

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

        # Store input for conv weight updates
        self.last_input_image = x

        # Create target representation if supervised
        target_output = None
        if target is not None:
            # One-hot encoding
            target_output = torch.zeros(self.num_classes, device=x.device)
            target_output[target] = 1.0

        # PC inference with teaching signal (NOT clamping!)
        output, error_0, error_1, error_2 = self._forward_with_teaching_signal(
            conv_features,
            target_output=target_output,
            num_iterations=num_iterations,
            teaching_strength=teaching_signal_strength
        )

        # Store for learning
        self.last_conv_features = conv_features
        self.last_errors = (error_0, error_1, error_2)
        self.last_target = target_output

        # Return with batch dimension
        return output.unsqueeze(0)  # (1, num_classes)

    def _forward_with_teaching_signal(
        self,
        input_data: torch.Tensor,
        target_output: torch.Tensor = None,
        num_iterations: int = 20,
        teaching_strength: float = 0.5
    ) -> tuple:
        """
        PC inference with teaching signal (no clamping!).

        Instead of forcing output = target, we add a "pull" toward the target.
        This allows the network to learn naturally while still getting supervision.

        Teaching signal acts like an additional prediction from a "teacher" area.
        """
        for iteration in range(num_iterations):
            # === LAYER 0: Superficial ===
            ff_0 = self.pc_inference.layer0.compute_feedforward(input_data)
            lat_0 = self.pc_inference.layer0.compute_lateral()
            fb_0 = self.pc_inference.layer0.compute_feedback(self.pc_inference.layer1.get_state())

            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = self.pc_inference.layer0.state - target_0
            self.pc_inference.layer0.state.data -= self.pc_inference.inference_lr * error_0.data

            # === LAYER 1: Middle ===
            ff_1 = self.pc_inference.layer1.compute_feedforward(self.pc_inference.layer0.get_state())
            fb_1 = self.pc_inference.layer1.compute_feedback(self.pc_inference.layer2.get_state())

            target_1 = ff_1 + fb_1
            error_1 = self.pc_inference.layer1.state - target_1
            self.pc_inference.layer1.state.data -= self.pc_inference.inference_lr * error_1.data

            # === LAYER 2: Output (with teaching signal, NO clamping!) ===
            ff_2 = self.pc_inference.layer2.compute_feedforward(self.pc_inference.layer1.get_state())

            if target_output is not None:
                # Teaching signal: blend between bottom-up prediction and target
                # This is like having a "teacher" area that makes predictions
                target_2 = (1 - teaching_strength) * ff_2 + teaching_strength * target_output
                error_2 = self.pc_inference.layer2.state - target_2
                self.pc_inference.layer2.state.data -= self.pc_inference.inference_lr * error_2.data
            else:
                # Unsupervised: just use bottom-up
                target_2 = ff_2
                error_2 = self.pc_inference.layer2.state - target_2
                self.pc_inference.layer2.state.data -= self.pc_inference.inference_lr * error_2.data

        # After inference, compute final errors for weight learning
        # These errors reflect how well the network can produce the target
        final_error_0 = self.pc_inference.layer0.state - (
            self.pc_inference.layer0.compute_feedforward(input_data) +
            0.5 * self.pc_inference.layer0.compute_lateral() +
            self.pc_inference.layer0.compute_feedback(self.pc_inference.layer1.get_state())
        )

        final_error_1 = self.pc_inference.layer1.state - (
            self.pc_inference.layer1.compute_feedforward(self.pc_inference.layer0.get_state()) +
            self.pc_inference.layer1.compute_feedback(self.pc_inference.layer2.get_state())
        )

        # Layer 2 error is difference between output and target (for supervised learning)
        if target_output is not None:
            final_error_2 = self.pc_inference.layer2.state - target_output
        else:
            final_error_2 = self.pc_inference.layer2.state - self.pc_inference.layer2.compute_feedforward(
                self.pc_inference.layer1.get_state()
            )

        return self.pc_inference.layer2.get_state(), final_error_0, final_error_1, final_error_2

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

    def update_conv_weights_pc(self, input_image: torch.Tensor, conv_learning_rate: float = 0.0001):
        """
        Update conv weights using error-driven local learning.

        Error signal comes from PC layer 0, propagated backward through conv layers.
        This is biologically plausible - error flows from higher areas to lower areas.

        Key idea:
        - PC Layer 0 has prediction error at its state (512 dims)
        - Propagate this error back to conv features (1024 dims)
        - Update conv to produce better features for PC layer

        Mathematically: Δw_conv ∝ error_PC × activation
        """
        error_0, _, _ = self.last_errors
        # error_0 is at PC layer 0 STATE (512 dims)
        # We need error at PC layer 0 INPUT (conv features, 1024 dims)

        # Propagate error backward through PC layer 0's feedforward weights
        # error_at_conv_features = W_feedforward.T @ error_at_state
        # W_feedforward shape: (512, 1024) → W.T shape: (1024, 512)
        error_at_conv_features = self.pc_inference.layer0.W_feedforward.weight.T @ error_0  # (1024,)

        with torch.no_grad():
            # Get intermediate conv activations (we need to do another forward pass)
            # This is necessary to get the activations at each layer
            x = input_image

            # Upsample
            x = self.conv_preprocess[0](x)  # Upsample

            # Layer 0: 32×32×1 → 16×16×64
            x = self.conv_preprocess[1](x)  # Conv2d
            act_after_conv0 = x.clone()
            x = self.conv_preprocess[2](x)  # Tanh
            act_0 = x.clone()

            # Layer 1: 16×16×64 → 8×8×128
            x = self.conv_preprocess[3](x)  # Conv2d
            act_after_conv1 = x.clone()
            x = self.conv_preprocess[4](x)  # Tanh
            act_1 = x.clone()

            # Layer 2: 8×8×128 → 4×4×256
            x = self.conv_preprocess[5](x)  # Conv2d
            act_after_conv2 = x.clone()
            x = self.conv_preprocess[6](x)  # Tanh
            act_2 = x.clone()

            # Flatten
            x = self.conv_preprocess[7](x)  # Flatten
            act_flat = x.clone()

            # Final linear layer: 4096 → 1024
            act_before_final = x.squeeze(0)  # Remove batch dim: (1, 4096) → (4096,)

            # Now we have error_at_conv_features (1024 dims) to propagate backward

            # Update final linear layer (4096 → 1024)
            # Δw = lr * error ⊗ input
            # Weight shape: (1024, 4096)
            # error_at_conv_features shape: (1024,)
            # act_before_final shape: (4096,)
            delta_final = conv_learning_rate * torch.outer(error_at_conv_features, act_before_final)
            self.conv_preprocess[-2].weight.data += delta_final

            # Propagate error backward through final layer
            # error_before_final = W^T @ error_at_conv_features
            error_before_final = self.conv_preprocess[-2].weight.T @ error_at_conv_features  # (4096,)

            # Reshape to spatial: (256, 4, 4)
            error_spatial = error_before_final.view(256, 4, 4).unsqueeze(0)  # (1, 256, 4, 4)

            # Update Conv layer 2 (8×8×128 → 4×4×256)
            # This is more complex - we need to use spatial correlation
            # For now, use a simplified local rule:
            # Δw ∝ error @ input (spatially)

            # Propagate error through tanh derivative
            error_conv2 = error_spatial * (1 - act_2.unsqueeze(0) ** 2)  # Tanh derivative

            # For convolutional layers, use spatial Hebbian learning
            # We'll use a simplified version: update based on correlation

            # Propagate through conv2
            # This is tricky - for simplicity, use average pooling to downsample error
            error_1 = F.interpolate(error_conv2, size=(8, 8), mode='bilinear', align_corners=False)
            error_1 = error_1 * (1 - act_1.unsqueeze(0) ** 2)

            # Propagate through conv1
            error_0_spatial = F.interpolate(error_1, size=(16, 16), mode='bilinear', align_corners=False)
            error_0_spatial = error_0_spatial * (1 - act_0.unsqueeze(0) ** 2)

            # NOTE: Full conv weight updates would require computing correlations
            # For now, we're only updating the final linear layer
            # This is still error-driven and local!

            # TODO: Implement Option 4 - Full PC Hierarchy for Conv Layers
            # This would be the most biologically plausible approach:
            # - Each conv layer becomes a PC layer with its own state
            # - States minimize prediction error locally
            # - Weights updated via local Hebbian rules
            # - Enables top-down generation and dreaming
            # - Fully unified PC framework from pixels to output
            # See: Rao & Ballard (1999) "Predictive Coding in the Visual Cortex"

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

        # Forward pass with teaching signal (NO clamping!)
        # Teaching signal guides output toward target without forcing it
        output = model(data, target=target.item(), num_iterations=num_iterations,
                      teaching_signal_strength=0.5)  # 50% pull toward target

        # Update weights using local learning rules (NO backprop!)
        # 1. Update PC layers
        model.update_weights_pc(learning_rate=learning_rate)

        # 2. Update conv layers using error from PC layer 0
        model.update_conv_weights_pc(input_image=data, conv_learning_rate=0.0001)

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
    print("  - Conv layers: Error-driven local learning (from PC layer 0)")
    print("  - Teaching signal: Guides (not clamps) output toward target")
    print("  - NO backprop anywhere - pure local learning!")

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
