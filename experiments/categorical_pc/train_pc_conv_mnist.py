"""
MNIST Training with FULL PC Conv Hierarchy + Active Inference

Implements Option 4: Full PC Conv Hierarchy
+ VERSES Optimizations (Precision Weighting + Bi-directional Propagation)
+ Direct Error Injection (replaces teaching signal)

Key Changes from Previous Approach:
1. ALL conv layers are PC layers with state and local learning
2. NO teaching signal blending - use direct error injection
3. Precision-weighted errors prevent decay in deep networks
4. Bi-directional predictions create richer error signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pc_conv_preprocessor import PCConvVisionPreprocessor
from categorical_network_impl import CanonicalMicrocircuit
from diagnostics_training import TrainingDiagnostics
from src.active_inference import ActiveCurriculumManager, LearningProgressTracker


class PCConvClassifier(nn.Module):
    """
    Full PC vision classifier with PC conv layers + PC inference layers.

    Architecture:
    - Input: 100×100×3 image
    - PC Conv Layers (Option 4): 3→64→128→256
    - PC Inference Layers: 1024→512→1024→10

    NO backprop, NO teaching signal - pure predictive coding!
    """

    def __init__(self, num_classes=10, dtype=torch.float32):
        super().__init__()

        self.num_classes = num_classes
        self.dtype = dtype

        # === PC CONVOLUTIONAL PREPROCESSOR ===
        # Precision weights: layer 0 = 1.0, layer 1 = 10.0, layer 2 = 100.0
        # Higher precisions prevent error decay in deeper layers
        self.pc_conv_preprocessor = PCConvVisionPreprocessor(
            dtype=dtype,
            precisions=[1.0, 10.0, 100.0]
        )

        # === PC INFERENCE LAYERS ===
        # 3-layer canonical microcircuit on top of conv features
        self.pc_inference = CanonicalMicrocircuit(
            num_classes=num_classes,
            input_features=1024,  # From conv preprocessor
            layer0_size=512,
            layer1_size=1024,
            layer2_size=num_classes,
            use_4bit=False,
            dtype=dtype
        )

        # Store last inputs for weight updates
        self.last_input_image = None
        self.last_conv_features = None
        self.last_target = None

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_conv_iterations: int = 20,
        num_inference_iterations: int = 20,
        conv_inference_lr: float = 0.1,
        pc_inference_lr: float = 0.1,
        error_injection_strength: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass through full PC hierarchy.

        Args:
            x: Input image (3, 100, 100) or (30000,)
            target: Target class (one-hot), optional
            num_conv_iterations: PC inference iterations for conv layers
            num_inference_iterations: PC inference iterations for output layers
            conv_inference_lr: Learning rate for conv state updates
            pc_inference_lr: Learning rate for PC layer state updates
            error_injection_strength: How strongly to inject output error

        Returns:
            Output (num_classes,) class predictions
        """
        # Store input for weight updates
        if x.dim() == 3:
            self.last_input_image = x.unsqueeze(0)  # (1, 3, 100, 100)
        elif x.dim() == 1:
            self.last_input_image = x.view(3, 100, 100).unsqueeze(0)

        # === PHASE 1: PC CONV INFERENCE ===
        # Run PC inference through convolutional layers
        conv_features = self.pc_conv_preprocessor.forward(
            x,
            num_iterations=num_conv_iterations,
            inference_lr=conv_inference_lr,
            use_lateral=True
        )

        self.last_conv_features = conv_features

        # === PHASE 2: PC OUTPUT INFERENCE (NO teaching signal!) ===
        # Run pure PC inference to equilibrium
        output, error_0, error_1, error_2 = self._pc_inference_pure(
            conv_features,
            num_iterations=num_inference_iterations,
            inference_lr=pc_inference_lr
        )

        # === PHASE 3: ERROR INJECTION (if supervised) ===
        if target is not None:
            self.last_target = target

            # Direct error injection at output layer
            # This is the proper PC way to provide supervision!
            # Instead of blending predictions, we inject a "surprise" error
            output_error = error_injection_strength * (target - output)

            # Store injected error for weight learning
            self.pc_inference.layer2.state.data += output_error.data

            # Let error propagate backward through one more inference iteration
            # This updates intermediate layer states based on the injected error
            output, error_0, error_1, error_2 = self._pc_inference_pure(
                conv_features,
                num_iterations=5,  # Just a few iterations to propagate error
                inference_lr=pc_inference_lr
            )

        return output.unsqueeze(0) if output.dim() == 1 else output

    def _pc_inference_pure(
        self,
        input_data: torch.Tensor,
        num_iterations: int = 20,
        inference_lr: float = 0.1
    ) -> tuple:
        """
        Pure PC inference with NO supervision.

        Just minimize prediction errors using bottom-up, top-down, and lateral predictions.
        """
        for iteration in range(num_iterations):
            # === LAYER 0: Superficial ===
            ff_0 = self.pc_inference.layer0.compute_feedforward(input_data)
            lat_0 = self.pc_inference.layer0.compute_lateral()
            fb_0 = self.pc_inference.layer0.compute_feedback(
                self.pc_inference.layer1.get_state()
            )

            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = self.pc_inference.layer0.state - target_0
            self.pc_inference.layer0.state.data -= inference_lr * error_0.data

            # === LAYER 1: Middle ===
            ff_1 = self.pc_inference.layer1.compute_feedforward(
                self.pc_inference.layer0.get_state()
            )
            fb_1 = self.pc_inference.layer1.compute_feedback(
                self.pc_inference.layer2.get_state()
            )

            target_1 = ff_1 + fb_1
            error_1 = self.pc_inference.layer1.state - target_1
            self.pc_inference.layer1.state.data -= inference_lr * error_1.data

            # === LAYER 2: Output (pure bottom-up, NO supervision yet) ===
            ff_2 = self.pc_inference.layer2.compute_feedforward(
                self.pc_inference.layer1.get_state()
            )

            target_2 = ff_2
            error_2 = self.pc_inference.layer2.state - target_2
            self.pc_inference.layer2.state.data -= inference_lr * error_2.data

        # Compute final errors after equilibrium
        final_ff_0 = self.pc_inference.layer0.compute_feedforward(input_data)
        final_lat_0 = self.pc_inference.layer0.compute_lateral()
        final_fb_0 = self.pc_inference.layer0.compute_feedback(
            self.pc_inference.layer1.get_state()
        )
        final_error_0 = self.pc_inference.layer0.state - (
            final_ff_0 + 0.5 * final_lat_0 + final_fb_0
        )

        final_ff_1 = self.pc_inference.layer1.compute_feedforward(
            self.pc_inference.layer0.get_state()
        )
        final_fb_1 = self.pc_inference.layer1.compute_feedback(
            self.pc_inference.layer2.get_state()
        )
        final_error_1 = self.pc_inference.layer1.state - (final_ff_1 + final_fb_1)

        final_ff_2 = self.pc_inference.layer2.compute_feedforward(
            self.pc_inference.layer1.get_state()
        )
        final_error_2 = self.pc_inference.layer2.state - final_ff_2

        output = self.pc_inference.layer2.get_state()

        return output, final_error_0, final_error_1, final_error_2

    def update_weights_pc(self, learning_rate=0.01, weight_decay=0.01):
        """Update PC inference layer weights using local learning."""
        # Update PC output layers
        self.pc_inference.update_weights(
            input_data=self.last_conv_features,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

    def update_conv_weights_pc(
        self,
        input_image: torch.Tensor,
        conv_learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Update PC conv layer weights using local Hebbian learning.

        This is where Option 4 shines - ALL conv layers update locally!
        """
        # Ensure input has batch dimension
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        elif input_image.dim() == 1:
            input_image = input_image.view(1, 3, 100, 100)

        # Update each PC conv layer with its local error and input
        # Layer 0: updates based on error_0 and input image
        self.pc_conv_preprocessor.pc_conv0.update_weights(
            input_below=input_image,
            input_above=self.pc_conv_preprocessor.pc_conv1.state,
            learning_rate=conv_learning_rate,
            weight_decay=weight_decay
        )

        # Layer 1: updates based on error_1 and layer 0 state
        self.pc_conv_preprocessor.pc_conv1.update_weights(
            input_below=self.pc_conv_preprocessor.pc_conv0.state,
            input_above=self.pc_conv_preprocessor.pc_conv2.state,
            learning_rate=conv_learning_rate,
            weight_decay=weight_decay
        )

        # Layer 2: updates based on error_2 and layer 1 state
        self.pc_conv_preprocessor.pc_conv2.update_weights(
            input_below=self.pc_conv_preprocessor.pc_conv1.state,
            input_above=None,  # No layer above
            learning_rate=conv_learning_rate,
            weight_decay=weight_decay
        )

    def reset_states(self):
        """Reset all layer states (call between samples)."""
        self.pc_conv_preprocessor.reset_states()
        self.pc_inference.reset_states()


def train_epoch(
    model,
    train_dataset,
    curriculum_manager,
    device,
    epoch,
    prioritize_learnable=True,
    learning_rate=0.01,
    conv_learning_rate=0.001
):
    """Train for one epoch with active curriculum."""
    model.train()

    # Get sample ordering from curriculum manager
    epoch_indices = curriculum_manager.get_epoch_indices(
        prioritize_learnable=prioritize_learnable
    )

    correct = 0
    total = 0
    epoch_errors = []

    for sample_idx in epoch_indices:
        # Get sample
        image, label = train_dataset[sample_idx]
        image = image.to(device)

        # One-hot encode label
        target = torch.zeros(10, dtype=model.dtype, device=device)
        target[label] = 1.0

        # Reset states
        model.reset_states()

        # Forward pass with error injection
        output = model(
            image.view(3, 100, 100),
            target=target,
            num_conv_iterations=20,
            num_inference_iterations=20,
            error_injection_strength=1.0
        )

        # Compute error for curriculum
        error = torch.sum((output.squeeze() - target) ** 2).item()
        epoch_errors.append(error)

        # Track accuracy
        pred = output.squeeze().argmax().item()
        correct += (pred == label)
        total += 1

        # Update PC layer weights
        model.update_weights_pc(learning_rate=learning_rate)

        # Update PC conv weights (Option 4!)
        model.update_conv_weights_pc(
            input_image=image.view(3, 100, 100),
            conv_learning_rate=conv_learning_rate
        )

        # Update curriculum with sample error
        curriculum_manager.update_sample(sample_idx, error, epoch)

    accuracy = 100. * correct / total
    mean_error = np.mean(epoch_errors)

    return accuracy, mean_error


def test(model, test_loader, device):
    """Test the model."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Process one sample at a time
            for i in range(data.size(0)):
                image = data[i].to(device)
                label = target[i].item()

                # Reset states
                model.reset_states()

                # Forward (no target = unsupervised inference)
                output = model(
                    image.view(3, 100, 100),
                    target=None,
                    num_conv_iterations=20,
                    num_inference_iterations=20
                )

                pred = output.squeeze().argmax().item()
                correct += (pred == label)
                total += 1

                if total >= 1000:  # Test on 1000 samples
                    break

            if total >= 1000:
                break

    accuracy = 100. * correct / total
    return accuracy


def main():
    print("=" * 80)
    print("MNIST Training with FULL PC Conv Hierarchy + Active Inference")
    print("Implements: Option 4 + VERSES Optimizations + Direct Error Injection")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.01  # PC layer learning rate
    conv_learning_rate = 0.001  # PC conv layer learning rate (10x smaller)

    print(f"\nHyperparameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  PC Layer LR: {learning_rate}")
    print(f"  PC Conv Layer LR: {conv_learning_rate}")
    print(f"  Error Injection: Direct (NO teaching signal!)")
    print(f"  Precision Weighting: [1.0, 10.0, 100.0]")
    print(f"  Bi-directional Propagation: Enabled")

    # Load MNIST
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Use subset for faster training
    train_size = 6000
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))

    # Create model
    model = PCConvClassifier(num_classes=10, dtype=torch.float32).to(device)
    print(f"\nModel: {model.pc_conv_preprocessor}")
    print(f"PC Conv Layers: {len(model.pc_conv_preprocessor.pc_layers)}")

    # Active curriculum manager
    curriculum_manager = ActiveCurriculumManager(
        num_samples=train_size,
        strategy='learning_progress',
        window_size=10,
        mastery_threshold=0.01,
        noise_threshold=-0.005
    )

    # Training diagnostics
    diagnostics = TrainingDiagnostics()

    # Training loop
    print("\n" + "=" * 80)
    print("Training Start")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        # Train
        train_acc, train_error = train_epoch(
            model,
            train_dataset,
            curriculum_manager,
            device,
            epoch,
            learning_rate=learning_rate,
            conv_learning_rate=conv_learning_rate
        )

        # Test
        test_acc = test(model, test_loader, device)

        # Get conv weight stats
        conv_stats = model.pc_conv_preprocessor.get_weight_stats()

        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Train Error: {train_error:.4f}")
        print("\nPC Conv Layer Stats:")
        for layer_name, stats in conv_stats.items():
            print(f"  {layer_name}: precision={stats['precision']:.2f}, "
                  f"weight_std={stats['std']:.4f}")

        # Log diagnostics
        diagnostics.log_epoch(epoch, train_acc, test_acc, train_error)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Get curriculum statistics
    curriculum_stats = curriculum_manager.get_statistics()
    print("\nCurriculum Statistics:")
    print(f"  Mastered: {curriculum_stats['mastered_count']} samples")
    print(f"  Learnable: {curriculum_stats['learnable_count']} samples")
    print(f"  Noise: {curriculum_stats['noise_count']} samples")
    print(f"  Unvisited: {curriculum_stats['unvisited_count']} samples")


if __name__ == '__main__':
    main()
