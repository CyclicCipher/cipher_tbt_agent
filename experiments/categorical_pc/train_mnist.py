"""
MNIST Training with PC Convolutional Hierarchy + Active Inference

Architecture:
- PC Conv Layers: 3→64→128→256 (local learning, no backprop)
- PC Inference Layers: 1024→512→1024→10 (local learning, no backprop)

Features:
- Precision-weighted errors ([1.0, 10.0, 100.0])
- Bi-directional predictions
- Direct error injection (no teaching signal blending)
- Active curriculum learning
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

from categorical_network import PCConvVisionPreprocessor, CanonicalMicrocircuit
from src.active_inference import ActiveCurriculumManager


class PCConvClassifier(nn.Module):
    """PC vision classifier with PC conv + PC inference layers."""

    def __init__(self, num_classes=10, dtype=torch.float32):
        super().__init__()

        self.num_classes = num_classes
        self.dtype = dtype

        # PC convolutional preprocessor
        self.pc_conv_preprocessor = PCConvVisionPreprocessor(
            dtype=dtype,
            precisions=[1.0, 10.0, 100.0]
        )

        # PC inference layers
        self.pc_inference = CanonicalMicrocircuit(
            num_classes=num_classes,
            input_features=1024,
            layer0_size=512,
            layer1_size=1024,
            layer2_size=num_classes,
            use_4bit=False,
            dtype=dtype
        )

        self.last_input_image = None
        self.last_conv_features = None

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
        """Forward pass through full PC hierarchy."""
        # Store input for weight updates
        if x.dim() == 3:
            self.last_input_image = x.unsqueeze(0)
        elif x.dim() == 1:
            self.last_input_image = x.view(3, 100, 100).unsqueeze(0)

        # PC conv inference
        conv_features = self.pc_conv_preprocessor.forward(
            x,
            num_iterations=num_conv_iterations,
            inference_lr=conv_inference_lr,
            use_lateral=True
        )

        self.last_conv_features = conv_features

        # PC output inference
        output = self._pc_inference_pure(
            conv_features,
            num_iterations=num_inference_iterations,
            inference_lr=pc_inference_lr
        )

        # Error injection (if supervised)
        if target is not None:
            output_error = error_injection_strength * (target - output)
            self.pc_inference.layer2.state.data += output_error.data

            # Propagate error backward
            output = self._pc_inference_pure(
                conv_features,
                num_iterations=5,
                inference_lr=pc_inference_lr
            )

        return output.unsqueeze(0) if output.dim() == 1 else output

    def _pc_inference_pure(
        self,
        input_data: torch.Tensor,
        num_iterations: int = 20,
        inference_lr: float = 0.1
    ) -> torch.Tensor:
        """Pure PC inference with NO supervision."""
        for iteration in range(num_iterations):
            # Layer 0
            ff_0 = self.pc_inference.layer0.compute_feedforward(input_data)
            lat_0 = self.pc_inference.layer0.compute_lateral()
            fb_0 = self.pc_inference.layer0.compute_feedback(
                self.pc_inference.layer1.get_state()
            )

            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = self.pc_inference.layer0.state - target_0
            self.pc_inference.layer0.state.data -= inference_lr * error_0.data

            # Layer 1
            ff_1 = self.pc_inference.layer1.compute_feedforward(
                self.pc_inference.layer0.get_state()
            )
            fb_1 = self.pc_inference.layer1.compute_feedback(
                self.pc_inference.layer2.get_state()
            )

            target_1 = ff_1 + fb_1
            error_1 = self.pc_inference.layer1.state - target_1
            self.pc_inference.layer1.state.data -= inference_lr * error_1.data

            # Layer 2
            ff_2 = self.pc_inference.layer2.compute_feedforward(
                self.pc_inference.layer1.get_state()
            )

            target_2 = ff_2
            error_2 = self.pc_inference.layer2.state - target_2
            self.pc_inference.layer2.state.data -= inference_lr * error_2.data

        return self.pc_inference.layer2.get_state()

    def update_weights_pc(self, learning_rate=0.01, weight_decay=0.01):
        """Update PC inference layer weights."""
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
        """Update PC conv layer weights."""
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        elif input_image.dim() == 1:
            input_image = input_image.view(1, 3, 100, 100)

        # Update each PC conv layer
        self.pc_conv_preprocessor.pc_conv0.update_weights(
            input_below=input_image,
            input_above=self.pc_conv_preprocessor.pc_conv1.state,
            learning_rate=conv_learning_rate,
            weight_decay=weight_decay
        )

        self.pc_conv_preprocessor.pc_conv1.update_weights(
            input_below=self.pc_conv_preprocessor.pc_conv0.state,
            input_above=self.pc_conv_preprocessor.pc_conv2.state,
            learning_rate=conv_learning_rate,
            weight_decay=weight_decay
        )

        self.pc_conv_preprocessor.pc_conv2.update_weights(
            input_below=self.pc_conv_preprocessor.pc_conv1.state,
            input_above=None,
            learning_rate=conv_learning_rate,
            weight_decay=weight_decay
        )

    def reset_states(self):
        """Reset all layer states."""
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

    epoch_indices = curriculum_manager.get_epoch_indices(
        prioritize_learnable=prioritize_learnable
    )

    correct = 0
    total = 0
    epoch_errors = []

    for sample_idx in epoch_indices:
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

        # Update weights
        model.update_weights_pc(learning_rate=learning_rate)
        model.update_conv_weights_pc(
            input_image=image.view(3, 100, 100),
            conv_learning_rate=conv_learning_rate
        )

        # Update curriculum
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
            for i in range(data.size(0)):
                image = data[i].to(device)
                label = target[i].item()

                model.reset_states()

                output = model(
                    image.view(3, 100, 100),
                    target=None,
                    num_conv_iterations=20,
                    num_inference_iterations=20
                )

                pred = output.squeeze().argmax().item()
                correct += (pred == label)
                total += 1

                if total >= 1000:
                    break

            if total >= 1000:
                break

    accuracy = 100. * correct / total
    return accuracy


def main():
    print("=" * 80)
    print("MNIST Training with PC Conv Hierarchy + Active Inference")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Hyperparameters
    num_epochs = 5
    learning_rate = 0.01
    conv_learning_rate = 0.001

    print(f"\nHyperparameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  PC Layer LR: {learning_rate}")
    print(f"  PC Conv Layer LR: {conv_learning_rate}")
    print(f"  Precision Weighting: [1.0, 10.0, 100.0]")

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
    print(f"\nModel created with {len(model.pc_conv_preprocessor.pc_layers)} PC conv layers")

    # Active curriculum
    curriculum_manager = ActiveCurriculumManager(
        num_samples=train_size,
        sampling_strategy='learning_progress',
        window_size=10,
        mastery_threshold=0.01,
        noise_threshold=-0.005
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Training Start")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        train_acc, train_error = train_epoch(
            model,
            train_dataset,
            curriculum_manager,
            device,
            epoch,
            learning_rate=learning_rate,
            conv_learning_rate=conv_learning_rate
        )

        test_acc = test(model, test_loader, device)

        conv_stats = model.pc_conv_preprocessor.get_weight_stats()

        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Train Error: {train_error:.4f}")
        print("\nPC Conv Layer Stats:")
        for layer_name, stats in conv_stats.items():
            print(f"  {layer_name}: precision={stats['precision']:.2f}, "
                  f"weight_std={stats['std']:.4f}")

    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)

    # Curriculum stats
    curriculum_stats = curriculum_manager.get_statistics()
    print("\nCurriculum Statistics:")
    print(f"  Mastered: {curriculum_stats['mastered_count']} samples")
    print(f"  Learnable: {curriculum_stats['learnable_count']} samples")
    print(f"  Noise: {curriculum_stats['noise_count']} samples")


if __name__ == '__main__':
    main()
