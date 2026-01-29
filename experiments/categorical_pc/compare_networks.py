"""
Compare vanilla vs. categorical predictive coding networks.

Tests whether categorical constraints (compositional predictions) help or hurt learning.

Experiment: MNIST digit recognition with identical architectures
- Baseline: Standard ModularNetwork
- Categorical: CategoricalNetwork with λ_composition = 0.1

Metrics:
- Sample efficiency (accuracy vs. training examples)
- Final test accuracy
- Composition error (categorical only)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from src.network.modular import SubNetwork, ModularNetwork
from categorical_network import CategoricalNetwork
from src.pretraining.grounded_math_curriculum import GroundedMathCurriculum
from src.vision.retinal_preprocessing import retinal_preprocessing
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("CATEGORICAL VS VANILLA PREDICTIVE CODING COMPARISON")
print("=" * 70)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
patch_size = 100
num_digits = 10
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

print(f"\nConfiguration:")
print(f"  Device: {device}")
print(f"  Seed: {seed}")
print(f"  Patch size: {patch_size}×{patch_size}")


def create_network(network_type: str, lambda_composition: float = 0.1):
    """Create either vanilla or categorical network with identical architecture."""

    # Position 0: Vision + Motor
    vision_input_size = 3 * patch_size * patch_size
    vision_subnet = SubNetwork(
        name="vision",
        layer_sizes=[256, 128, 64],
        input_size=vision_input_size,
        position=0,
        dtype=dtype,
        device=device
    )

    num_motor_latent = 32
    motor_subnet = SubNetwork(
        name="motor",
        layer_sizes=[num_digits, num_motor_latent],
        input_size=num_digits,
        position=0,
        dtype=dtype,
        device=device
    )

    # Position 1: Association
    association_input_size = 64 + num_motor_latent
    association_subnet = SubNetwork(
        name="association",
        layer_sizes=[128, 64, num_digits],
        input_size=association_input_size,
        position=1,
        dtype=dtype,
        device=device
    )

    # Create network
    if network_type == "categorical":
        network = CategoricalNetwork(
            subnetworks=[vision_subnet, motor_subnet, association_subnet],
            inference_lr=0.05,
            temperature=0.0,
            dtype=dtype,
            device=device,
            use_stable=True,
            stable_lr=0.001,
            stable_max_iterations=1000,
            lambda_composition=lambda_composition
        )
        print(f"  Created CATEGORICAL network (λ={lambda_composition})")
    else:
        network = ModularNetwork(
            subnetworks=[vision_subnet, motor_subnet, association_subnet],
            inference_lr=0.05,
            temperature=0.0,
            dtype=dtype,
            device=device,
            use_stable=True,
            stable_lr=0.001,
            stable_max_iterations=1000
        )
        print(f"  Created VANILLA network")

    return network, motor_subnet


def train_and_evaluate(network, motor_subnet, train_data, test_data, epochs=5):
    """Train network and track metrics."""

    inference_iterations = 30

    # Track metrics
    train_accuracies = []
    test_accuracies = []
    composition_errors = []

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch + 1}/{epochs}")

        # Shuffle training data
        np.random.shuffle(train_data)

        correct = 0
        total_loss = 0.0

        for i, (img, label) in enumerate(train_data):
            # Preprocess
            features = retinal_preprocessing(img)
            visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
            motor_input = torch.zeros(num_digits, dtype=dtype, device=device)

            # Target
            target_one_hot = torch.zeros(num_digits, dtype=dtype, device=device)
            target_one_hot[label] = 1.0

            # Forward with clamping
            clamp_layers = {
                "vision": visual_input,
                "motor": target_one_hot
            }

            output = network.forward(
                {"vision": visual_input, "motor": motor_input},
                num_iterations=inference_iterations,
                clamp_layers=clamp_layers
            )

            # Get prediction
            motor_prediction = motor_subnet.layers[0].get_state()

            # Update weights
            network.update_weights(
                lr=0.01,
                weight_decay=0.01,
                motor_targets={"motor": target_one_hot}
            )

            # Track metrics
            loss = F.cross_entropy(
                motor_prediction.detach().unsqueeze(0),
                torch.tensor([label], device=device)
            )
            total_loss += loss.item()

            predicted = torch.argmax(motor_prediction.detach()).item()
            if predicted == label:
                correct += 1

        # Training accuracy
        train_acc = correct / len(train_data)
        train_accuracies.append(train_acc)

        # Test accuracy
        test_correct = 0
        for img, label in test_data:
            features = retinal_preprocessing(img)
            visual_input = torch.from_numpy(features.flatten()).to(dtype).to(device)
            motor_input = torch.zeros(num_digits, dtype=dtype, device=device)

            with torch.no_grad():
                clamp_layers = {"vision": visual_input}

                output = network.forward(
                    {"vision": visual_input, "motor": motor_input},
                    num_iterations=inference_iterations,
                    clamp_layers=clamp_layers
                )

                motor_prediction = motor_subnet.layers[0].get_state()
                predicted = torch.argmax(motor_prediction).item()

                if predicted == label:
                    test_correct += 1

        test_acc = test_correct / len(test_data)
        test_accuracies.append(test_acc)

        # Composition error (categorical only)
        if hasattr(network, 'get_composition_error'):
            comp_err = network.get_composition_error()
            composition_errors.append(comp_err)
            print(f"    Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%, Composition Error: {comp_err:.4f}")
        else:
            print(f"    Train: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")

    return {
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'composition_errors': composition_errors
    }


# Generate data
print("\n" + "=" * 70)
print("GENERATING DATA")
print("=" * 70)

curriculum = GroundedMathCurriculum(seed=seed)
train_dataset = curriculum.generate_digit_recognition_dataset(samples_per_digit=30)
test_dataset = curriculum.generate_digit_recognition_dataset(samples_per_digit=10)

print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# Train vanilla network
print("\n" + "=" * 70)
print("TRAINING VANILLA NETWORK")
print("=" * 70)

vanilla_net, vanilla_motor = create_network("vanilla")
vanilla_results = train_and_evaluate(vanilla_net, vanilla_motor, train_dataset, test_dataset, epochs=5)

# Train categorical network
print("\n" + "=" * 70)
print("TRAINING CATEGORICAL NETWORK")
print("=" * 70)

categorical_net, categorical_motor = create_network("categorical", lambda_composition=0.1)
categorical_results = train_and_evaluate(categorical_net, categorical_motor, train_dataset, test_dataset, epochs=5)

# Compare results
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print("\nFinal Test Accuracy:")
print(f"  Vanilla:     {vanilla_results['test_accuracies'][-1]*100:.1f}%")
print(f"  Categorical: {categorical_results['test_accuracies'][-1]*100:.1f}%")

if categorical_results['test_accuracies'][-1] > vanilla_results['test_accuracies'][-1]:
    diff = (categorical_results['test_accuracies'][-1] - vanilla_results['test_accuracies'][-1]) * 100
    print(f"\n✓ Categorical network BETTER by {diff:.1f}%")
elif categorical_results['test_accuracies'][-1] < vanilla_results['test_accuracies'][-1]:
    diff = (vanilla_results['test_accuracies'][-1] - categorical_results['test_accuracies'][-1]) * 100
    print(f"\n✗ Categorical network WORSE by {diff:.1f}%")
else:
    print(f"\n= Networks performed EQUALLY")

# Plot results
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Test accuracy over epochs
axes[0].plot(vanilla_results['test_accuracies'], 'o-', label='Vanilla', linewidth=2)
axes[0].plot(categorical_results['test_accuracies'], 's-', label='Categorical', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Test Accuracy Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Composition error (categorical only)
if categorical_results['composition_errors']:
    axes[1].plot(categorical_results['composition_errors'], 'o-', color='C1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Composition Error')
    axes[1].set_title('Composition Error (Categorical Network)')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, transform=axes[1].transAxes)
    axes[1].set_title('Composition Error (N/A)')

plt.tight_layout()
plot_path = 'experiments/categorical_pc/comparison_results.png'
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved: {plot_path}")

# Summary
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("\nWhat these results tell us:")
print("  - If categorical ≈ vanilla: Constraint doesn't help/hurt (neutral)")
print("  - If categorical > vanilla: Composition enforcement improves learning!")
print("  - If categorical < vanilla: Constraint is harmful (over-regularization)")
print("\nComposition error:")
print("  - High: Network violates composition (predictions don't factor)")
print("  - Low: Network respects hierarchical structure")
print("  - Decreasing: Learning to be more compositional over time")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
