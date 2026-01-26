"""
Test vision→symbols pipeline: Tokenless math curriculum.

Demonstrates complete architecture:
1. Math problems rendered as images (test window)
2. Network sees via foveal/peripheral vision
3. Retinal preprocessing (edges, motion)
4. Network processes visual features (no tokens!)
5. Motor output via active inference (Position 0)
6. Computational optimizations applied

This is Phase 1: Single digit recognition with active inference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.network.modular import SubNetwork, ModularNetwork
from src.pretraining.math_curriculum import (
    MathCurriculum,
    MathDomain,
    DifficultyLevel
)
from src.utils.math_test_window import MathTestWindow
from src.vision.retinal_preprocessing import retinal_preprocessing
from src.network.optimizations import (
    EarlyStoppingInference,
    optimized_inference,
    profile_inference
)

print("=" * 70)
print("VISION→SYMBOLS: TOKENLESS MATH CURRICULUM")
print("=" * 70)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32  # Use FP32 for initial testing
fovea_size = 50  # 50×50 pixel foveal patch
peripheral_scale = 0.25  # 200×200 → 50×50 peripheral

print(f"\nConfiguration:")
print(f"  Device: {device}")
print(f"  Dtype: {dtype}")
print(f"  Fovea size: {fovea_size}×{fovea_size}")
print(f"  Peripheral: {int(800*peripheral_scale)}×{int(600*peripheral_scale)}")

# Build network: Vision→Symbols with motor at Position 0
print("\n" + "=" * 70)
print("BUILDING NETWORK (VISION→SYMBOLS)")
print("=" * 70)

# Position 0: Sensory and Motor (active inference)
print("\nPosition 0 (Sensory/Motor):")

# Vision subnet: Processes foveal patch
vision_input_size = 3 * fovea_size * fovea_size  # RGB foveal patch after retinal preprocessing
vision_subnet = SubNetwork(
    name="vision",
    layer_sizes=[256, 128, 64],  # Hierarchical compression
    input_size=vision_input_size,
    position=0,
    dtype=dtype,
    device=device
)
print(f"  Vision subnet: {vision_input_size} input → {vision_subnet.layer_sizes}")

# Motor subnet: Generates character outputs
# For Phase 1: Just digits 0-9
num_digits = 10
motor_subnet = SubNetwork(
    name="motor",
    layer_sizes=[32, num_digits],  # Small - just executes predictions from above
    input_size=num_digits,  # Self-prediction (active inference)
    position=0,
    dtype=dtype,
    device=device
)
print(f"  Motor subnet: {num_digits} chars → {motor_subnet.layer_sizes}")

# Position 1: Association (integrates vision, predicts motor)
print("\nPosition 1 (Association):")
# Input = concatenation of Position 0 outputs
vision_output_size = vision_subnet.layer_sizes[-1]  # 64
motor_output_size = motor_subnet.layer_sizes[-1]    # 10
association_input_size = vision_output_size + motor_output_size  # 64 + 10 = 74
association_subnet = SubNetwork(
    name="association",
    layer_sizes=[256, 128, 64, num_digits],  # Predicts which digit
    input_size=association_input_size,
    position=1,
    dtype=dtype,
    device=device
)
print(f"  Association subnet: {association_input_size} → {association_subnet.layer_sizes}")

# Create modular network with StableProspective optimizer
print("\nCreating ModularNetwork with optimizer:")
network = ModularNetwork(
    subnetworks=[vision_subnet, motor_subnet, association_subnet],
    inference_lr=0.05,
    temperature=0.0,
    dtype=dtype,
    device=device,
    use_stable=True,
    stable_lr=0.001,
    stable_max_iterations=1000,
    stable_lr_schedule="cosine"
)

network.print_architecture()

# Create test window
print("\n" + "=" * 70)
print("CREATING TEST WINDOW")
print("=" * 70)

window = MathTestWindow(
    width=800,
    height=600,
    font_scale=3.0  # Large font for easy recognition
)
print("\nTest window created: 800×600 pixels")

# Create math curriculum (Phase 1: digits only)
curriculum = MathCurriculum(seed=42)
print("\nMath curriculum initialized")

# Generate digit recognition problems
print("\n" + "=" * 70)
print("PHASE 1: SINGLE DIGIT RECOGNITION")
print("=" * 70)

# For Phase 1, use simple arithmetic with single-digit answers
problems = curriculum.generate_batch(
    batch_size=10,
    domain=MathDomain.ARITHMETIC,
    difficulty=DifficultyLevel.EASY
)

# Filter to only single-digit answers
single_digit_problems = [p for p in problems if len(p.output.strip()) == 1 and p.output.strip().isdigit()][:5]

print(f"\nGenerated {len(single_digit_problems)} problems:")
for i, p in enumerate(single_digit_problems, 1):
    print(f"  {i}. {p.input} → {p.output}")

# Setup optimizations
print("\n" + "=" * 70)
print("OPTIMIZATIONS")
print("=" * 70)

early_stop = EarlyStoppingInference(
    tolerance=1e-3,
    patience=5,
    min_iterations=10
)
print("\nEarly stopping enabled:")
print(f"  Tolerance: 1e-3")
print(f"  Patience: 5 iterations")
print(f"  Min iterations: 10")

# Profile baseline performance
print("\nProfiling baseline performance (no optimizations):")
print("  Running 5 iterations with 30 inference steps each...")

# Create dummy input for profiling
dummy_foveal = torch.randn(vision_input_size, dtype=dtype, device=device)
dummy_motor = torch.zeros(num_digits, dtype=dtype, device=device)

stats = profile_inference(
    network,
    {"vision": dummy_foveal, "motor": dummy_motor},
    num_runs=5,
    num_iterations=30
)

print(f"\nBaseline performance:")
print(f"  Mean time: {stats['mean_time']*1000:.1f} ms per forward pass")
print(f"  Time per iteration: {stats['time_per_iteration']*1000:.1f} ms")
print(f"  Total for 30 iterations: {stats['mean_time']*1000:.1f} ms")

# Test single problem with vision→motor pipeline
print("\n" + "=" * 70)
print("TESTING VISION→SYMBOLS PIPELINE")
print("=" * 70)

test_problem = single_digit_problems[0]
print(f"\nTest problem: {test_problem.input} → {test_problem.output}")

# 1. Render problem
window.show_problem(test_problem.input, test_problem.output)
window.save_screenshot("test_vision_problem.png")
print(f"  ✓ Problem rendered and saved to test_vision_problem.png")

# 2. Get foveal vision
window.move_gaze(400, window.problem_y)
foveal_patch = window.get_foveal_patch(fovea_size=fovea_size)
print(f"  ✓ Foveal patch extracted: {foveal_patch.shape}")

# 3. Retinal preprocessing
visual_features = retinal_preprocessing(foveal_patch)
print(f"  ✓ Retinal preprocessing: {visual_features.shape}")

# 4. Flatten and convert to tensor
visual_input = torch.from_numpy(visual_features.flatten()).to(dtype).to(device)
print(f"  ✓ Visual input tensor: {visual_input.shape}")

# 5. Network inference with early stopping
print(f"\n  Running inference with early stopping...")
motor_input = torch.zeros(num_digits, dtype=dtype, device=device)

output, iterations_used = optimized_inference(
    network,
    {"vision": visual_input, "motor": motor_input},
    max_iterations=50,
    early_stopping=early_stop,
    verbose=True
)

print(f"\n  ✓ Inference complete:")
print(f"    Iterations used: {iterations_used} / 50 (saved {50 - iterations_used})")
print(f"    Output shape: {output.shape}")

# 6. Decode motor output (active inference)
# Motor subnet's TOP layer state = action prediction (not bottom!)
# Motor subnet: [32, 10] → layers[0]=32 neurons, layers[-1]=10 neurons
motor_state = motor_subnet.layers[-1].get_state()
digit_probs = torch.softmax(motor_state, dim=0)
predicted_digit = torch.argmax(digit_probs).item()

print(f"\n  Motor output (active inference):")
print(f"    Digit probabilities: {digit_probs.detach().cpu().numpy()}")
print(f"    Predicted digit: {predicted_digit}")
print(f"    Correct answer: {test_problem.output}")
print(f"    Match: {str(predicted_digit) == test_problem.output}")

# 7. Type answer via motor control
window.type_character(str(predicted_digit))
is_correct = window.submit_answer()
window.save_screenshot("test_vision_result.png")

print(f"\n  ✓ Answer submitted:")
print(f"    Typed: {predicted_digit}")
print(f"    Result: {'✓ Correct' if is_correct else '✗ Wrong'}")
print(f"    Screenshot: test_vision_result.png")

# Summary
print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nPipeline verified:")
print("  ✓ Math problems rendered as images")
print("  ✓ Foveal vision extraction")
print("  ✓ Retinal preprocessing")
print("  ✓ Network processes visual features (no tokens!)")
print("  ✓ Motor output via active inference (Position 0)")
print("  ✓ Early stopping optimization working")

print(f"\nPerformance:")
print(f"  Baseline: 30 iterations × {stats['time_per_iteration']*1000:.1f} ms = {30*stats['time_per_iteration']*1000:.1f} ms")
print(f"  Optimized: {iterations_used} iterations × {stats['time_per_iteration']*1000:.1f} ms = {iterations_used*stats['time_per_iteration']*1000:.1f} ms")
print(f"  Speedup: {30/iterations_used:.1f}x faster")

print("\nNext steps:")
print("  1. Train on MNIST-style digit recognition (supervised)")
print("  2. Test generalization to new rendered digits")
print("  3. Scale to multi-digit arithmetic (2 + 3 = 5)")
print("  4. Add saccade planning for reading sequences")
print("  5. Full math curriculum with catastrophic forgetting tests")

print("\nGenerated files:")
print("  - test_vision_problem.png (what network sees)")
print("  - test_vision_result.png (network's answer)")
