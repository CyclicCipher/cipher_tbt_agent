"""
Performance profiling diagnostic for PC training.

Tests a SINGLE sample through the entire pipeline and identifies bottlenecks.
"""

import torch
import time
import numpy as np
from torchvision import datasets, transforms
from categorical_network import PCConvClassifier
import tracemalloc
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"

def format_memory(bytes):
    """Format memory in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"

def profile_step(name, func, *args, **kwargs):
    """Profile a single step and return result + timing."""
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Start memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

    # Time execution
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    # End memory tracking
    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        end_mem = tracemalloc.get_traced_memory()[0]
        peak_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

    elapsed = end_time - start_time
    mem_delta = end_mem - start_mem

    print(f"  {name:40s} | {format_time(elapsed):>10s} | Δmem: {format_memory(mem_delta):>10s} | Peak: {format_memory(peak_mem):>10s}")

    return result, elapsed, mem_delta

print("=" * 100)
print("PREDICTIVE CODING TRAINING PERFORMANCE PROFILE")
print("=" * 100)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load single MNIST sample
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
image, label = dataset[0]
image = image.to(device)

print(f"Sample: Image shape {image.shape}, Label {label}")

# Create model
model = PCConvClassifier(num_classes=10, dtype=torch.float32).to(device)
target = torch.zeros(10, dtype=torch.float32, device=device)
target[label] = 1.0

print("\n" + "=" * 100)
print("PROFILING SINGLE FORWARD PASS (1 sample)")
print("=" * 100)

# Test different iteration counts
test_configs = [
    (1, 1, "Minimal (1 conv, 1 inference)"),
    (5, 5, "Low (5 conv, 5 inference)"),
    (10, 10, "Medium (10 conv, 10 inference)"),
    (20, 20, "Current config (20 conv, 20 inference)"),
]

for num_conv_iter, num_inf_iter, config_name in test_configs:
    print(f"\n{config_name}")
    print("-" * 100)

    model.reset_states()
    total_start = time.time()

    # Forward pass
    output, fwd_time, fwd_mem = profile_step(
        "Full forward pass",
        lambda: model(
            image,
            target=target,
            num_conv_iterations=num_conv_iter,
            num_inference_iterations=num_inf_iter,
            error_injection_strength=1.0
        )
    )

    # Compute error
    def compute_error():
        return torch.sum((output.squeeze() - target) ** 2).item()

    error, err_time, err_mem = profile_step(
        "Compute error",
        compute_error
    )

    # Weight updates - PC layers
    _, pc_update_time, pc_update_mem = profile_step(
        "Update PC weights",
        lambda: model.update_weights_pc(learning_rate=0.01)
    )

    # Weight updates - Conv layers
    _, conv_update_time, conv_update_mem = profile_step(
        "Update Conv weights",
        lambda: model.update_conv_weights_pc(
            input_image=image.view(3, 100, 100),
            conv_learning_rate=0.001
        )
    )

    total_time = time.time() - total_start

    print(f"  {'─' * 40}   {'─' * 10}   {'─' * 11}   {'─' * 11}")
    print(f"  {'TOTAL PER SAMPLE':40s} | {format_time(total_time):>10s}")

    # Extrapolate to full epoch
    samples_per_epoch = 6000
    estimated_epoch_time = total_time * samples_per_epoch
    print(f"  {'Estimated epoch time (6000 samples)':40s} | {format_time(estimated_epoch_time):>10s}")

    if estimated_epoch_time > 600:  # > 10 minutes
        print(f"  ⚠ WARNING: This config would take {estimated_epoch_time/60:.1f} minutes per epoch!")

# Detailed breakdown of forward pass components
print("\n" + "=" * 100)
print("DETAILED FORWARD PASS BREAKDOWN (20/20 iterations)")
print("=" * 100)

model.reset_states()

# Manually step through forward pass
print("\nConvolutional preprocessing:")
_, conv_prep_time, _ = profile_step(
    "PC Conv forward (20 iterations)",
    lambda: model.pc_conv_preprocessor.forward(
        image, num_iterations=20, inference_lr=0.1, use_lateral=True
    )
)

print(f"\nAverage time per conv iteration: {format_time(conv_prep_time / 20)}")

print("\n" + "=" * 100)
print("MEMORY USAGE SUMMARY")
print("=" * 100)

if torch.cuda.is_available():
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model parameters: {format_memory(total_params)}")
    print(f"Current GPU memory: {format_memory(torch.cuda.memory_allocated())}")
    print(f"Peak GPU memory: {format_memory(torch.cuda.max_memory_allocated())}")

print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)

print("""
Based on the profiling results:

1. If 20/20 iterations takes > 1 minute per sample:
   → Reduce to 5/5 or even 3/3 iterations for initial training

2. If memory is growing over time:
   → Check for gradient accumulation bugs
   → Ensure torch.no_grad() is used during inference

3. If conv preprocessing is slow:
   → Consider reducing precision computation frequency
   → Check if lateral connections are needed

4. For faster experimentation:
   → Use smaller batch of samples (e.g., 1000 instead of 6000)
   → Start with minimal iterations and increase gradually

5. Expected reasonable performance:
   → 5/5 iterations: ~1-5 seconds per sample
   → Full epoch (6000 samples): ~10-30 minutes
""")
