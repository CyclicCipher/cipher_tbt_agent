"""
Memory and Performance Profiling for MNIST Training

This script instruments the training loop to identify:
1. Memory usage and leaks
2. Operation timing
3. Computational bottlenecks
4. Whether the program is making progress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import os
import time
import psutil
import gc
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from categorical_network import PCConvVisionPreprocessor, CanonicalMicrocircuit
from src.active_inference import ActiveCurriculumManager
from train_mnist import PCConvClassifier


class MemoryProfiler:
    """Track memory usage and identify leaks."""

    def __init__(self):
        self.process = psutil.Process()
        self.checkpoints = []

    def get_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_gpu_memory_mb(self):
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def checkpoint(self, label):
        """Record a memory checkpoint."""
        ram = self.get_memory_mb()
        gpu = self.get_gpu_memory_mb()
        self.checkpoints.append({
            'label': label,
            'ram_mb': ram,
            'gpu_mb': gpu,
            'timestamp': time.time()
        })
        return ram, gpu

    def report(self):
        """Print memory usage report."""
        if len(self.checkpoints) < 2:
            return

        print("\n" + "=" * 80)
        print("MEMORY USAGE REPORT")
        print("=" * 80)

        first = self.checkpoints[0]
        for i, cp in enumerate(self.checkpoints):
            ram_delta = cp['ram_mb'] - first['ram_mb']
            gpu_delta = cp['gpu_mb'] - first['gpu_mb']

            if i > 0:
                prev = self.checkpoints[i-1]
                ram_step = cp['ram_mb'] - prev['ram_mb']
                gpu_step = cp['gpu_mb'] - prev['gpu_mb']
                time_delta = cp['timestamp'] - prev['timestamp']

                print(f"\n{cp['label']}:")
                print(f"  RAM: {cp['ram_mb']:.1f} MB (Δ{ram_step:+.1f} MB, Total Δ{ram_delta:+.1f} MB)")
                print(f"  GPU: {cp['gpu_mb']:.1f} MB (Δ{gpu_step:+.1f} MB, Total Δ{gpu_delta:+.1f} MB)")
                print(f"  Time: {time_delta:.2f}s")
            else:
                print(f"\n{cp['label']} (baseline):")
                print(f"  RAM: {cp['ram_mb']:.1f} MB")
                print(f"  GPU: {cp['gpu_mb']:.1f} MB")


@contextmanager
def timer(label, profiler=None):
    """Context manager to time operations."""
    start = time.time()
    if profiler:
        mem_start_ram, mem_start_gpu = profiler.checkpoint(f"{label} - START")

    yield

    elapsed = time.time() - start
    if profiler:
        mem_end_ram, mem_end_gpu = profiler.checkpoint(f"{label} - END")
        ram_delta = mem_end_ram - mem_start_ram
        gpu_delta = mem_end_gpu - mem_start_gpu
        print(f"  ⏱️  {label}: {elapsed:.2f}s | RAM: Δ{ram_delta:+.1f}MB | GPU: Δ{gpu_delta:+.1f}MB")
    else:
        print(f"  ⏱️  {label}: {elapsed:.2f}s")


def profile_single_forward_pass(model, image, target, device, profiler):
    """Profile a single forward pass in detail."""
    print("\n" + "-" * 80)
    print("PROFILING SINGLE FORWARD PASS")
    print("-" * 80)

    model.reset_states()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    profiler.checkpoint("Before forward pass")

    # Profile PC conv iterations
    print("\nPC Conv Inference (20 iterations):")
    with timer("PC Conv Total", profiler):
        # Manually run conv preprocessing to track iterations
        x = image.unsqueeze(0) if image.dim() == 3 else image

        if model.pc_conv_preprocessor.pc_conv0.state is None:
            model.pc_conv_preprocessor.init_states(x.size(0), device)

        # Profile each iteration
        num_iterations = 20
        iteration_times = []

        for iteration in range(num_iterations):
            iter_start = time.time()

            # Update states (same as in forward)
            model.pc_conv_preprocessor.pc_conv2.update_state(
                input_below=model.pc_conv_preprocessor.pc_conv1.state,
                input_above=None,
                inference_lr=0.1,
                use_lateral=True
            )

            model.pc_conv_preprocessor.pc_conv1.update_state(
                input_below=model.pc_conv_preprocessor.pc_conv0.state,
                input_above=model.pc_conv_preprocessor.pc_conv2.state,
                inference_lr=0.1,
                use_lateral=True
            )

            model.pc_conv_preprocessor.pc_conv0.update_state(
                input_below=x,
                input_above=model.pc_conv_preprocessor.pc_conv1.state,
                inference_lr=0.1,
                use_lateral=True
            )

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)

            if iteration < 5 or iteration == num_iterations - 1:
                print(f"    Iteration {iteration+1}/{num_iterations}: {iter_time:.3f}s")
            elif iteration == 5:
                print(f"    ...")

        avg_iter_time = sum(iteration_times) / len(iteration_times)
        print(f"    Average iteration time: {avg_iter_time:.3f}s")
        print(f"    Total conv inference: {sum(iteration_times):.2f}s")

        # Get conv features
        conv_features = model.pc_conv_preprocessor.pool2(model.pc_conv_preprocessor.pc_conv2.state)
        conv_features = conv_features.flatten(1)
        if conv_features.size(0) == 1:
            conv_features = conv_features.squeeze(0)

    profiler.checkpoint("After PC conv inference")

    # Profile PC inference iterations
    print("\nPC Inference (20 iterations):")
    with timer("PC Inference Total", profiler):
        num_iterations = 20
        iteration_times = []

        for iteration in range(num_iterations):
            iter_start = time.time()

            # Layer 0
            ff_0 = model.pc_inference.layer0.compute_feedforward(conv_features)
            lat_0 = model.pc_inference.layer0.compute_lateral()
            fb_0 = model.pc_inference.layer0.compute_feedback(
                model.pc_inference.layer1.get_state()
            )
            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = model.pc_inference.layer0.state - target_0
            model.pc_inference.layer0.state.data -= 0.1 * error_0.data

            # Layer 1
            ff_1 = model.pc_inference.layer1.compute_feedforward(
                model.pc_inference.layer0.get_state()
            )
            fb_1 = model.pc_inference.layer1.compute_feedback(
                model.pc_inference.layer2.get_state()
            )
            target_1 = ff_1 + fb_1
            error_1 = model.pc_inference.layer1.state - target_1
            model.pc_inference.layer1.state.data -= 0.1 * error_1.data

            # Layer 2
            ff_2 = model.pc_inference.layer2.compute_feedforward(
                model.pc_inference.layer1.get_state()
            )
            target_2 = ff_2
            error_2 = model.pc_inference.layer2.state - target_2
            model.pc_inference.layer2.state.data -= 0.1 * error_2.data

            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)

            if iteration < 5 or iteration == num_iterations - 1:
                print(f"    Iteration {iteration+1}/{num_iterations}: {iter_time:.3f}s")
            elif iteration == 5:
                print(f"    ...")

        avg_iter_time = sum(iteration_times) / len(iteration_times)
        print(f"    Average iteration time: {avg_iter_time:.3f}s")
        print(f"    Total PC inference: {sum(iteration_times):.2f}s")

    profiler.checkpoint("After PC inference")

    # Error injection
    print("\nError Injection:")
    with timer("Error injection", profiler):
        output = model.pc_inference.layer2.get_state()
        output_error = 1.0 * (target - output)
        model.pc_inference.layer2.state.data += output_error.data

        # Run 5 more iterations
        for _ in range(5):
            ff_0 = model.pc_inference.layer0.compute_feedforward(conv_features)
            lat_0 = model.pc_inference.layer0.compute_lateral()
            fb_0 = model.pc_inference.layer0.compute_feedback(
                model.pc_inference.layer1.get_state()
            )
            target_0 = ff_0 + 0.5 * lat_0 + fb_0
            error_0 = model.pc_inference.layer0.state - target_0
            model.pc_inference.layer0.state.data -= 0.1 * error_0.data

            ff_1 = model.pc_inference.layer1.compute_feedforward(
                model.pc_inference.layer0.get_state()
            )
            fb_1 = model.pc_inference.layer1.compute_feedback(
                model.pc_inference.layer2.get_state()
            )
            target_1 = ff_1 + fb_1
            error_1 = model.pc_inference.layer1.state - target_1
            model.pc_inference.layer1.state.data -= 0.1 * error_1.data

            ff_2 = model.pc_inference.layer2.compute_feedforward(
                model.pc_inference.layer1.get_state()
            )
            target_2 = ff_2
            error_2 = model.pc_inference.layer2.state - target_2
            model.pc_inference.layer2.state.data -= 0.1 * error_2.data

    profiler.checkpoint("After error injection")

    # Weight updates
    print("\nWeight Updates:")
    with timer("PC layer weight update", profiler):
        model.update_weights_pc(learning_rate=0.01)

    profiler.checkpoint("After PC weight update")

    with timer("Conv layer weight update", profiler):
        model.update_conv_weights_pc(
            input_image=image.view(3, 100, 100),
            conv_learning_rate=0.001
        )

    profiler.checkpoint("After conv weight update")

    print("\n" + "-" * 80)


def profile_multiple_samples(model, train_dataset, device, num_samples=10):
    """Profile multiple samples to detect memory leaks."""
    print("\n" + "=" * 80)
    print(f"PROFILING {num_samples} SAMPLES")
    print("=" * 80)

    profiler = MemoryProfiler()
    profiler.checkpoint("Start")

    sample_times = []

    for i in range(num_samples):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sample_start = time.time()

        image, label = train_dataset[i]
        image = image.to(device)
        target = torch.zeros(10, dtype=torch.float32, device=device)
        target[label] = 1.0

        model.reset_states()

        # Forward pass
        output = model(
            image,
            target=target,
            num_conv_iterations=20,
            num_inference_iterations=20,
            error_injection_strength=1.0
        )

        # Weight updates
        model.update_weights_pc(learning_rate=0.01)
        model.update_conv_weights_pc(
            input_image=image.view(3, 100, 100),
            conv_learning_rate=0.001
        )

        sample_time = time.time() - sample_start
        sample_times.append(sample_time)

        ram, gpu = profiler.checkpoint(f"Sample {i+1}/{num_samples}")
        print(f"Sample {i+1}: {sample_time:.2f}s | RAM: {ram:.1f}MB | GPU: {gpu:.1f}MB")

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 40)
    print(f"Average sample time: {sum(sample_times)/len(sample_times):.2f}s")
    print(f"Min sample time: {min(sample_times):.2f}s")
    print(f"Max sample time: {max(sample_times):.2f}s")
    print(f"Total time: {sum(sample_times):.2f}s")
    print(f"Estimated time for 6000 samples: {sum(sample_times)/len(sample_times) * 6000 / 60:.1f} minutes")

    profiler.report()


def main():
    print("=" * 80)
    print("MNIST TRAINING PROFILER")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load MNIST
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_size = 6000
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))

    # Create model
    print("\nCreating model...")
    profiler = MemoryProfiler()
    profiler.checkpoint("Before model creation")

    model = PCConvClassifier(num_classes=10, dtype=torch.float32).to(device)

    profiler.checkpoint("After model creation")
    profiler.report()

    # Profile single forward pass in detail
    print("\n" + "=" * 80)
    print("DETAILED SINGLE SAMPLE PROFILING")
    print("=" * 80)

    image, label = train_dataset[0]
    image = image.to(device)
    target = torch.zeros(10, dtype=torch.float32, device=device)
    target[label] = 1.0

    profiler = MemoryProfiler()
    profile_single_forward_pass(model, image, target, device, profiler)

    # Profile multiple samples to detect memory leaks
    profile_multiple_samples(model, train_dataset, device, num_samples=10)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
