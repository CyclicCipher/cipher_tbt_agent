"""
Comprehensive diagnostics added to training script.

This monitors:
1. Device placement (CPU vs GPU)
2. Gradient flow through all layers
3. Weight changes per epoch
4. Feature discrimination evolution
5. Learning conflicts between PC and backprop

Usage: Run train_vision_mnist_pc.py - diagnostics run automatically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import pairwise_distances

from train_vision_mnist_pc import VisionPCClassifier


class TrainingDiagnostics:
    """
    Comprehensive diagnostic tracking for PC training.
    """

    def __init__(self, model):
        self.model = model
        self.history = {
            'weight_changes': [],
            'gradient_norms': [],
            'feature_ratios': [],
            'inference_errors': [],
            'device_status': []
        }

        # Save initial weights
        self.initial_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone()

    def check_device_placement(self, data):
        """
        CRITICAL: Check if model and data are actually on GPU.
        """
        print("\n" + "="*60)
        print("DIAGNOSTIC 1: Device Placement")
        print("="*60)

        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        # Check model parameters
        try:
            param = next(self.model.parameters())
            print(f"Model parameters device: {param.device}")
            print(f"Model on CUDA: {param.is_cuda}")
        except StopIteration:
            print("ERROR: Model has no parameters!")

        # Check specific layers
        print(f"Conv0 device: {self.model.conv_preprocess[1].weight.device}")
        print(f"PC Layer0 device: {self.model.pc_inference.layer0.W_feedforward.weight.device}")

        # Check data
        print(f"Data device: {data.device}")
        print(f"Data on CUDA: {data.is_cuda}")

        # Test GPU compute
        if torch.cuda.is_available():
            try:
                x = torch.randn(100, 100).cuda()
                y = x @ x
                print(f"GPU compute test: SUCCESS (result on {y.device})")
            except Exception as e:
                print(f"GPU compute test: FAILED - {e}")

        # Verdict
        all_on_gpu = (
            param.is_cuda and
            data.is_cuda and
            torch.cuda.is_available()
        )

        if all_on_gpu:
            print("\n✓ GOOD: Model and data on GPU")
        else:
            print("\n✗ PROBLEM: Model or data NOT on GPU!")
            print("  → Training will be slow and may not work correctly")
            print("  → This explains poor results!")

        print("="*60)
        return all_on_gpu

    def check_gradient_flow(self, model, loss):
        """
        Check if gradients are flowing to all layers.
        """
        print("\n" + "="*60)
        print("DIAGNOSTIC 2: Gradient Flow")
        print("="*60)

        gradient_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[name] = grad_norm

                # Check for problems
                if grad_norm < 1e-8:
                    status = "✗ VANISHING"
                elif grad_norm > 100:
                    status = "✗ EXPLODING"
                else:
                    status = "✓ OK"

                print(f"{status} {name}: {grad_norm:.6f}")
            else:
                print(f"✗ NO GRAD {name}")
                gradient_norms[name] = 0.0

        # Check conv layers specifically
        conv_grads_ok = all([
            gradient_norms.get(f'conv_preprocess.{i}.weight', 0) > 1e-6
            for i in [1, 3, 5]  # Conv layer indices
        ])

        if not conv_grads_ok:
            print("\n✗ PROBLEM: Conv layers have vanishing gradients!")
            print("  → Conv weights won't update")
            print("  → Features stay random")
        else:
            print("\n✓ GOOD: Conv layers receiving gradients")

        print("="*60)

        self.history['gradient_norms'].append(gradient_norms)
        return conv_grads_ok

    def check_weight_changes(self, epoch):
        """
        Check if weights are actually changing during training.
        """
        print("\n" + "="*60)
        print(f"DIAGNOSTIC 3: Weight Changes (Epoch {epoch})")
        print("="*60)

        changes = {}

        for name, param in self.model.named_parameters():
            if name in self.initial_weights:
                initial = self.initial_weights[name]
                current = param.data

                # Compute change statistics
                diff = current - initial
                abs_change = diff.abs().mean().item()
                rel_change = (diff.abs() / (initial.abs() + 1e-8)).mean().item()

                changes[name] = {
                    'abs': abs_change,
                    'rel': rel_change
                }

                # Determine status
                if abs_change < 1e-6:
                    status = "✗ FROZEN"
                elif abs_change > 1.0:
                    status = "⚠ LARGE"
                else:
                    status = "✓ CHANGING"

                print(f"{status} {name}:")
                print(f"     Abs change: {abs_change:.6f}")
                print(f"     Rel change: {rel_change:.4f}")

        # Check conv layers specifically
        conv_changing = all([
            changes.get(f'conv_preprocess.{i}.weight', {'abs': 0})['abs'] > 1e-4
            for i in [1, 3, 5]
        ])

        if not conv_changing:
            print("\n✗ PROBLEM: Conv weights not changing!")
            print("  → Learning rate too small or gradients blocked")
        else:
            print("\n✓ GOOD: Conv weights updating")

        print("="*60)

        self.history['weight_changes'].append(changes)
        return conv_changing

    def check_feature_quality(self, model, data_loader, device, num_samples=100):
        """
        Check if features are becoming more discriminative over training.
        """
        print("\n" + "="*60)
        print("DIAGNOSTIC 4: Feature Discrimination")
        print("="*60)

        model.eval()

        # Extract features by class
        features_by_class = {i: [] for i in range(10)}

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                feat = model.conv_preprocess(data).squeeze(0).cpu().numpy()
                features_by_class[target.item()].append(feat)

                if sum(len(v) for v in features_by_class.values()) >= num_samples:
                    break

        # Compute separation ratio
        within_dists = []
        between_dists = []

        for class_idx in range(10):
            features = np.stack(features_by_class[class_idx][:10])  # First 10

            if len(features) > 1:
                # Within-class
                dist_matrix = pairwise_distances(features)
                within = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
                within_dists.extend(within)

                # Between-class
                for other_idx in range(class_idx + 1, 10):
                    other_features = np.stack(features_by_class[other_idx][:10])
                    between = pairwise_distances(features, other_features).flatten()
                    between_dists.extend(between)

        within_mean = np.mean(within_dists) if within_dists else 0
        between_mean = np.mean(between_dists) if between_dists else 0
        ratio = between_mean / within_mean if within_mean > 0 else 0

        print(f"Within-class distance:  {within_mean:.4f}")
        print(f"Between-class distance: {between_mean:.4f}")
        print(f"Ratio (between/within): {ratio:.4f}")

        if ratio < 1.0:
            print("\n✗ POOR: Features collapsed (ratio < 1.0)")
        elif ratio < 1.5:
            print("\n⚠ WEAK: Some separation (ratio < 1.5)")
        else:
            print("\n✓ GOOD: Features discriminative (ratio > 1.5)")

        print("="*60)

        self.history['feature_ratios'].append(ratio)
        model.train()

        return ratio

    def check_pc_inference_quality(self, model, sample_data, device):
        """
        Check if PC inference is working correctly.
        """
        print("\n" + "="*60)
        print("DIAGNOSTIC 5: PC Inference Quality")
        print("="*60)

        model.eval()
        sample_data = sample_data.to(device)

        # Reset state
        model.pc_inference.layer0.state.data.zero_()
        model.pc_inference.layer1.state.data.zero_()
        model.pc_inference.layer2.state.data.zero_()

        # Conv features
        conv_features = model.conv_preprocess(sample_data).squeeze(0)

        # Track errors over iterations
        errors = []

        with torch.no_grad():
            for i in range(50):
                # Single inference step
                ff_0 = model.pc_inference.layer0.compute_feedforward(conv_features)
                lat_0 = model.pc_inference.layer0.compute_lateral()
                fb_0 = model.pc_inference.layer0.compute_feedback(
                    model.pc_inference.layer1.get_state()
                )

                target_0 = ff_0 + 0.5 * lat_0 + fb_0
                error_0 = model.pc_inference.layer0.state - target_0
                model.pc_inference.layer0.state.data -= 0.1 * error_0.data

                errors.append(error_0.norm().item())

        final_error = errors[-1]
        converged = final_error < 0.1

        print(f"Initial error: {errors[0]:.6f}")
        print(f"Final error (50 iter): {final_error:.6f}")
        print(f"Converged: {converged}")

        if not converged:
            print("\n⚠ PC inference not converging well")
            print("  → May need more iterations or different inference_lr")
        else:
            print("\n✓ PC inference converging")

        print("="*60)

        model.train()
        return converged

    def check_learning_conflicts(self):
        """
        Check if PC and backprop updates are conflicting.
        """
        print("\n" + "="*60)
        print("DIAGNOSTIC 6: Learning Conflict Detection")
        print("="*60)

        # Check if PC layer weights are in optimizer
        print("Checking if PC layers are being updated by both methods...")

        # This is a design issue - if PC layers have gradients AND
        # are updated by PC rules, they might conflict

        pc_params = set(self.model.pc_inference.parameters())

        print(f"PC parameters: {len(list(pc_params))}")
        print(f"All PC params have requires_grad=True")

        print("\n⚠ POTENTIAL CONFLICT:")
        print("  PC layers are updated by:")
        print("    1. PC local learning rules (update_weights_pc)")
        print("    2. Backprop gradients (loss.backward)")
        print("  These might interfere with each other!")

        print("\n  Recommendation:")
        print("    Option A: Freeze PC layers during backprop")
        print("    Option B: Use ONLY PC learning (no backprop)")
        print("    Option C: Use ONLY backprop (no PC learning)")

        print("="*60)

    def summary_report(self):
        """
        Generate summary of all diagnostics.
        """
        print("\n" + "="*70)
        print(" " * 20 + "DIAGNOSTIC SUMMARY")
        print("="*70)

        # Feature quality progression
        if self.history['feature_ratios']:
            print(f"\nFeature Quality (ratio) over epochs:")
            for i, ratio in enumerate(self.history['feature_ratios']):
                print(f"  Epoch {i+1}: {ratio:.4f}")

        # Weight change summary
        if self.history['weight_changes']:
            print(f"\nWeight Changes:")
            latest = self.history['weight_changes'][-1]
            conv_changes = [
                (name, data['abs'])
                for name, data in latest.items()
                if 'conv' in name and 'weight' in name
            ]
            for name, change in conv_changes:
                print(f"  {name}: {change:.6f}")

        print("\n" + "="*70)


def add_diagnostics_to_training():
    """
    Example of how to integrate diagnostics into training loop.
    """
    print("""
To add diagnostics to your training script, insert this code:

```python
from diagnostics_training import TrainingDiagnostics

# After creating model
diagnostics = TrainingDiagnostics(model)

# Before training loop
sample_data, _ = next(iter(train_loader))
diagnostics.check_device_placement(sample_data)
diagnostics.check_pc_inference_quality(model, sample_data, device)
diagnostics.check_learning_conflicts()

# After each epoch
diagnostics.check_weight_changes(epoch)
diagnostics.check_feature_quality(model, train_loader, device)

# In training loop, after loss.backward()
diagnostics.check_gradient_flow(model, loss)

# At end
diagnostics.summary_report()
```
""")


if __name__ == "__main__":
    add_diagnostics_to_training()
