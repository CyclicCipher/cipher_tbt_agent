"""
Stable Prospective Learning Optimizer

Custom optimizer designed specifically for prospective learning / predictive coding networks.

Addresses the error rebound problem:
- Manual GD finds excellent solutions (error 3.57) but can't maintain them
- Weight decay destroys good solutions over 250+ iterations
- Learning continues past minimum, causing drift

Solution:
- Cosine annealing LR schedule (high early, low late)
- Adaptive weight decay (strong when weights large, weak near solution)
- Stability detection (reduce updates when at good solution)
- Optional early stopping
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class StableProspectiveLearning(Optimizer):
    """
    Optimizer for stable prospective learning.

    Key features:
    1. LR scheduling: Cosine annealing (smooth decay to near-zero)
    2. Adaptive decay: Strong (0.01) far from solution, weak (0.001) near it
    3. Stability detection: Track if in good solution region
    4. Early stopping: Optional freeze when stable

    Args:
        params: Network parameters
        lr: Initial learning rate (default: 0.001)
        max_iterations: Total training iterations for LR schedule (default: 400)
        lr_schedule: Schedule type ("cosine", "linear", "exponential")
        weight_decay_strong: Decay when far from solution (default: 0.01)
        weight_decay_weak: Decay when near solution (default: 0.001)
        stability_threshold: Error multiplier for "good solution" (default: 1.2)
        early_stopping: Whether to stop when stable (default: False)
        patience: Iterations of stability required for early stopping (default: 50)
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        max_iterations: int = 400,
        lr_schedule: str = "cosine",
        weight_decay_strong: float = 0.01,
        weight_decay_weak: float = 0.001,
        stability_threshold: float = 1.2,
        early_stopping: bool = False,
        patience: int = 50,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay_strong < 0.0:
            raise ValueError(f"Invalid weight_decay_strong: {weight_decay_strong}")
        if weight_decay_weak < 0.0:
            raise ValueError(f"Invalid weight_decay_weak: {weight_decay_weak}")

        defaults = dict(
            lr=lr,
            max_iterations=max_iterations,
            lr_schedule=lr_schedule,
            weight_decay_strong=weight_decay_strong,
            weight_decay_weak=weight_decay_weak,
            stability_threshold=stability_threshold,
            early_stopping=early_stopping,
            patience=patience,
        )
        super().__init__(params, defaults)

        # Track best error and stability
        self.best_error = float('inf')
        self.stable_count = 0
        self.current_iteration = 0
        self.converged = False
        self.last_decay_used = weight_decay_strong  # Track actual decay used

    def step(self, closure=None, current_error: float = None):
        """
        Perform single optimization step.

        Args:
            closure: Optional callable to reevaluate model and return loss
            current_error: Current reconstruction error (for adaptive decay)

        Returns:
            "converged" if early stopping triggered, None otherwise
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Update tracking
        if current_error is not None:
            if current_error < self.best_error:
                self.best_error = current_error
            self._check_stability(current_error)

        # Check if already converged
        if self.converged:
            return "converged"

        for group in self.param_groups:
            # Get scheduled learning rate
            lr = self._get_scheduled_lr(
                initial_lr=group['lr'],
                iteration=self.current_iteration,
                max_iterations=group['max_iterations'],
                schedule=group['lr_schedule']
            )

            # Get adaptive weight decay
            weight_decay = self._get_adaptive_decay(
                current_error=current_error,
                strong=group['weight_decay_strong'],
                weak=group['weight_decay_weak'],
                threshold=group['stability_threshold']
            )

            # Save for stats
            self.last_decay_used = weight_decay

            # Update each parameter
            for p in group['params']:
                if p.grad is None:
                    continue

                # Gradient descent: W -= lr * grad
                p.data.add_(p.grad, alpha=-lr)

                # Weight decay: W -= decay * W
                p.data.add_(p.data, alpha=-weight_decay)

        self.current_iteration += 1

        # Check early stopping
        if group['early_stopping'] and self.stable_count >= group['patience']:
            self.converged = True
            return "converged"

        return loss

    def _get_scheduled_lr(self, initial_lr, iteration, max_iterations, schedule):
        """Compute learning rate based on schedule."""
        if schedule == "cosine":
            # Cosine annealing: smooth decay from initial_lr to ~0
            return initial_lr * 0.5 * (1 + math.cos(math.pi * iteration / max_iterations))

        elif schedule == "linear":
            # Linear decay: initial_lr * (1 - progress)
            progress = min(iteration / max_iterations, 1.0)
            return initial_lr * (1 - progress)

        elif schedule == "exponential":
            # Exponential decay: initial_lr * 0.99^(iteration/10)
            return initial_lr * (0.99 ** (iteration / 10))

        elif schedule == "constant":
            return initial_lr

        else:
            raise ValueError(f"Unknown LR schedule: {schedule}")

    def _get_adaptive_decay(self, current_error, strong, weak, threshold):
        """
        Adaptive weight decay based on solution quality.

        Args:
            current_error: Current reconstruction error
            strong: Decay when far from solution
            weak: Decay when near solution
            threshold: Error multiplier for "good solution region"

        Returns:
            Decay value to use
        """
        if current_error is None:
            # No error info, use strong decay (conservative)
            return strong

        # Check if in good solution region (within threshold of best)
        if current_error < self.best_error * threshold:
            # Near good solution: use weak decay to preserve it
            return weak
        else:
            # Far from solution: use strong decay for regularization
            return strong

    def _check_stability(self, current_error):
        """
        Track stability for early stopping.

        Stability = being in good solution region for patience iterations
        """
        # Get threshold from first param group (assume all same)
        threshold = self.param_groups[0]['stability_threshold']

        if current_error < self.best_error * threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

    def reset(self):
        """Reset optimizer state for new training run."""
        self.best_error = float('inf')
        self.stable_count = 0
        self.current_iteration = 0
        self.converged = False
        self.last_decay_used = self.param_groups[0]['weight_decay_strong']

    def get_stats(self):
        """Get optimizer statistics for debugging."""
        return {
            'iteration': self.current_iteration,
            'best_error': self.best_error,
            'stable_count': self.stable_count,
            'converged': self.converged,
            'current_lr': self._get_scheduled_lr(
                self.param_groups[0]['lr'],
                self.current_iteration,
                self.param_groups[0]['max_iterations'],
                self.param_groups[0]['lr_schedule']
            ),
            'current_decay': self.last_decay_used  # Use actual decay value
        }
