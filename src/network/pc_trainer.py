"""
Minimal Predictive Coding Trainer

Implements two-phase training:
1. Inference phase: Optimize value nodes (x) to minimize energy for T iterations
2. Learning phase: Update weights based on converged value nodes

Based on standard PC algorithm from Bogacz Group and Whittington & Bogacz (2017).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List


class PCTrainer:
    """Trainer for Predictive Coding networks.

    Implements the standard two-phase PC algorithm:
    - Phase 1 (Inference): Iteratively update value nodes to minimize free energy
    - Phase 2 (Learning): Update weights based on stabilized value nodes
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 35,  # Number of inference iterations (5*L for L layers)
        inference_lr: float = 0.1,  # Learning rate for value nodes
        weight_lr: float = 0.001,  # Learning rate for weights
        optimizer_x_fn: Callable = optim.SGD,  # Optimizer for value nodes
        optimizer_p_fn: Callable = optim.Adam,  # Optimizer for weights
        update_x_at: str = "all",  # When to update x: "all", "last", "never"
        update_p_at: str = "last",  # When to update weights: "all", "last", "never"
        device: str = "cpu",
    ):
        """Initialize PC Trainer.

        Args:
            model: The predictive coding network to train
            T: Number of inference iterations per batch
            inference_lr: Learning rate for inference (updating value nodes)
            weight_lr: Learning rate for weights
            optimizer_x_fn: Optimizer class for value nodes
            optimizer_p_fn: Optimizer class for weights
            update_x_at: When to update value nodes during inference
            update_p_at: When to update weights during inference
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.T = T
        self.device = device

        # Create optimizer for weights (parameters)
        # CRITICAL: Use get_network_parameters() to exclude value nodes
        # Value nodes should ONLY be optimized by optimizer_x, not optimizer_p
        self.optimizer_p = optimizer_p_fn(
            self.model.get_network_parameters(),
            lr=weight_lr
        )

        # Optimizer for value nodes will be created during training
        # (after first forward pass initializes them)
        self.optimizer_x_fn = optimizer_x_fn
        self.inference_lr = inference_lr
        self.optimizer_x = None

        # Parse update schedules
        self.update_x_at = self._parse_update_schedule(update_x_at, T)
        self.update_p_at = self._parse_update_schedule(update_p_at, T)

    def _parse_update_schedule(self, schedule: str, T: int) -> List[int]:
        """Parse update schedule string to list of iteration indices."""
        if schedule == "all":
            return list(range(T))
        elif schedule == "last":
            return [T - 1]
        elif schedule == "never":
            return []
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _recreate_optimizer_x(self):
        """Recreate optimizer for value nodes."""
        value_nodes = self.model.get_value_nodes()
        if len(value_nodes) > 0:
            self.optimizer_x = self.optimizer_x_fn(
                value_nodes,
                lr=self.inference_lr
            )

    def train_on_batch(
        self,
        inputs: torch.Tensor,
        loss_fn: Callable,
        targets: torch.Tensor = None,
    ) -> Dict:
        """Train on a single batch using two-phase PC algorithm.

        Args:
            inputs: Input data [batch_size, ...]
            loss_fn: Loss function that takes (outputs, targets) and returns scalar
            targets: Target labels (passed to loss_fn)

        Returns:
            Dictionary with training statistics
        """
        # Ensure model is in training mode
        assert self.model.training, "Call model.train() before training"

        # Move to device
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Initialize value nodes at start of batch
        self.model.set_sample_x(True)

        # Statistics
        losses = []
        energies = []
        overalls = []

        # ===== INFERENCE PHASE =====
        # Iteratively update value nodes to minimize free energy
        for t in range(self.T):

            # Forward pass (initializes or uses value nodes)
            outputs = self.model(inputs)

            # Compute task loss
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs)

            # Compute total energy from all PC layers
            layer_energies = self.model.get_energies()
            if len(layer_energies) > 0:
                total_energy = sum(layer_energies)
            else:
                total_energy = torch.tensor(0.0, device=self.device)

            # Free energy = task loss + prediction errors
            free_energy = loss + total_energy

            # Record statistics
            losses.append(loss.item())
            energies.append(total_energy.item())
            overalls.append(free_energy.item())

            # Create optimizer for value nodes on first iteration
            if t == 0:
                self._recreate_optimizer_x()

            # Zero gradients
            if self.optimizer_x is not None and t in self.update_x_at:
                self.optimizer_x.zero_grad()

            if t in self.update_p_at:
                self.optimizer_p.zero_grad()

            # Backward pass
            free_energy.backward()

            # Update value nodes (inference)
            if self.optimizer_x is not None and t in self.update_x_at:
                self.optimizer_x.step()

            # Update weights (learning)
            if t in self.update_p_at:
                self.optimizer_p.step()

        # Return statistics
        return {
            "loss": losses[-1],
            "energy": energies[-1],
            "free_energy": overalls[-1],
            "loss_history": losses,
            "energy_history": energies,
            "free_energy_history": overalls,
        }

    def test_on_batch(
        self,
        inputs: torch.Tensor,
        loss_fn: Callable,
        targets: torch.Tensor = None,
    ) -> Dict:
        """Evaluate on a batch (no training).

        Args:
            inputs: Input data
            loss_fn: Loss function
            targets: Target labels

        Returns:
            Dictionary with evaluation statistics
        """
        # Ensure model is in eval mode
        assert not self.model.training, "Call model.eval() before evaluation"

        # Move to device
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Forward pass (no PC dynamics in eval mode)
        with torch.no_grad():
            outputs = self.model(inputs)

            # Compute loss
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs)

        return {
            "loss": loss.item(),
            "outputs": outputs,
        }
