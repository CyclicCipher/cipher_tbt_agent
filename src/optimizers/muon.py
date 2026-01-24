"""
Muon optimizer implementation for predictive coding networks.

Based on "Muon: Momentum in Neuron Space" (KellerJordan, 2024)
Key idea: Apply momentum to neuron outputs rather than weights directly.

Simplified for our use case - full implementation would include:
- Orthogonalization of updates
- Separate learning rates for different parameter types
- Gradient clipping
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional


class Muon(Optimizer):
    """
    Muon optimizer: Momentum applied in neuron output space.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        weight_decay: Weight decay (L2 penalty) (default: 0.01)
        nesterov: Whether to use Nesterov momentum (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradient
                grad = p.grad

                # Add weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Initialize momentum buffer if needed
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    buf = param_state['momentum_buffer']

                # Update momentum buffer: m = β*m + (1-β)*∇f
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                # Nesterov momentum: look ahead
                if nesterov:
                    # Update using: p = p - lr * (β*m + (1-β)*∇f)
                    update = grad.mul(1 - momentum).add(buf, alpha=momentum)
                else:
                    # Standard momentum: p = p - lr * m
                    update = buf

                # Apply update
                p.add_(update, alpha=-lr)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients of all optimized parameters.

        Args:
            set_to_none: Instead of setting to zero, set to None for better memory efficiency
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()


class MuonWithActivityReg(Muon):
    """
    Muon optimizer with activity regularization for preventing pathologies.

    Adds penalties for:
    - Saturation (neurons stuck at ±1)
    - Dead neurons (no activation)
    - Too-high activation (seizure-like)

    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: L2 regularization
        saturation_penalty: Penalty for saturated activations (default: 0.01)
        activity_target: Target mean activation level (default: 0.3)
        activity_penalty: Penalty for deviating from target (default: 0.001)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        saturation_penalty: float = 0.01,
        activity_target: float = 0.3,
        activity_penalty: float = 0.001,
    ):
        super().__init__(params, lr, momentum, weight_decay)
        self.saturation_penalty = saturation_penalty
        self.activity_target = activity_target
        self.activity_penalty = activity_penalty

        # Track activation statistics
        self.activation_stats = {
            'mean_activation': [],
            'saturation_rate': [],
            'dead_neurons': []
        }

    def add_activity_regularization(self, activations: List[torch.Tensor]):
        """
        Add activity regularization gradients based on layer activations.

        Args:
            activations: List of activation tensors from each layer
        """
        for i, act in enumerate(activations):
            if act.requires_grad:
                # Saturation penalty: penalize neurons near ±1
                saturation_mask = (act.abs() > 0.9).float()
                saturation_loss = self.saturation_penalty * saturation_mask.mean()

                # Activity penalty: maintain target activation level
                mean_act = act.abs().mean()
                activity_loss = self.activity_penalty * (mean_act - self.activity_target) ** 2

                # Track statistics
                self.activation_stats['mean_activation'].append(mean_act.item())
                self.activation_stats['saturation_rate'].append(saturation_mask.mean().item())
                self.activation_stats['dead_neurons'].append((act.abs() < 0.01).float().mean().item())

                # Add to gradient
                total_reg_loss = saturation_loss + activity_loss
                if total_reg_loss.requires_grad:
                    total_reg_loss.backward(retain_graph=True)

    def get_activation_stats(self):
        """Get recent activation statistics."""
        return {
            key: sum(vals[-10:]) / min(len(vals), 10) if vals else 0
            for key, vals in self.activation_stats.items()
        }
