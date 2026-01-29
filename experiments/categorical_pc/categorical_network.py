"""
Categorical Predictive Coding Network.

Extends ModularNetwork with categorical constraints:
1. Compositional predictions: W_i→j = W_i→k ∘ W_k→j enforced via regularization
2. (Future) Functorial cross-network mappings
3. (Future) Universal properties for architecture

This is EXPERIMENTAL. We don't know if these constraints help or hurt learning.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from network.modular import ModularNetwork, SubNetwork
from typing import List, Dict, Optional


class CategoricalNetwork(ModularNetwork):
    """
    Predictive coding network with categorical constraints.

    Additional parameters beyond ModularNetwork:
        lambda_composition: Strength of compositional constraint (default: 0.1)
                           0 = no constraint (vanilla PC)
                           Higher = stronger enforcement
    """

    def __init__(
        self,
        subnetworks: List[SubNetwork],
        inference_lr: float = 0.1,
        temperature: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
        use_stable: bool = False,
        stable_lr: float = 0.001,
        stable_max_iterations: int = 400,
        stable_lr_schedule: str = "cosine",
        stable_decay_strong: float = 0.01,
        stable_decay_weak: float = 0.001,
        saturation_penalty: float = 0.01,
        activity_target: float = 0.3,
        lambda_composition: float = 0.1
    ):
        # Initialize parent
        super().__init__(
            subnetworks=subnetworks,
            inference_lr=inference_lr,
            temperature=temperature,
            dtype=dtype,
            device=device,
            use_stable=use_stable,
            stable_lr=stable_lr,
            stable_max_iterations=stable_max_iterations,
            stable_lr_schedule=stable_lr_schedule,
            stable_decay_strong=stable_decay_strong,
            stable_decay_weak=stable_decay_weak,
            saturation_penalty=saturation_penalty,
            activity_target=activity_target
        )

        self.lambda_composition = lambda_composition

        # Track composition violations for analysis
        self.composition_error = 0.0

    def update_weights(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        motor_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """
        Update weights with categorical constraints.

        Adds compositional constraint: For layers i > k > j,
        the prediction W_i→j should equal W_k→j ∘ W_i→k
        """
        # Standard PC weight update
        super().update_weights(lr=lr, weight_decay=weight_decay, motor_targets=motor_targets)

        # Add categorical constraints
        if self.lambda_composition > 0:
            self._enforce_composition(lr, weight_decay)

    def _enforce_composition(self, lr: float, weight_decay: float) -> None:
        """
        Enforce compositional structure: W_i→j = W_k→j ∘ W_i→k

        For each subnet, enforce that long-range predictions are compositions
        of intermediate predictions.

        This is implemented as a soft constraint (regularization), not hard constraint.
        """
        total_composition_error = 0.0
        num_constraints = 0

        for subnet in self.all_subnetworks:
            num_layers = len(subnet.layers)

            # For each triple of layers (i > k > j), enforce composition
            for i in range(2, num_layers):  # Start at layer 2 (need at least 3 layers)
                for j in range(i - 1):  # j < i
                    for k in range(j + 1, i):  # j < k < i
                        # Get prediction weights
                        # W_k→j: layer k predicts layer j
                        W_k_to_j = subnet.layers[k].neurons.W_apical  # Top-down from k

                        # W_i→k: layer i predicts layer k
                        W_i_to_k = subnet.layers[i].neurons.W_apical  # Top-down from i

                        # Check if both exist (some layers may not have apical weights)
                        if W_k_to_j is None or W_i_to_k is None:
                            continue

                        # Also need W_i→j to compare against
                        # For now, we don't explicitly store long-range predictions
                        # So we'll enforce composition on the basal weights instead
                        #
                        # Alternative interpretation: bottom-up predictions should compose
                        # W_j→k: layer j predicts layer k (basal)
                        W_j_to_k = subnet.layers[k].neurons.W_basal

                        # W_k→i: layer k predicts layer i (basal)
                        W_k_to_i = subnet.layers[i].neurons.W_basal

                        # Composed prediction: j → k → i should equal j → i
                        # (k → i) ∘ (j → k)
                        composed = W_k_to_i @ W_j_to_k

                        # We don't have explicit j → i weights, but we can enforce
                        # that intermediate layer k mediates the relationship
                        #
                        # Actually, let's take a simpler approach for now:
                        # Just enforce consistency between adjacent layers

            # SIMPLIFIED APPROACH: Enforce composition for adjacent pairs
            # For layers 0, 1, 2: enforce that 0→2 = (1→2) ∘ (0→1)
            if num_layers >= 3:
                # Bottom-up composition: layer 0 → layer 1 → layer 2
                # Should be consistent with direct 0 → 2 prediction

                # Get weights
                W_0_to_1 = subnet.layers[1].neurons.W_basal  # Layer 1 predicts from layer 0
                W_1_to_2 = subnet.layers[2].neurons.W_basal  # Layer 2 predicts from layer 1

                # Composed prediction from layer 0 to layer 2
                # This is what layer 2 would predict if driven by layer 0 through layer 1
                # state_2 = tanh(W_1_to_2 @ tanh(W_0_to_1 @ state_0))
                #
                # For linear approximation (small activations):
                # state_2 ≈ (W_1_to_2 @ W_0_to_1) @ state_0

                # The composition is: W_1_to_2 @ W_0_to_1
                # This should equal a direct 0→2 prediction if it existed
                #
                # But we don't have direct 0→2 weights in standard PC
                # So instead, enforce that the factorization is stable:
                # Minimize ||W_1_to_2 @ W_0_to_1 - (W_1_to_2 @ W_0_to_1)_prev||
                #
                # Actually, better approach: Use the ACTUAL activations to enforce composition

                # Get layer states
                state_0 = subnet.layers[0].get_state()
                state_1 = subnet.layers[1].get_state()
                state_2 = subnet.layers[2].get_state()

                # Direct prediction: layer 1 from layer 0
                pred_1_from_0 = torch.tanh(W_0_to_1 @ state_0)

                # Composed prediction: layer 2 from layer 0 via layer 1
                pred_2_from_0_via_1 = torch.tanh(W_1_to_2 @ pred_1_from_0)

                # Actual prediction: layer 2 from layer 1
                pred_2_from_1 = torch.tanh(W_1_to_2 @ state_1)

                # Composition constraint:
                # If layer 1 accurately represents layer 0 (pred_1_from_0 ≈ state_1),
                # then pred_2_from_0_via_1 should equal pred_2_from_1
                #
                # Error: How much does composition fail?
                composition_error = (pred_2_from_0_via_1 - pred_2_from_1).pow(2).sum()

                total_composition_error += composition_error.item()
                num_constraints += 1

                # Apply penalty to encourage composition
                # Gradient: push W_1_to_2 and W_0_to_1 to maintain composition
                if self.optimizer is not None:
                    # Add gradient penalty
                    # ∂L/∂W_1_to_2 from composition constraint
                    error_vec = pred_2_from_0_via_1 - pred_2_from_1

                    # Gradient through tanh
                    tanh_deriv = 1 - pred_2_from_0_via_1.pow(2)
                    weighted_error = error_vec * tanh_deriv

                    # Gradient for W_1_to_2
                    grad_W_1_to_2 = weighted_error.unsqueeze(1) @ pred_1_from_0.unsqueeze(0)

                    # Add to existing gradient
                    if subnet.layers[2].neurons.W_basal.grad is not None:
                        subnet.layers[2].neurons.W_basal.grad += self.lambda_composition * grad_W_1_to_2
                    else:
                        subnet.layers[2].neurons.W_basal.grad = self.lambda_composition * grad_W_1_to_2
                else:
                    # Manual update
                    error_vec = pred_2_from_0_via_1 - pred_2_from_1
                    tanh_deriv = 1 - pred_2_from_0_via_1.pow(2)
                    weighted_error = error_vec * tanh_deriv
                    grad_W_1_to_2 = weighted_error.unsqueeze(1) @ pred_1_from_0.unsqueeze(0)

                    # Update W_1_to_2 to reduce composition error
                    subnet.layers[2].neurons.W_basal.data -= lr * self.lambda_composition * grad_W_1_to_2

        # Track composition error for analysis
        if num_constraints > 0:
            self.composition_error = total_composition_error / num_constraints
        else:
            self.composition_error = 0.0

    def get_composition_error(self) -> float:
        """Return current composition error for monitoring."""
        return self.composition_error
