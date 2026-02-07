"""
Bayesian Predictive Coding Trainer

Implements Algorithm 1 from:
"Bayesian Predictive Coding" (Tschantz et al., 2025, arXiv:2503.24016)

Two-phase algorithm:
1. E-step (Inference): Optimize hidden value nodes z via gradient descent
2. M-step (Learning): Closed-form Bayesian update of weight posterior (Equation 7)

Supervised training: "fixes the input nodes to z_0 = x and the output nodes to z_L = y" (page 3).
The output layer is CLAMPED to the target — there is no separate task loss.
The prediction errors at all layers (including the output) are the ONLY learning signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Dict, List


class BayesianPCTrainer:
    """Trainer for Bayesian Predictive Coding networks.

    Implements the EM algorithm on variational free energy:
    - E-step: Optimize hidden value nodes z* via gradient descent on E(Z, λ)
    - M-step: Closed-form Bayesian update of natural parameters η (Equation 7)

    Output layer is clamped to the target (supervised PC, page 3 of paper).
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 35,  # Number of inference iterations
        inference_lr: float = 0.1,  # Learning rate for value nodes
        kappa: float = 0.25,  # Learning rate decay exponent for natural params
        optimizer_x_fn: Callable = optim.Adam,  # Optimizer for value nodes
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.T = T
        self.device = device
        self.inference_lr = inference_lr
        self.kappa = kappa

        self.optimizer_x_fn = optimizer_x_fn
        self.optimizer_x = None

        self.num_updates = 0

    def _create_optimizer_x(self):
        """Create optimizer for HIDDEN value nodes only.

        Output layer is clamped to target (not optimized).
        """
        value_nodes = []
        for layer in self.model.layers[:-1]:  # Exclude output layer
            value_nodes.extend(layer.get_value_nodes())
        if len(value_nodes) > 0:
            self.optimizer_x = self.optimizer_x_fn(
                value_nodes,
                lr=self.inference_lr
            )

    def _bayesian_update_weights(self, inputs: torch.Tensor, z_star: List[torch.Tensor]):
        """Closed-form Bayesian update of weight posterior (Equation 7).

        η_l* = η_l^(0) + Σ_n [f(z*_{l-1})f(z*_{l-1})^T,
                               f(z*_{l-1})z*_l^T,
                               z*_l z*_l^T,
                               1]

        For the output layer, z*_L = target (clamped).
        """
        batch_size = inputs.size(0)

        self.num_updates += 1
        kappa_t = self.num_updates ** (-self.kappa)

        # Build list of pre-synaptic activations
        h0 = self.model.activation(inputs)
        pre_activations = [self.model._augment_with_bias(h0)]

        for z in z_star[:-1]:  # All layers except last
            h = self.model.activation(z.detach())
            pre_activations.append(self.model._augment_with_bias(h))

        # Update each layer
        for i, layer in enumerate(self.model.layers):
            f_pre = pre_activations[i]
            z_post = z_star[i].detach()

            # Sufficient statistics (Equation 7)
            ss1 = f_pre.T @ f_pre
            ss2 = z_post.T @ f_pre
            ss3 = z_post.T @ z_post
            ss4 = float(batch_size)

            # Optimal natural parameters
            eta1_new = layer.eta1_prior + ss1
            eta2_new = layer.eta2_prior + ss2
            eta3_new = layer.eta3_prior + ss3
            eta4_new = layer.eta4_prior + ss4

            # Minibatch update: η ← (1 - κ_t)·η + κ_t·η*
            with torch.no_grad():
                layer.eta1.data = (1 - kappa_t) * layer.eta1.data + kappa_t * eta1_new
                layer.eta2.data = (1 - kappa_t) * layer.eta2.data + kappa_t * eta2_new
                layer.eta3.data = (1 - kappa_t) * layer.eta3.data + kappa_t * eta3_new
                layer.eta4.data = (1 - kappa_t) * layer.eta4.data + kappa_t * eta4_new

    def train_on_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        """Train on a single batch using Algorithm 1.

        Supervised PC: clamp z_0 = inputs, z_L = one_hot(targets).
        Optimize hidden value nodes z_1,...,z_{L-1} via gradient descent on
        prediction error energy E(Z, λ). No separate task loss.

        Args:
            inputs: Input data [batch_size, input_dim]
            targets: Target class labels [batch_size]

        Returns:
            Dictionary with training statistics
        """
        assert self.model.training, "Call model.train() before training"

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # One-hot encode targets for output clamping
        num_classes = self.model.layers[-1].out_features
        target_one_hot = F.one_hot(targets, num_classes).float()

        # Reset value nodes for new batch
        self.model.set_sample_x(True)

        losses = []
        energies = []
        free_energies = []

        # ===== E-STEP: INFERENCE =====
        for t in range(self.T):
            # Forward pass (initializes value nodes on t=0)
            outputs = self.model(inputs, sample_x=True)

            if t == 0:
                # Clamp output layer: z_L = y (page 3 of paper)
                with torch.no_grad():
                    self.model.layers[-1]._x.data.copy_(target_one_hot)
                # Recompute output energy with clamped target
                self.model.layers[-1]._compute_energy(self.model._last_layer_input)

            # Free energy = sum of prediction errors only (Equation 5)
            layer_energies = self.model.get_energies()
            total_energy = sum(layer_energies) if layer_energies else torch.tensor(0.0, device=self.device)
            free_energy = total_energy

            # Compute cross-entropy loss for LOGGING only (not used in optimization)
            with torch.no_grad():
                M, _, _, _ = self.model.layers[-1].natural_to_standard()
                predicted = F.linear(self.model._last_layer_input.detach(), M)
                loss = F.cross_entropy(predicted, targets)

            losses.append(loss.item())
            energies.append(total_energy.item() if isinstance(total_energy, torch.Tensor) else total_energy)
            free_energies.append(free_energy.item())

            if t == 0:
                self._create_optimizer_x()

            # Gradient descent on HIDDEN value nodes only
            if self.optimizer_x is not None:
                self.optimizer_x.zero_grad()
                free_energy.backward()
                self.optimizer_x.step()

        # ===== M-STEP: LEARNING =====
        # Collect converged value nodes (output = clamped target)
        z_star = [layer._x.detach() for layer in self.model.layers]

        self._bayesian_update_weights(inputs, z_star)

        return {
            "loss": losses[-1],
            "energy": energies[-1],
            "free_energy": free_energies[-1],
            "loss_history": losses,
            "energy_history": energies,
            "free_energy_history": free_energies,
        }

    def test_on_batch(
        self,
        inputs: torch.Tensor,
        loss_fn: Callable,
        targets: torch.Tensor = None,
    ) -> Dict:
        """Evaluate on a batch (no training).

        Uses expected weights for a deterministic forward pass (Equation 9).
        """
        assert not self.model.training, "Call model.eval() before evaluation"

        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs, sample_x=False)

            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs)

        return {
            "loss": loss.item(),
            "outputs": outputs,
        }
