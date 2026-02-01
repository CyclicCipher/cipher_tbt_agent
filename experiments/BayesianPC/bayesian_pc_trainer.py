"""
Bayesian Predictive Coding Trainer

Implements Algorithm 1 from:
"Bayesian Predictive Coding" (Tschantz et al., 2025, arXiv:2503.24016)

Two-phase algorithm:
1. E-step (Inference): Optimize value nodes z via gradient descent
2. M-step (Learning): Closed-form Bayesian update of weight posterior (Equation 7)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List


class BayesianPCTrainer:
    """Trainer for Bayesian Predictive Coding networks.

    Implements the EM algorithm on variational free energy:
    - E-step: Optimize value nodes z* via gradient descent on E(Z, λ)
    - M-step: Closed-form Bayesian update of natural parameters η (Equation 7)
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
        """Initialize Bayesian PC Trainer.

        Args:
            model: BayesianPCNetwork to train
            T: Number of inference iterations per batch
            inference_lr: Learning rate for inference (updating value nodes)
            kappa: Learning rate decay for natural parameter updates (κ_t = t^{-κ})
            optimizer_x_fn: Optimizer class for value nodes
            device: Device to train on
        """
        self.model = model.to(device)
        self.T = T
        self.device = device
        self.inference_lr = inference_lr
        self.kappa = kappa

        # Optimizer for value nodes (created fresh each batch)
        self.optimizer_x_fn = optimizer_x_fn
        self.optimizer_x = None

        # Track number of parameter updates for learning rate schedule
        self.num_updates = 0

    def _create_optimizer_x(self):
        """Create optimizer for value nodes.

        From Appendix F.1 (page 12):
        "For BPC, we used the Adam optimizer for hidden states,
         with a learning rate of 0.01 and 10 iterations per batch."

        The precision matrix Σ^{-1} already appears in the gradient (Equations 15-16),
        providing adaptive weighting. No need to scale the learning rate.
        """
        value_nodes = self.model.get_value_nodes()
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

        This is a Hebbian update based on pre- and post-synaptic activity.

        Args:
            inputs: Original input to network [batch_size, input_dim]
            z_star: Converged value nodes from all layers
        """
        batch_size = inputs.size(0)

        # Learning rate schedule: κ_t = t^{-epsilon}
        self.num_updates += 1
        kappa_t = self.num_updates ** (-self.kappa)

        # Build list of pre-synaptic activations (input + hidden layers after activation)
        pre_activations = [self.model.activation(inputs)]  # f(z_0)

        for z in z_star[:-1]:  # All layers except last
            pre_activations.append(self.model.activation(z.detach()))  # f(z_l)

        # Update each layer
        for i, layer in enumerate(self.model.layers):
            f_pre = pre_activations[i]  # f(z_{l-1}) [batch, in_features]
            z_post = z_star[i].detach()  # z_l [batch, out_features]

            # Compute sufficient statistics (Equation 28)
            # Sum over batch dimension n

            # SS1 = Σ_n f(z_{l-1})f(z_{l-1})^T  [in_features, in_features]
            ss1 = torch.zeros(layer.in_features, layer.in_features, device=self.device)
            for b in range(batch_size):
                f_b = f_pre[b:b+1].T  # [in_features, 1]
                ss1 += f_b @ f_b.T

            # SS2 = Σ_n f(z_{l-1})z_l^T  [out_features, in_features]
            ss2 = torch.zeros(layer.out_features, layer.in_features, device=self.device)
            for b in range(batch_size):
                z_b = z_post[b:b+1].T  # [out_features, 1]
                f_b = f_pre[b:b+1].T   # [in_features, 1]
                ss2 += z_b @ f_b.T

            # SS3 = Σ_n z_l z_l^T  [out_features, out_features]
            ss3 = torch.zeros(layer.out_features, layer.out_features, device=self.device)
            for b in range(batch_size):
                z_b = z_post[b:b+1].T  # [out_features, 1]
                ss3 += z_b @ z_b.T

            # SS4 = Σ_n 1 = batch_size
            ss4 = float(batch_size)

            # Compute optimal natural parameters (Equation 28)
            # η* = η_prior + sufficient_statistics
            eta1_new = layer.eta1_prior + ss1
            eta2_new = layer.eta2_prior + ss2
            eta3_new = layer.eta3_prior + ss3
            eta4_new = layer.eta4_prior + ss4

            # Minibatch update with learning rate schedule (bottom of page 3)
            # η ← (1 - κ_t)·η + κ_t·η*
            with torch.no_grad():
                layer.eta1.data = (1 - kappa_t) * layer.eta1.data + kappa_t * eta1_new
                layer.eta2.data = (1 - kappa_t) * layer.eta2.data + kappa_t * eta2_new
                layer.eta3.data = (1 - kappa_t) * layer.eta3.data + kappa_t * eta3_new
                layer.eta4.data = (1 - kappa_t) * layer.eta4.data + kappa_t * eta4_new

    def train_on_batch(
        self,
        inputs: torch.Tensor,
        loss_fn: Callable,
        targets: torch.Tensor = None,
    ) -> Dict:
        """Train on a single batch using Bayesian PC algorithm.

        Args:
            inputs: Input data [batch_size, ...]
            loss_fn: Loss function (outputs, targets) -> scalar
            targets: Target labels

        Returns:
            Dictionary with training statistics
        """
        assert self.model.training, "Call model.train() before training"

        # Move to device
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Initialize value nodes
        self.model.set_sample_x(True)

        # Statistics
        losses = []
        energies = []
        free_energies = []

        # ===== E-STEP: INFERENCE =====
        # Optimize value nodes z to minimize free energy E(Z, λ)
        for t in range(self.T):
            # Forward pass (initializes or uses value nodes)
            outputs = self.model(inputs, sample_x=True)

            # Compute task loss
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs)

            # Compute total energy from all layers
            layer_energies = self.model.get_energies()
            total_energy = sum(layer_energies) if layer_energies else torch.tensor(0.0, device=self.device)

            # Free energy = task loss + prediction errors
            free_energy = loss + total_energy

            # Record statistics
            losses.append(loss.item())
            energies.append(total_energy.item() if isinstance(total_energy, torch.Tensor) else total_energy)
            free_energies.append(free_energy.item())

            # Create optimizer on first iteration
            if t == 0:
                self._create_optimizer_x()

            # Gradient descent on value nodes
            if self.optimizer_x is not None:
                self.optimizer_x.zero_grad()
                free_energy.backward()
                self.optimizer_x.step()

        # ===== M-STEP: LEARNING =====
        # Closed-form Bayesian update of weight posterior
        # Collect converged value nodes from all layers
        z_star = [layer._x.detach() for layer in self.model.layers]

        # Update natural parameters using Equation 7
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

        Args:
            inputs: Input data
            loss_fn: Loss function
            targets: Target labels

        Returns:
            Dictionary with evaluation statistics
        """
        assert not self.model.training, "Call model.eval() before evaluation"

        # Move to device
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        # Forward pass with expected weights (no value nodes)
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
