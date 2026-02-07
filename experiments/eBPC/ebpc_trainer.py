"""
eBPC Trainer — Error-based Bayesian Predictive Coding

Combines:
- ePC inference: Optimize error tensors ε via backprop through global graph
- BPC learning: Closed-form Hebbian update of MNW weight posterior (Equation 7)

Algorithm:
1. Initialize ε_i ← 0 for hidden layers (ePC, Algorithm 2)
2. E-step: Minimize precision-weighted energy w.r.t. ε via SGD (ePC)
   - s_i = M_i @ f(s_{i-1}) + ε_i builds global graph
   - E = Σ 0.5·ε_i^T·Σ_i^{-1}·ε_i + 0.5·(y-ŷ)^T·Σ_L^{-1}·(y-ŷ)
   - Backprop delivers gradients to ALL errors simultaneously (no signal decay)
3. M-step: Recover states z*_i = ŝ_i + ε*_i, apply BPC Hebbian update (Eq 7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Dict, List


class eBPCTrainer:
    """Trainer for eBPC networks.

    E-step: ePC error optimization (global backprop, no signal decay)
    M-step: BPC Hebbian weight update (closed-form, local)
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 5,              # Inference iterations (ePC needs fewer than sPC)
        e_lr: float = 0.001,     # Error learning rate (ePC default)
        kappa: float = 0.25,     # Hebbian update decay exponent
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.T = T
        self.e_lr = e_lr
        self.kappa = kappa
        self.device = device
        self.num_updates = 0

    def _compute_energy(
        self,
        errors: list,
        output: torch.Tensor,
        target_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute precision-weighted eBPC energy.

        Hidden layers: 0.5 · ε_i^T · E[Σ_i^{-1}] · ε_i
        Output layer:  0.5 · (y - ŷ)^T · E[Σ_L^{-1}] · (y - ŷ)

        Uses mean over batch (consistent with BPC convention).
        """
        E = torch.tensor(0.0, device=self.device)

        # Hidden layer error norms (precision-weighted)
        for i, layer in enumerate(self.model.layers[:-1]):
            Sigma_inv = layer.get_expected_precision()
            err = errors[i]
            E = E + 0.5 * torch.sum(err @ Sigma_inv * err, dim=1).mean()

        # Output prediction error (clamped target vs prediction)
        out_layer = self.model.layers[-1]
        Sigma_inv_L = out_layer.get_expected_precision()
        out_err = target_one_hot - output
        E = E + 0.5 * torch.sum(out_err @ Sigma_inv_L * out_err, dim=1).mean()

        return E

    def _bayesian_update_weights(self, inputs: torch.Tensor, z_star: List[torch.Tensor]):
        """Closed-form Bayesian update of weight posterior (Equation 7).

        η_l* = η_l^(0) + Σ_n [f(z*_{l-1})f(z*_{l-1})^T,
                               f(z*_{l-1})z*_l^T,
                               z*_l z*_l^T,
                               1]
        """
        self.num_updates += 1
        kappa_t = self.num_updates ** (-self.kappa)

        # Build pre-synaptic activations: f(z*_{l-1}) augmented with bias
        h0 = self.model.activation(inputs)
        pre_activations = [self.model._augment_with_bias(h0)]

        for z in z_star[:-1]:
            h = self.model.activation(z)
            pre_activations.append(self.model._augment_with_bias(h))

        # Update each layer's natural parameters
        for i, layer in enumerate(self.model.layers):
            f_pre = pre_activations[i]
            z_post = z_star[i]

            ss1 = f_pre.T @ f_pre
            ss2 = z_post.T @ f_pre
            ss3 = z_post.T @ z_post
            ss4 = float(inputs.size(0))

            eta1_new = layer.eta1_prior + ss1
            eta2_new = layer.eta2_prior + ss2
            eta3_new = layer.eta3_prior + ss3
            eta4_new = layer.eta4_prior + ss4

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
        """Train on one batch: ePC inference + BPC Hebbian update.

        1. Initialize errors ε_i ← 0
        2. Optimize errors via SGD on precision-weighted energy (global backprop)
        3. Recover converged states, apply BPC Hebbian update
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        num_classes = self.model.layers[-1].out_features
        target_one_hot = F.one_hot(targets, num_classes).float()

        # === E-STEP: ePC Error Optimization ===

        # 1. Initialize errors to zero
        errors = self.model.init_errors(inputs)

        # 2. Optimize errors via SGD (ePC uses SGD, not Adam)
        error_optim = optim.SGD(errors, lr=self.e_lr)

        energy_history = []
        for t in range(self.T):
            error_optim.zero_grad()
            output, states, last_input = self.model.epc_forward(inputs, errors)
            E = self._compute_energy(errors, output, target_one_hot)
            E.backward()
            error_optim.step()
            energy_history.append(E.item())

        # Cross-entropy for logging only
        with torch.no_grad():
            output, states, last_input = self.model.epc_forward(inputs, errors)
            loss = F.cross_entropy(output, targets)

        # === M-STEP: BPC Hebbian Update ===

        # Recover converged states: z*_i = ŝ_i + ε*_i
        z_star = [s.detach() for s in states] + [target_one_hot]
        self._bayesian_update_weights(inputs, z_star)

        # Per-layer energies for diagnostics
        layer_energies = []
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers[:-1]):
                Sigma_inv = layer.get_expected_precision()
                err = errors[i]
                e_i = 0.5 * torch.sum(err @ Sigma_inv * err, dim=1).mean()
                layer_energies.append(e_i.item())
            # Output layer energy
            Sigma_inv_L = self.model.layers[-1].get_expected_precision()
            out_err = target_one_hot - output
            e_out = 0.5 * torch.sum(out_err @ Sigma_inv_L * out_err, dim=1).mean()
            layer_energies.append(e_out.item())

        return {
            "loss": loss.item(),
            "energy": energy_history[-1],
            "energy_history": energy_history,
            "layer_energies": layer_energies,
        }

    def test_on_batch(
        self,
        inputs: torch.Tensor,
        loss_fn: Callable,
        targets: torch.Tensor = None,
    ) -> Dict:
        """Evaluate on a batch. Uses expected weights (no inference needed)."""
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)

        return {"loss": loss.item(), "outputs": outputs}
