"""
Diagonal eBPC Trainer — Scalable Error-based Bayesian Predictive Coding

Optimizations over base eBPC trainer:
1. Diagonal V/Ψ: element-wise ops replace matrix multiplies/inverses
2. Adaptive T: early-stop inference when energy converges
3. bfloat16 mixed precision: autocast for forward/backward passes

Energy with diagonal precision:
  E = Σ_i 0.5 · (ε_i² · diag(Σ_i^{-1})).sum().mean()
    + 0.5 · ((y-ŷ)² · diag(Σ_L^{-1})).sum().mean()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Dict, List


class DiagonaleBPCTrainer:
    """Trainer for diagonal eBPC networks.

    E-step: ePC error optimization with diagonal precision weighting
    M-step: BPC Hebbian update with diagonal sufficient statistics
    """

    def __init__(
        self,
        model: nn.Module,
        T: int = 5,
        e_lr: float = 0.01,
        kappa: float = 0.25,
        adaptive_T: bool = True,
        convergence_threshold: float = 1e-6,
        use_amp: bool = True,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.T = T
        self.e_lr = e_lr
        self.kappa = kappa
        self.adaptive_T = adaptive_T
        self.convergence_threshold = convergence_threshold
        self.use_amp = use_amp and device != "cpu"  # AMP only on GPU
        self.device = device
        self.num_updates = 0

        # Track average iterations for diagnostics
        self._recent_T = []

    def _compute_energy(self, errors, output, target_one_hot):
        """Compute precision-weighted energy with diagonal Σ^{-1}.

        Much cheaper than full matrix: element-wise multiply instead of matmul.
        """
        E = torch.tensor(0.0, device=self.device)

        for i, layer in enumerate(self.model.layers[:-1]):
            prec_diag = layer.get_expected_precision_diag()  # [out_features]
            err = errors[i]
            # ε² · diag(Σ^{-1}), sum over features, mean over batch
            E = E + 0.5 * (err ** 2 * prec_diag.unsqueeze(0)).sum(dim=1).mean()

        # Output error
        out_layer = self.model.layers[-1]
        prec_diag_L = out_layer.get_expected_precision_diag()
        out_err = target_one_hot - output
        E = E + 0.5 * (out_err ** 2 * prec_diag_L.unsqueeze(0)).sum(dim=1).mean()

        return E

    def _bayesian_update_weights_diagonal(self, inputs, z_star):
        """Diagonal Hebbian update.

        Sufficient statistics for diagonal approximation:
          ss1 = Σ_n f_n ⊙ f_n  (element-wise square, summed over batch) [in]
          ss2 = Σ_n z_n · f_n^T  (outer product summed over batch) [out, in]
          ss3 = Σ_n z_n ⊙ z_n  (element-wise square, summed over batch) [out]
          ss4 = N (batch size)

        Update: η ← (1-κ)η + κ(η_prior + ss)
        """
        self.num_updates += 1
        kappa_t = self.num_updates ** (-self.kappa)

        # Pre-synaptic activations
        h0 = self.model.activation(inputs)
        pre_activations = [self.model._augment_with_bias(h0)]

        for z in z_star[:-1]:
            h = self.model.activation(z)
            pre_activations.append(self.model._augment_with_bias(h))

        for i, layer in enumerate(self.model.layers):
            f_pre = pre_activations[i]   # [batch, in]
            z_post = z_star[i]           # [batch, out]

            # Diagonal sufficient statistics
            ss1 = (f_pre ** 2).sum(dim=0)          # [in]
            ss2 = z_post.T @ f_pre                  # [out, in]
            ss3 = (z_post ** 2).sum(dim=0)          # [out]
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

    def train_on_batch(self, inputs, targets):
        """Train on one batch: ePC inference + diagonal BPC Hebbian update."""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        num_classes = self.model.layers[-1].out_features
        target_one_hot = F.one_hot(targets, num_classes).float()

        # === E-STEP: ePC Error Optimization ===
        errors = self.model.init_errors(inputs)
        error_optim = optim.Adam(errors, lr=self.e_lr)

        energy_history = []
        actual_T = self.T

        for t in range(self.T):
            error_optim.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output, states, last_input = self.model.epc_forward(inputs, errors)
                    E = self._compute_energy(errors, output, target_one_hot)
            else:
                output, states, last_input = self.model.epc_forward(inputs, errors)
                E = self._compute_energy(errors, output, target_one_hot)

            E.backward()
            error_optim.step()
            energy_history.append(E.item())

            # Adaptive T: early stop if converged
            if self.adaptive_T and t > 0:
                reduction = energy_history[-2] - energy_history[-1]
                if abs(reduction) < self.convergence_threshold:
                    actual_T = t + 1
                    break

        self._recent_T.append(actual_T)
        if len(self._recent_T) > 100:
            self._recent_T.pop(0)

        # Cross-entropy for logging
        with torch.no_grad():
            output, states, last_input = self.model.epc_forward(inputs, errors)
            loss = F.cross_entropy(output, targets)

        # === M-STEP: Diagonal Hebbian Update ===
        z_star = [s.detach() for s in states] + [target_one_hot]
        self._bayesian_update_weights_diagonal(inputs, z_star)

        # Per-layer energies for diagnostics
        layer_energies = []
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers[:-1]):
                prec_diag = layer.get_expected_precision_diag()
                err = errors[i]
                e_i = 0.5 * (err ** 2 * prec_diag.unsqueeze(0)).sum(dim=1).mean()
                layer_energies.append(e_i.item())
            prec_diag_L = self.model.layers[-1].get_expected_precision_diag()
            out_err = target_one_hot - output
            e_out = 0.5 * (out_err ** 2 * prec_diag_L.unsqueeze(0)).sum(dim=1).mean()
            layer_energies.append(e_out.item())

        return {
            "loss": loss.item(),
            "energy": energy_history[-1],
            "energy_history": energy_history,
            "layer_energies": layer_energies,
            "actual_T": actual_T,
            "avg_T": sum(self._recent_T) / len(self._recent_T),
        }

    def test_on_batch(self, inputs, loss_fn, targets=None):
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets) if targets is not None else loss_fn(outputs)
        return {"loss": loss.item(), "outputs": outputs}
