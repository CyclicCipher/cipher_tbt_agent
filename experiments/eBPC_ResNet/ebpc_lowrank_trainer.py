"""
Low-Rank eBPC Trainer — Error-based Bayesian Predictive Coding with Low-Rank V

E-step: ePC error optimization (same as diagonal — Adam on errors, precision-weighted energy)
M-step: Hebbian update with low-rank η1 structure

Key difference from diagonal trainer:
  η1 = diag(d) + U·U^T requires special handling in the M-step.

  Full sufficient statistic: ss1 = F^T @ F  [in, in]  (outer product of pre-activations)
  This is a rank-B matrix (B = batch size). We can't store it as full [in, in],
  but we CAN extract its top-k modes via truncated SVD.

  Update strategy:
    1. Form combined low-rank factor: A = [sqrt(1-κ) · U_old | sqrt(κ) · F^T_normalized]
       where F^T_normalized accounts for the prior and batch contributions
    2. Truncated SVD of A → keep top-k singular vectors → new U
    3. Fold residual singular values into the diagonal d
    4. η2, η3, η4 updated with standard moving average (same as diagonal)

  This ensures η1 maintains its low-rank + diagonal structure across updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict


class LowRankeBPCTrainer:
    """Trainer for low-rank eBPC networks.

    E-step: ePC error optimization with diagonal precision weighting
    M-step: BPC Hebbian update with low-rank V structure
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
        self.use_amp = use_amp and device != "cpu"
        self.device = device
        self.num_updates = 0

        self._recent_T = []

    def _compute_energy(self, errors, output, target_one_hot):
        """Compute precision-weighted energy with diagonal Σ^{-1}."""
        E = torch.tensor(0.0, device=self.device)

        for i, layer in enumerate(self.model.layers[:-1]):
            prec_diag = layer.get_expected_precision_diag()
            err = errors[i]
            E = E + 0.5 * (err ** 2 * prec_diag.unsqueeze(0)).sum(dim=1).mean()

        out_layer = self.model.layers[-1]
        prec_diag_L = out_layer.get_expected_precision_diag()
        out_err = target_one_hot - output
        E = E + 0.5 * (out_err ** 2 * prec_diag_L.unsqueeze(0)).sum(dim=1).mean()

        return E

    def _update_eta1_lowrank(self, layer, f_pre, kappa_t):
        """Update η1 = diag(d) + U·U^T with new batch data.

        The full sufficient statistic is ss1 = F^T @ F [in, in].
        The new η1 should be: (1-κ)·η1_old + κ·(η1_prior + ss1)

        We decompose this into diagonal + low-rank:
          (1-κ)·(d_old + U_old·U_old^T) + κ·(d_prior + ss1)
          = [(1-κ)·d_old + κ·d_prior + κ·diag(ss1)] + (1-κ)·U_old·U_old^T + κ·(ss1 - diag(ss1))

        The off-diagonal part of ss1 is captured by F^T's low-rank structure.
        Strategy: form combined matrix, truncated SVD, fold residual into diagonal.
        """
        N = f_pre.size(0)
        in_features = layer.in_features
        rank_k = layer.rank_k

        # Diagonal part of ss1 = sum of f^2 over batch
        ss1_diag = (f_pre ** 2).sum(dim=0)  # [in]

        # Update diagonal: (1-κ)·d_old + κ·(d_prior + ss1_diag)
        d_new_base = (1 - kappa_t) * layer.eta1_d + kappa_t * (layer.eta1_d_prior + ss1_diag)

        # Off-diagonal contributions come from two sources:
        # 1. Old U: scaled by sqrt(1-κ)
        # 2. New batch: F^T captures the off-diagonal structure of ss1 = F^T @ F
        #    We need sqrt(κ) · F^T so that (sqrt(κ)·F^T)(sqrt(κ)·F^T)^T = κ · F^T·F = κ · ss1

        # But ss1 includes the diagonal part too. We want only the off-diagonal
        # contribution in U. However, the SVD naturally handles this:
        # we'll combine everything into a matrix, SVD it, and put the top-k
        # modes into U while folding the rest into d.

        # Form combined matrix of all low-rank contributions:
        # A = [ sqrt(1-κ)·U_old | sqrt(κ)·F^T ]
        # Then A·A^T = (1-κ)·U_old·U_old^T + κ·F^T·F = (1-κ)·U_old·U_old^T + κ·ss1
        # This is the total off-diagonal + some diagonal contribution from ss1

        sqrt_1mk = torch.sqrt(torch.tensor(1.0 - kappa_t, device=self.device))
        sqrt_k = torch.sqrt(torch.tensor(kappa_t, device=self.device))

        A_old = sqrt_1mk * layer.eta1_U  # [in, k]
        A_new = sqrt_k * f_pre.T  # [in, N]

        # Also add the prior U contribution (though prior U is 0 initially)
        # η1_prior contribution: κ · U_prior · U_prior^T (zero if U_prior = 0)
        # Skip if U_prior is zero
        has_prior_U = layer.eta1_U_prior.abs().max() > 0
        if has_prior_U:
            A_prior = sqrt_k * layer.eta1_U_prior  # [in, k]
            A = torch.cat([A_old, A_new, A_prior], dim=1)  # [in, k + N + k]
        else:
            A = torch.cat([A_old, A_new], dim=1)  # [in, k + N]

        # Truncated SVD of A → top-k singular vectors
        # A = P·S·Q^T, so A·A^T = P·S²·P^T
        # We want U_new such that U_new·U_new^T ≈ A·A^T (rank-k approximation)
        # → U_new = P[:, :k] · S[:k]

        # For numerical stability, use SVD of A (not A·A^T)
        # A is [in_features, k+N] — tall and thin when N < in, or fat when N > in
        try:
            P, S, Qt = torch.linalg.svd(A, full_matrices=False)
        except torch.linalg.LinAlgError:
            # SVD failed — keep old values (rare edge case)
            with torch.no_grad():
                layer.eta1_d.data = d_new_base
            return

        # Keep top-k modes in U, fold the rest into diagonal
        actual_k = min(rank_k, len(S))
        U_new = P[:, :actual_k] * S[:actual_k].unsqueeze(0)  # [in, k]

        # The residual modes (k+1, k+2, ...) contribute to the diagonal:
        # residual = Σ_{j>k} s_j² · p_j · p_j^T → diagonal contribution = Σ_{j>k} s_j² · p_j²
        if len(S) > actual_k:
            residual_P = P[:, actual_k:]  # [in, remaining]
            residual_S = S[actual_k:]  # [remaining]
            residual_diag = (residual_P ** 2 * (residual_S ** 2).unsqueeze(0)).sum(dim=1)  # [in]
        else:
            residual_diag = torch.zeros(in_features, device=self.device)

        # But d_new_base already includes κ·ss1_diag, and A·A^T already includes
        # the full κ·ss1 (including diagonal). So we need to subtract the diagonal
        # part that's now captured by U_new·U_new^T and residual:
        # Total from A·A^T diagonal = U_new contribution + residual
        U_diag_contribution = (U_new ** 2).sum(dim=1)  # [in]
        total_AA_diag = U_diag_contribution + residual_diag  # [in]

        # The diagonal we want: d_new_base captures (1-κ)·d_old + κ·(d_prior + ss1_diag)
        # But (1-κ)·d_old is part of (1-κ)·η1_old_diagonal, and we also moved
        # (1-κ)·U_old·U_old^T into A. The diagonal of A·A^T includes the diagonal
        # of (1-κ)·U_old·U_old^T which shouldn't be in d.
        #
        # Clean accounting:
        #   η1_new = d_new + U_new·U_new^T
        #   where η1_new should equal (1-κ)·d_old + κ·(d_prior + ss1_diag)  [diagonal]
        #                           + (1-κ)·U_old·U_old^T + κ·(ss1 - diag(ss1)·I)  [off-diagonal + extra diag]
        #
        # Actually, let's think more carefully. We want:
        #   d_new + U_new·U_new^T ≈ d_new_base + A·A^T - κ·diag(ss1)·I
        #
        # Wait, d_new_base already has κ·ss1_diag. And A·A^T = (1-κ)·U·U^T + κ·ss1.
        # The diagonal of A·A^T includes κ·diag(ss1).
        # So: d_final = d_new_base - diagonal(A·A^T captured in U + residual) + residual_diag
        #             = d_new_base - U_diag_contribution
        # No wait. Let me redo this cleanly.
        #
        # Target: η1_target = (1-κ)·η1_old + κ·(η1_prior + ss1_full)
        #   = (1-κ)·(d_old + U_old·U_old^T) + κ·(d_prior + U_prior·U_prior^T + ss1_full)
        #   = [(1-κ)·d_old + κ·d_prior] + [(1-κ)·U_old·U_old^T + κ·U_prior·U_prior^T + κ·ss1_full]
        #
        # We approximate this as d_new + U_new·U_new^T where:
        #   A·A^T = (1-κ)·U_old·U_old^T + κ·ss1_full + [κ·U_prior·U_prior^T if applicable]
        #   U_new·U_new^T ≈ top-k of A·A^T
        #   residual ≈ A·A^T - U_new·U_new^T (folded into diagonal)
        #
        # So: d_new = [(1-κ)·d_old + κ·d_prior] + residual_diag
        d_final = (1 - kappa_t) * layer.eta1_d + kappa_t * layer.eta1_d_prior + residual_diag

        # Ensure d stays positive (PD guarantee for η1)
        d_final = torch.clamp(d_final, min=1e-8)

        # Pad U_new if actual_k < rank_k
        if actual_k < rank_k:
            padding = torch.zeros(in_features, rank_k - actual_k, device=self.device)
            U_new = torch.cat([U_new, padding], dim=1)

        with torch.no_grad():
            layer.eta1_d.data = d_final
            layer.eta1_U.data = U_new

    def _bayesian_update_weights_lowrank(self, inputs, z_star):
        """Low-rank Hebbian update.

        η1: low-rank update via SVD (see _update_eta1_lowrank)
        η2, η3, η4: standard moving average (same as diagonal)
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

            # Update η1 with low-rank structure
            self._update_eta1_lowrank(layer, f_pre, kappa_t)

            # Standard sufficient statistics for η2, η3, η4
            ss2 = z_post.T @ f_pre                  # [out, in]
            ss3 = (z_post ** 2).sum(dim=0)          # [out]
            ss4 = float(inputs.size(0))

            eta2_new = layer.eta2_prior + ss2
            eta3_new = layer.eta3_prior + ss3
            eta4_new = layer.eta4_prior + ss4

            with torch.no_grad():
                layer.eta2.data = (1 - kappa_t) * layer.eta2.data + kappa_t * eta2_new
                layer.eta3.data = (1 - kappa_t) * layer.eta3.data + kappa_t * eta3_new
                layer.eta4.data = (1 - kappa_t) * layer.eta4.data + kappa_t * eta4_new

            # Invalidate and recompute M cache
            layer._cache_valid = False
            layer._update_M_cache()

    def train_on_batch(self, inputs, targets):
        """Train on one batch: ePC inference + low-rank BPC Hebbian update."""
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

        # === M-STEP: Low-Rank Hebbian Update ===
        z_star = [s.detach() for s in states] + [target_one_hot]
        self._bayesian_update_weights_lowrank(inputs, z_star)

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
