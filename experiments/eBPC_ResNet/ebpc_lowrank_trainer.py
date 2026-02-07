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
    1. Form combined low-rank factor: A = [sqrt(1-κ) · U_old | sqrt(κ) · F^T]
    2. Truncated SVD of A → keep top-k singular vectors → new U
    3. Fold residual singular values into the diagonal d
    4. η2 updated with standard moving average
    5. Ψ^{-1} updated from regression RESIDUALS (always ≥ 0, guarantees positivity)
    6. η4 updated with standard moving average

  Critical design choice: Ψ^{-1} is computed from residuals r = z - M·f,
  NOT from η3 = Ψ^{-1} + M·η1·M^T. The Schur complement Φ = η3 - M·η1·M^T
  amplifies approximation error in η1, potentially making Φ negative.
  Direct residual computation avoids this entirely.
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

    def _diagonal_only_update(self, layer, f_pre, kappa_t):
        """Fallback: update only the diagonal of η1, zero out U."""
        ss1_diag = (f_pre ** 2).sum(dim=0)
        with torch.no_grad():
            layer.eta1_d.data = (1 - kappa_t) * layer.eta1_d + kappa_t * (layer.eta1_d_prior + ss1_diag)
            layer.eta1_d.data = torch.clamp(layer.eta1_d.data, min=1e-8)
            # Fold old U into diagonal before zeroing
            if layer.eta1_U.abs().max() > 1e-10:
                old_U_diag = (layer.eta1_U ** 2).sum(dim=1) * (1 - kappa_t)
                layer.eta1_d.data += old_U_diag
            layer.eta1_U.data.zero_()

    def _update_eta1_lowrank(self, layer, f_pre, kappa_t):
        """Update η1 = diag(d) + U·U^T with new batch data.

        Target: η1_target = (1-κ)·η1_old + κ·(η1_prior + ss1)
        where ss1 = F^T @ F [in, in].

        Decomposed as: d_new + U_new·U_new^T where:
          A·A^T = (1-κ)·U_old·U_old^T + κ·F^T·F  (all off-diagonal + some diagonal)
          d_new = (1-κ)·d_old + κ·d_prior + residual_diag  (pure diagonal part)
        """
        in_features = layer.in_features
        rank_k = layer.rank_k

        sqrt_1mk = torch.sqrt(torch.clamp(torch.tensor(1.0 - kappa_t, device=self.device), min=0.0))
        sqrt_k = torch.sqrt(torch.tensor(kappa_t, device=self.device))

        A_old = sqrt_1mk * layer.eta1_U  # [in, k]
        A_new = sqrt_k * f_pre.T  # [in, N]

        A = torch.cat([A_old, A_new], dim=1)  # [in, k + N]

        # Truncated SVD using torch.svd_lowrank (randomized, CUDA-stable)
        # This directly computes rank-k approximation without full SVD
        P = None
        S = None
        try:
            P, S, Qt = torch.svd_lowrank(A, q=rank_k)
            # P: [in, k], S: [k], Qt: [k, k+N]
        except Exception:
            pass

        # Check for NaN in SVD output
        if P is None or S is None or torch.isnan(P).any() or torch.isnan(S).any():
            # Fallback: full SVD on CPU
            try:
                A_cpu = A.float().cpu()
                P_full, S_full, Qt_full = torch.linalg.svd(A_cpu, full_matrices=False)
                P = P_full[:, :rank_k].to(self.device)
                S = S_full[:rank_k].to(self.device)
            except Exception:
                self._diagonal_only_update(layer, f_pre, kappa_t)
                return

        # Final NaN check
        if torch.isnan(P).any() or torch.isnan(S).any():
            self._diagonal_only_update(layer, f_pre, kappa_t)
            return

        # U_new = P · diag(S)  [in, k]
        actual_k = min(rank_k, len(S))
        U_new = P[:, :actual_k] * S[:actual_k].unsqueeze(0)

        # Residual: A·A^T - U_new·U_new^T contributes to diagonal
        # Exact: residual_diag = diag(A·A^T) - diag(U_new·U_new^T)
        # diag(A·A^T) = row-wise sum of A² = (1-κ)·(U_old²).sum(1) + κ·(f_pre²).sum(0)
        AA_diag = (A ** 2).sum(dim=1)  # [in]
        UU_diag = (U_new ** 2).sum(dim=1)  # [in]
        residual_diag = torch.clamp(AA_diag - UU_diag, min=0.0)  # clamp for numerical safety

        # d_new = (1-κ)·d_old + κ·d_prior + residual_diag
        d_final = (1 - kappa_t) * layer.eta1_d + kappa_t * layer.eta1_d_prior + residual_diag

        # Ensure d stays positive
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

        Order of operations (critical for consistency):
        1. Update η1 (low-rank SVD)
        2. Update η2 (moving average)
        3. Recompute M from updated η1, η2
        4. Compute residuals r = z - M·f using the NEW M
        5. Update Ψ^{-1} from residuals (always non-negative!)
        6. Update η4 (moving average)
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

            # 1. Update η1 with low-rank structure
            self._update_eta1_lowrank(layer, f_pre, kappa_t)

            # 2. Update η2 (standard moving average)
            ss2 = z_post.T @ f_pre  # [out, in]
            eta2_new = layer.eta2_prior + ss2
            with torch.no_grad():
                layer.eta2.data = (1 - kappa_t) * layer.eta2.data + kappa_t * eta2_new

            # 3. Recompute M from updated η1, η2
            layer._cache_valid = False
            layer._update_M_cache()
            M_new = layer._M_cache  # [out, in]

            # 4. Compute residuals using NEW M
            # r = z_post - f_pre @ M^T  [batch, out]
            with torch.no_grad():
                residuals = z_post - f_pre @ M_new.T  # [batch, out]
                # Residual sum of squares (always ≥ 0)
                ss_psi = (residuals ** 2).sum(dim=0)  # [out]

            # 5. Update Ψ^{-1} from residuals (guaranteed positive!)
            psi_inv_new = layer.psi_inv_prior + ss_psi
            with torch.no_grad():
                layer.psi_inv_diag.data = (1 - kappa_t) * layer.psi_inv_diag.data + kappa_t * psi_inv_new

            # 6. Update η4 (moving average)
            ss4 = float(inputs.size(0))
            eta4_new = layer.eta4_prior + ss4
            with torch.no_grad():
                layer.eta4.data = (1 - kappa_t) * layer.eta4.data + kappa_t * eta4_new

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
