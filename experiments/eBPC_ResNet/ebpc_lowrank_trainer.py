"""
Low-Rank eBPC Trainer — Error-based Bayesian Predictive Coding with Low-Rank V

E-step: ePC error optimization (Adam on errors, precision-weighted energy)
M-step: Hebbian update with low-rank η1 structure

Key difference from full eBPC trainer:
  η1 = diag(d) + U·U^T requires eigendecomposition-based update.

  Update strategy:
    1. Form combined low-rank factor: A = [sqrt(1-κ) · U_old | sqrt(κ) · F^T]
    2. Eigendecomposition of Gram matrix A^T·A → keep top-k eigenvectors → new U
    3. Fold residual into the diagonal d
    4. η2 updated with standard moving average
    5. diag(η3) updated with standard moving average from ss3_diag = (z²).sum(0)
    6. η4 updated with standard moving average

  All four natural parameters (η1, η2, η3, η4) updated jointly from conjugate
  sufficient statistics, maintaining MNW consistency and self-correcting feedback.
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

        Target: η1_target = (1-κ)·η1_old + κ·(η1_prior + ss1)
        where ss1 = F^T @ F [in, in].

        Uses eigendecomposition of the Gram matrix B = A^T·A.
        torch.linalg.eigh is much more numerically stable than SVD on CUDA.
        """
        in_features = layer.in_features
        rank_k = layer.rank_k

        sqrt_1mk = torch.sqrt(torch.clamp(torch.tensor(1.0 - kappa_t, device=self.device), min=0.0))
        sqrt_k = torch.sqrt(torch.tensor(kappa_t, device=self.device))

        A_old = sqrt_1mk * layer.eta1_U  # [in, k]
        A_new = sqrt_k * f_pre.T  # [in, N]

        A = torch.cat([A_old, A_new], dim=1)  # [in, k + N]

        # Eigendecomposition of Gram matrix B = A^T·A  [k+N, k+N]
        B = A.T @ A  # [k+N, k+N]

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(B)  # sorted ascending
        except torch.linalg.LinAlgError:
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(B.float().cpu())
                eigenvalues = eigenvalues.to(self.device)
                eigenvectors = eigenvectors.to(self.device)
            except Exception:
                ss1_diag = (f_pre ** 2).sum(dim=0)
                with torch.no_grad():
                    layer.eta1_d.data = (1 - kappa_t) * layer.eta1_d + kappa_t * (layer.eta1_d_prior + ss1_diag)
                    layer.eta1_d.data = torch.clamp(layer.eta1_d.data, min=1e-8)
                return

        # Take top-k eigenvalues (last k, since eigh sorts ascending)
        actual_k = min(rank_k, len(eigenvalues))
        top_eigenvalues = eigenvalues[-actual_k:]  # [k]
        top_eigenvectors = eigenvectors[:, -actual_k:]  # [k+N, k]

        # Clamp negative eigenvalues (numerical noise in PSD matrix)
        top_eigenvalues = torch.clamp(top_eigenvalues, min=0.0)

        # U_new = A @ Q_topk  [in, k]
        U_new = A @ top_eigenvectors  # [in, k]

        # NaN check
        if torch.isnan(U_new).any():
            ss1_diag = (f_pre ** 2).sum(dim=0)
            with torch.no_grad():
                layer.eta1_d.data = (1 - kappa_t) * layer.eta1_d + kappa_t * (layer.eta1_d_prior + ss1_diag)
                layer.eta1_d.data = torch.clamp(layer.eta1_d.data, min=1e-8)
            return

        # Quadratic constraint preservation:
        # The MNW block matrix [[η1, η2^T], [η2, η3]] must be PSD.
        # When we truncate η1 to rank-k, the residual R = AA^T - U_new·U_new^T
        # has spectral norm = largest dropped eigenvalue.
        # Using only diag(R) gives η1_approx < η1_true (violates constraint).
        # Instead, inflate diagonal by spectral norm of R so η1_approx ≥ η1_true:
        #   λ_max(R)·I - R ≥ 0  (all eigenvalues of R ≤ λ_max(R))
        if actual_k < len(eigenvalues):
            max_dropped_eigenvalue = torch.clamp(eigenvalues[-(actual_k + 1)], min=0.0)
        else:
            max_dropped_eigenvalue = torch.tensor(0.0, device=self.device)

        # d_new = (1-κ)·d_old + κ·d_prior + λ_max(R) (spectral norm inflation)
        d_final = (1 - kappa_t) * layer.eta1_d + kappa_t * layer.eta1_d_prior + max_dropped_eigenvalue

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
        """Low-rank Hebbian update with conjugate MNW sufficient statistics.

        All four natural parameters updated jointly:
        1. η1 (low-rank eigendecomposition)
        2. η2 (moving average from ss2 = z^T·f)
        3. diag(η3) (moving average from ss3_diag = (z²).sum(0))
        4. η4 (moving average from ss4 = N)

        Then recompute M = η2 @ V for the cache.
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

            # 2. Update η2 (standard moving average, conjugate ss2 = z^T·f)
            ss2 = z_post.T @ f_pre  # [out, in]
            eta2_new = layer.eta2_prior + ss2
            with torch.no_grad():
                layer.eta2.data = (1 - kappa_t) * layer.eta2.data + kappa_t * eta2_new

            # 3. Update diag(η3) (standard moving average, conjugate ss3_diag = (z²).sum(0))
            ss3_diag = (z_post ** 2).sum(dim=0)  # [out]
            eta3_diag_new = layer.eta3_diag_prior + ss3_diag
            with torch.no_grad():
                layer.eta3_diag.data = (1 - kappa_t) * layer.eta3_diag.data + kappa_t * eta3_diag_new

            # 4. Update η4 (moving average)
            ss4 = float(inputs.size(0))
            eta4_new = layer.eta4_prior + ss4
            with torch.no_grad():
                layer.eta4.data = (1 - kappa_t) * layer.eta4.data + kappa_t * eta4_new

            # Recompute M cache from updated η1, η2
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
