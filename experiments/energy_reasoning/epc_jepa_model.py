"""
ePC-JEPA: Predictive Coding JEPA with Mamba3.

The first combination of error-based predictive coding (ePC) with
Joint Embedding Predictive Architecture (JEPA).

Architecture:
    Context Encoder (ePC, 4-layer Mamba3):
        input -> embed -> [Block0 + e0] -> ... -> [Block3 + e3] -> norm -> s_context
        Error nodes optimized via SGD to minimize JEPA representation loss.
        Block weights updated via E_local (local learning).

    Target Encoder (EMA of context encoder):
        input -> embed -> Block0 -> ... -> Block3 -> norm -> s_target
        Parameters updated via exponential moving average of context encoder.

    Predictor (narrow 2-layer Mamba3, d=64):
        s_context -> input_proj -> Mamba3 blocks -> output_proj -> s_pred
        Trained via backprop from JEPA loss (through E_local).

    Decoder (Linear d_model -> vocab_size):
        s_pred -> token logits. Trained via CE loss.

Training (next-step prediction mode):
    Phase 1 (error optimization):
        E = sum(p_i * 0.5 * ||e_i||^2) + JEPA_loss(pred(enc(x,e)), tgt_enc(x))
        SGD on errors for T iterations.

    Phase 2 (weight optimization):
        E_local = sum(p_i * MSE_local_i) + JEPA_loss + decode_CE
        Adam on encoder blocks (local) + predictor (global) + decoder (global).

    Phase 3 (EMA update):
        target_encoder = tau * target + (1-tau) * context

Key insight: encoder blocks learn LOCALLY (each block's gradient comes
only from its own prediction error), while predictor and decoder learn
GLOBALLY (from JEPA representation loss and decode CE). This is a
principled hybrid of local and global learning.

Reference: Goemaere et al. 2025 (ePC), LeCun 2022 (JEPA),
           Salvatori et al. 2025 (precision weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.Mamba3.mamba3_block import Mamba3Config, Mamba3Block, RMSNorm
from experiments.energy_reasoning.jepa_model import (
    Mamba3Predictor, vicreg_loss,
)


class ePCJEPAModel(nn.Module):
    """ePC-JEPA model: predictive coding encoder + JEPA framework.

    The context encoder uses ePC (error nodes + local learning).
    The target encoder uses EMA (exponential moving average).
    The predictor and decoder use standard backprop through E_local.

    Args:
        enc_config: Mamba3Config for encoder blocks.
        pred_config: Mamba3Config for predictor (narrow bottleneck).
        vocab_size: Token vocabulary size.
        d_z: Latent variable dimension (for future Langevin, unused in Stage 1).
        iters: Number of error optimization steps (T).
        e_lr: Error learning rate for SGD/Adam.
        error_optim: 'sgd' or 'adam'.
        precision_mode: 'none', 'linear', or 'geometric'.
        precision_base: Base for geometric precision (e.g., 3.0).
        ema_tau_start: EMA tau at training start (lower = faster tracking).
        ema_tau_end: EMA tau at training end (1.0 = frozen target).
        jepa_loss_type: 'cosine' (scale-invariant) or 'l2'.
        lambda_decode: Weight for decode CE loss in E_local.
        lambda_var: Weight for VICReg variance loss (anti-collapse).
        lambda_cov: Weight for VICReg covariance loss (decorrelation).
    """

    def __init__(
        self,
        enc_config: Mamba3Config,
        pred_config: Mamba3Config,
        vocab_size: int,
        d_z: int = 64,
        iters: int = 5,
        e_lr: float = 0.1,
        error_optim: str = 'sgd',
        precision_mode: str = 'geometric',
        precision_base: float = 3.0,
        ema_tau_start: float = 0.996,
        ema_tau_end: float = 1.0,
        jepa_loss_type: str = 'cosine',
        lambda_decode: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
    ):
        super().__init__()
        self.enc_config = enc_config
        self.vocab_size = vocab_size
        self.d_z = d_z
        self.iters = iters
        self.e_lr = e_lr
        self.error_optim_mode = error_optim
        self.energy_scale = min(1.0, e_lr * iters)
        self.jepa_loss_type = jepa_loss_type
        self.lambda_decode = lambda_decode
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

        d = enc_config.d_model
        n = enc_config.n_layer

        # --- Context encoder (ePC-enabled) ---
        self.embedding = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList(
            [Mamba3Block(enc_config) for _ in range(n)]
        )
        self.out_norm = RMSNorm(d)

        # --- Target encoder (EMA, frozen) ---
        self.target_embedding = nn.Embedding(vocab_size, d)
        self.target_layers = nn.ModuleList(
            [Mamba3Block(enc_config) for _ in range(n)]
        )
        self.target_norm = RMSNorm(d)

        # --- Predictor (narrow Mamba3 with z conditioning) ---
        self.predictor = Mamba3Predictor(d, pred_config, d_z)

        # --- Decoder (linear probe) ---
        self.decoder = nn.Linear(d, vocab_size)

        # --- Precision weighting (Salvatori et al. 2025) ---
        if precision_mode == 'geometric':
            raw = [precision_base ** (n - 1 - i) for i in range(n)]
            mean_p = sum(raw) / len(raw)
            self.precisions = [p / mean_p for p in raw]
        elif precision_mode == 'linear':
            raw = [float(n - i) for i in range(n)]
            mean_p = sum(raw) / len(raw)
            self.precisions = [p / mean_p for p in raw]
        else:
            self.precisions = [1.0] * n

        # --- EMA config ---
        self.ema_tau_start = ema_tau_start
        self.ema_tau_end = ema_tau_end
        self.ema_tau = ema_tau_start

        # --- ePC state ---
        self.errors = None
        self._E_initial = 0.0
        self._E_final = 0.0
        self._actual_iters = iters

        # Initialize target from context
        self._init_target()

    # ------------------------------------------------------------------
    # Target encoder management
    # ------------------------------------------------------------------

    def _context_params(self):
        """Iterate context encoder parameters (for EMA sync)."""
        yield from self.embedding.parameters()
        yield from self.layers.parameters()
        yield from self.out_norm.parameters()

    def _target_params(self):
        """Iterate target encoder parameters (for EMA sync)."""
        yield from self.target_embedding.parameters()
        yield from self.target_layers.parameters()
        yield from self.target_norm.parameters()

    def _init_target(self):
        """Copy context encoder to target encoder and freeze."""
        for p_s, p_t in zip(self._context_params(), self._target_params()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """EMA update: target = tau * target + (1 - tau) * context."""
        for p_s, p_t in zip(self._context_params(), self._target_params()):
            p_t.data.mul_(self.ema_tau).add_(p_s.data, alpha=1 - self.ema_tau)

    def set_ema_progress(self, progress: float):
        """Update EMA tau (linear schedule, 0.0=start, 1.0=end)."""
        self.ema_tau = (self.ema_tau_start
                        + progress * (self.ema_tau_end - self.ema_tau_start))

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_target(self, input_ids: Tensor) -> Tensor:
        """Target encoder forward (no errors, no grad)."""
        x = self.target_embedding(input_ids)
        for layer in self.target_layers:
            x = layer(x)
        return self.target_norm(x)

    def encode_context(self, x_emb: Tensor) -> Tensor:
        """Context encoder forward WITH error nodes."""
        s = x_emb
        for e, layer in zip(self.errors, self.layers):
            s = layer(s) + e
        return self.out_norm(s)

    def _jepa_loss_nextstep(self, s_pred: Tensor,
                             s_target: Tensor) -> Tensor:
        """JEPA loss for next-step prediction.

        Args:
            s_pred: (batch, seq_len, d) predictor output.
            s_target: (batch, seq_len, d) target encoder output.

        Returns:
            Scalar loss: cosine or L2 between shifted sequences.
        """
        pred = s_pred[:, :-1]            # (batch, seq_len-1, d)
        target = s_target[:, 1:].detach()  # (batch, seq_len-1, d)

        if self.jepa_loss_type == 'cosine':
            pred_flat = pred.reshape(-1, pred.shape[-1])
            tgt_flat = target.reshape(-1, target.shape[-1])
            sim = F.cosine_similarity(pred_flat, tgt_flat, dim=-1)
            return (1.0 - sim).mean()
        else:
            return (pred - target).pow(2).mean()

    # ------------------------------------------------------------------
    # Error management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_zero_errors(self, x_emb: Tensor):
        """Initialize zero fp32 errors matching block output shapes."""
        input_shape = x_emb.shape
        if (hasattr(self, '_cached_input_shape')
                and self._cached_input_shape == input_shape):
            device = x_emb.device
            self.errors = [
                torch.zeros(shape, dtype=torch.float32,
                            device=device, requires_grad=True)
                for shape in self._cached_error_shapes
            ]
            return

        # Discover shapes via forward
        self.errors = []
        s = x_emb
        for layer in self.layers:
            s = layer(s)
            self.errors.append(
                torch.zeros(s.shape, dtype=torch.float32,
                            device=s.device, requires_grad=True)
            )
        self._cached_input_shape = input_shape
        self._cached_error_shapes = [e.shape for e in self.errors]

    # ------------------------------------------------------------------
    # Energy functions
    # ------------------------------------------------------------------

    def E(self, x_emb: Tensor, s_target: Tensor,
          z: Tensor = None) -> Tensor:
        """Energy for error optimization (global graph).

        E = sum(p_i * 0.5 * ||e_i||^2) + JEPA_loss

        The JEPA loss flows through: errors -> encoder -> predictor -> loss.
        This is the signal that tells errors WHERE to push the encoder output.
        """
        E_errors = 0.5 * sum(
            pi * torch.linalg.vector_norm(e, ord=2, dim=None) ** 2
            for pi, e in zip(self.precisions, self.errors)
        )
        s_context = self.encode_context(x_emb)
        s_pred = self.predictor(s_context, z)
        return E_errors + self._jepa_loss_nextstep(s_pred, s_target)

    def E_local(self, x_emb: Tensor, input_ids: Tensor,
                s_target: Tensor, z: Tensor = None) -> Tensor:
        """Energy for weight optimization (local learning).

        Block weights: gradient from local MSE (detached errors).
        Predictor: gradient from JEPA loss + decode CE.
        Decoder: gradient from decode CE.
        Embedding: gradient from block 0's local MSE.
        out_norm: gradient from JEPA loss + decode CE.
        """
        E = 0.0
        s = x_emb
        for pi, e, layer in zip(self.precisions, self.errors, self.layers):
            s_pred_local = layer(s)
            s = (s_pred_local + e).detach()
            E += pi * 0.5 * F.mse_loss(s_pred_local, s, reduction='sum')

        s_context = self.out_norm(s)
        s_pred = self.predictor(s_context, z)

        # JEPA loss (next-step shifted)
        E += self._jepa_loss_nextstep(s_pred, s_target)

        # Decode CE loss (next-step shifted)
        if self.lambda_decode > 0:
            logits = self.decoder(s_pred[:, :-1])
            decode_ce = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                input_ids[:, 1:].reshape(-1),
            )
            E += self.lambda_decode * decode_ce

        # VICReg: prevent representation collapse
        if self.lambda_var > 0 or self.lambda_cov > 0:
            L_var, L_cov = vicreg_loss(s_context)
            E += self.lambda_var * L_var + self.lambda_cov * L_cov

        return E

    # ------------------------------------------------------------------
    # ePC training phases
    # ------------------------------------------------------------------

    def _freeze_all_weights(self):
        """Freeze all trainable parameters for error optimization."""
        for p in self.parameters():
            p.requires_grad_(False)
        # Target encoder is always frozen (handled by _init_target)

    def _unfreeze_all_weights(self):
        """Unfreeze all trainable parameters."""
        for p in self.parameters():
            p.requires_grad_(True)
        # Re-freeze target encoder
        for p in self._target_params():
            p.requires_grad_(False)

    def minimize_error_energy(self, input_ids: Tensor,
                               s_target: Tensor,
                               z: Tensor = None,
                               early_stop_rtol: float = 1e-3,
                               min_iters: int = 2) -> float:
        """Phase 1: optimize errors to minimize energy.

        Freezes all weights. Runs T iterations of SGD/Adam on errors.
        Errors are optimized so that the predictor (applied to the
        error-corrected encoder output) matches the target encoder.

        Args:
            input_ids: (batch, seq_len) token indices.
            s_target: (batch, seq_len, d) target encoder representations.
            z: Optional latent variable.
            early_stop_rtol: Stop if relative energy reduction < this.
            min_iters: Minimum iterations before early stopping is checked.

        Returns:
            Final energy value.
        """
        x_emb = self.embedding(input_ids).detach()

        self._freeze_all_weights()
        self.init_zero_errors(x_emb)

        if self.error_optim_mode == 'adam':
            optim = torch.optim.Adam(self.errors, lr=self.e_lr)
        else:
            optim = torch.optim.SGD(self.errors, lr=self.e_lr)

        E_val = 0.0
        E_prev = float('inf')
        actual_iters = self.iters

        for t in range(self.iters):
            optim.zero_grad()
            E = self.E(x_emb, s_target, z)
            E_val = E.item()

            if t == 0:
                self._E_initial = E_val

            # Only check early stopping after min_iters full steps
            if early_stop_rtol > 0 and t >= min_iters:
                rel_reduction = (E_prev - E_val) / (abs(E_prev) + 1e-10)
                if rel_reduction < early_stop_rtol:
                    actual_iters = t + 1
                    break

            E_prev = E_val
            E.backward()
            optim.step()

        self._E_final = E_val
        self._actual_iters = actual_iters
        self._unfreeze_all_weights()

        return E_val

    def compute_weight_loss(self, input_ids: Tensor,
                            s_target: Tensor,
                            batch_size: int,
                            z: Tensor = None) -> Tensor:
        """Phase 2: compute E_local for weight optimizer.

        x_emb is NOT detached here, so the embedding gets gradient
        through block 0's local MSE.

        Returns:
            Normalized E_local loss for backward().
        """
        x_emb = self.embedding(input_ids)
        return (self.E_local(x_emb, input_ids, s_target, z)
                / (batch_size * self.energy_scale))

    def ipc_train_step(self, input_ids: Tensor, s_target: Tensor,
                       weight_optimizer, batch_size: int,
                       z: Tensor = None,
                       w_clip: float = 1.0) -> float:
        """Incremental PC: interleave error and weight steps.

        Each error SGD step is immediately followed by a weight update.
        This gives T weight updates per batch instead of 1.

        Returns:
            Final energy value.
        """
        x_emb_detached = self.embedding(input_ids).detach()

        self._freeze_all_weights()
        self.init_zero_errors(x_emb_detached)

        if self.error_optim_mode == 'adam':
            e_optim = torch.optim.Adam(self.errors, lr=self.e_lr)
        else:
            e_optim = torch.optim.SGD(self.errors, lr=self.e_lr)

        E_val = 0.0

        for t in range(self.iters):
            # --- Error step ---
            e_optim.zero_grad()
            E = self.E(x_emb_detached, s_target, z)
            E_val = E.item()
            if t == 0:
                self._E_initial = E_val
            E.backward()
            e_optim.step()

            # --- Weight step (iPC) ---
            self._unfreeze_all_weights()
            weight_optimizer.zero_grad()
            # Recompute embedding WITH gradients for weight phase
            x_emb = self.embedding(input_ids)
            w_loss = (self.E_local(x_emb, input_ids, s_target, z)
                      / (batch_size * self.energy_scale))
            w_loss.backward()
            if w_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.get_trainable_params(), max_norm=w_clip)
            weight_optimizer.step()

            # Re-freeze for next error step
            self._freeze_all_weights()

        self._E_final = E_val
        self._unfreeze_all_weights()

        return E_val

    # ------------------------------------------------------------------
    # Inference (no ePC, for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_eval(self, input_ids: Tensor,
                     z: Tensor = None) -> Tensor:
        """Evaluation: standard forward (no errors), returns shifted logits.

        Returns: (batch, seq_len-1, vocab_size) logits for next-token.
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        s_context = self.out_norm(x)
        s_pred = self.predictor(s_context, z)
        return self.decoder(s_pred[:, :-1])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_trainable_params(self):
        """Parameters updated by optimizer (excludes target encoder)."""
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.layers.parameters())
        params.extend(self.out_norm.parameters())
        params.extend(self.predictor.parameters())
        params.extend(self.decoder.parameters())
        return params

    def get_diagnostics(self) -> dict:
        """Collect diagnostics after inference phase."""
        diag = {
            'E_initial': self._E_initial,
            'E_final': self._E_final,
            'convergence': self._E_initial - self._E_final,
            'actual_iters': self._actual_iters,
            'precisions': self.precisions,
            'error_norms': [],
            'layer_energies': [],
        }
        if self.errors is not None:
            for e in self.errors:
                if isinstance(e, Tensor):
                    norm = torch.linalg.vector_norm(
                        e, ord=2, dim=None).item()
                    diag['error_norms'].append(norm)
                    diag['layer_energies'].append(0.5 * norm ** 2)
        return diag
