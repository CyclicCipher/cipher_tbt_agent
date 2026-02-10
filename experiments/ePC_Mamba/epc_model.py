"""
Error-based Predictive Coding wrapper for sequence models (ePC-Mamba).

Adapts the PCE framework (from ePC_ResNet) for a stack of Mamba2 blocks.
Errors are fp32 tensors of shape [batch, seqlen, d_model] placed between
blocks. The ePC algorithm runs T Newton iterations to optimize errors,
then computes E_local for local weight updates.

Architecture:
  Embedding → RMSNorm → Mamba Block 0 → + e_0 (fp32)
            → RMSNorm → Mamba Block 1 → + e_1 (fp32)
            → ...
            → RMSNorm → Output projection

Errors are placed AFTER each block's output, BEFORE the next block's
RMSNorm. This way RMSNorm operates on "clean" signal + error, and
the error is not unpredictably rescaled.

Key differences from ePC_ResNet:
  - Errors are [batch, seqlen, d_model] (sequence dimension, no spatial shrinkage)
  - Each "layer" is RMSNorm → Mamba2Block (pre-norm pattern)
  - Output loss is cross-entropy over tokens (summed over sequence)
  - seqlen must be a multiple of chunk_size

Reference: Goemaere et al. 2025, arXiv:2505.20137
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mamba_block import Mamba2Block, Mamba2Config, RMSNorm


def _sync_time():
    """Synchronize CUDA and return wall-clock time for profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class MambaLayer(nn.Module):
    """Pre-norm wrapper: RMSNorm → Mamba2Block.

    This is one "layer" from ePC's perspective. Errors are added
    between layers (after this module's output).
    """

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba = Mamba2Block(config)

    def forward(self, x: Tensor) -> Tensor:
        return self.mamba(self.norm(x))


class PCESequence(nn.Module):
    """Error-based Predictive Coding for sequence models.

    Works with a stack of MambaLayers. Errors are added between layers
    during inference; weight updates use E_local for local learning.

    Args:
        config: Mamba2Config defining the block architecture.
        iters: Number of error optimization steps per batch (T).
        damping: Damping factor for Newton mode (higher = more conservative).
        early_stop_threshold: Stop when relative energy improvement < this.
        output_loss: 'ce' for cross-entropy, 'mse' for mean squared error.
    """

    def __init__(self, config: Mamba2Config, iters: int = 2,
                 damping: float = 1.0, early_stop_threshold: float = 0.0,
                 output_loss: str = 'ce'):
        super().__init__()
        self.config = config
        self.iters = iters
        self.damping = damping
        self.early_stop_threshold = early_stop_threshold
        self._iters_used = 0
        self._weight_phase_prediction = None
        self.profiling = False
        self._profile = {}

        # Layers: list of MambaLayers
        self.layers = nn.ModuleList([
            MambaLayer(config) for _ in range(config.n_layer)
        ])

        # Output head: norm + linear
        self.out_norm = RMSNorm(config.d_model)

        # Errors (set during forward)
        self.errors = None

        # Loss function
        if output_loss == 'mse':
            def _mse_loss(y_pred, y):
                return 0.5 * F.mse_loss(y_pred, y, reduction='sum')
            self._output_loss = _mse_loss
        elif output_loss == 'ce':
            # For token-level CE: y_pred is (batch, seqlen, vocab),
            # y is (batch, seqlen) of token indices.
            # F.cross_entropy expects (N, C) so we reshape.
            def _ce_loss(y_pred, y):
                b, l, v = y_pred.shape
                return F.cross_entropy(
                    y_pred.reshape(b * l, v), y.reshape(b * l),
                    reduction='sum'
                )
            self._output_loss = _ce_loss
        else:
            raise ValueError(f"Unknown output_loss: {output_loss}")

    def y_pred(self, x: Tensor) -> Tensor:
        """Forward pass with current errors.

        Args:
            x: (batch, seqlen, d_model) input embeddings.

        Returns:
            (batch, seqlen, d_model) pre-projection output.
        """
        s_i = x
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)
        return self.out_norm(s_i)

    def E(self, x: Tensor, y: Tensor, output_proj: nn.Module) -> Tensor:
        """Energy using errors (global graph — for error optimization).

        DO NOT use for weight optimization (would be standard backprop).

        Args:
            x: (batch, seqlen, d_model) input embeddings.
            y: target tensor (shape depends on task).
            output_proj: nn.Linear mapping d_model → output_dim.
        """
        E_errors = 0.5 * sum(
            torch.linalg.vector_norm(e, ord=2, dim=None) ** 2
            for e in self.errors
        )
        logits = output_proj(self.y_pred(x))
        return E_errors + self._output_loss(logits, y)

    def E_local(self, x: Tensor, y: Tensor, output_proj: nn.Module) -> Tensor:
        """Energy using local interactions (detached — for weight optimization).

        Same value as E, but the computational graph enforces local weight
        updates by detaching states between layers.

        Args:
            x: (batch, seqlen, d_model) input embeddings.
            y: target tensor.
            output_proj: nn.Linear mapping d_model → output_dim.
        """
        E = 0.0
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            E += 0.5 * F.mse_loss(s_i_pred, s_i, reduction='sum')
        # Last layer + output
        s_last_pred = self.layers[-1](s_i)
        s_last = self.out_norm(s_last_pred)
        logits = output_proj(s_last)
        self._weight_phase_prediction = logits.detach()
        return E + self._output_loss(logits, y)

    @torch.no_grad()
    def init_zero_errors(self, x: Tensor):
        """Initialize zero errors with shape caching.

        On first call (or shape change), runs a forward pass to discover
        error shapes. Subsequent calls create tensors from cached shapes.

        Errors are ALWAYS fp32. fp16 rounds Newton corrections to zero,
        defeating early stopping (see MISTAKES.md / MEMORY.md).
        """
        input_shape = x.shape
        if (hasattr(self, '_cached_error_shapes')
                and self._cached_input_shape == input_shape):
            device = x.device
            self.errors = [
                torch.zeros(shape, dtype=torch.float32,
                            device=device, requires_grad=True)
                for shape in self._cached_error_shapes
            ]
            return

        # Forward pass to discover shapes (N-1 errors for N layers)
        self.errors = []
        s_i = x
        for layer_i in self.layers[:-1]:
            s_i = layer_i(s_i)
            self.errors.append(
                torch.zeros(s_i.shape, dtype=torch.float32,
                            device=s_i.device, requires_grad=True)
            )

        self._cached_input_shape = input_shape
        self._cached_error_shapes = [e.shape for e in self.errors]

    def _newton_step(self):
        """Rank-1 LRPD Newton step for error optimization.

        H = I + J^T H_L J, g = e + J^T(dL/dy), u = g - e.
        Rank-1: H ≈ (1+damping)I + u*u^T.
        Woodbury: H^{-1}g = g/d - (u^T g)/(d^2 + d*||u||^2) * u

        Uses streaming per-layer dot products (no concatenation).
        """
        with torch.no_grad():
            d = 1.0 + self.damping

            # Accumulate decomposed dot products per-layer
            gTg = 0.0
            gTe = 0.0
            eTe = 0.0
            for e in self.errors:
                g_flat = e.grad.reshape(-1)
                e_flat = e.data.reshape(-1)
                gTg += torch.dot(g_flat, g_flat).item()
                gTe += torch.dot(g_flat, e_flat).item()
                eTe += torch.dot(e_flat, e_flat).item()

            uTg = gTg - gTe
            uTu = gTg - 2.0 * gTe + eTe

            # Woodbury coefficient
            coeff = uTg / (d * d + d * uTu)

            # In-place update: e_new = e*(1-coeff) + grad*(coeff - 1/d)
            c1 = 1.0 - coeff
            c2 = coeff - 1.0 / d
            for e in self.errors:
                e.data.mul_(c1).add_(e.grad, alpha=c2)

    def minimize_error_energy(self, x: Tensor, y: Tensor,
                              output_proj: nn.Module) -> float:
        """Inference phase: optimize errors to minimize energy.

        Args:
            x: (batch, seqlen, d_model) input embeddings.
            y: target tensor.
            output_proj: nn.Linear for output.

        Returns:
            Final energy value.
        """
        # Detach input from upstream graph (e.g. embedding). ePC only
        # optimizes errors during inference. Without this, the second
        # E.backward() fails because the embedding's saved tensors were
        # freed by the first backward. The weight phase (E_local) re-embeds
        # from scratch, so detaching here is safe.
        x = x.detach()

        prof = self.profiling

        if prof:
            _t = _sync_time()

        # Freeze weights during inference
        for p in self.layers.parameters():
            p.requires_grad_(False)
        for p in self.out_norm.parameters():
            p.requires_grad_(False)
        output_proj.requires_grad_(False)

        self.init_zero_errors(x)

        if prof:
            _t2 = _sync_time()
            prof_init = (_t2 - _t) * 1000
            prof_fwd = 0.0
            prof_bwd = 0.0
            prof_step = 0.0

        E_prev = None
        E_val = 0.0
        for t in range(self.iters):
            # Zero error gradients
            for e in self.errors:
                if e.grad is not None:
                    e.grad.zero_()

            if prof:
                _t = _sync_time()

            E = self.E(x, y, output_proj)
            E_val = E.item()

            if prof:
                _t2 = _sync_time()
                prof_fwd += (_t2 - _t) * 1000

            if t == 0:
                self._E_initial = E_val

            # Early stopping
            if t > 0 and self.early_stop_threshold > 0 and E_prev is not None:
                rel_improvement = (E_prev - E_val) / max(abs(E_prev), 1e-8)
                if rel_improvement < self.early_stop_threshold:
                    self._E_final = E_val
                    self._iters_used = t
                    break

            E_prev = E_val

            if prof:
                _t = _sync_time()

            E.backward()

            if prof:
                _t2 = _sync_time()
                prof_bwd += (_t2 - _t) * 1000
                _t = _t2

            self._newton_step()

            if prof:
                prof_step += (_sync_time() - _t) * 1000
        else:
            self._E_final = E_val
            self._iters_used = self.iters

        # Unfreeze weights
        for p in self.layers.parameters():
            p.requires_grad_(True)
        for p in self.out_norm.parameters():
            p.requires_grad_(True)
        output_proj.requires_grad_(True)

        if prof:
            self._profile = {
                'init_ms': prof_init,
                'forward_ms': prof_fwd,
                'backward_ms': prof_bwd,
                'step_ms': prof_step,
            }

        return self._E_final

    def get_diagnostics(self) -> dict:
        """Collect per-layer diagnostics after inference."""
        diag = {
            'E_initial': getattr(self, '_E_initial', 0.0),
            'E_final': getattr(self, '_E_final', 0.0),
            'convergence': getattr(self, '_E_initial', 0.0) - getattr(self, '_E_final', 0.0),
            'iters_used': getattr(self, '_iters_used', self.iters),
            'error_norms': [],
            'layer_energies': [],
        }
        if self.errors is not None:
            for e in self.errors:
                if isinstance(e, Tensor):
                    norm = torch.linalg.vector_norm(e, ord=2, dim=None).item()
                    diag['error_norms'].append(norm)
                    diag['layer_energies'].append(0.5 * norm ** 2)
        return diag


class ePCMambaLM(nn.Module):
    """Complete ePC-Mamba language model.

    Wraps PCESequence with token embedding and output projection.
    This is the top-level module for training.

    Args:
        config: Mamba2Config.
        vocab_size: Number of tokens.
        iters: Newton iterations for error optimization.
        damping: Newton damping factor.
        early_stop_threshold: Early stopping for inference.
    """

    def __init__(self, config: Mamba2Config, vocab_size: int,
                 iters: int = 2, damping: float = 1.0,
                 early_stop_threshold: float = 0.0):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pce = PCESequence(
            config, iters=iters, damping=damping,
            early_stop_threshold=early_stop_threshold,
            output_loss='ce',
        )
        self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying (standard for small LMs)
        self.out_proj.weight = self.embedding.weight

    def forward(self, input_ids: Tensor, targets: Tensor | None = None):
        """Forward pass.

        Args:
            input_ids: (batch, seqlen) token indices.
            targets: (batch, seqlen) target token indices. If None,
                returns logits without ePC inference.

        Returns:
            If targets: final energy (float).
            If no targets: logits (batch, seqlen, vocab_size).
        """
        x = self.embedding(input_ids)

        if targets is not None:
            return self.pce.minimize_error_energy(x, targets, self.out_proj)
        else:
            self.pce.errors = [0.0] * (len(self.pce.layers) - 1)
            hidden = self.pce.y_pred(x)
            return self.out_proj(hidden)

    def compute_weight_loss(self, input_ids: Tensor, targets: Tensor,
                            batch_size: int) -> Tensor:
        """Compute E_local for weight optimizer (call after forward with targets)."""
        x = self.embedding(input_ids)
        return self.pce.E_local(x, targets, self.out_proj) / batch_size

    def get_diagnostics(self) -> dict:
        return self.pce.get_diagnostics()
