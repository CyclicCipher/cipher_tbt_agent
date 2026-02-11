"""
Error-based Predictive Coding wrapper for Mamba-3 (ePC-Mamba3).

Adapts the ePC framework for a stack of Mamba3 layers. Each layer
is a pre-norm Mixer + pre-norm MLP WITHOUT block-level residual
connections. Errors are fp32 tensors placed between layers.

CRITICAL: ePC layers must NOT have residual connections. In ePC, the
errors serve the role of skip connections — they carry information
forward in the network. If blocks already have residuals, the errors
become redundant perturbations on a dominant residual stream, and
E_local gradients vanish. This is why Mamba3Block (with residuals)
fails at 7% while MNIST MLP (without residuals) reaches 95.74%.

Architecture:
  Embedding → Mamba3Layer 0 → + e_0 (fp32)
            → Mamba3Layer 1 → + e_1 (fp32)
            → ...
            → Mamba3Layer N-1 → (no error)
            → RMSNorm → Output projection

Key lessons applied (from MISTAKES.md):
  - Errors ALWAYS fp32 (fp16 rounds Newton corrections to zero)
  - Rank-1 Newton for errors (not Adam — #25, not diagonal — #24)
  - 4+ layers required (2-layer pathological — #26)
  - Weight gradient clipping prevents catastrophic forgetting
  - CE > MSE for sequence tasks
  - Collect diagnostics BEFORE accuracy eval (#23)
  - No block-level residuals in ePC layers (errors ARE the residuals)

Reference: Goemaere et al. 2025, arXiv:2505.20137
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mamba3_block import Mamba3Config, Mamba3Mixer, SwiGLUMLP, RMSNorm


class Mamba3Layer(nn.Module):
    """Mamba3 layer WITHOUT block-level residual connections (for ePC).

    Pre-norm Mixer → Pre-norm MLP. No residual skip around the block.
    In ePC, errors between layers serve the role that residual connections
    normally serve — they carry corrective information forward.

    The Mixer's internal D skip and SSD structure are preserved. Only the
    block-level x = x + Mixer(norm(x)) residual is removed.
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = Mamba3Mixer(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

    def forward(self, x: Tensor) -> Tensor:
        # No block-level residual: the mixer transforms, then MLP transforms
        # with an internal residual (MLP adds to mixer output, not to original x)
        h = self.mixer(self.mixer_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


def _sync_time():
    """Synchronize CUDA and return wall-clock time for profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class PCEMamba3(nn.Module):
    """Error-based Predictive Coding for Mamba-3 layers.

    Each Mamba3Layer (Mixer + MLP, NO residual) is one ePC "layer".
    Errors are added between layers during inference; weight updates
    use E_local for local learning.

    Args:
        config: Mamba3Config defining the block architecture.
        iters: Number of error optimization steps per batch (T).
        damping: Damping factor for Newton mode.
    """

    def __init__(self, config: Mamba3Config, iters: int = 2,
                 e_lr: float = 0.02, error_optim: str = 'newton',
                 damping: float = 1.0):
        super().__init__()
        self.config = config
        self.iters = iters
        self.e_lr = e_lr
        self.error_optim_mode = error_optim
        self.damping = damping
        self._iters_used = 0
        self._weight_phase_prediction = None
        self.profiling = False
        self._profile = {}

        # Scale factor to compensate for small errors from limited SGD/Adam
        # iterations. Newton converges better so errors are larger.
        if error_optim == 'newton':
            self.energy_scale = 1.0
        else:
            self.energy_scale = min(1.0, e_lr * iters)

        # Layers: Mamba3Layers WITHOUT block-level residuals
        # (ePC errors serve the residual role)
        self.layers = nn.ModuleList([
            Mamba3Layer(config) for _ in range(config.n_layer)
        ])

        # Output head: final norm
        self.out_norm = RMSNorm(config.d_model)

        # Errors (set during forward)
        self.errors = None

        # CE loss for sequences: y_pred (batch, seqlen, vocab), y (batch, seqlen)
        def _ce_loss(y_pred, y):
            b, l, v = y_pred.shape
            return F.cross_entropy(
                y_pred.reshape(b * l, v), y.reshape(b * l),
                reduction='sum'
            )
        self._output_loss = _ce_loss

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

        Errors are ALWAYS fp32. fp16 rounds Newton corrections to zero,
        defeating early stopping (see MISTAKES.md).
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

        The error Hessian is H = I + J^T H_L J where J = dy/de.
        Rank-1 approximation: H ≈ (1+damping)I + u*u^T
        where u = g - e = J^T(dL/dy).

        Woodbury inverse: H^{-1}g = g/d - (u^T g)/(d^2 + d*||u||^2) * u
        Cost: zero extra backward passes (reuses the gradient).
        """
        with torch.no_grad():
            d = 1.0 + self.damping

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

            coeff = uTg / (d * d + d * uTu)

            c1 = 1.0 - coeff
            c2 = coeff - 1.0 / d
            for e in self.errors:
                e.data.mul_(c1).add_(e.grad, alpha=c2)

            self._newton_diag = {
                'gTg': gTg,
                'uTu': uTu,
                'eTe': eTe,
                'coeff': coeff,
                'rank1_ratio': uTu / max(gTg, 1e-10),
            }

    def minimize_error_energy(self, x: Tensor, y: Tensor,
                              output_proj: nn.Module) -> float:
        """Inference phase: optimize errors to minimize energy.

        Returns:
            Final energy value.
        """
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

        # Create first-order optimizer if needed
        if self.error_optim_mode == 'sgd':
            optim = torch.optim.SGD(self.errors, lr=self.e_lr)
        elif self.error_optim_mode == 'adam':
            optim = torch.optim.Adam(self.errors, lr=self.e_lr)
        else:
            optim = None  # Newton mode

        E_val = 0.0
        for t in range(self.iters):
            if optim is not None:
                optim.zero_grad()
            else:
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

            if prof:
                _t = _sync_time()

            E.backward()

            if prof:
                _t2 = _sync_time()
                prof_bwd += (_t2 - _t) * 1000
                _t = _t2

            if optim is not None:
                optim.step()
            else:
                self._newton_step()

            if prof:
                prof_step += (_sync_time() - _t) * 1000

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

        newton = getattr(self, '_newton_diag', {})
        diag['newton_rank1_ratio'] = newton.get('rank1_ratio', 0.0)
        diag['newton_coeff'] = newton.get('coeff', 0.0)

        return diag


class ePCMamba3LM(nn.Module):
    """Complete ePC-Mamba3 language model.

    Wraps PCEMamba3 with token embedding and output projection.

    Args:
        config: Mamba3Config.
        vocab_size: Number of tokens.
        iters: Error optimization steps per batch.
        damping: Newton damping factor.
    """

    def __init__(self, config: Mamba3Config, vocab_size: int,
                 iters: int = 2, e_lr: float = 0.02,
                 error_optim: str = 'newton', damping: float = 1.0):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pce = PCEMamba3(config, iters=iters, e_lr=e_lr,
                             error_optim=error_optim, damping=damping)
        self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying
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
        return self.pce.E_local(x, targets, self.out_proj) / (batch_size * self.pce.energy_scale)

    def get_diagnostics(self) -> dict:
        return self.pce.get_diagnostics()
