"""
Error-based Predictive Coding wrapper for Mamba-3 (ePC-Mamba3).

Adapts the ePC framework for a stack of Mamba3 blocks. Each block
is a pre-norm Mixer + pre-norm MLP with block-level residual
connections. Errors are fp32 tensors placed between blocks.

Architecture:
  Embedding → Mamba3Block 0 → + e_0 (fp32)
            → Mamba3Block 1 → + e_1 (fp32)
            → ...
            → Mamba3Block N-1 → (no error)
            → RMSNorm → Output projection

Empirical findings on residuals:
  - Mamba3Block (WITH residuals): 7% accuracy — errors too small
  - Mamba3Layer (WITHOUT residuals): 4% accuracy — even worse
  - SGD errors without residuals: loss explodes to 94M
  - Residuals are NOT the problem. Using Mamba3Block.

Key lessons applied (from MISTAKES.md):
  - Errors ALWAYS fp32 (fp16 rounds Newton corrections to zero)
  - 4+ layers required (2-layer pathological — #26)
  - Weight gradient clipping prevents catastrophic forgetting
  - CE > MSE for sequence tasks
  - Collect diagnostics BEFORE accuracy eval (#23)

Reference: Goemaere et al. 2025, arXiv:2505.20137
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mamba3_block import Mamba3Config, Mamba3Block, RMSNorm


def _sync_time():
    """Synchronize CUDA and return wall-clock time for profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class PCEMamba3(nn.Module):
    """Error-based Predictive Coding for Mamba-3 blocks.

    Each Mamba3Block (Mixer + MLP, WITH residual) is one ePC "layer".
    Errors are added between blocks during inference; weight updates
    use E_local for local learning.

    Args:
        config: Mamba3Config defining the block architecture.
        iters: Number of error optimization steps per batch (T).
        e_lr: Learning rate for error optimization (SGD/Adam).
        error_optim: 'sgd', 'adam', or 'newton'.
        damping: Damping factor for Newton mode.
    """

    def __init__(self, config: Mamba3Config, iters: int = 2,
                 e_lr: float = 0.02, error_optim: str = 'newton',
                 damping: float = 1.0, precision_mode: str = 'none',
                 precision_base: float = 3.0):
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

        # Per-layer precision weighting (Salvatori et al. 2025).
        # Amplifies early-layer error contributions to counteract the
        # natural error decay (Layer 1 >> Layer N in error magnitude,
        # but E_local terms are ||e||^2 so early layers are starved).
        # Precisions are normalized to mean=1 to preserve the error-vs-output
        # loss balance.
        n_errors = config.n_layer - 1
        if precision_mode == 'none':
            self.precisions = [1.0] * n_errors
        elif precision_mode == 'linear':
            # Layer 0 (earliest) gets highest precision
            raw = [float(n_errors - i) for i in range(n_errors)]
            mean_p = sum(raw) / len(raw)
            self.precisions = [p / mean_p for p in raw]
        elif precision_mode == 'geometric':
            # Exponential scaling: base^(N-1-i) for layer i
            raw = [precision_base ** (n_errors - 1 - i) for i in range(n_errors)]
            mean_p = sum(raw) / len(raw)
            self.precisions = [p / mean_p for p in raw]
        else:
            raise ValueError(f"Unknown precision_mode: {precision_mode}")

        # Layers: Mamba3Blocks WITH block-level residuals
        self.layers = nn.ModuleList([
            Mamba3Block(config) for _ in range(config.n_layer)
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

        Precision-weighted: E = sum(pi_l * 0.5 * ||e_l||^2) + output_loss.
        Higher precision on early layers amplifies their contribution,
        encouraging Newton/SGD to find larger errors where they matter most.

        DO NOT use for weight optimization (would be standard backprop).
        """
        E_errors = 0.5 * sum(
            pi * torch.linalg.vector_norm(e, ord=2, dim=None) ** 2
            for pi, e in zip(self.precisions, self.errors)
        )
        logits = output_proj(self.y_pred(x))
        return E_errors + self._output_loss(logits, y)

    def E_local(self, x: Tensor, y: Tensor, output_proj: nn.Module) -> Tensor:
        """Energy using local interactions (detached — for weight optimization).

        Precision-weighted: early layers get amplified E_local gradients.
        Same value as E, but the computational graph enforces local weight
        updates by detaching states between layers.
        """
        E = 0.0
        s_i = x
        for pi, e_i, layer_i in zip(self.precisions, self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            E += pi * 0.5 * F.mse_loss(s_i_pred, s_i, reduction='sum')
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
        """Precision-aware rank-1 LRPD Newton step for error optimization.

        With per-layer precisions pi_l, the energy is:
          E = sum(pi_l * 0.5 * ||e_l||^2) + L(y_pred, y)

        Gradient: g_l = pi_l * e_l + J_l^T(dL/dy)
        Output component: u_l = g_l - pi_l * e_l = J_l^T(dL/dy)
        Hessian: H = D + u*u^T where D = diag(pi_l * (1+damping) * I)

        Woodbury: (D + u*u^T)^{-1} g = D^{-1}g - D^{-1}u (u^T D^{-1} g)/(1 + u^T D^{-1} u)

        Per-layer update:
          e_l_new = e_l * c1 + g_l * c2_l
        where c1 = 1 - c/(1+damp) is global,
              c2_l = -(1-c)/(pi_l*(1+damp)) varies by layer.

        When all precisions = 1, reduces to the original Newton step.
        """
        with torch.no_grad():
            damp = self.damping

            # Per-layer dot products
            uTDinv_g = 0.0  # u^T D^{-1} g
            uTDinv_u = 0.0  # u^T D^{-1} u
            gTg_total = 0.0

            for pi, e in zip(self.precisions, self.errors):
                g_flat = e.grad.reshape(-1)
                e_flat = e.data.reshape(-1)
                d_l = pi * (1.0 + damp)

                gTg_l = torch.dot(g_flat, g_flat).item()
                gTe_l = torch.dot(g_flat, e_flat).item()
                eTe_l = torch.dot(e_flat, e_flat).item()

                # u_l = g_l - pi*e_l
                # u_l^T g_l = gTg_l - pi*gTe_l
                # ||u_l||^2 = gTg_l - 2*pi*gTe_l + pi^2*eTe_l
                uTDinv_g += (gTg_l - pi * gTe_l) / d_l
                uTDinv_u += (gTg_l - 2.0 * pi * gTe_l + pi * pi * eTe_l) / d_l
                gTg_total += gTg_l

            # Woodbury coefficient
            c = uTDinv_g / (1.0 + uTDinv_u) if (1.0 + uTDinv_u) != 0 else 0.0

            # Global and per-layer update coefficients
            c1 = 1.0 - c / (1.0 + damp)

            for pi, e in zip(self.precisions, self.errors):
                c2_l = -(1.0 - c) / (pi * (1.0 + damp))
                e.data.mul_(c1).add_(e.grad, alpha=c2_l)

            self._newton_diag = {
                'gTg': gTg_total,
                'uTDinv_u': uTDinv_u,
                'uTDinv_g': uTDinv_g,
                'coeff': c,
                'rank1_ratio': uTDinv_u / max(uTDinv_g, 1e-10),
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
        diag['precisions'] = self.precisions

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
                 error_optim: str = 'newton', damping: float = 1.0,
                 precision_mode: str = 'none', precision_base: float = 3.0):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.pce = PCEMamba3(config, iters=iters, e_lr=e_lr,
                             error_optim=error_optim, damping=damping,
                             precision_mode=precision_mode,
                             precision_base=precision_base)
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
