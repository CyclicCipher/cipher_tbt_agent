"""
Error-based Predictive Coding (ePC) model.

Pure ePC with gradient descent weight updates (no BPC/Hebbian).

Algorithm (per batch):
  1. Inference: Initialize errors=0, then optimize errors to minimize
     E = 0.5 * sum(||e_i||^2) + output_loss(y_pred, y)
  2. Weight update: Compute E_local (detached states for local learning),
     backprop to weights via standard optimizer.

The key insight of ePC is error reparameterization:
  s_i = layer_i(s_{i-1}) + e_i
This creates a global computational graph where all errors are optimized
simultaneously via single backprop (no signal decay across layers).

Error optimizer modes:
  - 'sgd': Standard SGD (reference ePC paper, T=5 iterations)
  - 'adam': Adam optimizer for errors (adaptive learning rates)
  - 'newton': LRPD rank-1 Newton step (second-order, T=1-2 iterations)

The Newton optimizer exploits the fact that the error Hessian is
  H = I + J^T H_L J
where J = dy_pred/de has rank <= n_output (e.g., 10 for CIFAR-10).
This is an LRPD matrix, and the Woodbury identity gives exact inversion.
The rank-1 approximation uses the gradient decomposition g = e + J^T(dL/dy)
to extract the dominant curvature direction at zero extra backward-pass cost.

Reference: Goemaere et al. 2025, arXiv:2505.20137
Code reference: https://github.com/cgoemaere/error_based_PC (Apache 2.0)
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sync_time():
    """Synchronize CUDA and return wall-clock time for profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# --- Quantization-Aware Training (QAT) utilities ---

class FakeQuantize(torch.autograd.Function):
    """Symmetric per-tensor fake quantization with Straight-Through Estimator.

    Simulates INT8 (or any bit-width) quantization during forward pass.
    Backward pass uses STE: gradients pass through unchanged to FP32 weights.
    """

    @staticmethod
    def forward(ctx, x, num_bits):
        qmax = 2 ** (num_bits - 1) - 1
        scale = x.abs().max().clamp(min=1e-8) / qmax
        return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _FakeQuantParametrization(nn.Module):
    """Weight parametrization that applies fake quantization during forward."""

    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, weight):
        return FakeQuantize.apply(weight, self.num_bits)


def quantize_model_weights(model, num_bits=8):
    """Apply fake quantization to Conv2d/Linear weights for QAT.

    Uses torch.nn.utils.parametrize to intercept weight access. The original
    FP32 weights are preserved for optimizer updates; only the forward pass
    sees quantized weights. STE passes gradients through unchanged.

    Does NOT quantize biases or BatchNorm parameters.

    Returns the number of layers quantized.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.utils.parametrize.register_parametrization(
                module, 'weight', _FakeQuantParametrization(num_bits)
            )
            count += 1
    return count


class PCE(nn.Module):
    """Error-based Predictive Coding model.

    Works with any architecture expressed as a list of nn.Sequential layers.
    Errors are added between layers during inference, weight updates use
    E_local for local (biologically plausible) learning.

    Args:
        layers: List of nn.Sequential modules defining the architecture.
        iters: Number of error optimization steps per batch.
        e_lr: Learning rate for error optimization (SGD/Adam).
        output_loss: 'ce' (cross-entropy) or 'mse' (mean squared error).
        error_optim: 'sgd', 'adam', or 'newton'.
        damping: Damping factor for Newton mode (higher = more conservative).
    """

    def __init__(self, layers, iters=5, e_lr=0.001, output_loss='ce',
                 error_optim='sgd', damping=1.0, early_stop_threshold=0.0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.iters = iters
        self.e_lr = e_lr
        self.errors = None
        self.error_optim_mode = error_optim
        self.damping = damping
        self.early_stop_threshold = early_stop_threshold
        self._iters_used = 0
        self._weight_phase_prediction = None
        self.profiling = False
        self._profile = {}

        if output_loss == 'mse':
            def _mse_loss(y_pred, y):
                if y.dim() == 1:
                    y = F.one_hot(y, num_classes=y_pred.shape[-1]).float()
                return 0.5 * F.mse_loss(y_pred, y)  # mean reduction
            self._output_loss = _mse_loss
        elif output_loss == 'ce':
            self._output_loss = lambda y_pred, y: F.cross_entropy(y_pred, y)  # mean reduction
        else:
            raise ValueError(f"Unknown output_loss: {output_loss}")

        # (energy_scale removed: mean reduction eliminates the need for it)

    def y_pred(self, x):
        """Forward pass with current errors."""
        s_i = x
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)
        return s_i

    def E(self, x, y):
        """Energy using errors (global graph — for error optimization).

        E = 0.5 * sum(mean(e_i^2)) + output_loss.
        Mean-reduced error penalty matches mean-reduced output loss.

        IMPORTANT: Do not use this for weight optimization, or you'll be
        doing standard backprop instead of local learning.
        """
        E_errors = 0.5 * sum(e.pow(2).mean() for e in self.errors)
        return E_errors + self._output_loss(self.y_pred(x), y)

    def E_local(self, x, y):
        """Energy using local interactions (detached — for weight optimization).

        Same value as E, but the computational graph enforces local weight
        updates by detaching states between layers. Each layer's weights
        receive gradients only from local prediction errors.
        """
        E = 0.0
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            E += 0.5 * F.mse_loss(s_i_pred, s_i)  # mean reduction
        y_pred = self.layers[-1](s_i)
        self._weight_phase_prediction = y_pred.detach()
        return E + self._output_loss(y_pred, y)

    @torch.no_grad()
    def init_zero_errors(self, x):
        """Initialize zero errors, caching shapes to skip redundant forward passes.

        On first call (or when input shape changes), runs a full forward pass
        to discover error shapes. On subsequent calls with the same input shape,
        creates zero tensors directly from cached shapes — saving one full
        forward pass per batch (~26ms on ResNet-18).

        Errors are always fp32 regardless of autocast context. fp16 rounds
        small Newton corrections to zero, which defeats early stopping and
        degrades inference quality. fp32 errors are cheap (just vectors added
        between layers); the expensive ops (conv, linear) still run in fp16.
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

        self.errors = []
        for layer_i in self.layers[:-1]:
            x = layer_i(x)
            self.errors.append(
                torch.zeros(x.shape, dtype=torch.float32,
                            device=x.device, requires_grad=True)
            )
        self._cached_input_shape = input_shape
        self._cached_error_shapes = [e.shape for e in self.errors]

    def _newton_step(self):
        """Rank-1 LRPD Newton step for error optimization.

        The error Hessian is H = I + J^T H_L J where J = dy/de.
        The gradient decomposes as g = e + J^T(dL/dy), so
        u = g - e = J^T(dL/dy) is the output-driven component.

        Rank-1 approximation: H ≈ (1+damping)I + u*u^T
        Woodbury inverse: H^{-1}g = g/d - (u^T g)/(d^2 + d*||u||^2) * u
        where d = 1 + damping.

        Cost: zero extra backward passes (reuses the gradient).

        Optimized: streams per-layer to avoid concatenating all errors into
        a single 250MB+ vector. Uses decomposed dot products:
          uTg = gTg - gTe,  uTu = gTg - 2*gTe + eTe
        and in-place update:
          e_new = e*(1-c) + g*(c - 1/d)  where c = uTg/(d^2 + d*uTu)
        """
        with torch.no_grad():
            d = 1.0 + self.damping

            # Accumulate dot products per-layer (no concatenation needed).
            # Decompose: uTg = gTg - gTe, uTu = gTg - 2*gTe + eTe
            # where u = g - e. This avoids creating the u vector entirely.
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

            # Apply step in-place: e_new = e*(1-coeff) + g*(coeff - 1/d)
            # Derived from: e_new = e - (g/d - coeff*(g - e))
            c1 = 1.0 - coeff
            c2 = coeff - 1.0 / d
            for e in self.errors:
                e.data.mul_(c1).add_(e.grad, alpha=c2)

    def minimize_error_energy(self, x, y):
        """Inference phase: optimize errors to minimize energy.

        With early_stop_threshold > 0, stops iterating when the relative
        energy improvement drops below the threshold. This avoids wasted
        computation on batches that converge quickly.
        """
        prof = self.profiling

        if prof:
            _t = _sync_time()

        for p in self.layers.parameters():
            p.requires_grad_(False)

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

        E_prev = None
        for t in range(self.iters):
            # Zero gradients
            if optim is not None:
                optim.zero_grad()
            else:
                for e in self.errors:
                    if e.grad is not None:
                        e.grad.zero_()

            # Compute energy and gradients
            if prof:
                _t = _sync_time()

            E = self.E(x, y)
            E_val = E.item()

            if prof:
                _t2 = _sync_time()
                prof_fwd += (_t2 - _t) * 1000

            if t == 0:
                self._E_initial = E_val

            # Adaptive early stopping: if energy barely decreased, stop
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

            # Take optimization step
            if optim is not None:
                optim.step()
            else:
                self._newton_step()

            if prof:
                prof_step += (_sync_time() - _t) * 1000
        else:
            # Loop completed without break
            self._E_final = E_val
            self._iters_used = self.iters

        for p in self.layers.parameters():
            p.requires_grad_(True)

        if prof:
            self._profile = {
                'init_ms': prof_init,
                'forward_ms': prof_fwd,
                'backward_ms': prof_bwd,
                'step_ms': prof_step,
            }

        return self._E_final

    def get_diagnostics(self):
        """Collect per-layer diagnostics after inference.

        Returns dict with: E_initial, E_final, convergence,
        error_norms (per-layer), layer_energies (per-layer).
        """
        diag = {
            'E_initial': getattr(self, '_E_initial', 0.0),
            'E_final': getattr(self, '_E_final', 0.0),
            'convergence': getattr(self, '_E_initial', 0.0) - getattr(self, '_E_final', 0.0),
            'iters_used': getattr(self, '_iters_used', self.iters),
            'error_norms': [],
            'layer_energies': [],
        }
        for e in self.errors:
            if isinstance(e, torch.Tensor):
                norm = torch.linalg.vector_norm(e, ord=2, dim=None).item()
                diag['error_norms'].append(norm)
                diag['layer_energies'].append(0.5 * norm ** 2)
        return diag

    def forward(self, x, y=None):
        """Forward pass. With y: runs inference (training). Without y: feedforward."""
        if y is None:
            self.errors = [0.0] * (len(self.layers) - 1)
            return self.y_pred(x)
        else:
            return self.minimize_error_energy(x, y)

    def compute_weight_loss(self, x, y):
        """Compute loss for weight optimizer (call after forward with y)."""
        return self.E_local(x, y)


class PCESkipConnection(PCE):
    """ePC with skip connection support via (activity, identity) tuples.

    Layers use SaveIdentity/AddIdentity/LayerWithResidual wrappers to
    handle skip connections. The state flows as (activity, identity) tuples
    through the network, with errors added to the activity component.
    """

    def y_pred(self, x):
        s_i = (x, 0.0)
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = layer_i(s_i)
            s_i = (s_i[0] + e_i, s_i[1])
        return s_i[0]

    def E_local(self, x, y):
        E = 0.0
        s_i = (x, 0.0)
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred[0]).detach()
            E += 0.5 * F.mse_loss(s_i_pred[0], s_i)  # mean reduction
            s_i = (s_i, s_i_pred[1])
        y_pred = self.layers[-1](s_i)[0]
        self._weight_phase_prediction = y_pred.detach()
        return E + self._output_loss(y_pred, y)

    @torch.no_grad()
    def init_zero_errors(self, x):
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

        self.errors = []
        s_i = (x, 0.0)
        for layer_i in self.layers[:-1]:
            s_i = layer_i(s_i)
            self.errors.append(
                torch.zeros(s_i[0].shape, dtype=torch.float32,
                            device=s_i[0].device, requires_grad=True)
            )
        self._cached_input_shape = input_shape
        self._cached_error_shapes = [e.shape for e in self.errors]
