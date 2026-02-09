"""KRONOS: KRONecker-Optimized Second-order optimizer.

KFAC (Kronecker-Factored Approximate Curvature) with LRPD (Low-Rank Plus
Diagonal) factor decomposition for scalable second-order weight optimization.

For a layer y = Wx, the Fisher Information Matrix is:
    F_W ≈ A ⊗ G
where A = E[aa^T] (input covariance) and G = E[gg^T] (output gradient
covariance). The preconditioned (natural) gradient is:
    ΔW = (G + λ I)^{-1} @ ∇W @ (A + λ I)^{-1}

ePC-adapted damping:
    ePC's E_local detaches states between layers and uses local MSE losses.
    After good inference (Newton T=2), prediction errors are small, making
    the output gradient covariance G degenerate (trace(G) << trace(A)).
    Pi-correction would push all damping onto A, collapsing the preconditioner
    to a scaled identity. Instead, KRONOS uses flat damping (same λ for both)
    which preserves A's curvature structure. G^{-1} becomes approximately
    constant scaling, and the useful curvature comes from A^{-1} (input
    whitening). The G^{-1} amplification is controlled by per-layer gradient
    clipping, which preserves the curvature-corrected direction.

KRONOS decomposes A and G into LRPD form (diag(d) + UU^T) and uses the
Woodbury identity for O(n*k^2) inversion instead of O(n^3). Factor estimates
are maintained via streaming LRPD updates (EMA blending). Damping is applied
at inversion time (not baked into the factors).

Scalability:
    Memory: O(n*k) per factor instead of O(n^2)
    Inversion: O(n*k^2) instead of O(n^3)
    For k=32, n=4096: 120x less memory, 20000x faster inversion

ePC compatibility: ePC's E_local detaches states between layers, making the
weight Hessian block-diagonal. KRONOS exploits this: each layer gets its own
independent KFAC estimate. No cross-layer coupling to approximate away.

Conv2d support: im2col unfolding converts spatial convolutions to equivalent
linear operations for KFAC factor computation.

Reference:
    Martens & Grosse 2015, "Optimizing Neural Networks with
    Kronecker-factored Approximate Curvature" (KFAC)
    Yeon & Anitescu 2025, "Beyond Low Rank" (LRPD decomposition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lrpd import lrpd_solve, online_lrpd_update, alt_decompose


# Small floor for numerical stability (NOT damping)
_EPS_FLOOR = 1e-6


class _KronosState:
    """Per-layer KFAC state with LRPD factor decomposition.

    Maintains LRPD estimates of the input covariance A = E[aa^T] and
    output gradient covariance G = E[gg^T] for a single parametric layer.
    Factors store PURE curvature (no damping). Flat damping is applied at
    inversion time (no pi-correction — see module docstring for rationale).

    Attributes:
        d_a, U_a: LRPD factors for A [in_dim], [in_dim, rank_a]
        d_g, U_g: LRPD factors for G [out_dim], [out_dim, rank_g]
    """

    def __init__(self, module, rank, damping, ema_decay, max_samples):
        self.module = module
        self.damping = damping
        self.ema_decay = ema_decay
        self.max_samples = max_samples

        # Determine layer dimensions
        if isinstance(module, nn.Linear):
            self.in_dim = module.in_features
            self.out_dim = module.out_features
            self.is_conv = False
        elif isinstance(module, nn.Conv2d):
            self.in_dim = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            self.out_dim = module.out_channels
            self.is_conv = True
        else:
            raise ValueError(f"Unsupported layer type: {type(module)}")

        # Augment input dim for bias
        self.has_bias = module.bias is not None
        self.aug_in_dim = self.in_dim + (1 if self.has_bias else 0)

        # Cap rank at dim-1 (full rank minus 1 diagonal dof)
        self.rank_a = min(rank, max(1, self.aug_in_dim - 1))
        self.rank_g = min(rank, max(1, self.out_dim - 1))

        # LRPD factors (initialized on first update)
        self.d_a = None
        self.U_a = None
        self.d_g = None
        self.U_g = None

        # Hook caches
        self._cached_a = None
        self._cached_g = None

        self.initialized = False
        self.steps = 0

    def capture_forward(self, a):
        """Cache (subsampled) activations [N, aug_in_dim]."""
        if a.shape[0] > self.max_samples:
            idx = torch.randperm(a.shape[0], device=a.device)[:self.max_samples]
            a = a[idx]
        self._cached_a = a.detach()

    def capture_backward(self, g):
        """Cache (subsampled) output gradients [N, out_dim]."""
        if g.shape[0] > self.max_samples:
            idx = torch.randperm(g.shape[0], device=g.device)[:self.max_samples]
            g = g[idx]
        self._cached_g = g.detach()

    def _lrpd_trace(self, d, U):
        """Compute trace(diag(d) + UU^T) = sum(d) + ||U||_F^2."""
        return d.sum() + (U ** 2).sum()

    @torch.no_grad()
    def update_factors(self):
        """Update LRPD estimates of A and G from cached hook data.

        Factors store pure curvature (no damping). Uses FITC residual mode
        for exact diagonal preservation without over-inflation.

        Scaling:
            A = E[a a^T]: activations are raw, so F_a = a / sqrt(n)
                gives F_a^T F_a = a^T a / n = (1/B) sum a_i a_i^T.

            G = E[g_per_sample g_per_sample^T]: but PyTorch backward gives
                g_batch_mean = g_per_sample / B (from batch-mean loss).
                To recover the per-sample Fisher: F_g = g * sqrt(n)
                gives F_g^T F_g = n * g^T g = n * sum (g_true/B)^2
                = (1/B) sum g_true g_true^T.
        """
        a = self._cached_a
        g = self._cached_g
        if a is None or g is None:
            return

        n_a = a.shape[0]
        n_g = g.shape[0]
        device = a.device
        dtype = a.dtype

        # Small numerical floor (NOT damping)
        floor_a = torch.full((self.aug_in_dim,), _EPS_FLOOR,
                             device=device, dtype=dtype)
        floor_g = torch.full((self.out_dim,), _EPS_FLOOR,
                             device=device, dtype=dtype)

        if not self.initialized:
            # First update: full weight on batch data
            d_a_init = floor_a.clone()
            U_a_init = torch.zeros(self.aug_in_dim, self.rank_a,
                                   device=device, dtype=dtype)

            F_a = a / (n_a ** 0.5)  # A = a^T a / n (activations: raw)
            self.d_a, self.U_a = online_lrpd_update(
                d_a_init, U_a_init, F_a,
                alpha=1.0, beta=0.0,
                d_prior=floor_a,
                alpha_prior=1.0,
                residual_mode='fitc',
                rank_k=self.rank_a,
            )

            d_g_init = floor_g.clone()
            U_g_init = torch.zeros(self.out_dim, self.rank_g,
                                   device=device, dtype=dtype)

            F_g = g * (n_g ** 0.5)  # G = n * g^T g (undo batch-mean scaling)
            self.d_g, self.U_g = online_lrpd_update(
                d_g_init, U_g_init, F_g,
                alpha=1.0, beta=0.0,
                d_prior=floor_g,
                alpha_prior=1.0,
                residual_mode='fitc',
                rank_k=self.rank_g,
            )

            self.initialized = True
        else:
            # Streaming EMA update (pure curvature, no damping)
            F_a = a / (n_a ** 0.5)
            self.d_a, self.U_a = online_lrpd_update(
                self.d_a, self.U_a, F_a,
                alpha=1.0 - self.ema_decay,
                beta=self.ema_decay,
                d_prior=floor_a,
                alpha_prior=1.0 - self.ema_decay,
                residual_mode='fitc',
                rank_k=self.rank_a,
            )

            F_g = g * (n_g ** 0.5)  # G = n * g^T g (undo batch-mean scaling)
            self.d_g, self.U_g = online_lrpd_update(
                self.d_g, self.U_g, F_g,
                alpha=1.0 - self.ema_decay,
                beta=self.ema_decay,
                d_prior=floor_g,
                alpha_prior=1.0 - self.ema_decay,
                residual_mode='fitc',
                rank_k=self.rank_g,
            )

        self._cached_a = None
        self._cached_g = None
        self.steps += 1

    @torch.no_grad()
    def precondition(self, grad_w, grad_b=None):
        """Apply KFAC preconditioning with flat damping.

        Computes: ΔW = (G + λ I)^{-1} @ ∇W @ (A + λ I)^{-1}

        Uses flat damping (same λ for both factors) instead of pi-correction.
        For ePC's E_local, G is degenerate (tiny output gradients after
        inference), so G^{-1} ≈ (1/λ)*I (constant scaling). The useful
        curvature comes from A^{-1} (input whitening). The G^{-1} scaling
        is controlled by gradient clipping in step().

        Args:
            grad_w: [out_dim, in_features] or [out, in, kH, kW] for Conv2d
            grad_b: [out_dim] or None

        Returns:
            (preconditioned_grad_w, preconditioned_grad_b)
        """
        if not self.initialized:
            return grad_w, grad_b

        # Reshape conv gradient to 2D: [out, in*kH*kW]
        orig_shape = grad_w.shape
        if self.is_conv:
            grad_w = grad_w.reshape(self.out_dim, self.in_dim)

        # Build augmented gradient matrix [out, aug_in]
        if self.has_bias and grad_b is not None:
            M = torch.cat([grad_w, grad_b.unsqueeze(1)], dim=1)
        else:
            M = grad_w

        # Flat damping: same value for both A and G
        d_a_damped = self.d_a + self.damping
        d_g_damped = self.d_g + self.damping

        # Right-multiply: M @ (A + λ I)^{-1}
        M = lrpd_solve(d_a_damped, self.U_a, M)

        # Left-multiply: (G + λ I)^{-1} @ M = (M^T @ (G + λ I)^{-1})^T
        M = lrpd_solve(d_g_damped, self.U_g, M.T).T

        # Split augmented gradient back
        if self.has_bias and grad_b is not None:
            grad_w_out = M[:, :-1]
            grad_b_out = M[:, -1]
        else:
            grad_w_out = M
            grad_b_out = grad_b

        # Restore conv shape
        if self.is_conv:
            grad_w_out = grad_w_out.reshape(orig_shape)

        return grad_w_out, grad_b_out


class KRONOS(torch.optim.Optimizer):
    """KRONecker-Optimized Second-order optimizer.

    A KFAC-family optimizer using LRPD factor decomposition for scalable
    second-order preconditioning. Automatically registers hooks on all
    Linear and Conv2d layers in the model.

    Usage:
        model = PCESkipConnection(layers, ...)
        optimizer = KRONOS(model, lr=0.001)

        # Training loop:
        optimizer.zero_grad()
        loss = model.compute_weight_loss(x, y, batch_size)
        loss.backward()
        optimizer.step()

    The optimizer captures activations and gradients during the forward/backward
    pass of compute_weight_loss (E_local). During inference (minimize_error_energy),
    hooks are inactive because layer parameters have requires_grad=False.

    Per-layer gradient clipping controls step magnitude after preconditioning.
    Since G^{-1} ≈ (1/damping)*I for ePC (degenerate G factor), the
    preconditioned gradient is amplified by ~1/damping. Clipping at a fixed
    norm preserves the curvature-corrected DIRECTION from A^{-1} while
    capping the step magnitude.

    Args:
        model: nn.Module with Linear and/or Conv2d layers.
        lr: Learning rate for preconditioned updates.
        momentum: SGD momentum on preconditioned gradients.
        damping: Tikhonov damping (λ). Same value applied to both A and G
            factors (flat damping, no pi-correction). Should be small enough
            to not dominate A's eigenvalues (e.g., 0.01 for typical A with
            eigenvalues 0.01-0.5).
        rank: LRPD rank for factor decomposition. Capped at dim-1 per layer.
        ema_decay: Exponential moving average decay for factor updates.
        update_freq: Steps between factor LRPD updates. Factors are also
            always updated on the first step.
        weight_decay: L2 regularization.
        grad_clip: Per-layer gradient norm clipping (0 to disable).
        max_samples: Maximum activation/gradient samples per update.
            Controls memory for Conv2d layers with large spatial dims.
    """

    def __init__(self, model, lr=0.001, momentum=0.9, damping=0.01,
                 rank=32, ema_decay=0.95, update_freq=10,
                 weight_decay=0.0, grad_clip=1.0, max_samples=1024):
        self._states = {}
        self._hooks = []
        self._step_count = 0
        self._update_freq = update_freq
        self._grad_clip = grad_clip

        # Find all parametric layers and register hooks
        kfac_params = []
        kfac_param_ids = set()

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                state = _KronosState(module, rank, damping, ema_decay,
                                     max_samples)
                self._states[module] = state

                self._hooks.append(
                    module.register_forward_hook(self._fwd_hook))
                self._hooks.append(
                    module.register_full_backward_hook(self._bwd_hook))

                for p in module.parameters():
                    kfac_params.append(p)
                    kfac_param_ids.add(id(p))

        # Remaining parameters (BatchNorm etc) get first-order updates
        other_params = [p for p in model.parameters()
                        if id(p) not in kfac_param_ids and p.requires_grad]

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        param_groups = []
        if kfac_params:
            param_groups.append({'params': kfac_params, 'kfac': True})
        if other_params:
            param_groups.append({'params': other_params, 'kfac': False})

        super().__init__(param_groups, defaults)

        n_kfac = len(self._states)
        n_other = len(other_params)
        total_kfac = sum(p.numel() for p in kfac_params)
        total_other = sum(p.numel() for p in other_params)
        print(f"KRONOS: {n_kfac} KFAC layers ({total_kfac:,} params), "
              f"{n_other} first-order params ({total_other:,})")
        for module, state in self._states.items():
            name = module.__class__.__name__
            print(f"  {name}: in={state.aug_in_dim}, out={state.out_dim}, "
                  f"rank_a={state.rank_a}, rank_g={state.rank_g}")

    def _fwd_hook(self, module, input, output):
        """Capture activations during forward pass (weight update phase only)."""
        # Skip during inference: parameters have requires_grad=False
        if not module.weight.requires_grad:
            return

        state = self._states.get(module)
        if state is None:
            return

        a = input[0]

        if state.is_conv:
            # im2col: [B, C, H, W] → [B, C*kH*kW, L] → [B*L, C*kH*kW]
            a = F.unfold(a, module.kernel_size,
                         dilation=module.dilation,
                         padding=module.padding,
                         stride=module.stride)
            a = a.permute(0, 2, 1).reshape(-1, a.shape[1])
        else:
            a = a.reshape(-1, a.shape[-1])

        # Augment with ones for bias
        if state.has_bias:
            ones = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)

        state.capture_forward(a)

    def _bwd_hook(self, module, grad_input, grad_output):
        """Capture output gradients during backward pass."""
        if not module.weight.requires_grad:
            return

        state = self._states.get(module)
        if state is None:
            return

        g = grad_output[0]

        if state.is_conv:
            # [B, C_out, H_out, W_out] → [B*H_out*W_out, C_out]
            g = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        else:
            g = g.reshape(-1, g.shape[-1])

        state.capture_backward(g)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Updates LRPD factor estimates (if update_freq allows), then applies
        KFAC-preconditioned gradients to parametric layers and standard
        SGD+momentum to remaining parameters.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Update LRPD factors (always on first step, then every update_freq)
        if self._step_count == 1 or self._step_count % self._update_freq == 0:
            for state in self._states.values():
                state.update_factors()

        lr = self.param_groups[0]['lr']
        momentum = self.param_groups[0]['momentum']
        wd = self.param_groups[0]['weight_decay']

        # KFAC-preconditioned updates for Linear/Conv2d
        processed_ids = set()
        for module, state in self._states.items():
            w = module.weight
            b = module.bias

            if w.grad is None:
                continue

            grad_w = w.grad
            grad_b = b.grad if b is not None else None

            # Precondition with KFAC
            if state.initialized:
                grad_w, grad_b = state.precondition(grad_w, grad_b)

            # Per-layer gradient norm clipping
            if self._grad_clip > 0:
                grad_norm = grad_w.norm()
                if grad_b is not None:
                    grad_norm = (grad_norm ** 2 + grad_b.norm() ** 2) ** 0.5
                clip_coef = self._grad_clip / (grad_norm + 1e-6)
                if clip_coef < 1.0:
                    grad_w = grad_w * clip_coef
                    if grad_b is not None:
                        grad_b = grad_b * clip_coef

            # Weight decay
            if wd > 0:
                grad_w = grad_w + wd * w.data
                if b is not None and grad_b is not None:
                    grad_b = grad_b + wd * b.data

            # Momentum
            p_state = self.state[w]
            if 'momentum_w' not in p_state:
                p_state['momentum_w'] = torch.zeros_like(w)
            buf_w = p_state['momentum_w']
            buf_w.mul_(momentum).add_(grad_w)
            w.data.sub_(lr * buf_w)

            if b is not None and grad_b is not None:
                if 'momentum_b' not in p_state:
                    p_state['momentum_b'] = torch.zeros_like(b)
                buf_b = p_state['momentum_b']
                buf_b.mul_(momentum).add_(grad_b)
                b.data.sub_(lr * buf_b)

            processed_ids.add(id(w))
            if b is not None:
                processed_ids.add(id(b))

        # First-order SGD+momentum for remaining params (BatchNorm etc)
        for group in self.param_groups:
            if group.get('kfac', False):
                continue
            for p in group['params']:
                if p.grad is None or id(p) in processed_ids:
                    continue

                grad = p.grad
                if wd > 0:
                    grad = grad + wd * p.data

                p_state = self.state[p]
                if 'momentum_buf' not in p_state:
                    p_state['momentum_buf'] = torch.zeros_like(p)
                buf = p_state['momentum_buf']
                buf.mul_(momentum).add_(grad)
                p.data.sub_(lr * buf)

        return loss

    def remove_hooks(self):
        """Remove all registered forward/backward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_diagnostics(self):
        """Return per-layer curvature diagnostics.

        Returns:
            dict mapping layer description to curvature info including
            factor traces, damping, and eigenvalue ranges.
        """
        diag = {}
        for i, (module, state) in enumerate(self._states.items()):
            name = f"L{i+1}_{module.__class__.__name__}"
            info = {
                'initialized': state.initialized,
                'steps': state.steps,
                'rank_a': state.rank_a,
                'rank_g': state.rank_g,
            }
            if state.initialized:
                trace_a = state._lrpd_trace(state.d_a, state.U_a).item()
                trace_g = state._lrpd_trace(state.d_g, state.U_g).item()
                info['trace_a'] = trace_a
                info['trace_g'] = trace_g
                info['damping'] = state.damping
                info['damping_vs_median_d_a'] = state.damping / (state.d_a.median().item() + 1e-8)
                info['d_a_range'] = (state.d_a.min().item(),
                                     state.d_a.max().item())
                info['d_g_range'] = (state.d_g.min().item(),
                                     state.d_g.max().item())
            diag[name] = info
        return diag
