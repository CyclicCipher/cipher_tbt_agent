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

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 error_optim='sgd', damping=1.0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.iters = iters
        self.e_lr = e_lr
        self.errors = None
        self.error_optim_mode = error_optim
        self.damping = damping

        if output_loss == 'mse':
            def _mse_loss(y_pred, y):
                if y.dim() == 1:
                    y = F.one_hot(y, num_classes=y_pred.shape[-1]).float()
                return 0.5 * F.mse_loss(y_pred, y, reduction='sum')
            self._output_loss = _mse_loss
        elif output_loss == 'ce':
            self._output_loss = lambda y_pred, y: F.cross_entropy(y_pred, y, reduction='sum')
        else:
            raise ValueError(f"Unknown output_loss: {output_loss}")

        # Scale factor to compensate for small errors from limited inference.
        # Newton converges better, so errors are larger -> less scaling needed.
        if error_optim == 'newton':
            self.energy_scale = 1.0
        else:
            self.energy_scale = min(1.0, e_lr * iters)

    def y_pred(self, x):
        """Forward pass with current errors."""
        s_i = x
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)
        return s_i

    def E(self, x, y):
        """Energy using errors (global graph — for error optimization).

        IMPORTANT: Do not use this for weight optimization, or you'll be
        doing standard backprop instead of local learning.
        """
        E_errors = 0.5 * sum(
            torch.linalg.vector_norm(e, ord=2, dim=None) ** 2
            for e in self.errors
        )
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
            E += 0.5 * F.mse_loss(s_i_pred, s_i, reduction='sum')
        y_pred = self.layers[-1](s_i)
        return E + self._output_loss(y_pred, y)

    @torch.no_grad()
    def init_zero_errors(self, x):
        """Initialize zero errors by running a feedforward pass."""
        self.errors = [
            torch.zeros_like(x := layer_i(x), requires_grad=True)
            for layer_i in self.layers[:-1]
        ]

    def _newton_step(self):
        """Rank-1 LRPD Newton step for error optimization.

        The error Hessian is H = I + J^T H_L J where J = dy/de.
        The gradient decomposes as g = e + J^T(dL/dy), so
        u = g - e = J^T(dL/dy) is the output-driven component.

        Rank-1 approximation: H ≈ (1+damping)I + u*u^T
        Woodbury inverse: H^{-1}g = g/d - (u^T g)/(d^2 + d*||u||^2) * u
        where d = 1 + damping.

        Cost: zero extra backward passes (reuses the gradient).
        """
        with torch.no_grad():
            # Collect gradients and error values
            all_g = []
            all_e = []
            for e in self.errors:
                all_g.append(e.grad.flatten())
                all_e.append(e.data.flatten())

            g = torch.cat(all_g)
            e_vec = torch.cat(all_e)

            # Output-driven component: u = g - e = J^T(dL/dy)
            u = g - e_vec

            # Rank-1 Woodbury Newton step
            d = 1.0 + self.damping
            uTg = torch.dot(u, g)
            uTu = torch.dot(u, u)

            # H^{-1} g = g/d - (u^T g)/(d^2 + d*||u||^2) * u
            step = g / d - (uTg / (d * d + d * uTu)) * u

            # Apply step to individual errors
            offset = 0
            for e in self.errors:
                numel = e.numel()
                e.data -= step[offset:offset + numel].view_as(e)
                offset += numel

    def minimize_error_energy(self, x, y):
        """Inference phase: optimize errors to minimize energy."""
        for p in self.layers.parameters():
            p.requires_grad_(False)

        self.init_zero_errors(x)

        # Create first-order optimizer if needed
        if self.error_optim_mode == 'sgd':
            optim = torch.optim.SGD(self.errors, lr=self.e_lr)
        elif self.error_optim_mode == 'adam':
            optim = torch.optim.Adam(self.errors, lr=self.e_lr)
        else:
            optim = None  # Newton mode

        for t in range(self.iters):
            # Zero gradients
            if optim is not None:
                optim.zero_grad()
            else:
                for e in self.errors:
                    if e.grad is not None:
                        e.grad.zero_()

            # Compute energy and gradients
            E = self.E(x, y)
            if t == 0:
                self._E_initial = E.item()
            E.backward()

            # Take optimization step
            if optim is not None:
                optim.step()
            else:
                self._newton_step()

        self._E_final = E.item()

        for p in self.layers.parameters():
            p.requires_grad_(True)

        return E.item()

    def get_diagnostics(self):
        """Collect per-layer diagnostics after inference.

        Returns dict with: E_initial, E_final, convergence,
        error_norms (per-layer), layer_energies (per-layer).
        """
        diag = {
            'E_initial': getattr(self, '_E_initial', 0.0),
            'E_final': getattr(self, '_E_final', 0.0),
            'convergence': getattr(self, '_E_initial', 0.0) - getattr(self, '_E_final', 0.0),
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

    def compute_weight_loss(self, x, y, batch_size):
        """Compute loss for weight optimizer (call after forward with y)."""
        return self.E_local(x, y) / (batch_size * self.energy_scale)


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
            E += 0.5 * F.mse_loss(s_i_pred[0], s_i, reduction='sum')
            s_i = (s_i, s_i_pred[1])
        y_pred = self.layers[-1](s_i)[0]
        return E + self._output_loss(y_pred, y)

    @torch.no_grad()
    def init_zero_errors(self, x):
        self.errors = []
        s_i = (x, 0.0)
        for layer_i in self.layers[:-1]:
            s_i = layer_i(s_i)
            self.errors.append(torch.zeros_like(s_i[0], requires_grad=True))
