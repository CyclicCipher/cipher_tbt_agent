"""Deep-Equilibrium wrapper: iterate the settling block to a fixed point.

Solves ``h* = f_theta(h*, x)`` by Picard iteration (optionally Anderson-
accelerated), returns the equilibrium plus rich convergence diagnostics, and
offers O(1)-memory gradient paths.

Gradient modes (Docs/architecture.md §4b, implementation_plan §2.1):
  * "one_step"  -- (default) detach the solved fixed point, take ONE more
                   block step with grad enabled. This is the HRM / phantom-
                   gradient trick the Starting Docs explicitly endorse:
                   O(1) memory, stable, no implicit linear solve. Correct
                   enough to validate Stage 0; cheap.
  * "unrolled"  -- truncated BPTT: solve under no_grad, then re-run the last
                   ``backward_steps`` iterations WITH grad. A middle ground.
  * "ift"       -- full implicit-function-theorem gradient (Bai et al. DEQ).
                   NOT implemented in the skeleton; stub raises with guidance.

The whole point of Stage 0 is to find out whether the loop converges at all
(Risk 1). Every solve returns a ``FixedPointInfo`` so the convergence monitor
(probes/convergence.py) can aggregate behavior across a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

GradMode = Literal["one_step", "unrolled", "ift", "bptt"]


@dataclass
class DEQConfig:
    max_iter: int = 50
    tol: float = 1e-3            # relative residual threshold for convergence
    grad_mode: GradMode = "one_step"
    backward_steps: int = 1      # used by "unrolled"
    bptt_iters: int = 12         # used by "bptt": fixed iteration count, full BPTT
    state_norm: bool = False     # RMS-normalize state each iteration (contraction aid)
    anderson: bool = False       # Anderson acceleration of the forward solve
    anderson_m: int = 5          # history size
    anderson_beta: float = 1.0   # mixing
    detect_oscillation: bool = True


@dataclass
class FixedPointInfo:
    """Per-solve convergence diagnostics (one entry per forward call)."""

    iters: int
    converged: bool
    final_rel_residual: float
    residual_trace: list[float] = field(default_factory=list)
    oscillating: bool = False

    def as_dict(self) -> dict:
        return {
            "iters": self.iters,
            "converged": self.converged,
            "final_rel_residual": self.final_rel_residual,
            "oscillating": self.oscillating,
        }


def _rel_residual(h_new: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    num = (h_new - h).flatten(1).norm(dim=1)
    den = h_new.flatten(1).norm(dim=1).clamp_min(1e-8)
    return (num / den).mean()


def _looks_oscillating(trace: list[float], window: int = 6, tol: float = 1e-2) -> bool:
    """Heuristic limit-cycle flag: residual plateaus *above* tol without dropping."""
    if len(trace) < window:
        return False
    recent = trace[-window:]
    spread = max(recent) - min(recent)
    return (min(recent) > tol) and (spread / max(recent) < 0.2)


def _rms_normalize(h: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Project the state onto the unit-RMS sphere (parameter-free).

    Why: a pre-norm *residual* block iterated is an integrator (h <- h + g(h)),
    which drifts rather than contracting to a fixed point -- the Stage-0 failure.
    Renormalizing the state each iteration bounds it so the map can settle (or at
    worst oscillate on a bounded manifold) instead of drifting to infinity. This
    is the cheapest realization of the contraction lever (architecture A8 / §3b).
    """
    # unit RMS per element (L2 norm ~ sqrt(dim)) -- the scale attention/RMSNorm
    # already expect, so the bounded state stays in-distribution for the block.
    return h * h.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()


class DEQFixedPoint(nn.Module):
    """Run a ``SettlingBlock`` to equilibrium with diagnostics."""

    def __init__(self, block: nn.Module, cfg: DEQConfig | None = None) -> None:
        super().__init__()
        self.block = block
        self.cfg = cfg or DEQConfig()

    def _step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One iteration of the fixed-point map, with optional state bounding."""
        h = self.block(h, x)
        if self.cfg.state_norm:
            h = _rms_normalize(h)
        return h

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def _picard_solve(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, FixedPointInfo]:
        cfg = self.cfg
        h = h0
        trace: list[float] = []
        converged = False
        i = 0
        for i in range(1, cfg.max_iter + 1):
            h_new = self._step(h, x)
            r = _rel_residual(h_new, h).item()
            trace.append(r)
            h = h_new
            if r < cfg.tol:
                converged = True
                break
        osc = _looks_oscillating(trace) if (cfg.detect_oscillation and not converged) else False
        info = FixedPointInfo(
            iters=i, converged=converged,
            final_rel_residual=trace[-1] if trace else float("nan"),
            residual_trace=trace, oscillating=osc,
        )
        return h, info

    @torch.no_grad()
    def _anderson_solve(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, FixedPointInfo]:
        """Anderson acceleration -- usually far fewer iterations than Picard.

        Standard batched Anderson (Bai et al. DEQ reference implementation),
        flattened over all non-batch dims.
        """
        cfg = self.cfg
        b = h0.shape[0]
        shape = h0.shape
        d = h0[0].numel()
        m = cfg.anderson_m

        X = torch.zeros(b, m, d, dtype=h0.dtype, device=h0.device)
        Fm = torch.zeros(b, m, d, dtype=h0.dtype, device=h0.device)

        def g(h_flat: torch.Tensor) -> torch.Tensor:
            return self.block(h_flat.view(shape), x).reshape(b, d)

        x0 = h0.reshape(b, d)
        X[:, 0] = x0
        Fm[:, 0] = g(x0)
        X[:, 1] = Fm[:, 0]
        Fm[:, 1] = g(X[:, 1])

        trace: list[float] = []
        converged = False
        k = 1
        for k in range(2, cfg.max_iter):
            n = min(k, m)
            G = Fm[:, :n] - X[:, :n]
            # solve least squares for mixing coefficients alpha
            H = torch.zeros(b, n + 1, n + 1, dtype=h0.dtype, device=h0.device)
            H[:, 0, 1:] = 1.0
            H[:, 1:, 0] = 1.0
            H[:, 1:, 1:] = torch.bmm(G, G.transpose(1, 2)) + 1e-4 * torch.eye(
                n, dtype=h0.dtype, device=h0.device
            ).unsqueeze(0)
            y = torch.zeros(b, n + 1, 1, dtype=h0.dtype, device=h0.device)
            y[:, 0] = 1.0
            alpha = torch.linalg.solve(H, y)[:, 1:, 0]  # (b, n)

            idx = k % m
            X[:, idx] = (
                cfg.anderson_beta * (alpha.unsqueeze(1) @ Fm[:, :n])[:, 0]
                + (1 - cfg.anderson_beta) * (alpha.unsqueeze(1) @ X[:, :n])[:, 0]
            )
            Fm[:, idx] = g(X[:, idx])
            r = (Fm[:, idx] - X[:, idx]).norm(dim=1) / (Fm[:, idx].norm(dim=1).clamp_min(1e-8))
            r = r.mean().item()
            trace.append(r)
            if r < cfg.tol:
                converged = True
                break

        h = X[:, k % m].view(shape)
        osc = _looks_oscillating(trace) if (cfg.detect_oscillation and not converged) else False
        info = FixedPointInfo(
            iters=k, converged=converged,
            final_rel_residual=trace[-1] if trace else float("nan"),
            residual_trace=trace, oscillating=osc,
        )
        return h, info

    # ---------------------------------------------------------------- forward
    def forward(
        self, x: torch.Tensor, h0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, FixedPointInfo]:
        """Return (equilibrium, info). ``x`` is the constant input injection."""
        if h0 is None:
            h0 = torch.zeros_like(x)

        # BPTT: fixed iteration count, gradient through every step. Decouples
        # learning from convergence (the Ouro / looped-transformer regime) -- the
        # plan's documented fallback when the fixed-point solve doesn't converge.
        # Run the SAME fixed unroll in eval (no_grad) as in training -- otherwise
        # eval would fall through to the Picard solver and read the state at a
        # different, non-converged point than training optimized (the train/eval
        # mismatch behind the BPTT oscillation). Consistency, not convergence.
        if self.cfg.grad_mode == "bptt":
            h = h0
            trace: list[float] = []
            for _ in range(self.cfg.bptt_iters):
                h_new = self._step(h, x)
                trace.append(_rel_residual(h_new, h).item())
                h = h_new
            info = FixedPointInfo(
                iters=self.cfg.bptt_iters, converged=trace[-1] < self.cfg.tol,
                final_rel_residual=trace[-1], residual_trace=trace, oscillating=False,
            )
            return h, info

        solver = self._anderson_solve if self.cfg.anderson else self._picard_solve
        h_star, info = solver(x, h0)

        if not torch.is_grad_enabled():
            return h_star, info

        # reattach a differentiable path to the (no_grad-solved) fixed point
        if self.cfg.grad_mode in ("one_step", "bptt"):
            # "bptt" under no_grad (shouldn't happen in training) degrades to one_step
            h_star = self._step(h_star.detach(), x)
        elif self.cfg.grad_mode == "unrolled":
            h = h_star.detach()
            for _ in range(max(1, self.cfg.backward_steps)):
                h = self._step(h, x)
            h_star = h
        elif self.cfg.grad_mode == "ift":
            h_star = self._ift_reattach(h_star.detach(), x)
        else:  # pragma: no cover
            raise ValueError(f"unknown grad_mode {self.cfg.grad_mode!r}")

        return h_star, info

    def _ift_reattach(self, h_fixed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Implicit-function-theorem gradient at the fixed point (Bai et al. 2019).

        For a loss L on the equilibrium h* = f(h*, x), the exact gradient is
            dL/dtheta = (dL/dh*) (I - J_f)^{-1} (df/dtheta),   J_f = df/dh* .
        We avoid forming (I - J_f)^{-1}: a backward hook solves the adjoint system
            g = (dL/dh*) + J_f^T g
        by fixed-point iteration (each step a vector-Jacobian product via autograd),
        then the normal autograd through one re-engaged step f(h*, x) multiplies g by
        df/dtheta. O(1) memory in the number of forward iterations. The adjoint solve
        converges when J_f is contractive -- which is exactly what `state_norm` buys.
        """
        h_eng = self._step(h_fixed, x)               # graph: depends on theta (h_fixed const)
        z0 = h_fixed.clone().requires_grad_(True)    # separate graph for VJPs
        f0 = self._step(z0, x)
        max_it, tol = self.cfg.max_iter, self.cfg.tol

        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad
            for _ in range(max_it):
                vjp = torch.autograd.grad(f0, z0, g, retain_graph=True)[0]
                g_new = grad + vjp
                if _rel_residual(g_new, g).item() < tol:
                    g = g_new
                    break
                g = g_new
            return g

        h_eng.register_hook(backward_hook)
        return h_eng
