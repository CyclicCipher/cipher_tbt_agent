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

GradMode = Literal["one_step", "unrolled", "ift"]


@dataclass
class DEQConfig:
    max_iter: int = 50
    tol: float = 1e-3            # relative residual threshold for convergence
    grad_mode: GradMode = "one_step"
    backward_steps: int = 1      # used by "unrolled"
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


class DEQFixedPoint(nn.Module):
    """Run a ``SettlingBlock`` to equilibrium with diagnostics."""

    def __init__(self, block: nn.Module, cfg: DEQConfig | None = None) -> None:
        super().__init__()
        self.block = block
        self.cfg = cfg or DEQConfig()

    # ------------------------------------------------------------------ solve
    @torch.no_grad()
    def _picard_solve(self, x: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, FixedPointInfo]:
        cfg = self.cfg
        h = h0
        trace: list[float] = []
        converged = False
        i = 0
        for i in range(1, cfg.max_iter + 1):
            h_new = self.block(h, x)
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

        solver = self._anderson_solve if self.cfg.anderson else self._picard_solve
        h_star, info = solver(x, h0)

        if not torch.is_grad_enabled():
            return h_star, info

        # reattach a differentiable path to the (no_grad-solved) fixed point
        if self.cfg.grad_mode == "one_step":
            h_star = self.block(h_star.detach(), x)
        elif self.cfg.grad_mode == "unrolled":
            h = h_star.detach()
            for _ in range(max(1, self.cfg.backward_steps)):
                h = self.block(h, x)
            h_star = h
        elif self.cfg.grad_mode == "ift":
            raise NotImplementedError(
                "Full implicit (IFT) gradient is a Stage-0+ TODO. Implement the "
                "backward fixed-point solve (Bai et al. 2019) here, or use "
                "grad_mode='one_step' (default) / 'unrolled' for now."
            )
        else:  # pragma: no cover
            raise ValueError(f"unknown grad_mode {self.cfg.grad_mode!r}")

        return h_star, info
