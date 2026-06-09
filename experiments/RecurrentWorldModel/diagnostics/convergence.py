"""Risk 1 instrumentation: does the settling loop converge, reliably, adaptively?

Two tools:

  * ``ConvergenceMonitor`` -- aggregate ``FixedPointInfo`` across a run into the
    metrics the implementation plan asks for (convergence rate, iterations
    distribution, oscillation rate, residual stats).

  * ``basin_consistency`` -- the spurious-attractor probe: solve the SAME input
    from several random initial states and measure whether they land on the same
    attractor. Low agreement = unreliable basins / spurious fixed points (the
    HRM failure mode A8 warns about).

These are measurement only. They never train and never mutate the model.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

from core.deq import DEQFixedPoint, FixedPointInfo


@dataclass
class ConvergenceMonitor:
    """Accumulate per-solve diagnostics and summarize."""

    iters: list[int] = field(default_factory=list)
    converged_flags: list[bool] = field(default_factory=list)
    final_residuals: list[float] = field(default_factory=list)
    oscillating_flags: list[bool] = field(default_factory=list)

    def record(self, info: FixedPointInfo) -> None:
        self.iters.append(info.iters)
        self.converged_flags.append(info.converged)
        self.final_residuals.append(info.final_rel_residual)
        self.oscillating_flags.append(info.oscillating)

    def summary(self) -> dict:
        n = max(1, len(self.iters))
        return {
            "n_solves": len(self.iters),
            "convergence_rate": sum(self.converged_flags) / n,
            "oscillation_rate": sum(self.oscillating_flags) / n,
            "iters_mean": statistics.fmean(self.iters) if self.iters else float("nan"),
            "iters_median": statistics.median(self.iters) if self.iters else float("nan"),
            "iters_max": max(self.iters) if self.iters else 0,
            "final_residual_mean": (
                statistics.fmean(self.final_residuals) if self.final_residuals else float("nan")
            ),
        }

    def difficulty_correlation(self, difficulties: list[float]) -> float:
        """Pearson r between problem difficulty and iterations-to-converge.

        Positive r = adaptive compute working (harder inputs settle slower).
        """
        if len(difficulties) != len(self.iters) or len(self.iters) < 2:
            return float("nan")
        xs = [float(d) for d in difficulties]
        ys = [float(i) for i in self.iters]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        denom = (vx * vy) ** 0.5
        return cov / denom if denom > 0 else float("nan")


@torch.no_grad()
def basin_consistency(
    deq: DEQFixedPoint,
    x: torch.Tensor,
    n_restarts: int = 4,
    init_scale: float = 1.0,
) -> dict:
    """Solve the same input from several random inits; measure attractor agreement.

    Returns mean pairwise relative distance between the resulting equilibria
    (0 = identical attractor every time; large = spurious / multiple basins),
    plus how many of the restarts actually converged.
    """
    equilibria: list[torch.Tensor] = []
    converged = 0
    for _ in range(n_restarts):
        h0 = init_scale * torch.randn_like(x)
        h_star, info = deq(x, h0=h0)
        equilibria.append(h_star)
        converged += int(info.converged)

    dists: list[float] = []
    for i in range(len(equilibria)):
        for j in range(i + 1, len(equilibria)):
            num = (equilibria[i] - equilibria[j]).flatten(1).norm(dim=1)
            den = equilibria[i].flatten(1).norm(dim=1).clamp_min(1e-8)
            dists.append((num / den).mean().item())

    return {
        "n_restarts": n_restarts,
        "converged_restarts": converged,
        "mean_pairwise_rel_dist": statistics.fmean(dists) if dists else 0.0,
        "max_pairwise_rel_dist": max(dists) if dists else 0.0,
    }
