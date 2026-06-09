"""Convergence-based halting -- no EOS token (Docs/architecture.md §6).

Two nested signals, both reuses of the settling machinery:

  * Per-step (depth): the inner loop halts when the free units reach the fixed
    point (``rel_residual < tol``). This already lives in ``deq.py`` -- the
    ``converged`` helper here just exposes the predicate for callers that hold a
    residual directly.

  * Per-problem (chain): the reasoning loop halts when re-clamping + re-settling
    no longer changes the conclusion (the chain reaches its own fixed point), or
    when a value estimate says the goal is reached. ``ChainHalt`` is a minimal
    scaffold for that; the *value* path is Stage 3 and is left as a hook.
"""

from __future__ import annotations

import torch


def converged(rel_residual: float, tol: float = 1e-3) -> bool:
    """Per-step halting predicate."""
    return rel_residual < tol


class ChainHalt:
    """Per-problem (chain) halting scaffold -- Stage 2+.

    Tracks successive settled conclusions and signals halt when they stop
    changing. The value-based criterion (Stage 3) is a documented hook, not yet
    implemented.
    """

    def __init__(self, tol: float = 1e-3, max_steps: int = 16) -> None:
        self.tol = tol
        self.max_steps = max_steps
        self._prev: torch.Tensor | None = None
        self._steps = 0

    def reset(self) -> None:
        self._prev = None
        self._steps = 0

    def should_halt(self, conclusion: torch.Tensor, value: float | None = None) -> bool:
        """Return True if the chain should stop.

        Args:
            conclusion: the latest settled conclusion (read-off units).
            value: optional goal-proximity estimate (Stage 3). If provided and
                >= 1.0 it forces a halt; wiring is here, the estimator is not.
        """
        self._steps += 1
        if value is not None and value >= 1.0:
            return True
        if self._steps >= self.max_steps:
            return True
        if self._prev is None:
            self._prev = conclusion.detach()
            return False
        num = (conclusion - self._prev).flatten(1).norm(dim=1)
        den = conclusion.flatten(1).norm(dim=1).clamp_min(1e-8)
        changed = (num / den).mean().item()
        self._prev = conclusion.detach()
        return changed < self.tol
