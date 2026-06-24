"""PartitionedModel -- continual learning by novelty-gated allocation of parallel experts.

The paper's mechanism (Makushkin ai2026): "build other conditional distributions in parallel
instead of modifying existing distributions." Each expert is a FULL model; allocating a new
one FREEZES the previous, so prior regimes are never overwritten -> no catastrophic forgetting.

Modes:
  * single   -- one expert, always trained (the forgetful baseline)
  * oracle   -- allocate a fresh expert at a KNOWN regime boundary (the harness calls allocate()).
                Upper bound on retention: the old expert is frozen, the new one learns B from
                scratch.
  * surprise -- allocate when the ACTIVE expert's loss spikes (z-score > k past a warmup/cooldown):
                a general surprise signal (the ML reading of the paper's "timestamps of state
                changes" / HTM bursting), NOT a handcoded domain rule.

Routing here is oracle at eval (regime -> owning expert): A owns expert 0, B owns the allocated
expert. (Label-free confidence-routing is a documented follow-up, not needed to test retention.)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class PartitionedModel(nn.Module):
    def __init__(self, make_expert: Callable[[], nn.Module], mode: str = "single",
                 k: float = 4.0, cooldown: int = 50) -> None:
        super().__init__()
        assert mode in ("single", "oracle", "surprise")
        self.make_expert = make_expert
        self.mode, self.k, self.cooldown = mode, k, cooldown
        self.experts = nn.ModuleList([make_expert()])
        self.active = 0
        self._ema_mean: float | None = None
        self._ema_var: float = 1.0
        self._since_alloc = 0

    def forward(self, x: torch.Tensor, expert: int | None = None) -> torch.Tensor:
        return self.experts[self.active if expert is None else expert](x)

    def allocate(self) -> nn.Module:
        """Freeze the current expert, add a fresh one, make it active. Returns the new expert so
        the harness can add its parameters to the optimizer."""
        for p in self.experts[self.active].parameters():
            p.requires_grad_(False)
        new = self.make_expert()
        self.experts.append(new)
        self.active = len(self.experts) - 1
        self._ema_mean, self._since_alloc = None, 0
        return new

    def observe(self, loss: float) -> nn.Module | None:
        """Surprise detector. Returns a new expert iff it just allocated (else None)."""
        self._since_alloc += 1
        if self.mode != "surprise":
            return None
        if self._ema_mean is None:                          # initialise stats for the active expert
            self._ema_mean, self._ema_var = loss, max(0.1 * loss, 1e-3) ** 2
            return None
        z = (loss - self._ema_mean) / (self._ema_var ** 0.5 + 1e-6)
        self._ema_mean = 0.98 * self._ema_mean + 0.02 * loss
        self._ema_var = 0.98 * self._ema_var + 0.02 * (loss - self._ema_mean) ** 2
        if z > self.k and self._since_alloc > self.cooldown:
            return self.allocate()
        return None

    def n_experts(self) -> int:
        return len(self.experts)
