"""PeriodicField -- a FAIR, stable latent-rollout drift test for LeWorldModel.

A bounded triangle wave  v(t) = v_mid + A * tri(t/P + phase)  with random period P and phase
per sequence. Because t_obs >= P_max, the model always observes >= one full period, so the
entire future is DETERMINED and BOUNDED: a perfect latent rollout tracks the wave forever and
any drift is the architecture's (no information trap, unlike DriftField's unbounded line).

This is the latent-space home for the TBAF test the token task couldn't provide: LeWM's
rollout genuinely iterates a latent (feeds its own predictions back), so phase/amplitude drift
can actually accumulate. Question: does inserting the corrected common-mode-rejecting
activation in LeWM's predictor flatten the rollout-decay curve vs a plain (GELU) sublayer?
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PeriodicBatch:
    obs_v: torch.Tensor        # (B, K)   observed wave values  -> LeWM continuous input
    obs_t: torch.Tensor        # (B, K)   observation times     -> action / coordinate
    field_target: torch.Tensor # (B, Tb, Vb) one-hot true value bin at each time centre
    time_mask: torch.Tensor    # (Tb,)    1 = within observation window, 0 = OOD rollout region
    period: torch.Tensor       # (B,)     diagnostics
    phase: torch.Tensor        # (B,)

    def to(self, device) -> "PeriodicBatch":
        return PeriodicBatch(self.obs_v.to(device), self.obs_t.to(device),
                             self.field_target.to(device), self.time_mask.to(device),
                             self.period.to(device), self.phase.to(device))


def _tri(x: torch.Tensor) -> torch.Tensor:
    """Period-1 triangle wave in [-1, 1]."""
    f = x - torch.floor(x + 0.5)                            # in [-0.5, 0.5]
    return 4.0 * f.abs() - 1.0


class PeriodicField:
    def __init__(self, n_obs: int = 20, t_obs: float = 15.0, t_max: float = 30.0, t_bins: int = 30,
                 v_min: float = 0.0, v_max: float = 24.0, v_bins: int = 24,
                 period_range: tuple[float, float] = (3.0, 7.0), amp_frac: float = 0.45,
                 seed: int = 0) -> None:
        assert t_obs >= period_range[1], "must observe >= one full period (t_obs >= P_max)"
        self.n_obs, self.t_obs, self.t_max = n_obs, t_obs, t_max
        self.t_bins, self.v_bins = t_bins, v_bins
        self.v_min, self.v_max = v_min, v_max
        self.period_range, self.seed = period_range, seed
        self.v_mid = 0.5 * (v_min + v_max)
        self.A = amp_frac * (v_max - v_min)
        self.t_edges = torch.linspace(0.0, t_max, t_bins + 1, dtype=torch.float32)
        self.v_edges = torch.linspace(v_min, v_max, v_bins + 1, dtype=torch.float32)
        self.t_centers = 0.5 * (self.t_edges[:-1] + self.t_edges[1:])
        self.time_mask = (self.t_centers <= t_obs).to(torch.float32)

    def _wave(self, t: torch.Tensor, P: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.v_mid + self.A * _tri(t / P + phase)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> PeriodicBatch:
        g = generator
        lo, hi = self.period_range
        P = lo + (hi - lo) * torch.rand(batch_size, generator=g)            # (B,)
        phase = torch.rand(batch_size, generator=g)                         # (B,)
        obs_t = torch.sort(torch.rand(batch_size, self.n_obs, generator=g) * self.t_obs, dim=1).values
        obs_v = self._wave(obs_t, P[:, None], phase[:, None])               # (B, K)
        tc = self.t_centers                                                 # (Tb,)
        vals = self._wave(tc[None, :], P[:, None], phase[:, None])          # (B, Tb) true value at centres
        idx = (torch.bucketize(vals, self.v_edges) - 1).clamp(0, self.v_bins - 1)
        field_target = torch.zeros(batch_size, self.t_bins, self.v_bins)
        field_target.scatter_(2, idx[..., None], 1.0)
        return PeriodicBatch(obs_v=obs_v.to(torch.float32), obs_t=obs_t.to(torch.float32),
                             field_target=field_target.to(torch.float32),
                             time_mask=self.time_mask.clone(), period=P, phase=phase)
