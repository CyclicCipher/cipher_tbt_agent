"""DriftField -- Stage 3 of the temporal fork. The generator behind the
unified (time x value) field predictor vs separate-heads experiment.

A latent value follows Brownian-motion-with-drift, observed at irregular times:

    v(t) = v0 + mu * t + sigma * W(t)          v0, mu random per sequence

The model sees a partial, irregularly-timed path {(t_i, v_i)}, must infer the
dynamics, and answer two queries that are two *reads* of one predicted object:

    what(tau)  = p(value | time = tau)            -- a column of the field
    when(theta)= p(first-passage time | value=theta) -- a survival read

sigma = 0 -> DETERMINISTIC (the field is a sharp ridge; both queries exact).
sigma > 0 -> STOCHASTIC   (the field is a diffuse band):
    what(tau)   ~ Normal(v0 + mu*tau, sigma^2 * tau)         [Gaussian column]
    when(theta) ~ InverseGaussian(m = a/mu, lam = a^2/sigma^2), a = theta - v0
                  [first passage of drifted Brownian motion -- closed form]

Closed forms give the analytic floor H(p_true) per query, so the stochastic
metric is KL(true||model) = CE - H(p_true), not raw CE (see the Theory doc and
the Stage-2 result). v0 and mu are BOTH randomised so neither query is solvable
from the time axis alone or the value axis alone -- the field cannot collapse to
a 1-D marginal (the `fixed_dist`-style confound discipline).

OOD = horizon extrapolation: time bins are supervised only within [0, t_obs]
(the observation window); bins in (t_obs, t_max] are never supervised and test
whether the model learned the rate law (extrapolates) or a lookup (caps).

This module only *generates* targets. Readouts + training live in train_field.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

SQRT2 = math.sqrt(2.0)


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF via erf (float64 in, float64 out)."""
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


@dataclass
class DriftBatch:
    obs_v: torch.Tensor        # (B, K)  observed values  -> continuous input to the trunk
    obs_t: torch.Tensor        # (B, K)  observation times -> PoPE coordinate
    field_target: torch.Tensor # (B, Tb, Vb)  p(value | time bin)        -- the "what" stack
    when_target: torch.Tensor  # (B, Vb, Tb)  p(first-passage | threshold)-- the "when" stack
    time_mask: torch.Tensor    # (Tb,)  1.0 where a time bin is in-dist (supervised), 0.0 OOD
    thr_mask: torch.Tensor     # (Vb,)  1.0 where a threshold bin is in-dist, 0.0 OOD
    when_valid: torch.Tensor   # (B, Vb) 1.0 where the threshold is genuinely reached within the
                               #         horizon (above v0, mean crossing <= t_max); 0.0 = censored
                               #         /already-crossed -> a trivial answer, excluded from the metric
    v0: torch.Tensor           # (B,)   latent intercept (diagnostics)
    mu: torch.Tensor           # (B,)   latent drift     (diagnostics)

    def to(self, device) -> "DriftBatch":
        return DriftBatch(
            self.obs_v.to(device), self.obs_t.to(device), self.field_target.to(device),
            self.when_target.to(device), self.time_mask.to(device), self.thr_mask.to(device),
            self.when_valid.to(device), self.v0.to(device), self.mu.to(device),
        )


class DriftField:
    def __init__(self, n_obs: int = 12, t_obs: float = 10.0, t_max: float = 20.0,
                 t_bins: int = 20, v_min: float = 0.0, v_max: float = 64.0, v_bins: int = 32,
                 v0_range: tuple[float, float] = (0.0, 20.0),
                 mu_range: tuple[float, float] = (0.5, 2.0),
                 sigma: float = 0.0, seed: int = 0) -> None:
        assert 0.0 < t_obs <= t_max
        self.n_obs = n_obs
        self.t_obs, self.t_max = t_obs, t_max
        self.t_bins, self.v_bins = t_bins, v_bins
        self.v_min, self.v_max = v_min, v_max
        self.v0_range, self.mu_range = v0_range, mu_range
        self.sigma = sigma
        self.seed = seed

        # bin edges + centres (float64 for target math; cast to float32 at the end)
        self.t_edges = torch.linspace(0.0, t_max, t_bins + 1, dtype=torch.float64)
        self.v_edges = torch.linspace(v_min, v_max, v_bins + 1, dtype=torch.float64)
        self.t_centers = 0.5 * (self.t_edges[:-1] + self.t_edges[1:])
        self.v_centers = 0.5 * (self.v_edges[:-1] + self.v_edges[1:])

        # in-dist = within the observation window; OOD = the unobserved future / range
        self.time_mask = (self.t_centers <= t_obs).to(torch.float32)
        self.thr_mask = (self.v_centers <= v_min + (v_max - v_min) * 0.5).to(torch.float32)

    # ---- discretisation helpers (all float64) -----------------------------------
    def _gauss_bins(self, mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
        """p(value bin | Normal(mean, sd^2)). mean,sd broadcast to (..., 1); -> (..., Vb)."""
        edges = self.v_edges.to(mean.device)                       # (Vb+1,)
        sd = sd.clamp_min(1e-6)
        cdf = _phi((edges - mean[..., None]) / sd[..., None])       # (..., Vb+1)
        probs = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
        # mass outside [v_min, v_max] folded onto the nearest edge bin (censoring)
        probs[..., 0] += cdf[..., 0]
        probs[..., -1] += 1.0 - cdf[..., -1]
        return probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)

    def _ig_cdf(self, t: torch.Tensor, m: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Inverse-Gaussian CDF (first passage of BM-with-drift). t,m,lam broadcast."""
        t = t.clamp_min(1e-9)
        u = torch.sqrt(lam / t)
        a = u * (t / m - 1.0)
        b = -u * (t / m + 1.0)
        # exp(2 lam/m) can overflow; clamp the exponent -- Phi(b) -> 0 when the barrier is
        # far, so the product stays negligible. final CDF clamped to [0,1].
        term2 = torch.exp((2.0 * lam / m).clamp(max=30.0)) * _phi(b)
        return (_phi(a) + term2).clamp(0.0, 1.0)

    def _ig_bins(self, m: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """p(time bin | first-passage). m,lam shape (...,); -> (..., Tb). Mass beyond
        t_max (never reached within the horizon) is censored onto the last time bin."""
        edges = self.t_edges.to(m.device)                          # (Tb+1,)
        cdf = self._ig_cdf(edges, m[..., None], lam[..., None])     # (..., Tb+1)
        probs = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
        probs[..., -1] += 1.0 - cdf[..., -1]                        # censor the tail
        return probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)

    # ---- sampling ---------------------------------------------------------------
    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> DriftBatch:
        g = generator
        B, K = batch_size, self.n_obs

        def rand(*shape):
            return torch.rand(*shape, generator=g, dtype=torch.float64)

        v0 = self.v0_range[0] + (self.v0_range[1] - self.v0_range[0]) * rand(B)
        mu = self.mu_range[0] + (self.mu_range[1] - self.mu_range[0]) * rand(B)

        # irregular observation times in (0, t_obs], sorted ascending per row
        obs_t = torch.sort(rand(B, K) * self.t_obs, dim=1).values            # (B,K)
        mean_obs = v0[:, None] + mu[:, None] * obs_t                         # (B,K)
        if self.sigma > 0.0:
            dt = torch.diff(obs_t, dim=1, prepend=torch.zeros(B, 1, dtype=torch.float64))
            z = torch.randn(B, K, generator=g, dtype=torch.float64)
            w = torch.cumsum(torch.sqrt(dt.clamp_min(0.0)) * z, dim=1)       # Brownian path
            obs_v = mean_obs + self.sigma * w
        else:
            obs_v = mean_obs

        tc = self.t_centers.to(v0.device)                                   # (Tb,)
        vc = self.v_centers.to(v0.device)                                   # (Vb,)

        # field: p(value | time) for every time bin
        field_mean = v0[:, None] + mu[:, None] * tc[None, :]                # (B,Tb)
        if self.sigma > 0.0:
            field_sd = self.sigma * torch.sqrt(tc.clamp_min(0.0))[None, :].expand(B, -1)
            field_target = self._gauss_bins(field_mean, field_sd)          # (B,Tb,Vb)
        else:
            idx = torch.bucketize(field_mean, self.v_edges.to(v0.device)) - 1
            idx = idx.clamp(0, self.v_bins - 1)
            field_target = torch.zeros(B, self.t_bins, self.v_bins, dtype=torch.float64)
            field_target.scatter_(2, idx[..., None], 1.0)

        # when: p(first-passage time | threshold) for every threshold bin
        a = vc[None, :] - v0[:, None]                                       # (B,Vb) barrier
        already = a <= 0.0                                                  # threshold at/below start
        # a threshold is a genuine timing question only if it sits above the start AND is reached
        # within the horizon (mean crossing a/mu <= t_max); otherwise the answer is trivial
        # (bin 0 if already crossed, last bin if censored) -- excluded from the when metric.
        when_valid = ((a > 0.0) & ((a / mu[:, None]) <= self.t_max)).to(torch.float32)
        if self.sigma > 0.0:
            m = (a / mu[:, None]).clamp_min(1e-6)
            lam = (a * a) / (self.sigma ** 2)
            when_target = self._ig_bins(m.clamp_min(1e-6), lam.clamp_min(1e-9))  # (B,Vb,Tb)
        else:
            t_pass = a / mu[:, None]                                        # (B,Vb)
            jdx = torch.bucketize(t_pass, self.t_edges.to(v0.device)) - 1
            jdx = jdx.clamp(0, self.t_bins - 1)
            when_target = torch.zeros(B, self.v_bins, self.t_bins, dtype=torch.float64)
            when_target.scatter_(2, jdx[..., None], 1.0)
        # thresholds already crossed at t=0 -> one-hot at the first time bin
        when_target[already] = 0.0
        when_target[already, 0] = 1.0

        return DriftBatch(
            obs_v=obs_v.to(torch.float32), obs_t=obs_t.to(torch.float32),
            field_target=field_target.to(torch.float32),
            when_target=when_target.to(torch.float32),
            time_mask=self.time_mask.clone(), thr_mask=self.thr_mask.clone(),
            when_valid=when_valid, v0=v0.to(torch.float32), mu=mu.to(torch.float32),
        )

    # ---- metrics helpers --------------------------------------------------------
    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        """Row-wise entropy in nats (last dim is the distribution). The KL-above-floor
        metric reports CE(model) - H(p_true); this gives H(p_true)."""
        p = probs.clamp_min(1e-12)
        return -(p * p.log()).sum(-1)
