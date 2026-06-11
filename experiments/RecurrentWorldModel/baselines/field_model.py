"""Stage 3 readouts on the continuous-input PoPE trunk.

The single variable under test is the *readout*; the trunk (continuous-input,
PoPE, causal) is identical across both. Both expose the same interface so the
harness compares them directly:

    what_logp(obs_v, obs_t) -> (B, Tb, Vb)   log p(value | time bin)
    when_logp(obs_v, obs_t) -> (B, Vb, Tb)   log p(first-passage time | threshold)

* UnifiedFieldModel -- predicts ONE (time x value) field as a fixed grid from the
  summary; `what` is a column slice, `when` is a PARAMETER-FREE survival read of the
  same field (so the zero-shot-inverse control C1 is testable: train `what` only,
  read `when` cold).
* FunctionalFieldModel -- predicts a latent code `z`, then EVALUATES a learned
  function g(z, coord) at each query coordinate (delivered as positional/Fourier
  features, the strong channel from data point #1) instead of indexing a fixed grid.
  "Predict the rule, evaluate it" -- the readout point 3 found missing for OOD. Same
  `when` survival read, so C1 still applies.
* SeparateHeadsModel -- two query-conditioned MLP heads, no shared field.

The trunk is causal, so the last observation's hidden state has attended to the
whole path; we use it as the dynamics summary.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.fixed_depth import FixedDepthConfig, FixedDepthTransformer


def _trunk(dim: int, n_heads: int, n_layers: int, n_obs: int, v_bins: int) -> FixedDepthTransformer:
    return FixedDepthTransformer(FixedDepthConfig(
        vocab_size=v_bins, dim=dim, n_heads=n_heads, n_layers=n_layers,
        max_seq=n_obs, pos_mode="pope", continuous_input=True))


def _summary(trunk: FixedDepthTransformer, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
    return trunk.encode(obs_v, coord=obs_t)[:, -1]            # (B, dim) -- causal last token


def _survival_when_logp(what_logp: torch.Tensor) -> torch.Tensor:
    """Parameter-free first-passage read from a (B,Tb,Vb) what-field. Exceedance
    c[k,t]=P(value>=threshold_k | t) is increasing in t for positive drift; its
    positive time-increments are the first-passage hazard. Tail mass (never crossed
    within the horizon) is censored onto the last time bin, matching the target."""
    p = what_logp.exp()                                      # (B, Tb, Vb)
    rev = torch.flip(torch.cumsum(torch.flip(p, [-1]), -1), [-1])  # P(value>=v_bin | t)
    c = rev.permute(0, 2, 1)                                 # (B, Vb, Tb) exceedance vs time
    haz = (c[..., 1:] - c[..., :-1]).clamp_min(0.0)          # (B, Vb, Tb-1)
    dist = torch.cat([c[..., :1], haz], dim=-1)              # (B, Vb, Tb)
    dist[..., -1] = dist[..., -1] + (1.0 - c[..., -1]).clamp_min(0.0)   # censor tail
    dist = dist / dist.sum(-1, keepdim=True).clamp_min(1e-9)
    return dist.clamp_min(1e-9).log()


def _fourier(coord: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Positional features of a normalised scalar coordinate: [c, sin/cos(2^k pi c)].
    The strong 'coordinate-as-position' channel (data point #1); valid for OOD coords
    (the function just gets evaluated at an unseen but well-defined input)."""
    feats = [coord.unsqueeze(-1)]
    for k in range(n_freq):
        w = (2.0 ** k) * math.pi
        feats.append(torch.sin(w * coord).unsqueeze(-1))
        feats.append(torch.cos(w * coord).unsqueeze(-1))
    return torch.cat(feats, dim=-1)                          # (..., 1 + 2*n_freq)


class UnifiedFieldModel(nn.Module):
    """One (time x value) field; both queries are reads of it."""

    def __init__(self, dim: int, n_heads: int, n_layers: int, n_obs: int,
                 t_bins: int, v_bins: int) -> None:
        super().__init__()
        self.trunk = _trunk(dim, n_heads, n_layers, n_obs, v_bins)
        self.field = nn.Linear(dim, t_bins * v_bins)
        self.t_bins, self.v_bins = t_bins, v_bins

    def what_logp(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        s = _summary(self.trunk, obs_v, obs_t)               # (B, dim)
        logits = self.field(s).view(-1, self.t_bins, self.v_bins)
        return F.log_softmax(logits, dim=-1)                 # (B, Tb, Vb)

    def when_logp(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        return _survival_when_logp(self.what_logp(obs_v, obs_t))


class SeparateHeadsModel(nn.Module):
    """Two query-conditioned heads; no shared field. The structural baseline for C1
    (a `when` head that was never trained cannot answer -- by construction)."""

    def __init__(self, dim: int, n_heads: int, n_layers: int, n_obs: int,
                 t_bins: int, v_bins: int,
                 t_centers_norm: torch.Tensor, v_centers_norm: torch.Tensor) -> None:
        super().__init__()
        self.trunk = _trunk(dim, n_heads, n_layers, n_obs, v_bins)
        self.t_embed = nn.Linear(1, dim)                     # query-time -> dim
        self.v_embed = nn.Linear(1, dim)                     # query-threshold -> dim
        self.what_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, v_bins))
        self.when_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, t_bins))
        self.register_buffer("t_centers", t_centers_norm.view(1, -1, 1))   # (1, Tb, 1)
        self.register_buffer("v_centers", v_centers_norm.view(1, -1, 1))   # (1, Vb, 1)

    def what_logp(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        s = _summary(self.trunk, obs_v, obs_t)[:, None, :]   # (B, 1, dim)
        q = self.t_embed(self.t_centers)                     # (1, Tb, dim)
        return F.log_softmax(self.what_mlp(s + q), dim=-1)   # (B, Tb, Vb)

    def when_logp(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        s = _summary(self.trunk, obs_v, obs_t)[:, None, :]   # (B, 1, dim)
        q = self.v_embed(self.v_centers)                     # (1, Vb, dim)
        return F.log_softmax(self.when_mlp(s + q), dim=-1)   # (B, Vb, Tb)


class FunctionalFieldModel(nn.Module):
    """Predict a latent `z` from the path, then EVALUATE a learned g(z, coord) at each
    (time, value) cell -- the coordinates delivered as Fourier/positional features.

    Unlike UnifiedFieldModel (a fixed grid read off the summary), an unobserved time is
    answered by evaluating the *same* g at that time's coordinate, so extrapolation is a
    function evaluation rather than a memorised output slot. Still one shared field, so
    `when` is the same parameter-free survival read and C1 applies. Bitter-lesson clean:
    g is a generic MLP -- nothing assumes the dynamics are linear."""

    def __init__(self, dim: int, n_heads: int, n_layers: int, n_obs: int,
                 t_bins: int, v_bins: int,
                 t_centers_norm: torch.Tensor, v_centers_norm: torch.Tensor,
                 n_freq: int = 6, hidden: int | None = None) -> None:
        super().__init__()
        self.trunk = _trunk(dim, n_heads, n_layers, n_obs, v_bins)
        self.t_bins, self.v_bins = t_bins, v_bins
        hidden = hidden or dim
        cf = 1 + 2 * n_freq                                  # per-axis coordinate-feature dim
        # fixed (Tb*Vb) grid of [time-coord-feats || value-coord-feats]
        tf = _fourier(t_centers_norm, n_freq)                # (Tb, cf)
        vf = _fourier(v_centers_norm, n_freq)                # (Vb, cf)
        grid = torch.cat([tf[:, None, :].expand(t_bins, v_bins, cf),
                          vf[None, :, :].expand(t_bins, v_bins, cf)], dim=-1)
        self.register_buffer("grid_feat", grid.reshape(t_bins * v_bins, 2 * cf))  # (Tb*Vb, 2cf)
        self.g = nn.Sequential(
            nn.Linear(dim + 2 * cf, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def what_logp(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        z = _summary(self.trunk, obs_v, obs_t)               # (B, dim)
        n = self.grid_feat.shape[0]
        x = torch.cat([z[:, None, :].expand(-1, n, -1),
                       self.grid_feat[None].expand(z.shape[0], -1, -1)], dim=-1)
        logits = self.g(x).squeeze(-1).view(-1, self.t_bins, self.v_bins)
        return F.log_softmax(logits, dim=-1)                 # (B, Tb, Vb)

    def when_logp(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        return _survival_when_logp(self.what_logp(obs_v, obs_t))
