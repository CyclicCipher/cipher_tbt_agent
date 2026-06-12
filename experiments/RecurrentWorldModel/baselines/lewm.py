"""LeWorldModel (LeWM) for DriftField -- a latent world model that rolls dynamics forward.

Faithful to Maes/Le Lidec/LeCun/Balestriero, arXiv:2603.19312 (code: lucas-maes/le-wm):
  * Encoder        z_t = enc(o_t)              -- per-observation embedder (no temporal mixing)
  * Predictor      z_hat_{t+1} = pred(z_t, a_t) -- autoregressive transformer, the ACTION a_t
                   conditions every block via AdaLN-zero (DiT-style), exactly as le-wm's
                   ARPredictor / ConditionalBlock.
  * Loss           L = MSE(z_hat_{t+1}, z_{t+1}.detach()) + lambda * SIGReg(Z)
                   (target embedding is DETACHED, per le-wm/jepa.py; SIGReg prevents collapse,
                   so no EMA / teacher is needed).
  * Rollout        autoregressive -- compose the predictor over a sequence of actions.

DriftField mapping: o_i = value v_i, action a_i = elapsed time Delta_t to the next
observation. Reaching an OOD future time is done by COMPOSING in-distribution time-steps
(latent rollout) -- the model never extrapolates its inputs, the mechanism the supervised
readouts lacked. Positions use continuous-time rotary (our PoPE channel; data point #1).
A small decoder z -> value-bin distribution provides the what/when readout for the metric.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.bottleneck import ActivationFFN
from baselines.field_model import _survival_when_logp
from baselines.sigreg import MultiSubspaceSIGReg, SIGReg


def _rotary(x: torch.Tensor, t: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Continuous-time rotary on (B, H, K, hd) using real time coordinate t (B, K).
    The PoPE/continuous-coordinate positional channel (the data-point-#1 winner)."""
    *_, hd = x.shape
    half = hd // 2
    freqs = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = t[:, None, :, None] * freqs                       # (B, 1, K, half)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ConditionalBlock(nn.Module):
    """Causal self-attention + MLP, with AdaLN-zero conditioning on the per-step action."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)               # AdaLN-zero: start as identity
        nn.init.zeros_(self.adaLN[-1].bias)

    def _attn(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, K, dim = h.shape
        H, hd = self.heads, dim // self.heads
        qkv = self.qkv(h).view(B, K, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B, H, K, hd)
        q, k = _rotary(q, t), _rotary(k, t)
        mask = torch.triu(torch.full((K, K), float("-inf"), device=h.device), 1)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask
        out = att.softmax(-1) @ v                           # (B, H, K, hd)
        return self.proj(out.transpose(1, 2).reshape(B, K, dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + g1 * self._attn(self.norm1(x) * (1 + sc1) + sh1, t)
        x = x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)
        return x


class ARPredictor(nn.Module):
    """z_hat_{i+1} = pred(z_i, a_i): predicts the NEXT latent at each position; the action
    a_i (the time-gap to the predicted step) conditions every block via AdaLN-zero."""

    def __init__(self, dim: int, heads: int, depth: int, inject_act: str = "none") -> None:
        super().__init__()
        self.act_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([ConditionalBlock(dim, heads) for _ in range(depth)])
        # the TBAF drift test: one common-mode-rejecting activation sublayer in the iterated
        # latent path (default off). This is where rollout drift actually accumulates.
        self.inject = ActivationFFN(dim, inject_act) if inject_act != "none" else None
        self.inject_layer = depth // 2
        self.head = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        c = self.act_embed(a.unsqueeze(-1))                 # (B, K, dim)
        x = z
        for i, blk in enumerate(self.blocks):
            x = blk(x, t, c)
            if self.inject is not None and i == self.inject_layer:
                x = x + self.inject(x)
        return self.head(x)                                 # (B, K, dim) predicted next latents


class LeWorldModel(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, depth: int = 4, v_bins: int = 32,
                 v_min: float = 0.0, v_max: float = 64.0, num_proj: int = 256,
                 lam: float = 1.0, reg: str = "subjepa", num_subspaces: int = 32,
                 inject_act: str = "none") -> None:
        super().__init__()
        self.v_bins, self.v_min, self.v_max, self.lam = v_bins, v_min, v_max, lam
        self.enc = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.predictor = ARPredictor(dim, heads, depth, inject_act=inject_act)
        self.dec = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, v_bins))
        # default: Sub-JEPA subspace regularizer; "sigreg" = the ambient SIGReg it improves on
        self.reg_kind = reg
        if reg == "subjepa":
            self.reg = MultiSubspaceSIGReg(embed_dim=dim, num_subspaces=num_subspaces, num_proj=num_proj)
        elif reg == "sigreg":
            self.reg = SIGReg(n_slices=num_proj)
        else:
            raise ValueError(f"reg must be 'subjepa' or 'sigreg', got {reg!r}")

    # ---- core ------------------------------------------------------------------
    def encode(self, obs_v: torch.Tensor) -> torch.Tensor:
        return self.enc(obs_v.unsqueeze(-1))                # (B, K, dim)

    def _actions(self, obs_t: torch.Tensor) -> torch.Tensor:
        """a_i = gap from t_i to t_{i+1} (last position's action is a copy -- unused by the loss)."""
        a = torch.zeros_like(obs_t)
        a[:, :-1] = obs_t[:, 1:] - obs_t[:, :-1]
        a[:, -1] = a[:, -2] if obs_t.shape[1] > 1 else 0.0
        return a

    def losses(self, obs_v: torch.Tensor, obs_t: torch.Tensor) -> dict:
        z = self.encode(obs_v)                              # (B, K, d)
        a = self._actions(obs_t)
        pred = self.predictor(z, obs_t, a)                  # (B, K, d) predicted next
        mse = F.mse_loss(pred[:, :-1], z[:, 1:].detach())   # detached target (le-wm)
        # subjepa expects (B, T, D); ambient SIGReg expects a flat (N, D) embedding batch
        sig = self.reg(z) if self.reg_kind == "subjepa" else self.reg(z.reshape(-1, z.shape[-1]))
        # decoder probe: trained on BOTH encoder latents AND predicted latents, so the eval
        # (which decodes rolled-out predicted latents) is on-manifold -- the bug-fix.
        edges = torch.linspace(self.v_min, self.v_max, self.v_bins + 1, device=obs_v.device)
        tgt = (torch.bucketize(obs_v, edges) - 1).clamp(0, self.v_bins - 1)
        dec_enc = F.cross_entropy(self.dec(z).reshape(-1, self.v_bins), tgt.reshape(-1))
        dec_pred = F.cross_entropy(self.dec(pred[:, :-1]).reshape(-1, self.v_bins),
                                   tgt[:, 1:].reshape(-1))   # decode the NEXT value from pred
        dec = 0.5 * (dec_enc + dec_pred)
        total = mse + self.lam * sig + dec
        return {"total": total, "mse": mse, "sigreg": sig, "decode": dec}

    # ---- evaluation: autoregressive rollout over the query-time grid ------------
    @torch.no_grad()
    def rollout_what(self, obs_v: torch.Tensor, obs_t: torch.Tensor,
                     t_centers: torch.Tensor, t_obs: float) -> torch.Tensor:
        """Value distribution at each ascending time-bin centre. WITHIN the observation
        window (tc <= t_obs): decode the encoder latent of the nearest preceding observation
        (causal read; the decoder's encoder-latent manifold). BEYOND it (OOD): forward-only
        autoregressive rollout -- reach the future by COMPOSING in-distribution steps. Both
        keep actions positive and decode on-manifold (the eval bug-fixes)."""
        z = self.encode(obs_v)                              # (B, K, d)
        B, dev = obs_v.shape[0], obs_v.device
        rows = torch.arange(B, device=dev)
        zh, th = z, obs_t                                   # history grows only for OOD steps
        logps = []
        for tc in t_centers.tolist():
            tcb = torch.full((B,), tc, device=dev)
            if tc <= t_obs:                                 # in-window: nearest preceding obs
                idx = ((obs_t <= tc).sum(1).clamp(min=1) - 1)
                zc = z[rows, idx]                           # (B, d)
                logps.append(F.log_softmax(self.dec(zc), dim=-1))
            else:                                           # OOD: forward autoregressive step
                a = th.clone()
                if th.shape[1] > 1:
                    a[:, :-1] = th[:, 1:] - th[:, :-1]
                a[:, -1] = tcb - th[:, -1]                  # positive: tc > t_obs >= last obs
                znew = self.predictor(zh, th, a)[:, -1]
                logps.append(F.log_softmax(self.dec(znew), dim=-1))
                zh = torch.cat([zh, znew[:, None]], dim=1)
                th = torch.cat([th, tcb[:, None]], dim=1)
        return torch.stack(logps, dim=1)                    # (B, Tb, v_bins)

    @torch.no_grad()
    def what_when(self, obs_v, obs_t, t_centers, t_obs):
        what = self.rollout_what(obs_v, obs_t, t_centers, t_obs)
        return what, _survival_when_logp(what)              # when via the shared survival read
