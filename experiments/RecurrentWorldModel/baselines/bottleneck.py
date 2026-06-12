"""Swappable activations for the TBAF drift test + a one-shot FFN sublayer to inject them.

The point of the experiment is whether a single "common-mode-rejecting" activation reduces
free-running rollout drift. All activations operate on (B, T, H) and are parameter-free.

* TBAFPerToken    -- the CORRECTED Triangle-Based op: per token, split H into groups of 3
  and emit the 3 pairwise |differences| (a V/"triangle" shape). |a-b| is invariant to a
  common shift of a,b -> common-mode rejection at the channel-pair level (the principle, =
  delta-encoding / Prediction P1 at the activation level). Dim-preserving.
* TBAFVerbatim    -- the repo code as written (intcomp note / Skull18500/TBAF): distances
  taken over the FLATTENED batch*time axis and broadcast back, so every position gets the
  SAME vector and it depends on the rest of the batch. Almost certainly a bug; included as a
  CONTROL to test the "it 'works' only by going near-constant" critique. Batch-coupled.
* CommonMode      -- subtract the per-token mean (linear common-mode removal). Isolates the
  invariance from the triangle nonlinearity.
* GELU            -- the standard activation; the capacity-matched baseline for "does TBAF
  beat a normal activation in the SAME inserted sublayer".

ActivationFFN wraps any of them as one residual FFN sublayer (norm -> Linear -> act ->
Linear), inserted ONCE in the trunk -- faithful to the repo's "used only once" finding and
to the agenda's "does it beat SwiGLU there" framing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.block import RMSNorm


class TBAFPerToken(nn.Module):
    """Per-token pairwise |differences| of channel triples (the corrected, principled op)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        v = x.reshape(B, T, H // 3, 3)
        a, b, c = v[..., 0], v[..., 1], v[..., 2]            # (B, T, H/3)
        out = torch.stack([(a - b).abs(), (a - c).abs(), (b - c).abs()], dim=-1)
        return out.reshape(B, T, H)


class TBAFVerbatim(nn.Module):
    """The repo op verbatim: distances over the flattened batch*time axis, broadcast to all
    positions (so every token gets the same, batch-dependent vector). Control only."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        n = B * T
        pts = x.reshape(n, H // 3, 3).permute(1, 2, 0)       # (groups, 3, n)
        d01 = torch.norm(pts[:, 0] - pts[:, 1], dim=-1, keepdim=True)   # (groups, 1)
        d02 = torch.norm(pts[:, 0] - pts[:, 2], dim=-1, keepdim=True)
        d12 = torch.norm(pts[:, 1] - pts[:, 2], dim=-1, keepdim=True)
        dists = torch.cat([d01, d02, d12], dim=-1).reshape(-1)          # (H,)
        return dists[None, None].expand(B, T, H)                        # same everywhere


class CommonMode(nn.Module):
    """Remove the per-token mean -- pure common-mode rejection, no triangle nonlinearity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)


def make_activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "tbaf": TBAFPerToken(), "tbaf_verbatim": TBAFVerbatim(),
            "commonmode": CommonMode()}[name]


class ActivationFFN(nn.Module):
    """One residual FFN sublayer with a swappable activation -- inserted once in the trunk.
    hidden is forced to a multiple of 3 (TBAF needs it); all arms share the same shape so the
    only difference between them is the activation."""

    def __init__(self, dim: int, act_name: str, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 3 * dim
        assert hidden % 3 == 0, "hidden must be divisible by 3 for TBAF"
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden)
        self.act = make_activation(act_name)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(self.norm(x))))
