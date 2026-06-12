"""Fixed-depth transformer baseline -- the control for the Stage 0 gate.

Same block architecture as the settling core (so the comparison is fair: identical
attention/FFN/norm), but ``n_layers`` **distinct** layers applied **once**, with no
iteration and no DEQ. This isolates the single variable under test: depth-via-
iteration (settling core, one shared block) vs depth-via-distinct-layers (this),
at a matched parameter budget.

Implementation note: we reuse ``SettlingBlock`` as a plain transformer layer by
feeding a zero injection (``x = 0``), so ``f(h, 0)`` is an ordinary pre-norm block.
Each layer is its own instance with its own weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.block import RMSNorm, SettlingBlock, SettlingBlockConfig
from core.model import count_parameters
from baselines.bottleneck import ActivationFFN


@dataclass
class FixedDepthConfig:
    vocab_size: int = 17
    dim: int = 256
    n_heads: int = 4
    n_layers: int = 6
    max_seq: int = 64
    tie_head: bool = True
    emb_scale: float = 1.0
    pos_mode: str = "pope"   # match the settling model's positional scheme
    residual_gate: bool = False
    gate_init: float = 0.1
    n_axes: int = 1          # PoPE coordinate axes (continuous-time / multi-axis)
    time_input: bool = False  # add a learned projection of log(1+time) to the embedding
                              # (the "time as content" control vs "time as position")
    continuous_input: bool = False  # input is REAL values (B,T) projected via Linear(1,dim),
                                    # not tokens -- for Δ-encoding / numeric-sequence tasks
    inject_act: str = "none"  # insert ONE ActivationFFN sublayer (TBAF drift test); one of
                              # none|gelu|tbaf|tbaf_verbatim|commonmode
    inject_layer: int = -1    # after which layer to inject (-1 -> n_layers//2)


class FixedDepthTransformer(nn.Module):
    def __init__(self, cfg: FixedDepthConfig) -> None:
        super().__init__()
        self.cfg = cfg
        block_cfg = SettlingBlockConfig(
            dim=cfg.dim, n_heads=cfg.n_heads, causal=True, max_seq=cfg.max_seq,
            pos_enc=cfg.pos_mode if cfg.pos_mode in ("rope", "pope") else "none",
            residual_gate=cfg.residual_gate, gate_init=cfg.gate_init, n_axes=cfg.n_axes,
        )
        # input: real values (continuous) or a token table
        self.input_proj = nn.Linear(1, cfg.dim) if cfg.continuous_input else None
        self.embed = None if cfg.continuous_input else nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = (nn.Parameter(torch.randn(1, cfg.max_seq, cfg.dim) * 0.02)
                    if cfg.pos_mode == "learned" else None)
        self.layers = nn.ModuleList([SettlingBlock(block_cfg) for _ in range(cfg.n_layers)])
        # optional one-shot activation sublayer (the TBAF drift test); default off
        self.inject = (ActivationFFN(cfg.dim, cfg.inject_act) if cfg.inject_act != "none" else None)
        self.inject_layer = cfg.inject_layer if cfg.inject_layer >= 0 else cfg.n_layers // 2
        self.norm_out = RMSNorm(cfg.dim, block_cfg.rmsnorm_eps)
        # continuous input can't tie the head to a (nonexistent) embedding
        self.head_proj = (nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
                          if (not cfg.tie_head or cfg.continuous_input) else None)
        self.time_proj = nn.Linear(1, cfg.dim) if cfg.time_input else None
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, tokens: torch.Tensor, coord: torch.Tensor | None = None,
               time_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Trunk only: returns the normed hidden state (b, t, dim), pre-head. Custom
        readouts (e.g. the Stage-3 field heads) build on this instead of the logits."""
        t = tokens.shape[1]
        if self.input_proj is not None:                       # continuous real-valued input
            h = self.input_proj(tokens.float().unsqueeze(-1))
        else:
            h = self.embed(tokens) * self.cfg.emb_scale
        if self.pos is not None:
            h = h + self.pos[:, :t]
        if self.time_proj is not None and time_feat is not None:
            h = h + self.time_proj(torch.log1p(time_feat.clamp_min(0)).unsqueeze(-1))
        zero = torch.zeros_like(h)
        for i, layer in enumerate(self.layers):
            h = layer(h, zero, coord=coord)
            if self.inject is not None and i == self.inject_layer:
                h = h + self.inject(h)                         # one-shot activation sublayer
        return self.norm_out(h)

    def forward(self, tokens: torch.Tensor, coord: torch.Tensor | None = None,
                time_feat: torch.Tensor | None = None) -> torch.Tensor:
        """``coord`` (b, t[, n_axes]) feeds continuous-time / multi-axis PoPE (None ->
        integer positions). ``time_feat`` (b, t) optionally adds log(1+time) to the
        embedding -- the 'time as content' control vs PoPE's 'time as position'."""
        h = self.encode(tokens, coord, time_feat)
        if self.head_proj is None:
            return F.linear(h, self.embed.weight)
        return self.head_proj(h)


def matched_baseline(
    target_params: int,
    *,
    vocab_size: int,
    n_heads: int = 4,
    n_layers: int = 6,
    max_seq: int = 64,
    tie_head: bool = True,
    pos_mode: str = "pope",
    residual_gate: bool = False,
    gate_init: float = 0.1,
    max_dim_mult: int = 8,
) -> tuple[FixedDepthTransformer, FixedDepthConfig, int]:
    """Build an ``n_layers``-layer transformer whose param count is closest to
    ``target_params`` (the settling model's count), sweeping width.

    Returns (model, config, param_count). Lets the Stage 0 comparison hold the
    parameter budget fixed and vary only how it is spent (iteration vs layers).
    """
    best: tuple[FixedDepthTransformer, FixedDepthConfig, int] | None = None
    best_gap = float("inf")
    # sweep dims that are multiples of n_heads up to a generous ceiling
    base = max(n_heads, 8)
    ceiling = base * max_dim_mult * 8
    for dim in range(base, ceiling + 1, n_heads):
        if pos_mode == "rope" and (dim // n_heads) % 2 != 0:
            continue  # RoPE needs an even head_dim (PoPE does not)
        cfg = FixedDepthConfig(
            vocab_size=vocab_size, dim=dim, n_heads=n_heads,
            n_layers=n_layers, max_seq=max_seq, tie_head=tie_head, pos_mode=pos_mode,
            residual_gate=residual_gate, gate_init=gate_init,
        )
        model = FixedDepthTransformer(cfg)
        n = count_parameters(model)
        gap = abs(n - target_params)
        if gap < best_gap:
            best_gap, best = gap, (model, cfg, n)
        if n > target_params and best_gap < target_params:
            break  # past the target; closest already found
    assert best is not None
    return best
