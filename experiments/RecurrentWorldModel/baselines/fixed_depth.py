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


class FixedDepthTransformer(nn.Module):
    def __init__(self, cfg: FixedDepthConfig) -> None:
        super().__init__()
        self.cfg = cfg
        block_cfg = SettlingBlockConfig(
            dim=cfg.dim, n_heads=cfg.n_heads, causal=True, max_seq=cfg.max_seq,
            pos_enc=cfg.pos_mode if cfg.pos_mode in ("rope", "pope") else "none",
            residual_gate=cfg.residual_gate, gate_init=cfg.gate_init,
        )
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = (nn.Parameter(torch.randn(1, cfg.max_seq, cfg.dim) * 0.02)
                    if cfg.pos_mode == "learned" else None)
        self.layers = nn.ModuleList([SettlingBlock(block_cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.dim, block_cfg.rmsnorm_eps)
        self.head_proj = None if cfg.tie_head else nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        h = self.embed(tokens) * self.cfg.emb_scale
        if self.pos is not None:
            h = h + self.pos[:, :t]
        zero = torch.zeros_like(h)
        for layer in self.layers:
            h = layer(h, zero)
        h = self.norm_out(h)
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
