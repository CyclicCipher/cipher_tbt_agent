"""``SettlingLM`` -- the Stage 0 model: token adapter -> settling core -> head.

This is the thin wrapper that turns the settling operator (``SettlingBlock`` run
to equilibrium by ``DEQFixedPoint``) into a language model we can train and probe:

    tokens --embed+pos--> x (the clamped input injection)
                          |
                          v   (DEQ settles the free state to an attractor)
                       h* = f_theta(h*, x)
                          |
                       RMSNorm -> head -> logits

Deep supervision (the HRM data-efficiency device that survives the hierarchy
demotion): run the core for ``n_supervision_segments`` segments, apply the loss
after each, detaching the equilibrium between segments so each segment carries a
local (one-step) gradient. With one segment this is a plain DEQ forward.

No reconstructive decoder. Text only at Stage 0. Nothing here trains -- the loss
and optimizer live in ``train_stage0.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import RMSNorm, SettlingBlock, SettlingBlockConfig
from .deq import DEQConfig, DEQFixedPoint, FixedPointInfo


@dataclass
class SettlingLMConfig:
    vocab_size: int = 17
    dim: int = 256
    n_heads: int = 4
    max_seq: int = 64
    n_supervision_segments: int = 1
    tie_head: bool = True
    emb_scale: float = 1.0
    pos_mode: str = "pope"  # "pope" | "rope" | "learned" (absolute)
    warm_start: str = "zeros"  # "zeros" | "input" | "proposal" (settle init; Solve-the-Loop)
    n_warm: int = 2            # block steps for warm_start="proposal"
    residual_gate: bool = False  # LayerScale contraction gate (see SettlingBlockConfig)
    gate_init: float = 0.1
    block: SettlingBlockConfig = field(default=None)  # filled in __post_init__
    deq: DEQConfig = field(default_factory=DEQConfig)

    def __post_init__(self) -> None:
        if self.pos_mode not in ("pope", "rope", "learned"):
            raise ValueError(f"pos_mode must be pope|rope|learned, got {self.pos_mode!r}")
        if self.warm_start not in ("zeros", "input", "proposal"):
            raise ValueError(f"warm_start must be zeros|input|proposal, got {self.warm_start!r}")
        if self.block is None:
            self.block = SettlingBlockConfig(
                dim=self.dim, n_heads=self.n_heads, causal=True, max_seq=self.max_seq,
                pos_enc=self.pos_mode if self.pos_mode in ("rope", "pope") else "none",
                residual_gate=self.residual_gate, gate_init=self.gate_init,
            )


class SettlingLM(nn.Module):
    def __init__(self, cfg: SettlingLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        # absolute position table only in "learned" mode; rope/pope carry position in attn
        self.pos = (nn.Parameter(torch.randn(1, cfg.max_seq, cfg.dim) * 0.02)
                    if cfg.pos_mode == "learned" else None)
        self.block = SettlingBlock(cfg.block)
        self.deq = DEQFixedPoint(self.block, cfg.deq)
        self.norm_out = RMSNorm(cfg.dim, cfg.block.rmsnorm_eps)
        if cfg.tie_head:
            self.head_proj = None  # uses embed.weight
        else:
            self.head_proj = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _inject(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        if t > self.cfg.max_seq:
            raise ValueError(f"sequence length {t} exceeds max_seq {self.cfg.max_seq}")
        x = self.embed(tokens) * self.cfg.emb_scale
        if self.pos is not None:
            x = x + self.pos[:, :t]
        return x

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_out(h)
        if self.head_proj is None:
            return F.linear(h, self.embed.weight)
        return self.head_proj(h)

    def _warm_h0(self, x: torch.Tensor) -> torch.Tensor:
        """Settle-init (Solve-the-Loop warm-start). 'zeros' = cold start; 'input' =
        start from the embedded input; 'proposal' = a cheap no-grad few-step forward
        as a coherent starting point, which converges faster and more stably."""
        mode = self.cfg.warm_start
        if mode == "input":
            return x
        if mode == "proposal":
            with torch.no_grad():
                h = torch.zeros_like(x)
                for _ in range(self.cfg.n_warm):
                    h = self.deq._step(h, x)
            return h
        return torch.zeros_like(x)

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[FixedPointInfo]]:
        """Return (final logits, per-segment logits, per-segment fixed-point info)."""
        x = self._inject(tokens)
        h = self._warm_h0(x)
        seg_logits: list[torch.Tensor] = []
        infos: list[FixedPointInfo] = []
        for _ in range(max(1, self.cfg.n_supervision_segments)):
            h_star, info = self.deq(x, h0=h.detach())
            seg_logits.append(self._logits(h_star))
            infos.append(info)
            h = h_star
        return seg_logits[-1], seg_logits, infos


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (tied weights counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
