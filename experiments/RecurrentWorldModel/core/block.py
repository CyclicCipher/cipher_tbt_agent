"""The weight-shared relational settling block ``f_theta``.

One block, applied repeatedly by the DEQ wrapper (``deq.py``) to refine a latent
state ``h`` toward an equilibrium ``h* = f_theta(h*, x)``. ``x`` is the *input
injection* (the clamped sensory drive), held constant across the inner loop and
re-added every iteration -- this is what makes the loop a fixed-point solve over
a fixed input rather than a feed-forward stack.

Design notes (see Docs/architecture.md §2.2):
  * Pre-norm transformer block: RMSNorm -> attention -> RMSNorm -> SwiGLU FFN.
  * QK-Norm on attention (stabilizes logits in deep recurrence; also the natural
    place to control the magnitude channel of the polar split later).
  * The polar hyle/morphe (magnitude/phase) split is a FLAG, default OFF.
    Per the implementation plan: start without it; add it once the loop is
    stable, so convergence debugging is not entangled with the relational
    machinery.

Nothing here trains. This is the operator only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SettlingBlockConfig:
    dim: int = 256
    n_heads: int = 4
    ffn_mult: float = 8.0 / 3.0  # SwiGLU keeps ~2/3 to match param count of 4x MLP
    causal: bool = True          # Stage 0 text LM uses causal attention
    qk_norm: bool = True
    rmsnorm_eps: float = 1e-5
    inject_scale: float = 1.0    # weight on the constant input injection each step
    polar_split: bool = False    # hyle/morphe magnitude/phase split (Stage 1+; OFF by default)
    rope: bool = False           # rotary (relative) position in attention -- extrapolates
    max_seq: int = 64            # for the RoPE cos/sin tables
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        if self.rope and (self.dim // self.n_heads) % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim; got {self.dim // self.n_heads}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN (item 13). hidden ~= ffn_mult * dim, rounded to a multiple of 8."""

    def __init__(self, dim: int, ffn_mult: float) -> None:
        super().__init__()
        hidden = int(round(ffn_mult * dim))
        hidden = ((hidden + 7) // 8) * 8
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE (item 3 / the rotary core of PoPE): rotate Q,K by position so the
    attention score depends on *relative* position. Unlike learned absolute
    position embeddings, this extrapolates to sequence lengths/positions never
    seen in training -- the fix for the Stage-0 OOD positional confound.

    Full PoPE (the magnitude=content / phase=position decoupling, repr §3b) is a
    further step flagged open (Q11); this is the rotary part that PoPE generalizes.
    """

    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0) -> None:
        super().__init__()
        theta = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(pos, theta)              # (max_seq, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)      # (max_seq, head_dim)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)  # (1,1,S,hd)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = q.shape[-2]
        cos, sin = self.cos[..., :t, :], self.sin[..., :t, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    """Multi-head self-attention with optional QK-Norm and RoPE.

    QK-Norm (item 56): L2-normalize per-head queries and keys, then scale by a
    learned temperature. Stabilizes attention logits across many settling
    iterations (the §1c stability hazard) and bounds the magnitude channel.
    RoPE applies *after* QK-Norm (rotation preserves norm, so the two compose).
    """

    def __init__(self, cfg: SettlingBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        if cfg.qk_norm:
            # learned per-head temperature; init so effective scale ~= 1/sqrt(head_dim)
            self.q_scale = nn.Parameter(torch.zeros(cfg.n_heads))
            self.k_scale = nn.Parameter(torch.zeros(cfg.n_heads))
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq, cfg.rope_base) if cfg.rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(self.cfg.dim, dim=-1)
        # (b, h, t, d)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.cfg.qk_norm:
            q = F.normalize(q, dim=-1) * F.softplus(self.q_scale).view(1, -1, 1, 1)
            k = F.normalize(k, dim=-1) * F.softplus(self.k_scale).view(1, -1, 1, 1)
            scale = 1.0  # normalization already sets the scale
        else:
            scale = self.head_dim ** -0.5

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.cfg.causal, scale=scale
        )
        attn = attn.transpose(1, 2).reshape(b, t, self.cfg.dim)
        return self.out(attn)


class SettlingBlock(nn.Module):
    """``h_next = f_theta(h, x)`` -- one iteration of the settling loop.

    The same instance is called repeatedly by the DEQ wrapper. ``x`` (the input
    injection) is added each call so the fixed point depends on the input.
    """

    def __init__(self, cfg: SettlingBlockConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm_attn = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.attn = Attention(cfg)
        self.norm_ffn = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_mult)
        if cfg.polar_split:
            # placeholder for the magnitude/phase relational channel (Stage 1+).
            # Intentionally not implemented yet; flag exists so the wiring is in
            # place. See Docs/architecture.md §2.2 and representation_learning §3b.
            raise NotImplementedError(
                "polar_split is Stage 1+; keep it OFF until the settling loop is stable."
            )

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # constant input injection (clamped sensory drive)
        h = h + self.cfg.inject_scale * x
        h = h + self.attn(self.norm_attn(h))
        h = h + self.ffn(self.norm_ffn(h))
        return h

    @torch.no_grad()
    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        """A zero starting state shaped like the injection ``x``."""
        return torch.zeros_like(x)
