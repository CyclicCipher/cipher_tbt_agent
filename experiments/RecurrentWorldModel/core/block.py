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
    pos_enc: str = "none"        # "none" | "rope" | "pope" (positional scheme in attention)
    max_seq: int = 64            # for the position cos/sin tables
    pos_base: float = 10000.0
    n_axes: int = 1              # PoPE coordinate axes (1=time/position; >1 for 2D/3D+time)
    residual_gate: bool = False  # LayerScale/ReZero gate on attn+ffn residuals (contraction lever)
    gate_init: float = 0.1       # small init -> map ~ norm(h + input), strongly contractive

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} not divisible by n_heads {self.n_heads}")
        if self.pos_enc not in ("none", "rope", "pope"):
            raise ValueError(f"pos_enc must be none|rope|pope, got {self.pos_enc!r}")
        # RoPE rotates feature PAIRS, so it needs an even head_dim; PoPE is per-feature
        # (each scalar -> its own (cos,sin) plane) and has no parity requirement.
        if self.pos_enc == "rope" and (self.dim // self.n_heads) % 2 != 0:
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
    """RoPE: rotate Q,K *pairs* by position so the score depends on relative
    position. Extrapolates better than learned absolute positions, but the rotary
    inner product entangles content and position (the cos(phi_k - phi_q) cross-term
    in the score). PoPE below removes that. Kept as an available control.
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


class PolarPositionalEmbedding(nn.Module):
    """PoPE -- Polar Coordinate Positional Embeddings (Gopalakrishnan et al. 2025,
    arXiv:2509.10534). The clean realization of the §3b hyle/morphe split:
    **content lives entirely in magnitude, position entirely in phase**, so the
    score factorizes as (what-match) x (where-match) with NO cross-term.

        magnitude (content):  mu = softplus(x)          -- position-independent
        phase    (position):  phi_c = pos * theta_c     -- content-independent
        polar form:           [mu*cos(phi), mu*sin(phi)]  (head_dim -> 2*head_dim)
        score:  q~ . k~ = sum_c mu_q,c mu_k,c cos((s - t) theta_c + delta_c)

    Each scalar feature gets its own (cos,sin) plane (no pairing, no parity needed),
    so the rotation leaves magnitude exactly invariant -- by construction, not a
    learned penalty. delta_c is the paper's optional per-frequency phase bias
    (in [-2pi, 0]); applied to keys, it is the only residual what-where coupling.
    Cost: the QK score is computed in 2*head_dim (the paper's "d frequencies").
    The what/where decoupling is a RESOLVED question -- see Docs/architecture.md §2.2.

    Coordinate-driven & multi-axis: the phase is computed from a per-position REAL
    coordinate passed at forward time (not a fixed integer table), so the "where" can
    be continuous time (irregular event streams) -- the score then depends on relative
    *elapsed time* (t_i - t_j), unbounded-safe since only differences matter and the
    phase wraps. With ``n_axes > 1`` the head_dim is split into per-axis bands, each
    driven by its own coordinate (e.g. row/col for vision, or space + time). Default
    coordinate = integer positions, recovering standard PoPE.
    """

    def __init__(self, head_dim: int, max_seq: int, base: float = 10000.0, n_axes: int = 1) -> None:
        super().__init__()
        if head_dim % n_axes != 0:
            raise ValueError(f"head_dim {head_dim} not divisible by n_axes {n_axes}")
        self.n_axes = n_axes
        self.max_seq = max_seq
        per = head_dim // n_axes                                        # freqs per axis
        inv_freq = base ** (-torch.arange(0, per, dtype=torch.float32) / per)  # (per,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.delta = nn.Parameter(torch.zeros(head_dim))  # learnable phase bias (keys)

    def _phase(self, coord: torch.Tensor) -> torch.Tensor:
        """coord (b, t, n_axes) real -> phase (b, t, head_dim)."""
        ph = coord.unsqueeze(-1) * self.inv_freq      # (b, t, n_axes, per)
        return ph.flatten(-2)                         # (b, t, head_dim)

    def encode(self, x: torch.Tensor, is_key: bool, coord: torch.Tensor | None = None) -> torch.Tensor:
        """(b, h, t, head_dim) raw features -> (b, h, t, 2*head_dim) polar form.
        ``coord`` is (b, t) or (b, t, n_axes) real positions; None -> integer positions."""
        b, h, t, _ = x.shape
        if coord is None:
            idx = torch.arange(t, device=x.device, dtype=torch.float32)
            coord = idx[None, :, None].expand(b, t, self.n_axes)
        elif coord.dim() == 2:
            coord = coord.unsqueeze(-1)               # (b, t) -> (b, t, 1)
        phase = self._phase(coord.to(torch.float32))  # (b, t, head_dim)
        cos, sin = phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)   # (b, 1, t, head_dim)
        if is_key:                                    # rotate key phase by delta -> cos(phi+delta)
            cd, sd = torch.cos(self.delta), torch.sin(self.delta)
            cos, sin = cos * cd - sin * sd, sin * cd + cos * sd
        mu = F.softplus(x)                            # magnitude = content
        return torch.cat((mu * cos, mu * sin), dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with QK-Norm and an optional positional scheme.

    QK-Norm (item 56): L2-normalize per-head queries and keys, then scale by a
    learned temperature. Stabilizes attention logits across many settling
    iterations (the §1c stability hazard) and bounds the magnitude channel.

    Positional scheme (cfg.pos_enc):
      * "rope" -- rotary, applied after QK-Norm (rotation preserves norm).
      * "pope" -- polar; magnitude (content) = softplus(features). QK-Norm, if on,
        is applied to the RAW features *before* softplus (bounds magnitudes for a
        more contractive settle; preserves decoupling). Q,K score runs in 2*hd.
      * "none" -- no positional info in attention (use a model-level abs-pos table).
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
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq, cfg.pos_base) if cfg.pos_enc == "rope" else None
        self.pope = (PolarPositionalEmbedding(self.head_dim, cfg.max_seq, cfg.pos_base, cfg.n_axes)
                     if cfg.pos_enc == "pope" else None)

    def forward(self, x: torch.Tensor, coord: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(self.cfg.dim, dim=-1)
        # (b, h, t, d)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pope is not None:
            # QK-Norm on the RAW features (before softplus), not on the magnitude:
            # bounds the magnitudes -> tighter, more contractive settling -- while
            # preserving PoPE's decoupling (magnitude stays position-independent).
            # The PoPE paper omits QK-Norm (it is feed-forward); this is the natural
            # extrapolation for the recurrent/settling setting. See Docs/LESSONS.md §3.
            if self.cfg.qk_norm:
                q = F.normalize(q, dim=-1) * F.softplus(self.q_scale).view(1, -1, 1, 1)
                k = F.normalize(k, dim=-1) * F.softplus(self.k_scale).view(1, -1, 1, 1)
            # Q,K become 2*head_dim polar vectors; v stays head_dim.
            q = self.pope.encode(q, is_key=False, coord=coord)
            k = self.pope.encode(k, is_key=True, coord=coord)
            scale = self.head_dim ** -0.5
        else:
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
        if cfg.residual_gate:
            # LayerScale/ReZero gates: small init makes the iterated map start near
            # norm(h + input), which is strongly contractive (Jacobian ~ 1/sqrt(dim)),
            # so the forward solve converges fast -- what IFT needs. Training can grow
            # the gates as expressivity demands.
            self.attn_gate = nn.Parameter(torch.full((cfg.dim,), cfg.gate_init))
            self.ffn_gate = nn.Parameter(torch.full((cfg.dim,), cfg.gate_init))
        else:
            self.attn_gate = None
            self.ffn_gate = None
        if cfg.polar_split:
            # placeholder for the magnitude/phase relational channel (Stage 1+).
            # Intentionally not implemented yet; flag exists so the wiring is in
            # place. See Docs/architecture.md §2.2 and representation_learning §3b.
            raise NotImplementedError(
                "polar_split is Stage 1+; keep it OFF until the settling loop is stable."
            )

    def forward(self, h: torch.Tensor, x: torch.Tensor, coord: torch.Tensor | None = None) -> torch.Tensor:
        # constant input injection (clamped sensory drive)
        h = h + self.cfg.inject_scale * x
        ag = self.attn_gate if self.attn_gate is not None else 1.0
        fg = self.ffn_gate if self.ffn_gate is not None else 1.0
        h = h + ag * self.attn(self.norm_attn(h), coord=coord)
        h = h + fg * self.ffn(self.norm_ffn(h))
        return h

    @torch.no_grad()
    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        """A zero starting state shaped like the injection ``x``."""
        return torch.zeros_like(x)
