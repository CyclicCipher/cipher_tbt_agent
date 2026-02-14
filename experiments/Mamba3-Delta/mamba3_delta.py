"""
Mamba3-Delta: Mamba3 + Delta Rule + MIMO + PoPE orthogonal pair.

Combines Mamba3's continuous-time SSM dynamics with the delta rule's targeted
write/erase memory.  See DESIGN.md for the full mathematical specification.

Key differences from mamba3_block.py:
  1. MIMO: B, C are rank-r matrices (not rank-1 vectors)
  2. Delta rule: Householder erase before write
  3. PoPE orthogonal pair: B₂ = (-μ·sin(θ), μ·cos(θ)) for rotation
  4. Per-channel decay: diagonal α_t replaces scalar exp(Δ·A)
  5. Surprise-modulated β gates (placeholder — requires external signal)

Implementation notes:
  - Phase 1: Naive sequential recurrence (no chunkwise parallelism)
  - Do NOT run full training on CPU (Mistake #36)
  - Chunkwise WY parallelism is Phase 5
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Mamba3DeltaConfig:
    """Configuration for Mamba3-Delta."""
    d_model: int = 128
    d_state: int = 64
    expand: int = 2
    headdim: int = 64
    n_layer: int = 4
    mlp_expand: int = 4

    # --- MIMO ---
    mimo_rank: int = 1          # r=1 is SISO (Mamba3 default), r>1 is MIMO

    # --- Delta rule ---
    use_delta_rule: bool = True
    use_pope_perp: bool = True  # PoPE orthogonal pair for n_h=2 Householder

    # --- Decay ---
    per_channel_decay: bool = True   # KDA-style per-channel α
    stable_reparam: bool = False     # StableSSM reparameterization

    # --- Surprise ---
    use_surprise_gate: bool = False  # Phase 4: surprise-modulated β

    # --- Mamba3 features ---
    use_pope: bool = True
    use_trapezoidal: bool = True

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim


# ---------------------------------------------------------------------------
# Utilities (shared with mamba3_block.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_pope(x: Tensor, theta: Tensor, delta: Tensor) -> Tensor:
    """PoPE: magnitude encodes content, phase encodes position."""
    mu = F.softplus(x + delta)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.cat([mu * cos_t, mu * sin_t], dim=-1)


def apply_pope_perp(x: Tensor, theta: Tensor, delta: Tensor) -> Tensor:
    """PoPE orthogonal partner: rotate phase by π/2.

    If B₁ = (μ·cos(θ), μ·sin(θ)), then B₂ = (-μ·sin(θ), μ·cos(θ)).
    B₁ · B₂ = 0 by construction (orthogonal).
    Product of two Householder reflections about orthogonal axes = rotation.
    """
    mu = F.softplus(x + delta)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.cat([-mu * sin_t, mu * cos_t], dim=-1)


def apply_rope(x: Tensor, theta: Tensor) -> Tensor:
    """Standard data-dependent RoPE (fallback if PoPE disabled)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)


# ---------------------------------------------------------------------------
# Delta Rule Recurrence (naive sequential, Phase 1)
# ---------------------------------------------------------------------------

def delta_recurrence(
    x: Tensor,             # (batch, seqlen, nheads, headdim)
    x_prev: Tensor,        # (batch, seqlen, nheads, headdim) shifted for trapezoidal
    B1: Tensor,            # (batch, seqlen, nheads_bc, d_state) primary key
    B1_prev: Tensor,       # shifted B1 for trapezoidal
    B2: Optional[Tensor],  # (batch, seqlen, nheads_bc, d_state) orthogonal key (or None)
    C: Tensor,             # (batch, seqlen, nheads_bc, d_state)
    alpha: Tensor,         # (batch, seqlen, nheads, d_state) per-channel decay
    beta1: Tensor,         # (batch, seqlen, nheads, 1) write gate
    beta2: Optional[Tensor],  # (batch, seqlen, nheads, 1) rotation gate (or None)
    lam: Tensor,           # (batch, seqlen, 1, 1) trapezoidal mixing
    D: Tensor,             # (nheads,) skip connection
    use_trapezoidal: bool = True,
    use_delta: bool = True,
) -> Tensor:
    """Naive sequential delta-rule recurrence.

    This is the reference implementation for correctness. Not optimized.
    Chunkwise WY parallelism is Phase 5.

    State shape per head: (headdim, d_state).
    """
    batch, seqlen, nheads, headdim = x.shape
    d_state = B1.shape[-1]
    nheads_bc = B1.shape[2]

    device = x.device
    dtype = x.dtype

    # Initialize state: (batch, nheads, headdim, d_state)
    h = torch.zeros(batch, nheads, headdim, d_state, device=device, dtype=dtype)
    outputs = []

    for t in range(seqlen):
        # --- Decay: per-channel diagonal ---
        # alpha: (batch, nheads, d_state) applied to h: (batch, nheads, headdim, d_state)
        a_t = alpha[:, t]  # (batch, nheads, d_state)
        h = h * a_t.unsqueeze(2)  # broadcast over headdim

        # --- Delta rule: Householder erase ---
        if use_delta:
            # B1 key for erase (broadcast over heads if nheads_bc < nheads)
            b1_t = B1[:, t]  # (batch, nheads_bc, d_state)
            if nheads_bc < nheads:
                b1_t = b1_t.expand(-1, nheads, -1)

            # L2 normalize for Householder
            b1_hat = F.normalize(b1_t, dim=-1)  # (batch, nheads, d_state)
            bt1 = beta1[:, t]  # (batch, nheads, 1)

            # Householder 1: h = h · (I - β₁ · b̂₁ · b̂₁^T)
            # Equivalent to: h = h - β₁ · (h · b̂₁) · b̂₁^T
            proj1 = torch.einsum('bnpd,bnd->bnp', h, b1_hat)  # (batch, nheads, headdim)
            h = h - bt1.unsqueeze(2) * torch.einsum('bnp,bnd->bnpd', proj1, b1_hat)

            # Householder 2 (PoPE orthogonal pair): h = h · (I - β₂ · b̂₂ · b̂₂^T)
            if B2 is not None and beta2 is not None:
                b2_t = B2[:, t]
                if nheads_bc < nheads:
                    b2_t = b2_t.expand(-1, nheads, -1)

                b2_hat = F.normalize(b2_t, dim=-1)
                bt2 = beta2[:, t]

                proj2 = torch.einsum('bnpd,bnd->bnp', h, b2_hat)
                h = h - bt2.unsqueeze(2) * torch.einsum('bnp,bnd->bnpd', proj2, b2_hat)

        # --- Write: add new key-value associations ---
        x_t = x[:, t]  # (batch, nheads, headdim)
        b1_write = B1[:, t]  # (batch, nheads_bc, d_state)
        if nheads_bc < nheads:
            b1_write = b1_write.expand(-1, nheads, -1)

        if use_trapezoidal and t > 0:
            x_prev_t = x_prev[:, t]
            b1_prev_t = B1_prev[:, t]
            if nheads_bc < nheads:
                b1_prev_t = b1_prev_t.expand(-1, nheads, -1)

            lam_t = lam[:, t]  # (batch, 1, 1)
            # Trapezoidal: blend current and previous input terms
            write_euler = torch.einsum('bnp,bnd->bnpd', x_t, b1_write)
            write_trapz = torch.einsum('bnp,bnd->bnpd', x_prev_t, b1_prev_t)
            write = lam_t.unsqueeze(1) * write_euler + (1.0 - lam_t.unsqueeze(1)) * write_trapz
        else:
            write = torch.einsum('bnp,bnd->bnpd', x_t, b1_write)

        if use_delta:
            write = beta1[:, t].unsqueeze(2) * write

        h = h + write

        # Write via B2 (orthogonal direction) if enabled
        if B2 is not None and beta2 is not None and use_delta:
            b2_write = B2[:, t]
            if nheads_bc < nheads:
                b2_write = b2_write.expand(-1, nheads, -1)
            write2 = beta2[:, t].unsqueeze(2) * torch.einsum('bnp,bnd->bnpd', x_t, b2_write)
            h = h + write2

        # --- Readout ---
        c_t = C[:, t]  # (batch, nheads_bc, d_state)
        if nheads_bc < nheads:
            c_t = c_t.expand(-1, nheads, -1)

        y_t = torch.einsum('bnpd,bnd->bnp', h, c_t)  # (batch, nheads, headdim)

        # Skip connection
        y_t = y_t + x_t * D.unsqueeze(-1)

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)  # (batch, seqlen, nheads, headdim)


# ---------------------------------------------------------------------------
# Mamba3-Delta Mixer
# ---------------------------------------------------------------------------

class Mamba3DeltaMixer(nn.Module):
    """Mamba3-Delta mixer block.

    Combines:
    - PoPE (content/position decoupling)
    - Delta rule (targeted erase+write)
    - PoPE orthogonal pair (rotation via n_h=2 Householder)
    - Per-channel decay (KDA-style multi-scale)
    - MIMO (rank-r B, C, X projections)
    - Trapezoidal discretization (retained from Mamba3)
    - Surprise-modulated beta gates (Phase 4 placeholder)
    """

    def __init__(self, config: Mamba3DeltaConfig):
        super().__init__()
        self.config = config
        d = config.d_inner
        n = config.d_state
        r = config.mimo_rank

        # PoPE projects to half-size then expands via polar encoding
        d_bc = n // 2 if config.use_pope else n

        # --- Input projection ---
        # z: gating (d_inner)
        # x: SSM input (d_inner)
        # B: input projection (d_bc * r for MIMO)
        # C: output projection (d_bc * r for MIMO)
        # theta: rotation angles (d_state // 2)
        # lam: trapezoidal mixing (1 scalar)
        d_in_proj = (
            2 * d                      # z, x
            + 2 * d_bc * r             # B, C (MIMO rank-r)
            + n // 2                   # theta
            + 1                        # lambda (trapezoidal)
        )
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        # --- BC bias + QK-norm ---
        self.B_bias = nn.Parameter(torch.zeros(d_bc * r))
        self.C_bias = nn.Parameter(torch.zeros(d_bc * r))
        self.B_norm = RMSNorm(d_bc * r)
        self.C_norm = RMSNorm(d_bc * r)

        # --- PoPE phase bias ---
        if config.use_pope:
            self.pope_delta_B = nn.Parameter(torch.zeros(d_bc))
            self.pope_delta_C = nn.Parameter(torch.zeros(d_bc))

        # --- Decay parameters ---
        if config.per_channel_decay:
            # Per-channel decay via low-rank MLP (KDA-style)
            decay_bottleneck = max(n // 4, 16)
            self.decay_down = nn.Linear(config.d_model, decay_bottleneck, bias=False)
            self.decay_up = nn.Linear(decay_bottleneck, config.nheads * n, bias=False)
            self.decay_bias = nn.Parameter(torch.zeros(config.nheads * n))
        else:
            # Scalar decay (Mamba3 style)
            self.A_log = nn.Parameter(torch.empty(config.nheads))
            self.dt_bias = nn.Parameter(torch.empty(config.nheads))
            # dt projection from input
            self.dt_proj = nn.Linear(config.d_model, config.nheads, bias=False)

        # --- Delta rule gates ---
        if config.use_delta_rule:
            self.beta1_proj = nn.Linear(config.d_model, config.nheads, bias=True)
            if config.use_pope_perp:
                self.beta2_proj = nn.Linear(config.d_model, config.nheads, bias=True)

        # --- MIMO projections ---
        if r > 1:
            # X: d_inner -> d_inner * r (expand input for MIMO)
            self.mimo_x_proj = nn.Linear(d, d * r, bias=False)
            # Output: d_inner * r -> d_inner (contract MIMO output)
            self.mimo_out_proj = nn.Linear(d * r, d, bias=False)

        # --- Skip connection, output ---
        self.D = nn.Parameter(torch.empty(config.nheads))
        self.out_norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, config.d_model, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        cfg = self.config
        nn.init.ones_(self.D)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if cfg.per_channel_decay:
            # Initialize decay bias to produce α ≈ 0.95 (moderate retention)
            nn.init.constant_(self.decay_bias, 3.0)
            nn.init.xavier_uniform_(self.decay_down.weight)
            nn.init.xavier_uniform_(self.decay_up.weight)
        else:
            nn.init.uniform_(self.A_log, -5.0, -1.0)
            nn.init.uniform_(self.dt_bias, -1.0, 1.0)

        if cfg.use_delta_rule:
            # Initialize beta bias to produce β ≈ 0.5
            nn.init.constant_(self.beta1_proj.bias, 0.0)
            if cfg.use_pope_perp:
                # Start β₂ small — rotation is initially off
                nn.init.constant_(self.beta2_proj.bias, -2.0)

    def forward(self, u: Tensor, surprise: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            u: (batch, seqlen, d_model) input.
            surprise: (batch, seqlen) optional surprise signal for gating.

        Returns:
            (batch, seqlen, d_model)
        """
        cfg = self.config
        batch, seqlen, _ = u.shape
        r = cfg.mimo_rank
        d_bc = cfg.d_state // 2 if cfg.use_pope else cfg.d_state

        # --- Project input ---
        proj = self.in_proj(u)
        z, x, B_raw, C_raw, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, d_bc * r, d_bc * r,
             cfg.d_state // 2, 1],
            dim=-1,
        )

        x = F.silu(x)
        lam = torch.sigmoid(lam_logit)

        # --- B, C processing ---
        B_raw = self.B_norm(B_raw + self.B_bias)
        C_raw = self.C_norm(C_raw + self.C_bias)

        # Accumulate theta for positional encoding
        theta_cumsum = torch.cumsum(theta, dim=1)

        if cfg.use_pope:
            if r > 1:
                # MIMO: reshape to (batch, seqlen, r, d_bc), apply PoPE per column
                B_reshaped = B_raw.reshape(batch, seqlen, r, d_bc)
                C_reshaped = C_raw.reshape(batch, seqlen, r, d_bc)
                B_list, C_list = [], []
                B2_list = []
                for i in range(r):
                    b_i = apply_pope(B_reshaped[:, :, i], theta_cumsum, self.pope_delta_B)
                    c_i = apply_pope(C_reshaped[:, :, i], theta_cumsum, self.pope_delta_C)
                    B_list.append(b_i)
                    C_list.append(c_i)
                    if cfg.use_pope_perp and cfg.use_delta_rule:
                        b2_i = apply_pope_perp(B_reshaped[:, :, i], theta_cumsum, self.pope_delta_B)
                        B2_list.append(b2_i)
                # Stack: (batch, seqlen, r, d_state) -> reshape for recurrence
                B1 = torch.stack(B_list, dim=2)  # (batch, seqlen, r, d_state)
                C = torch.stack(C_list, dim=2)
                B2 = torch.stack(B2_list, dim=2) if B2_list else None
            else:
                B1 = apply_pope(B_raw, theta_cumsum, self.pope_delta_B)
                C = apply_pope(C_raw, theta_cumsum, self.pope_delta_C)
                B1 = B1.unsqueeze(2)  # (batch, seqlen, 1, d_state)
                C = C.unsqueeze(2)
                if cfg.use_pope_perp and cfg.use_delta_rule:
                    B2 = apply_pope_perp(B_raw, theta_cumsum, self.pope_delta_B)
                    B2 = B2.unsqueeze(2)
                else:
                    B2 = None
        else:
            B1 = apply_rope(B_raw, theta_cumsum)
            C = apply_rope(C_raw, theta_cumsum)
            B1 = B1.unsqueeze(2)
            C = C.unsqueeze(2)
            B2 = None

        # --- Compute decay α ---
        if cfg.per_channel_decay:
            decay_hidden = F.silu(self.decay_down(u))
            alpha_logits = self.decay_up(decay_hidden) + self.decay_bias
            alpha_logits = alpha_logits.reshape(batch, seqlen, cfg.nheads, cfg.d_state)
            if cfg.stable_reparam:
                # StableSSM: α = 1 - 1/(z² + 0.5)
                alpha = 1.0 - 1.0 / (alpha_logits.pow(2) + 0.5)
            else:
                alpha = torch.sigmoid(alpha_logits)
        else:
            # Scalar decay (Mamba3 style)
            A = -torch.exp(self.A_log)
            dt = F.softplus(self.dt_proj(u) + self.dt_bias)
            # exp(Δ·A): (batch, seqlen, nheads) -> expand to per-channel
            scalar_alpha = torch.exp(dt * A)
            alpha = scalar_alpha.unsqueeze(-1).expand(-1, -1, -1, cfg.d_state)

        # --- Compute beta (write gates) ---
        if cfg.use_delta_rule:
            beta1_input = self.beta1_proj(u)  # (batch, seqlen, nheads)
            if cfg.use_surprise_gate and surprise is not None:
                # Surprise modulation (stop-gradiented)
                s = surprise.detach().unsqueeze(-1)  # (batch, seqlen, 1)
                beta1_input = beta1_input + s
            beta1 = torch.sigmoid(beta1_input).unsqueeze(-1)  # (batch, seqlen, nheads, 1)

            if cfg.use_pope_perp:
                beta2_input = self.beta2_proj(u)
                if cfg.use_surprise_gate and surprise is not None:
                    beta2_input = beta2_input + s
                beta2 = torch.sigmoid(beta2_input).unsqueeze(-1)
            else:
                beta2 = None
        else:
            beta1 = None
            beta2 = None

        # --- Reshape x for multi-head ---
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)

        # --- MIMO: expand x to rank-r ---
        if r > 1:
            x_flat = x.reshape(batch, seqlen, cfg.d_inner)
            x_mimo = self.mimo_x_proj(x_flat)  # (batch, seqlen, d_inner * r)
            x_mimo = x_mimo.reshape(batch, seqlen, cfg.nheads, cfg.headdim, r)
            # For the recurrence, we need x per MIMO column
            # For now in Phase 1, sum across MIMO rank for simplicity
            # Full MIMO: B_t · X_t^T is a rank-r matrix product
            # TODO: implement proper MIMO recurrence in Phase 5
            x = x_mimo.mean(dim=-1)  # placeholder: average across MIMO rank

        # --- Trapezoidal: create shifted versions ---
        if cfg.use_trapezoidal:
            x_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
            B1_prev = F.pad(B1[:, :-1], (0, 0, 0, 0, 1, 0))
        else:
            x_prev = torch.zeros_like(x)
            B1_prev = torch.zeros_like(B1)

        # --- Run recurrence ---
        y = delta_recurrence(
            x=x,
            x_prev=x_prev,
            B1=B1,
            B1_prev=B1_prev,
            B2=B2,
            C=C,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            lam=lam.unsqueeze(-1),  # (batch, seqlen, 1, 1)
            D=self.D,
            use_trapezoidal=cfg.use_trapezoidal,
            use_delta=cfg.use_delta_rule,
        )

        # --- MIMO: contract output ---
        y = y.reshape(batch, seqlen, cfg.d_inner)
        if r > 1:
            # TODO: proper MIMO output contraction in Phase 5
            pass

        # --- Gated output ---
        y = self.out_norm(y * F.silu(z))
        y = self.out_proj(y)
        return y


# ---------------------------------------------------------------------------
# SwiGLU MLP (same as Mamba3)
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full Mamba3-Delta Block and Model
# ---------------------------------------------------------------------------

class Mamba3DeltaBlock(nn.Module):
    """One Mamba3-Delta layer: pre-norm Mixer + pre-norm MLP with residuals."""

    def __init__(self, config: Mamba3DeltaConfig):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = Mamba3DeltaMixer(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

    def forward(self, x: Tensor, surprise: Optional[Tensor] = None) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x), surprise=surprise)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Mamba3DeltaLM(nn.Module):
    """Complete Mamba3-Delta language model.

    Architecture:
        Embedding -> N x Mamba3DeltaBlock -> RMSNorm -> Linear(vocab)
    """

    def __init__(self, config: Mamba3DeltaConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Mamba3DeltaBlock(config) for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying
        self.out_proj.weight = self.embedding.weight

    def forward(self, input_ids: Tensor, surprise: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seqlen) token indices.
            surprise: (batch, seqlen) optional surprise signal.

        Returns:
            logits: (batch, seqlen, vocab_size)
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, surprise=surprise)
        x = self.norm(x)
        return self.out_proj(x)
