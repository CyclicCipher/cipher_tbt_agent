"""
Mamba-3 block in pure PyTorch.

Implements the three core improvements from the Mamba-3 paper
(ICLR 2026 submission, OpenReview HwCvaJOiCj):

1. Trapezoidal discretization (second-order, replaces Euler):
   h_t = exp(Δ*A)*h_{t-1} + (1-λ)*Δ*exp(Δ*A)*B_{t-1}*x_{t-1} + λ*Δ*B_t*x_t
   where λ_t = σ(u_t) is data-dependent.

2. Complex-valued SSM via data-dependent RoPE:
   Rotation matrices R_t (from θ_t) applied to B, C projections.
   This enables state tracking (parity, modular arithmetic) that
   real-valued Mamba2 cannot solve.

3. Llama-style architecture:
   Alternating Mamba3Mixer + SwiGLU MLP blocks with pre-RMSNorm.
   QK-norm on B, C projections. No short causal convolution.

The SSD core is reused from Mamba2 with modifications for
trapezoidal discretization.

No official code has been released. This is based on the paper's
equations and descriptions.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Mamba3Config:
    """Configuration for a Mamba-3 model."""
    d_model: int = 128       # model dimension (D)
    d_state: int = 64        # SSM state dimension (N)
    expand: int = 2          # expansion factor (E)
    headdim: int = 64        # head dimension (P)
    chunk_size: int = 64     # matrix partition size (Q)
    n_layer: int = 4         # number of (Mixer + MLP) pairs
    mlp_expand: int = 4      # SwiGLU MLP expansion (typical: 4x or 8/3x)
    use_conv: bool = False   # optional short convolution (Mamba3 removes it)
    d_conv: int = 4          # convolution kernel size (if use_conv=True)
    use_pope: bool = True    # PoPE (Polar Positional Embeddings) instead of RoPE

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, (
            f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
        )
        self.nheads = self.d_inner // self.headdim


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: Tensor, theta: Tensor) -> Tensor:
    """Apply rotary position embedding (data-dependent RoPE).

    Implements the complex SSM equivalence from Proposition 3:
    rotations are absorbed into B/C projections.

    Args:
        x: (..., d_state) tensor to rotate.
        theta: (..., d_state // 2) rotation angles.

    Returns:
        Rotated tensor of same shape.
    """
    d = x.shape[-1]
    assert d % 2 == 0, f"d_state must be even for RoPE, got {d}"
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rx1 = x1 * cos_t - x2 * sin_t
    rx2 = x1 * sin_t + x2 * cos_t
    return torch.cat([rx1, rx2], dim=-1)


def apply_pope(x: Tensor, theta: Tensor, delta: Tensor) -> Tensor:
    """Apply Polar Coordinate Positional Embedding (PoPE).

    Decouples content ("what") from position ("where") by encoding
    content purely in magnitude (via softplus) and position purely
    in phase (via rotation).  Each scalar feature becomes a 2D
    (cos, sin) pair, so d features → 2d output.

    Reference: Gopalakrishnan et al. 2024, "Decoupling the 'What'
    and 'Where' With Polar Coordinate Positional Embeddings".

    Args:
        x: (..., d) raw features (pre-softplus).
        theta: (..., d) rotation angles (one per feature).
        delta: (d,) learnable bias added before softplus.

    Returns:
        (..., 2*d) polar-encoded tensor.
    """
    mu = F.softplus(x + delta)  # (..., d) positive magnitudes
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.cat([mu * cos_t, mu * sin_t], dim=-1)


# ---------------------------------------------------------------------------
# SSD core with trapezoidal discretization
# ---------------------------------------------------------------------------

def segsum(x: Tensor) -> Tensor:
    """Stable segment sum in log-space (same as Mamba2).

    Uses a large finite negative instead of -inf for the upper-triangle mask.
    exp(-1e10) = 0 in fp32/bf16, same as exp(-inf), but -1e10 is finite so
    second-order autograd (HVP in CG) won't produce 0 * (-inf) = NaN.
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device),
                      diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device),
                      diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -1e10)
    return x_segsum


def ssd_trapz(x_curr: Tensor, x_prev: Tensor,
              A: Tensor, B_curr: Tensor, B_prev: Tensor,
              C: Tensor, lam: Tensor,
              chunk_size: int) -> tuple[Tensor, Tensor]:
    """SSD with trapezoidal discretization.

    The trapezoidal rule adds a dependence on the PREVIOUS timestep's B*x:
        h_t = exp(Δ*A)*h_{t-1} + (1-λ)*Δ*exp(Δ*A)*B_{t-1}*x_{t-1} + λ*Δ*B_t*x_t

    The effective input at each position is a sum of two outer products:
        input_t = λ_t * (B_t ⊗ x_t) + (1-λ_t) * exp(A_t) * (B_{t-1} ⊗ x_{t-1})

    Because B and x have different shapes (B is head-shared, x is per-head),
    blending them separately and then multiplying creates spurious cross-terms.
    Instead, we run the SSD einsums twice — once for the Euler (current) term
    and once for the trapezoidal (previous) correction — and sum the results.
    This is exact: no cross-terms, no head-averaging.

    The per-head decay exp(A_t) is absorbed into x_prev, correctly applying
    head-specific temporal dynamics to the previous-step contribution.

    Args:
        x_curr: (batch, seqlen, nheads, headdim) — current input (dt-scaled)
        x_prev: (batch, seqlen, nheads, headdim) — previous input (dt-scaled, shifted)
        A: (batch, seqlen, nheads) — log-space decay * dt
        B_curr: (batch, seqlen, 1, d_state) — current B
        B_prev: (batch, seqlen, 1, d_state) — previous B (shifted)
        C: (batch, seqlen, 1, d_state) — output projection
        lam: (batch, seqlen, 1, 1) — trapezoidal mixing coeff in [0,1]
        chunk_size: partition size Q

    Returns:
        y: (batch, seqlen, nheads, headdim)
        final_state: (batch, nheads, headdim, d_state)
    """
    batch, seqlen, nheads, headdim = x_curr.shape
    assert seqlen % chunk_size == 0

    # Pre-weight x with trapezoidal coefficients.
    # Euler term: λ * x_curr (current B/x contribution)
    # Trapz term: (1-λ) * exp(A) * x_prev (previous B/x, decayed one step)
    # lam: (batch, seqlen, 1, 1) broadcasts over (nheads, headdim)
    # Per-head decay applied to x_prev (B is head-shared so doesn't need it)
    step_decay = torch.exp(A)  # (batch, seqlen, nheads)
    x_euler = lam * x_curr
    x_trapz = (1.0 - lam) * step_decay.unsqueeze(-1) * x_prev

    def _chunk(t):
        return t.reshape(batch, seqlen // chunk_size, chunk_size, *t.shape[2:])

    xE_c, xT_c = _chunk(x_euler), _chunk(x_trapz)
    A_c = _chunk(A)
    BE_c, BT_c = _chunk(B_curr), _chunk(B_prev)
    C_c = _chunk(C)

    A_c = A_c.permute(0, 3, 1, 2)  # (batch, nheads, chunks, chunk_size)
    A_cumsum = torch.cumsum(A_c, dim=-1)

    # Step 1: Intra-chunk (diagonal blocks) — two terms, summed
    L = torch.exp(segsum(A_c))
    Y_diag = (
        torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C_c, BE_c, L, xE_c)
        + torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C_c, BT_c, L, xT_c)
    )

    # Step 2: State accumulation — two terms, summed
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = (
        torch.einsum("bclhn, bhcl, bclhp -> bchpn", BE_c, decay_states, xE_c)
        + torch.einsum("bclhn, bhcl, bclhp -> bchpn", BT_c, decay_states, xT_c)
    )

    # Step 3: Inter-chunk recurrence (operates on accumulated states, unchanged)
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Step 4: State-to-output (operates on accumulated states, unchanged)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C_c, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(batch, seqlen, nheads, headdim)
    return Y, final_state


# ---------------------------------------------------------------------------
# Mamba-3 Mixer Block
# ---------------------------------------------------------------------------

class Mamba3Mixer(nn.Module):
    """Mamba-3 mixer block (replaces Mamba2Block).

    Key changes from Mamba2:
    - Trapezoidal discretization (blends current + previous input)
    - Data-dependent RoPE on B, C (complex SSM equivalence)
    - Learnable bias on B, C projections
    - QK-norm on B, C after projection
    - Short convolution optional (default: off)

    Input:  (batch, seqlen, d_model)
    Output: (batch, seqlen, d_model)
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d = config.d_inner
        n = config.d_state

        # With PoPE, B/C are projected to half-size (n//2) then expanded
        # to full d_state via polar encoding.  Theta dimension is the same.
        d_bc = n // 2 if config.use_pope else n

        # Input projection: [z, x, B, C, dt, theta, lambda]
        # z: gating (d_inner)
        # x: SSM input (d_inner)
        # B: input projection (d_bc)
        # C: output projection (d_bc)
        # dt: timestep (nheads)
        # theta: rotation angles (d_state // 2)
        # lambda: trapezoidal mixing (1 scalar)
        d_in_proj = 2 * d + 2 * d_bc + config.nheads + n // 2 + 1
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        # BC bias (learnable, channel-wise)
        # For PoPE these also serve as the learnable delta offset
        self.B_bias = nn.Parameter(torch.zeros(d_bc))
        self.C_bias = nn.Parameter(torch.zeros(d_bc))

        # QK-norm on B, C (applied on raw features before rotation/polar encoding)
        self.B_norm = RMSNorm(d_bc)
        self.C_norm = RMSNorm(d_bc)

        # PoPE learnable phase bias (delta) — controls softplus operating point
        if config.use_pope:
            self.pope_delta_B = nn.Parameter(torch.zeros(d_bc))
            self.pope_delta_C = nn.Parameter(torch.zeros(d_bc))

        # Optional short convolution
        if config.use_conv:
            conv_dim = d + 2 * n
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim, out_channels=conv_dim,
                kernel_size=config.d_conv, groups=conv_dim,
                padding=config.d_conv - 1,
            )
        else:
            self.conv1d = None

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        self.A_log = nn.Parameter(torch.empty(config.nheads))
        self.D = nn.Parameter(torch.empty(config.nheads))

        # Output
        self.out_norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, config.d_model, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        cfg = self.config
        nn.init.uniform_(self.A_log, -5.0, -1.0)
        nn.init.ones_(self.D)
        nn.init.uniform_(self.dt_bias, -1.0, 1.0)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, u: Tensor) -> Tensor:
        """Forward pass.

        Args:
            u: (batch, seqlen, d_model)

        Returns:
            (batch, seqlen, d_model)
        """
        cfg = self.config
        batch, seqlen, _ = u.shape

        A = -torch.exp(self.A_log)  # (nheads,)

        # Project input
        d_bc = cfg.d_state // 2 if cfg.use_pope else cfg.d_state
        proj = self.in_proj(u)
        z, x, B, C, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, d_bc, d_bc,
             cfg.nheads, cfg.d_state // 2, 1],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)
        lam = torch.sigmoid(lam_logit)       # (batch, seqlen, 1)

        # Optional convolution on x
        if self.conv1d is not None:
            x = F.silu(
                self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
            )
        else:
            x = F.silu(x)

        # Add BC bias + QK-norm
        B = self.B_norm(B + self.B_bias)
        C = self.C_norm(C + self.C_bias)

        # Accumulate theta across positions for positional encoding
        theta_cumsum = torch.cumsum(theta, dim=1)

        if cfg.use_pope:
            # PoPE: softplus magnitudes + phase-only rotation
            # B, C: (batch, seqlen, d_state//2) → (batch, seqlen, d_state)
            B = apply_pope(B, theta_cumsum, self.pope_delta_B)
            C = apply_pope(C, theta_cumsum, self.pope_delta_C)
        else:
            # Standard data-dependent RoPE on B and C
            B = apply_rope(B, theta_cumsum)
            C = apply_rope(C, theta_cumsum)

        # Reshape for multi-head SSD
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        x_dt = x * dt.unsqueeze(-1)  # scale by dt

        # Create shifted (previous) versions for trapezoidal rule
        x_prev = F.pad(x_dt[:, :-1], (0, 0, 0, 0, 1, 0))  # shift right, pad zeros
        B_prev = F.pad(B[:, :-1].unsqueeze(2), (0, 0, 0, 0, 1, 0))  # (batch, seqlen, 1, d_state)
        B_curr = B.unsqueeze(2)
        C_curr = C.unsqueeze(2)
        lam_expand = lam.unsqueeze(-1)  # (batch, seqlen, 1, 1)

        # SSD with trapezoidal discretization
        y, _ = ssd_trapz(
            x_dt, x_prev,
            A * dt,
            B_curr, B_prev,
            C_curr, lam_expand,
            cfg.chunk_size,
        )

        # D skip connection
        y = y + x * self.D.unsqueeze(-1)

        # Flatten heads, gated norm, output projection
        y = y.reshape(batch, seqlen, cfg.d_inner)
        y = self.out_norm(y * F.silu(z))
        y = self.out_proj(y)

        return y


# ---------------------------------------------------------------------------
# SwiGLU MLP (Llama-style)
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP block (Llama-style).

    gate = SiLU(W_gate @ x)
    up   = W_up @ x
    out  = W_down @ (gate * up)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full Mamba-3 Model (Llama-style: alternating Mixer + MLP)
# ---------------------------------------------------------------------------

class Mamba3Block(nn.Module):
    """One Mamba-3 layer: pre-norm Mixer + pre-norm MLP with residuals.

    Follows Llama-style layout:
        x = x + Mixer(RMSNorm(x))
        x = x + MLP(RMSNorm(x))
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = Mamba3Mixer(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Mamba3LM(nn.Module):
    """Complete Mamba-3 language model.

    Architecture:
        Embedding → N x Mamba3Block → RMSNorm → Linear(vocab)

    Args:
        config: Mamba3Config.
        vocab_size: Number of tokens.
    """

    def __init__(self, config: Mamba3Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Mamba3Block(config) for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)

        # Weight tying
        self.out_proj.weight = self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seqlen) token indices.

        Returns:
            logits: (batch, seqlen, vocab_size)
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.out_proj(x)
