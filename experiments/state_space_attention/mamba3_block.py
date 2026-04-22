"""
Mamba-3 block — SISO baseline for the Slot Workspace experiment.

Copied from experiments/Mamba3/mamba3_block.py and stripped to SISO only:
  - MIMO removed (r=1 always; mimo_rank, mimo_x_proj, mimo_out_proj gone)
  - Manifold-constrained hyperconnections (mHC) removed
  - PoPE, StableSSM, trapezoidal discretization, QK-norm all preserved

This file is the SSM backbone.  The slot-structured state extension with
intra-slot attention lives in slot_workspace.py, which imports the SSD
utilities (ssd_trapz, stable_log_decay, segsum) and helpers (RMSNorm,
SwiGLUMLP, apply_pope, apply_rope) from here.

Original paper: Mamba-3 (ICLR 2026 submission, OpenReview HwCvaJOiCj).
Modifications: PoPE (Gopalakrishnan et al. 2024), StableSSM (Wang & Li 2024).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Mamba3Config:
    """Configuration for a Mamba-3 model (SISO)."""
    d_model: int = 128       # model dimension (D)
    d_state: int = 64        # SSM state dimension (N)
    expand: int = 2          # expansion factor (E)
    headdim: int = 64        # head dimension (P)
    chunk_size: int = 64     # matrix partition size (Q)
    n_layer: int = 4         # number of (Mixer + MLP) pairs
    mlp_expand: int = 4      # SwiGLU MLP expansion
    use_conv: bool = False   # optional short convolution (Mamba3 removes it)
    d_conv: int = 4          # convolution kernel size (if use_conv=True)
    use_pope: bool = True    # PoPE instead of RoPE
    stable_ssm: bool = False # StableSSM A-matrix reparameterization
    use_triton: bool = False # Triton-accelerated SSD kernels

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
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------

def apply_rope(x: Tensor, theta: Tensor) -> Tensor:
    """Apply rotary position embedding (data-dependent RoPE)."""
    d = x.shape[-1]
    assert d % 2 == 0
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.cat([x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t], dim=-1)


def apply_pope(x: Tensor, theta: Tensor, delta: Tensor) -> Tensor:
    """Apply Polar Coordinate Positional Embedding (PoPE).

    Decouples content (magnitude via softplus) from position (phase via
    rotation).  d features -> 2d output.

    Ref: Gopalakrishnan et al. 2024.
    """
    mu = F.softplus(x + delta)   # positive magnitudes
    return torch.cat([mu * torch.cos(theta), mu * torch.sin(theta)], dim=-1)


# ---------------------------------------------------------------------------
# SSD core with trapezoidal discretization
# ---------------------------------------------------------------------------

def stable_log_decay(w: Tensor) -> Tensor:
    """StableSSM 'best' reparameterization in log-space.

    decay = 1 - 1/(w^2 + 0.5)   ->  log(decay)
    Ref: Wang & Li 2024 (arXiv:2311.14495, ICML 2024).
    """
    decay = 1.0 - 1.0 / (w * w + 0.5)
    return torch.log(decay.clamp(min=1e-6))


def segsum(x: Tensor) -> Tensor:
    """Stable segment sum in log-space (same as Mamba2)."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    return x_segsum.masked_fill(~mask, -1e10)


def ssd_trapz(x_curr: Tensor, x_prev: Tensor,
              A: Tensor, B_curr: Tensor, B_prev: Tensor,
              C: Tensor, lam: Tensor,
              chunk_size: int) -> tuple[Tensor, Tensor]:
    """SSD with trapezoidal discretization.

    h_t = exp(Δ*A)*h_{t-1} + (1-λ)*Δ*exp(Δ*A)*B_{t-1}*x_{t-1} + λ*Δ*B_t*x_t

    Args:
        x_curr: (batch, seqlen, nheads, headdim)
        x_prev: (batch, seqlen, nheads, headdim)
        A:      (batch, seqlen, nheads) log-space decay * dt
        B_curr: (batch, seqlen, 1, d_state)
        B_prev: (batch, seqlen, 1, d_state)
        C:      (batch, seqlen, 1, d_state)
        lam:    (batch, seqlen, 1, 1)
        chunk_size: partition size Q

    Returns:
        y:           (batch, seqlen, nheads, headdim)
        final_state: (batch, nheads, headdim, d_state)
    """
    batch, seqlen, nheads, headdim = x_curr.shape
    assert seqlen % chunk_size == 0

    step_decay = torch.exp(A)
    x_euler = lam * x_curr
    x_trapz = (1.0 - lam) * step_decay.unsqueeze(-1) * x_prev

    def _chunk(t):
        return t.reshape(batch, seqlen // chunk_size, chunk_size, *t.shape[2:])

    xE_c, xT_c = _chunk(x_euler), _chunk(x_trapz)
    A_c = _chunk(A)
    BE_c, BT_c = _chunk(B_curr), _chunk(B_prev)
    C_c = _chunk(C)

    A_c = A_c.permute(0, 3, 1, 2)   # (batch, nheads, chunks, chunk_size)
    A_cumsum = torch.cumsum(A_c, dim=-1)

    L = torch.exp(segsum(A_c))
    Y_diag = (
        torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C_c, BE_c, L, xE_c)
        + torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C_c, BT_c, L, xT_c)
    )

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = (
        torch.einsum("bclhn, bhcl, bclhp -> bchpn", BE_c, decay_states, xE_c)
        + torch.einsum("bclhn, bhcl, bclhp -> bchpn", BT_c, decay_states, xT_c)
    )

    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C_c, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(batch, seqlen, nheads, headdim)
    return Y, final_state


# ---------------------------------------------------------------------------
# Mamba-3 Mixer Block (SISO)
# ---------------------------------------------------------------------------

class Mamba3Mixer(nn.Module):
    """Mamba-3 mixer block, SISO (r=1).

    Key features (all from Mamba-3 paper):
    - Trapezoidal discretization
    - Data-dependent PoPE or RoPE on B, C
    - QK-norm on B, C after projection
    - No pre-output-projection RMSNorm

    Input/output: (batch, seqlen, d_model)
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d = config.d_inner
        n = config.d_state

        d_bc = n // 2 if config.use_pope else n

        # Input projection: [z, x, B, C, dt, theta, lambda]
        d_in_proj = 2 * d + 2 * d_bc + config.nheads + n // 2 + 1
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        self.B_bias = nn.Parameter(torch.ones(d_bc))
        self.C_bias = nn.Parameter(torch.ones(d_bc))
        self.B_norm = RMSNorm(d_bc)
        self.C_norm = RMSNorm(d_bc)

        if config.use_pope:
            self.pope_delta_B = nn.Parameter(torch.zeros(d_bc))
            self.pope_delta_C = nn.Parameter(torch.zeros(d_bc))

        if config.use_conv:
            self.conv1d = nn.Conv1d(
                d, d, kernel_size=config.d_conv, groups=d,
                padding=config.d_conv - 1,
            )
        else:
            self.conv1d = None

        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        if config.stable_ssm:
            self.A_raw = nn.Parameter(torch.empty(config.nheads))
        else:
            self.A_log = nn.Parameter(torch.empty(config.nheads))
        self.D = nn.Parameter(torch.empty(config.nheads))

        self.out_proj = nn.Linear(d, config.d_model, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        cfg = self.config
        if cfg.stable_ssm:
            nn.init.uniform_(self.A_raw, -3.0, -0.5)
        else:
            nn.init.uniform_(self.A_log, -5.0, -1.0)
        nn.init.ones_(self.D)
        dt_target = torch.exp(
            torch.empty(cfg.nheads).uniform_(math.log(0.001), math.log(0.1))
        )
        self.dt_bias.data.copy_(torch.log(torch.exp(dt_target) - 1))
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, u: Tensor) -> Tensor:
        cfg = self.config
        batch, seqlen, _ = u.shape

        if cfg.stable_ssm:
            A_raw = self.A_raw
        else:
            A_raw = -torch.exp(self.A_log)

        d_bc = cfg.d_state // 2 if cfg.use_pope else cfg.d_state
        proj = self.in_proj(u)
        z, x, B_raw, C_raw, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, d_bc, d_bc, cfg.nheads, cfg.d_state // 2, 1],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)
        lam = torch.sigmoid(lam_logit)

        if self.conv1d is not None:
            x = F.silu(
                self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
            )

        B_raw = self.B_norm(B_raw) + self.B_bias
        C_raw = self.C_norm(C_raw) + self.C_bias

        theta_cumsum = torch.cumsum(theta, dim=1)

        if cfg.use_pope:
            B = apply_pope(B_raw, theta_cumsum, self.pope_delta_B)
            C = apply_pope(C_raw, theta_cumsum, self.pope_delta_C)
        else:
            B = apply_rope(B_raw, theta_cumsum)
            C = apply_rope(C_raw, theta_cumsum)

        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        dt_exp = dt.unsqueeze(-1)
        x_dt = x * dt_exp

        if cfg.stable_ssm:
            A_dt = stable_log_decay(A_raw * dt)
        else:
            A_dt = A_raw * dt

        _ssd_fn = ssd_trapz
        if cfg.use_triton:
            try:
                from triton_ssd import ssd_trapz_triton
                _ssd_fn = ssd_trapz_triton
            except ImportError:
                pass

        lam_expand = lam.unsqueeze(-1)
        x_raw_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = x_raw_prev * dt_exp
        B_prev = F.pad(B[:, :-1].unsqueeze(2), (0, 0, 0, 0, 1, 0))
        B_curr = B.unsqueeze(2)
        C_curr = C.unsqueeze(2)

        y, _ = _ssd_fn(
            x_dt, x_prev, A_dt,
            B_curr, B_prev, C_curr,
            lam_expand, cfg.chunk_size,
        )

        y = y + x * self.D.unsqueeze(-1)
        y = y.reshape(batch, seqlen, cfg.d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Full Mamba-3 Block and LM (SISO, no mHC)
# ---------------------------------------------------------------------------

class Mamba3Block(nn.Module):
    """One Mamba-3 layer: pre-norm Mixer + pre-norm MLP."""

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
    """Complete Mamba-3 language model (SISO baseline)."""

    def __init__(self, config: Mamba3Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model)
        self.out_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embedding.weight   # weight tying

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.out_proj(self.norm(x))
