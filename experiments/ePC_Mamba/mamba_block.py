"""
Minimal Mamba2 block in pure PyTorch.

Implements the Structured State Space Duality (SSD) algorithm from
Gu & Dao (2024), "Transformers are SSMs". No CUDA kernels — the SSD
scan uses ~30 lines of einsums that run on any device.

Reference: tommyip/mamba2-minimal (MIT license)

Architecture of a single block:
  in_proj: Linear(d_model → [z, x, B, C, dt])
  conv1d:  Depthwise causal Conv1d on [x, B, C]
  SSD scan: chunked quadratic + linear recurrence
  D skip:  direct input-to-output residual
  norm:    RMSNorm with SiLU gating (output = norm(x * silu(z)))
  out_proj: Linear(d_inner → d_model)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class Mamba2Config:
    """Configuration for a Mamba2 block stack."""
    d_model: int = 128      # model dimension (D)
    d_state: int = 64       # state dimension (N)
    d_conv: int = 4         # convolution kernel size
    expand: int = 2         # expansion factor (E)
    headdim: int = 64       # head dimension (P)
    chunk_size: int = 64    # matrix partition size (Q)
    n_layer: int = 2        # number of Mamba2 layers

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, (
            f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
        )
        self.nheads = self.d_inner // self.headdim


# ---------------------------------------------------------------------------
# SSD core (pure PyTorch)
# ---------------------------------------------------------------------------

def segsum(x: Tensor) -> Tensor:
    """Stable segment sum in log-space.

    exp(segsum(A)) produces a 1-semiseparable matrix equivalent to a
    scalar SSM. Uses cumsum + lower-triangular masking.

    Args:
        x: (..., T) log-space decay values.

    Returns:
        (..., T, T) lower-triangular cumulative sums.
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)               # (..., T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device),
                      diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device),
                      diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x: Tensor, A: Tensor, B: Tensor, C: Tensor,
        chunk_size: int) -> tuple[Tensor, Tensor]:
    """Structured State Space Duality — the core of Mamba-2.

    4-step chunked algorithm:
      1. Intra-chunk: quadratic attention within each chunk
      2. State accumulation: B-weighted input states per chunk
      3. Inter-chunk: SSM recurrence across chunk boundaries
      4. State-to-output: C-weighted output from accumulated states

    Args:
        x: (batch, seqlen, nheads, headdim)  — input scaled by dt
        A: (batch, seqlen, nheads)            — log-space decay (negative)
        B: (batch, seqlen, 1, d_state)        — input projection
        C: (batch, seqlen, 1, d_state)        — output projection
        chunk_size: partition size Q

    Returns:
        y: (batch, seqlen, nheads, headdim)
        final_state: (batch, nheads, headdim, d_state)
    """
    batch, seqlen, nheads, headdim = x.shape
    assert seqlen % chunk_size == 0, (
        f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
    )

    # Reshape into chunks: (batch, n_chunks, chunk_size, ...)
    def _chunk(t):
        return t.reshape(batch, seqlen // chunk_size, chunk_size, *t.shape[2:])

    x, A, B, C = _chunk(x), _chunk(A), _chunk(B), _chunk(C)

    # A: (batch, chunks, chunk_size, nheads) → (batch, nheads, chunks, chunk_size)
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # Step 1: Intra-chunk (diagonal blocks) — quadratic in chunk_size
    L = torch.exp(segsum(A))    # (batch, nheads, chunks, chunk_size, chunk_size)
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # Step 2: State accumulation per chunk (B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # Step 3: Inter-chunk recurrence (A terms)
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Step 4: State-to-output (C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Combine intra-chunk and inter-chunk
    Y = (Y_diag + Y_off).reshape(batch, seqlen, nheads, headdim)
    return Y, final_state


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Gated Root Mean Square Layer Normalization.

    When z is provided: output = RMSNorm(x * silu(z))
    This is the "gated" variant used inside each Mamba block.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor, z: Tensor | None = None) -> Tensor:
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# Mamba2 block
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """A single Mamba-2 block (pure PyTorch, no CUDA kernels).

    Input:  (batch, seqlen, d_model)
    Output: (batch, seqlen, d_model)

    seqlen must be a multiple of chunk_size.
    """

    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        d = config.d_inner
        n = config.d_state

        # Combined input projection: [z, x, B, C, dt]
        d_in_proj = 2 * d + 2 * n + config.nheads
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        # Depthwise causal convolution on [x, B, C]
        conv_dim = d + 2 * n
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=config.d_conv,
            groups=conv_dim,
            padding=config.d_conv - 1,
        )

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        self.A_log = nn.Parameter(torch.empty(config.nheads))
        self.D = nn.Parameter(torch.empty(config.nheads))

        # Output
        self.norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, config.d_model, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize SSM parameters following Mamba2 conventions."""
        cfg = self.config
        # A_log: log of negative decay, initialized so A ∈ [-1, -0.01]
        nn.init.uniform_(self.A_log, -5.0, -1.0)
        # D: direct skip connection, small positive
        nn.init.ones_(self.D)
        # dt_bias: softplus(dt_bias) gives initial dt ∈ [0.1, 1.0]
        # softplus(x) ≈ x for x > 5, so bias ~0.5 → dt ~1.1
        nn.init.uniform_(self.dt_bias, -1.0, 1.0)
        # Projections: default Xavier is fine for in_proj/out_proj
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, u: Tensor) -> Tensor:
        """Forward pass through one Mamba2 block.

        Args:
            u: (batch, seqlen, d_model) input tensor.
                seqlen must be divisible by chunk_size.

        Returns:
            (batch, seqlen, d_model) output tensor.
        """
        cfg = self.config
        batch, seqlen, _ = u.shape

        A = -torch.exp(self.A_log)                          # (nheads,)

        # Project input → [z, xBC, dt]
        zxbcdt = self.in_proj(u)                             # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.d_inner + 2 * cfg.d_state, cfg.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)                   # (batch, seqlen, nheads)

        # Causal convolution on [x, B, C]
        xBC = F.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
        )

        # Split into x, B, C
        x, B, C = torch.split(
            xBC, [cfg.d_inner, cfg.d_state, cfg.d_state], dim=-1
        )

        # Reshape for multi-head SSD
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)

        # SSD scan
        y, _ = ssd(
            x * dt.unsqueeze(-1),                            # scale input by dt
            A * dt,                                          # decay per timestep
            B.unsqueeze(2),                                  # (batch, seqlen, 1, d_state)
            C.unsqueeze(2),                                  # (batch, seqlen, 1, d_state)
            cfg.chunk_size,
        )

        # D skip connection
        y = y + x * self.D.unsqueeze(-1)

        # Flatten heads, gated norm, output projection
        y = y.reshape(batch, seqlen, cfg.d_inner)
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y
