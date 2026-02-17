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
    mimo_rank: int = 1       # r=1 is SISO (default), r>1 is MIMO
    stable_ssm: bool = False # StableSSM "best" reparameterization for A-matrix
                             # (Wang & Li 2024, arXiv:2311.14495)
    use_triton: bool = False # Use Triton-accelerated SSD kernels when available
    use_mhc: bool = False    # Manifold-constrained hyperconnections (Xiao et al. 2025)
    mhc_n_streams: int = 4   # Number of residual streams (expansion rate)
    mhc_alpha_init: float = 0.01  # Initial gating parameter (small → ≈uniform mixing)
    mhc_sinkhorn_iters: int = 20  # Sinkhorn-Knopp iterations for column-stochastic

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


# ---------------------------------------------------------------------------
# Manifold-Constrained Hyperconnections (mHC)
# ---------------------------------------------------------------------------
# Ref: Xiao et al. 2025, "Manifold-Constrained Hyperconnections"
#
# Replaces standard residual connections (x + f(x)) with multi-stream
# gated communication. Each sublayer reads from a weighted combination of
# n streams and writes its output back to all streams via a column-stochastic
# gating matrix. The constraint (each column sums to 1) preserves the
# "residual manifold", preventing norm drift across layers.
#
# Compute overhead: ~2-7% (dominated by stream mixing, not Sinkhorn on tiny
# n×n matrices). Parameter overhead: negligible (~200 params for n=4, L=4).
# ---------------------------------------------------------------------------

def sinkhorn_normalize(M_raw: Tensor, n_iters: int = 20) -> Tensor:
    """Column-stochastic normalization via Sinkhorn-Knopp.

    Uses straight-through estimator: forward pass uses full Sinkhorn result,
    backward pass sees only single-step column normalization. This avoids
    backpropagating through the iterative loop (20 iters → 1 step in grad).

    Args:
        M_raw: raw parameter matrix, typically (n+1, n)
        n_iters: Sinkhorn iterations (paper default: 20)

    Returns:
        Column-stochastic matrix (same shape, each column sums to 1)
    """
    # Forward: full Sinkhorn in no_grad (no graph built for iterations)
    with torch.no_grad():
        M = M_raw.exp()
        for _ in range(n_iters):
            M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
        M_target = M / (M.sum(dim=0, keepdim=True) + 1e-8)

    # Backward: single-step column softmax (differentiable, cheap)
    M_soft = M_raw.exp()
    M_soft = M_soft / (M_soft.sum(dim=0, keepdim=True) + 1e-8)

    # Straight-through: forward value = M_target, gradient from M_soft
    return M_soft + (M_target - M_soft).detach()


class HyperConnection(nn.Module):
    """Manifold-constrained hyperconnection for one sublayer.

    Manages the multi-stream residual connection. The sublayer reads from
    a softmax-weighted combination of n streams and writes its output back
    to all n streams via a column-stochastic gating matrix.

    The gating matrix M ∈ R^((n+1)×n) has:
      - Rows 0..n-1: how each existing stream contributes to each new stream
      - Row n: how the sublayer output contributes to each new stream
      - Column-stochastic: each new stream's source weights sum to 1

    This generalizes standard residual connections: when the diagonal of
    rows 0..n-1 dominates, streams bypass the layer (residual); when row n
    dominates, the sublayer output replaces the streams.

    Args:
        n_streams: number of residual streams (expansion rate)
        alpha_init: initial gating value (small → approx uniform mixing)
        sinkhorn_iters: iterations for manifold constraint enforcement
    """

    def __init__(self, n_streams: int, alpha_init: float = 0.01,
                 sinkhorn_iters: int = 20):
        super().__init__()
        self.n = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Gate: (n+1) sources × n destinations
        self.gate = nn.Parameter(
            torch.full((n_streams + 1, n_streams), alpha_init))

        # Input combination weights (softmax-normalized)
        # Initialize to strongly prefer stream 0 (primary residual)
        init_w = torch.zeros(n_streams)
        init_w[0] = 5.0  # After softmax: ~98% weight on stream 0
        self.input_logits = nn.Parameter(init_w)

    def get_input(self, streams: Tensor) -> Tensor:
        """Combine n streams → sublayer input.

        Args:
            streams: (batch, seq_len, n, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        w = F.softmax(self.input_logits, dim=0)  # (n,)
        return torch.einsum('n, bsnd -> bsd', w, streams)

    def update(self, streams: Tensor, sublayer_output: Tensor) -> Tensor:
        """Mix sublayer output back into all streams.

        Args:
            streams: (batch, seq_len, n, d_model) current streams
            sublayer_output: (batch, seq_len, d_model) sublayer output

        Returns:
            (batch, seq_len, n, d_model) updated streams
        """
        M = sinkhorn_normalize(self.gate, self.sinkhorn_iters)  # (n+1, n)
        # Split gating: stream-to-stream mixing + output injection
        # Avoids torch.cat temporary tensor allocation
        M_streams = M[:self.n]  # (n, n)
        M_output = M[self.n]    # (n,)
        new = torch.einsum('jn, bsjd -> bsnd', M_streams, streams)
        new = new + M_output[None, None, :, None] * sublayer_output.unsqueeze(2)
        return new


# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------

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

def stable_log_decay(w: Tensor) -> Tensor:
    """StableSSM 'best' reparameterization in log-space.

    Maps unconstrained w to log-space decay compatible with the SSD core.
    Ref: Wang & Li 2024, "StableSSM" (arXiv:2311.14495, ICML 2024).

    The "best" stable reparameterization:
        decay = 1 - 1/(w² + 0.5)

    For |w| > sqrt(0.5) ≈ 0.707: decay ∈ (0, 1) — valid positive decay.
    For smaller |w|: decay ≤ 0 — clamped to avoid log of non-positive.

    Returns log(decay) (negative), suitable for segsum/exp in SSD.
    """
    decay = 1.0 - 1.0 / (w * w + 0.5)
    # Clamp to small positive value to keep log defined
    decay = decay.clamp(min=1e-6)
    return torch.log(decay)


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


def mamba3_mimo_recurrence(
    x_write: Tensor,       # (batch, seqlen, nheads, headdim, r) MIMO write input
    x_write_prev: Tensor,  # (batch, seqlen, nheads, headdim, r) shifted for trapezoidal
    A_dt: Tensor,          # (batch, seqlen, nheads) log-decay (dt * A, negative)
    B: Tensor,             # (batch, seqlen, r, d_state)
    B_prev: Tensor,        # (batch, seqlen, r, d_state) shifted for trapezoidal
    C: Tensor,             # (batch, seqlen, r, d_state)
    lam: Tensor,           # (batch, seqlen, 1, 1) trapezoidal mixing
) -> Tensor:
    """Sequential MIMO recurrence for Mamba3 (no delta rule).

    Reference implementation for MIMO comparison. For SISO (r=1),
    prefer ssd_trapz for speed.

    Returns:
        (batch, seqlen, nheads, headdim, r)
    """
    batch, seqlen, nheads, headdim, r = x_write.shape
    d_state = B.shape[-1]
    device, dtype = x_write.device, x_write.dtype

    h = torch.zeros(batch, nheads, headdim, d_state, device=device, dtype=dtype)
    outputs = []

    for t in range(seqlen):
        # Decay: scalar per head, broadcast to all channels
        decay = torch.exp(A_dt[:, t])  # (batch, nheads)
        h = h * decay[:, :, None, None]

        # Write: rank-r MIMO outer product sum
        xw_t = x_write[:, t]  # (batch, nheads, headdim, r)
        b_t = B[:, t]         # (batch, r, d_state)

        if t > 0:
            xw_prev_t = x_write_prev[:, t]
            b_prev_t = B_prev[:, t]
            lam_t = lam[:, t]  # (batch, 1, 1)

            write_euler = torch.einsum('bnpi,bid->bnpd', xw_t, b_t)
            # Previous term gets this step's decay (trapezoidal rule)
            write_trapz = decay[:, :, None, None] * torch.einsum(
                'bnpi,bid->bnpd', xw_prev_t, b_prev_t)
            write = lam_t.unsqueeze(1) * write_euler + (1.0 - lam_t.unsqueeze(1)) * write_trapz
        else:
            write = torch.einsum('bnpi,bid->bnpd', xw_t, b_t)

        h = h + write

        # Readout: rank-r
        c_t = C[:, t]  # (batch, r, d_state)
        y_t = torch.einsum('bnpd,bid->bnpi', h, c_t)  # (batch, nheads, headdim, r)

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)


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
        r = config.mimo_rank

        # With PoPE, B/C are projected to half-size (n//2) then expanded
        # to full d_state via polar encoding.  Theta dimension is the same.
        d_bc = n // 2 if config.use_pope else n

        # Input projection: [z, x, B, C, dt, theta, lambda]
        # z: gating (d_inner)
        # x: SSM input (d_inner)
        # B: input projection (d_bc * r for MIMO)
        # C: output projection (d_bc * r for MIMO)
        # dt: timestep (nheads)
        # theta: rotation angles (d_state // 2)
        # lambda: trapezoidal mixing (1 scalar)
        d_in_proj = 2 * d + 2 * d_bc * r + config.nheads + n // 2 + 1
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        # BC bias (learnable, channel-wise, applied AFTER QK-norm)
        # Paper Section 3.4: "initialized to all ones"
        self.B_bias = nn.Parameter(torch.ones(d_bc * r))
        self.C_bias = nn.Parameter(torch.ones(d_bc * r))

        # QK-norm on B, C (applied on raw features before rotation/polar encoding)
        self.B_norm = RMSNorm(d_bc * r)
        self.C_norm = RMSNorm(d_bc * r)

        # PoPE learnable phase bias (delta) — controls softplus operating point
        # Shared across MIMO columns (position encoding is column-independent)
        if config.use_pope:
            self.pope_delta_B = nn.Parameter(torch.zeros(d_bc))
            self.pope_delta_C = nn.Parameter(torch.zeros(d_bc))

        # Optional short convolution (on x only; Mamba3 removes this by default)
        if config.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=d, out_channels=d,
                kernel_size=config.d_conv, groups=d,
                padding=config.d_conv - 1,
            )
        else:
            self.conv1d = None

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        if config.stable_ssm:
            # StableSSM "best" reparameterization (Wang & Li 2024):
            #   decay = 1 - 1/(w^2 + 0.5)  where w = A_raw * dt
            # Unconstrained A_raw; negative values give decay < 1.
            self.A_raw = nn.Parameter(torch.empty(config.nheads))
        else:
            # Standard Mamba parameterization: A = -exp(A_log)
            self.A_log = nn.Parameter(torch.empty(config.nheads))
        self.D = nn.Parameter(torch.empty(config.nheads))

        # MIMO projections
        if r > 1:
            self.mimo_x_proj = nn.Linear(d, d * r, bias=False)
            self.mimo_out_proj = nn.Linear(d * r, d, bias=False)

        # Output
        self.out_norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, config.d_model, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        cfg = self.config
        if cfg.stable_ssm:
            # Initialize A_raw so that initial decay ≈ exp(-exp(uniform(-5,-1))*dt)
            # With typical dt~1: w~-2 gives decay≈0.78, w~-1 gives decay≈0.33
            nn.init.uniform_(self.A_raw, -3.0, -0.5)
        else:
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
        r = cfg.mimo_rank

        # A-matrix / decay computation
        if cfg.stable_ssm:
            A_raw = self.A_raw  # (nheads,) unconstrained
        else:
            A_raw = -torch.exp(self.A_log)  # (nheads,) guaranteed negative

        # Project input
        d_bc = cfg.d_state // 2 if cfg.use_pope else cfg.d_state
        proj = self.in_proj(u)
        z, x, B_raw, C_raw, dt, theta, lam_logit = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, d_bc * r, d_bc * r,
             cfg.nheads, cfg.d_state // 2, 1],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)
        lam = torch.sigmoid(lam_logit)       # (batch, seqlen, 1)

        # Optional convolution on x
        # Paper Section 3.4: trapezoidal discretization eliminates the need
        # for "the original short causal convolution and its accompanying
        # activation function."  No SiLU on x when conv is off.
        if self.conv1d is not None:
            x = F.silu(
                self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen, :]
            )

        # QK-norm then BC bias (paper: "bias ... after its normalization")
        B_raw = self.B_norm(B_raw) + self.B_bias
        C_raw = self.C_norm(C_raw) + self.C_bias

        # Accumulate theta across positions for positional encoding
        theta_cumsum = torch.cumsum(theta, dim=1)

        # Apply PoPE/RoPE to B, C (per MIMO column if r > 1)
        if cfg.use_pope:
            if r > 1:
                B_res = B_raw.reshape(batch, seqlen, r, d_bc)
                C_res = C_raw.reshape(batch, seqlen, r, d_bc)
                B_list = [apply_pope(B_res[:, :, i], theta_cumsum, self.pope_delta_B) for i in range(r)]
                C_list = [apply_pope(C_res[:, :, i], theta_cumsum, self.pope_delta_C) for i in range(r)]
                B = torch.stack(B_list, dim=2)  # (batch, seqlen, r, d_state)
                C = torch.stack(C_list, dim=2)
            else:
                B = apply_pope(B_raw, theta_cumsum, self.pope_delta_B)
                C = apply_pope(C_raw, theta_cumsum, self.pope_delta_C)
        else:
            if r > 1:
                B_res = B_raw.reshape(batch, seqlen, r, d_bc)
                C_res = C_raw.reshape(batch, seqlen, r, d_bc)
                B_list = [apply_rope(B_res[:, :, i], theta_cumsum) for i in range(r)]
                C_list = [apply_rope(C_res[:, :, i], theta_cumsum) for i in range(r)]
                B = torch.stack(B_list, dim=2)
                C = torch.stack(C_list, dim=2)
            else:
                B = apply_rope(B_raw, theta_cumsum)
                C = apply_rope(C_raw, theta_cumsum)

        # Reshape for multi-head
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        dt_exp = dt.unsqueeze(-1)  # (batch, seqlen, nheads, 1) for broadcasting
        x_dt = x * dt_exp  # scale by current dt

        # Log-space decay: A_dt < 0 so exp(A_dt) ∈ (0, 1)
        if cfg.stable_ssm:
            # StableSSM: decay = 1 - 1/((A_raw*dt)² + 0.5), in log-space
            A_dt = stable_log_decay(A_raw * dt)  # (batch, seqlen, nheads)
        else:
            A_dt = A_raw * dt  # standard: A_raw = -exp(A_log) < 0

        if r > 1:
            # --- MIMO path: sequential recurrence (reference implementation) ---
            x_flat = x_dt.reshape(batch, seqlen, cfg.d_inner)
            x_write = self.mimo_x_proj(x_flat).reshape(batch, seqlen, cfg.nheads, cfg.headdim, r)

            x_write_prev = F.pad(x_write[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
            B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))
            lam_expand = lam.unsqueeze(-1)  # (batch, seqlen, 1, 1)

            y = mamba3_mimo_recurrence(
                x_write, x_write_prev,
                A_dt,
                B, B_prev, C,
                lam_expand,
            )
            # y: (batch, seqlen, nheads, headdim, r)

            # Contract MIMO output
            y = y.reshape(batch, seqlen, cfg.d_inner * r)
            y = self.mimo_out_proj(y)
            y = y.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        else:
            # --- SISO path: SSD (fast, chunkwise parallel) ---
            # Paper Eq. 4: β_t = (1-λ_t) * Δ_t * exp(Δ_t A_t)
            # Both γ_t and β_t use the CURRENT Δ_t, so x_prev must be
            # shifted raw x scaled by current dt (not previous dt).
            x_raw_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
            x_prev = x_raw_prev * dt_exp  # current dt applied to previous x
            B_prev = F.pad(B[:, :-1].unsqueeze(2), (0, 0, 0, 0, 1, 0))
            B_curr = B.unsqueeze(2)
            C_curr = C.unsqueeze(2)
            lam_expand = lam.unsqueeze(-1)  # (batch, seqlen, 1, 1)

            _ssd_fn = ssd_trapz
            if cfg.use_triton:
                try:
                    from .triton_ssd import ssd_trapz_triton
                    _ssd_fn = ssd_trapz_triton
                except ImportError:
                    pass  # fall back to PyTorch

            y, _ = _ssd_fn(
                x_dt, x_prev,
                A_dt,
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
        Standard:  Embedding → N x Mamba3Block → RMSNorm → Linear(vocab)
        With mHC:  Embedding → Expand(n) → N x (mHC-Mixer + mHC-MLP) → Contract → RMSNorm → Linear(vocab)

    When use_mhc=True, the residual stream is expanded to n parallel streams.
    Each sublayer (mixer, MLP) reads from a weighted combination of streams
    and writes back via a column-stochastic gating matrix (manifold constraint).
    Ref: Xiao et al. 2025, "Manifold-Constrained Hyperconnections"

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

        # mHC: per-sublayer hyperconnections (mixer + MLP per block)
        if config.use_mhc:
            n = config.mhc_n_streams
            a = config.mhc_alpha_init
            s = config.mhc_sinkhorn_iters
            self.mixer_hcs = nn.ModuleList([
                HyperConnection(n, a, s) for _ in range(config.n_layer)
            ])
            self.mlp_hcs = nn.ModuleList([
                HyperConnection(n, a, s) for _ in range(config.n_layer)
            ])
            # Contraction weights (combine n streams → 1 at model exit)
            self.contract_logits = nn.Parameter(torch.zeros(n))

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seqlen) token indices.

        Returns:
            logits: (batch, seqlen, vocab_size)
        """
        x = self.embedding(input_ids)

        if self.config.use_mhc:
            n = self.config.mhc_n_streams
            # Expand to n streams: (batch, seq, d) → (batch, seq, n, d)
            # expand() creates a view (no copy); update() produces new tensors via einsum
            streams = x.unsqueeze(2).expand(-1, -1, n, -1)

            for i, block in enumerate(self.layers):
                # Mixer sublayer with mHC
                mixer_in = self.mixer_hcs[i].get_input(streams)
                mixer_out = block.mixer(block.mixer_norm(mixer_in))
                streams = self.mixer_hcs[i].update(streams, mixer_out)

                # MLP sublayer with mHC
                mlp_in = self.mlp_hcs[i].get_input(streams)
                mlp_out = block.mlp(block.mlp_norm(mlp_in))
                streams = self.mlp_hcs[i].update(streams, mlp_out)

            # Contract: (batch, seq, n, d) → (batch, seq, d)
            w = F.softmax(self.contract_logits, dim=0)  # (n,)
            x = torch.einsum('n, bsnd -> bsd', w, streams)
        else:
            for layer in self.layers:
                x = layer(x)

        x = self.norm(x)
        return self.out_proj(x)
