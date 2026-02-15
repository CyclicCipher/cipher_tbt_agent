"""
Naja: Mamba3 + Delta Rule + MIMO + PoPE orthogonal pair.

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
from torch.utils.checkpoint import checkpoint


@dataclass
class NajaConfig:
    """Configuration for Naja."""
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

    # --- Phase 5: Chunkwise ---
    chunk_size: int = 64
    use_chunkwise: bool = False   # gradient-checkpointed chunk processing
    use_wy_chunkwise: bool = False  # real WY chunkwise parallelism (Phase 5a)

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
    x_write: Tensor,       # (batch, seqlen, nheads, headdim, r) MIMO write input
    x_write_prev: Tensor,  # (batch, seqlen, nheads, headdim, r) shifted for trapezoidal
    B1: Tensor,            # (batch, seqlen, r, d_state) primary key (r MIMO columns)
    B1_prev: Tensor,       # shifted B1 for trapezoidal
    B2: Optional[Tensor],  # (batch, seqlen, r, d_state) orthogonal key (or None)
    C: Tensor,             # (batch, seqlen, r, d_state)
    alpha: Tensor,         # (batch, seqlen, nheads, d_state) per-channel decay
    beta1: Optional[Tensor],  # (batch, seqlen, nheads, 1) write gate
    beta2: Optional[Tensor],  # (batch, seqlen, nheads, 1) rotation gate (or None)
    lam: Tensor,           # (batch, seqlen, 1, 1) trapezoidal mixing
    use_trapezoidal: bool = True,
    use_delta: bool = True,
) -> Tensor:
    """Naive sequential delta-rule recurrence with MIMO support.

    This is the reference implementation for correctness. Not optimized.
    Chunkwise WY parallelism is Phase 5.

    State shape per head: (headdim, d_state) — same for SISO and MIMO.
    MIMO increases write/read rank without growing state size.

    Write: rank-r outer product sum  Σ_i x_write[:,:,:,i] ⊗ B1[:,:,i,:]
    Read:  rank-r readout            y[:,:,:,i] = h · C[:,:,i,:]
    Erase: uses first MIMO column as Householder key direction.

    Returns:
        (batch, seqlen, nheads, headdim, r) — r readout columns.
        Caller contracts r dimension (via mimo_out_proj or squeeze).
    """
    batch, seqlen, nheads, headdim, r = x_write.shape
    d_state = B1.shape[-1]
    device, dtype = x_write.device, x_write.dtype

    # State: (batch, nheads, headdim, d_state) — unchanged by MIMO
    h = torch.zeros(batch, nheads, headdim, d_state, device=device, dtype=dtype)
    outputs = []

    for t in range(seqlen):
        # --- Decay: per-channel diagonal ---
        a_t = alpha[:, t]  # (batch, nheads, d_state)
        h = h * a_t.unsqueeze(2)  # broadcast over headdim

        # --- Delta rule: Householder erase (first MIMO column as key) ---
        if use_delta and beta1 is not None:
            # Primary erase direction: first column of B1, shared across heads
            b1_key = B1[:, t, 0, :]  # (batch, d_state)
            b1_hat = F.normalize(b1_key, dim=-1)
            bt1 = beta1[:, t]  # (batch, nheads, 1)

            # Householder 1: h -= β₁ · (h · b̂₁) ⊗ b̂₁
            proj1 = torch.einsum('bnpd,bd->bnp', h, b1_hat)
            h = h - bt1.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj1, b1_hat)

            # Householder 2: PoPE orthogonal pair
            if B2 is not None and beta2 is not None:
                b2_key = B2[:, t, 0, :]  # (batch, d_state)
                b2_hat = F.normalize(b2_key, dim=-1)
                bt2 = beta2[:, t]

                proj2 = torch.einsum('bnpd,bd->bnp', h, b2_hat)
                h = h - bt2.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj2, b2_hat)

        # --- Write: rank-r MIMO outer product sum ---
        # Σ_i x_write[:,:,:,i] ⊗ B1[:,:,i,:] contracts over i (MIMO rank)
        xw_t = x_write[:, t]  # (batch, nheads, headdim, r)
        b1_all = B1[:, t]     # (batch, r, d_state) — shared across heads

        if use_trapezoidal and t > 0:
            xw_prev_t = x_write_prev[:, t]  # (batch, nheads, headdim, r)
            b1_prev_t = B1_prev[:, t]        # (batch, r, d_state)
            lam_t = lam[:, t]                # (batch, 1, 1)

            write_euler = torch.einsum('bnpi,bid->bnpd', xw_t, b1_all)
            write_trapz = torch.einsum('bnpi,bid->bnpd', xw_prev_t, b1_prev_t)
            write = lam_t.unsqueeze(1) * write_euler + (1.0 - lam_t.unsqueeze(1)) * write_trapz
        else:
            write = torch.einsum('bnpi,bid->bnpd', xw_t, b1_all)

        if use_delta and beta1 is not None:
            write = beta1[:, t].unsqueeze(2) * write

        h = h + write

        # Write via B2 (orthogonal direction, rank-r)
        if B2 is not None and beta2 is not None and use_delta:
            b2_all = B2[:, t]  # (batch, r, d_state)
            write2 = beta2[:, t].unsqueeze(2) * torch.einsum('bnpi,bid->bnpd', xw_t, b2_all)
            h = h + write2

        # --- Readout: rank-r ---
        # Each MIMO column i gives a separate headdim-dimensional readout
        c_all = C[:, t]  # (batch, r, d_state)
        y_t = torch.einsum('bnpd,bid->bnpi', h, c_all)  # (batch, nheads, headdim, r)

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)  # (batch, seqlen, nheads, headdim, r)


# ---------------------------------------------------------------------------
# Phase 5: Chunkwise Recurrence with Gradient Checkpointing
# ---------------------------------------------------------------------------

def _process_chunk(
    h: Tensor,
    x_write: Tensor, x_write_prev: Tensor,
    B1: Tensor, B1_prev: Tensor, B2: Optional[Tensor],
    C: Tensor, alpha: Tensor,
    beta1: Optional[Tensor], beta2: Optional[Tensor],
    lam: Tensor,
    use_trapezoidal: bool, use_delta: bool,
    chunk_start: int,
) -> tuple:
    """Process one chunk of the recurrence. Designed for gradient checkpointing.

    Args:
        h: (batch, nheads, headdim, d_state) incoming state.
        Remaining args: sliced to this chunk's time range.
        chunk_start: absolute position of chunk start (for trapezoidal t>0 check).

    Returns:
        (y_chunk, h_out): chunk outputs and outgoing state.
    """
    chunk_len = x_write.shape[1]
    outputs = []

    for t in range(chunk_len):
        abs_t = chunk_start + t

        # Decay
        a_t = alpha[:, t]
        h = h * a_t.unsqueeze(2)

        # Delta rule erase
        if use_delta and beta1 is not None:
            b1_key = B1[:, t, 0, :]
            b1_hat = F.normalize(b1_key, dim=-1)
            bt1 = beta1[:, t]
            proj1 = torch.einsum('bnpd,bd->bnp', h, b1_hat)
            h = h - bt1.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj1, b1_hat)

            if B2 is not None and beta2 is not None:
                b2_key = B2[:, t, 0, :]
                b2_hat = F.normalize(b2_key, dim=-1)
                bt2 = beta2[:, t]
                proj2 = torch.einsum('bnpd,bd->bnp', h, b2_hat)
                h = h - bt2.unsqueeze(2) * torch.einsum('bnp,bd->bnpd', proj2, b2_hat)

        # Write
        xw_t = x_write[:, t]
        b1_all = B1[:, t]

        if use_trapezoidal and abs_t > 0:
            xw_prev_t = x_write_prev[:, t]
            b1_prev_t = B1_prev[:, t]
            lam_t = lam[:, t]
            write_euler = torch.einsum('bnpi,bid->bnpd', xw_t, b1_all)
            write_trapz = torch.einsum('bnpi,bid->bnpd', xw_prev_t, b1_prev_t)
            write = lam_t.unsqueeze(1) * write_euler + (1.0 - lam_t.unsqueeze(1)) * write_trapz
        else:
            write = torch.einsum('bnpi,bid->bnpd', xw_t, b1_all)

        if use_delta and beta1 is not None:
            write = beta1[:, t].unsqueeze(2) * write
        h = h + write

        if B2 is not None and beta2 is not None and use_delta:
            b2_all = B2[:, t]
            write2 = beta2[:, t].unsqueeze(2) * torch.einsum('bnpi,bid->bnpd', xw_t, b2_all)
            h = h + write2

        # Readout
        c_all = C[:, t]
        y_t = torch.einsum('bnpd,bid->bnpi', h, c_all)
        outputs.append(y_t)

    return torch.stack(outputs, dim=1), h


def delta_recurrence_chunkwise(
    x_write: Tensor, x_write_prev: Tensor,
    B1: Tensor, B1_prev: Tensor, B2: Optional[Tensor],
    C: Tensor, alpha: Tensor,
    beta1: Optional[Tensor], beta2: Optional[Tensor],
    lam: Tensor,
    use_trapezoidal: bool = True, use_delta: bool = True,
    chunk_size: int = 64,
) -> Tensor:
    """Chunkwise delta-rule recurrence with gradient checkpointing.

    Splits the sequence into chunks and applies torch.utils.checkpoint
    to each chunk, trading compute for memory during backward pass.
    Activations within each chunk are recomputed during backward instead
    of stored, reducing peak memory from O(seqlen) to O(chunk_size).

    Same interface as delta_recurrence; drop-in replacement.
    """
    batch, seqlen, nheads, headdim, r = x_write.shape
    d_state = B1.shape[-1]
    device, dtype = x_write.device, x_write.dtype

    h = torch.zeros(batch, nheads, headdim, d_state, device=device, dtype=dtype)
    all_outputs = []

    for start in range(0, seqlen, chunk_size):
        end = min(start + chunk_size, seqlen)
        sl = slice(start, end)

        # Slice all inputs to this chunk
        xw_c = x_write[:, sl]
        xwp_c = x_write_prev[:, sl]
        b1_c = B1[:, sl]
        b1p_c = B1_prev[:, sl]
        b2_c = B2[:, sl] if B2 is not None else None
        c_c = C[:, sl]
        a_c = alpha[:, sl]
        bt1_c = beta1[:, sl] if beta1 is not None else None
        bt2_c = beta2[:, sl] if beta2 is not None else None
        lam_c = lam[:, sl]

        # Gradient checkpoint: recompute activations during backward
        y_chunk, h = checkpoint(
            _process_chunk,
            h, xw_c, xwp_c, b1_c, b1p_c, b2_c, c_c, a_c, bt1_c, bt2_c,
            lam_c, use_trapezoidal, use_delta, start,
            use_reentrant=False,
        )
        all_outputs.append(y_chunk)

    return torch.cat(all_outputs, dim=1)


# ---------------------------------------------------------------------------
# Phase 5a: WY Chunkwise Delta Recurrence (Real Parallelism)
# ---------------------------------------------------------------------------
#
# Implements the 4-step WY chunkwise algorithm from:
#   Yang et al. 2024 — "Parallelizing Linear Transformers with the Delta Rule"
#   (arXiv:2406.06484), Section 3 and Appendix B.
#
# Phase 5a simplifications (to be lifted in Phase 5b):
#   - SISO only (r=1): single MIMO column for erase key
#   - Single Householder (B1 only): no PoPE pair B2
#   - Scalar decay: mean of per-channel α (not full diagonal)
#   - Trapezoidal blending baked into V before WY transform
#
# The algorithm reduces sequential steps from L to L/C by computing
# intra-chunk work via matrix multiplications (parallel on GPU).

def _chunk_scaled_dot_kkt(K: Tensor, beta: Tensor, log_alpha_cumsum: Tensor) -> Tensor:
    """Compute A = tril(diag(β) · Γ ⊙ (K · K^T), -1) per chunk.

    Includes intra-chunk decay: A_{i,j} = β_j · γ_{i,j} · (k_i · k_j)
    where γ_{i,j} = exp(cumsum_log_α[i] - cumsum_log_α[j]) is the
    cumulative decay from position j to position i within the chunk.

    Args:
        K: (batch, nheads, n_chunks, C, d_k) — keys within each chunk.
        beta: (batch, nheads, n_chunks, C) — write gates.
        log_alpha_cumsum: (batch, nheads, n_chunks, C) — cumulative log-decay.

    Returns:
        A: (batch, nheads, n_chunks, C, C) — strictly lower triangular.
    """
    # K K^T: (batch, nheads, n_chunks, C, C)
    KKt = torch.einsum('bncid, bncjd -> bncij', K, K)
    # Scale columns by beta_j
    A = KKt * beta.unsqueeze(-2)  # broadcast beta over row dim
    # Intra-chunk decay matrix: γ_{i,j} = exp(cumsum[i] - cumsum[j])
    decay_matrix = torch.exp(
        log_alpha_cumsum.unsqueeze(-1) - log_alpha_cumsum.unsqueeze(-2)
    )  # (batch, nheads, n_chunks, C, C)
    A = A * decay_matrix
    # Strictly lower triangular
    C_sz = A.shape[-1]
    mask = torch.tril(torch.ones(C_sz, C_sz, device=A.device, dtype=torch.bool), diagonal=-1)
    A = A.masked_fill(~mask, 0.0)
    return A


def _solve_tril(A: Tensor, beta: Tensor) -> Tensor:
    """Compute T = (I + A)^{-1} · diag(β) via forward substitution.

    Args:
        A: (batch, nheads, n_chunks, C, C) — strictly lower triangular.
        beta: (batch, nheads, n_chunks, C) — write gates.

    Returns:
        T: (batch, nheads, n_chunks, C, C) — UT transform matrix.
    """
    C_sz = A.shape[-1]
    # (I + A) is unit lower triangular — solve (I+A)T = diag(β)
    # Use torch.linalg.solve_triangular for batched solve
    # RHS is diag(beta): (batch, nheads, n_chunks, C, C)
    eye_beta = torch.diag_embed(beta)  # (batch, nheads, n_chunks, C, C)
    IpA = torch.eye(C_sz, device=A.device, dtype=A.dtype) + A
    # solve_triangular: solve IpA @ T = eye_beta for T
    # IpA is lower triangular, unitriangular=True
    T = torch.linalg.solve_triangular(IpA, eye_beta, upper=False, unitriangular=True)
    return T


def _prepare_wy_repr(K: Tensor, V: Tensor, beta: Tensor, log_alpha_cumsum: Tensor) -> tuple:
    """Full UT transform: compute pseudo-keys W and pseudo-values U.

    Args:
        K: (batch, nheads, n_chunks, C, d_k) — keys.
        V: (batch, nheads, n_chunks, C, d_v) — values (write vectors).
        beta: (batch, nheads, n_chunks, C) — write gates.
        log_alpha_cumsum: (batch, nheads, n_chunks, C) — cumulative log-decay.

    Returns:
        W: (batch, nheads, n_chunks, C, d_k) — pseudo-keys.
        U: (batch, nheads, n_chunks, C, d_v) — pseudo-values.
        T: (batch, nheads, n_chunks, C, C) — UT matrix (for debugging).
    """
    A = _chunk_scaled_dot_kkt(K, beta, log_alpha_cumsum)
    T = _solve_tril(A, beta)  # (batch, nheads, n_chunks, C, C)
    W = torch.einsum('bncij, bncjd -> bncid', T, K)  # pseudo-keys
    U = torch.einsum('bncij, bncjd -> bncid', T, V)  # pseudo-values
    return W, U, T


def delta_recurrence_wy(
    x_write: Tensor,       # (batch, seqlen, nheads, headdim, r) MIMO write input
    x_write_prev: Tensor,  # (batch, seqlen, nheads, headdim, r) shifted for trapezoidal
    B1: Tensor,            # (batch, seqlen, r, d_state) primary key (r MIMO columns)
    B1_prev: Tensor,       # shifted B1 for trapezoidal
    B2: Optional[Tensor],  # ignored in Phase 5a (single Householder only)
    C: Tensor,             # (batch, seqlen, r, d_state)
    alpha: Tensor,         # (batch, seqlen, nheads, d_state) per-channel decay
    beta1: Optional[Tensor],  # (batch, seqlen, nheads, 1) write gate
    beta2: Optional[Tensor],  # ignored in Phase 5a
    lam: Tensor,           # (batch, seqlen, 1, 1) trapezoidal mixing
    use_trapezoidal: bool = True,
    use_delta: bool = True,
    chunk_size: int = 64,
) -> Tensor:
    """WY chunkwise delta-rule recurrence (Phase 5a).

    Implements the 4-step algorithm from Yang et al. 2024 with Naja's
    specific parameterization. Same interface as delta_recurrence.

    Phase 5a limitations:
    - SISO only (uses first MIMO column r=0 for erase key)
    - Single Householder (B1 only, B2 ignored)
    - Scalar decay (mean of per-channel α within each chunk)

    Returns:
        (batch, seqlen, nheads, headdim, r) — same shape as delta_recurrence.
    """
    batch, seqlen, nheads, headdim, r = x_write.shape
    d_state = B1.shape[-1]
    device, dtype = x_write.device, x_write.dtype

    # Pad sequence to multiple of chunk_size if needed
    pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        x_write = F.pad(x_write, (0, 0, 0, 0, 0, 0, 0, pad_len))
        x_write_prev = F.pad(x_write_prev, (0, 0, 0, 0, 0, 0, 0, pad_len))
        B1 = F.pad(B1, (0, 0, 0, 0, 0, pad_len))
        B1_prev = F.pad(B1_prev, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, 0, 0, pad_len), value=1.0)  # decay=1 for padding
        if beta1 is not None:
            beta1 = F.pad(beta1, (0, 0, 0, 0, 0, pad_len))
        lam = F.pad(lam, (0, 0, 0, 0, 0, pad_len), value=0.5)
    L = seqlen + pad_len
    n_chunks = L // chunk_size
    Cs = chunk_size

    # =====================================================================
    # Map Naja's parameterization to DeltaNet's (K, V, Q, beta)
    # =====================================================================

    # K = normalized first MIMO column of B1 (erase key direction)
    # Shape: (batch, L, d_state)
    k_raw = B1[:, :, 0, :]  # first MIMO column
    K = F.normalize(k_raw, dim=-1)

    # Q = first MIMO column of C (readout direction)
    Q = C[:, :, 0, :]  # (batch, L, d_state)

    # beta = write gate (scalar per head)
    if beta1 is not None and use_delta:
        beta_flat = beta1[:, :, :, 0]  # (batch, L, nheads)
    else:
        beta_flat = torch.ones(batch, L, nheads, device=device, dtype=dtype)

    # V = write vector: the value written into state at each position
    # For DeltaNet: V_t = write contribution (before delta correction)
    # With trapezoidal: V_t = λ_t * (x_t ⊗ B1_t) + (1-λ_t) * (x_{t-1} ⊗ B1_{t-1})
    # We need V as (batch, L, nheads, headdim) — the value that gets written

    # Compute write vectors per timestep
    # x_write: (batch, L, nheads, headdim, r), B1: (batch, L, r, d_state)
    # write = einsum('bnpi,bid->bnpd', x_write_t, B1_t) → (batch, nheads, headdim, d_state)
    # But for WY, V should be (batch, L, headdim) with K handling the d_state direction
    # In DeltaNet: state update is S += β_t * (v_t - S @ k_t) ⊗ k_t
    # Which expands to: S += β_t * v_t ⊗ k_t - β_t * (S @ k_t) ⊗ k_t
    # The erase term is the Householder: S -= β_t * (S @ k_t) ⊗ k_t
    # The write term is: S += β_t * v_t ⊗ k_t

    # So V_t (per head) is the value vector written along key direction K_t:
    # For SISO: v_t = x_write[:,:,:,:,0] contracted with B1 but separated as v ⊗ k
    # Actually in DeltaNet formulation: the write is β * v ⊗ k where:
    #   v_t is the "value" (headdim-dimensional)
    #   k_t is the "key" (d_state-dimensional)
    # So our x_write[:,:,:,:,0] IS the value vector (headdim-dim),
    # and B1 normalized is the key.

    # SISO extraction with ||B1|| scaling.
    # The naive recurrence writes: β * (x_write ⊗ B1_unnormalized)
    # WY uses K = normalize(B1) as key, so the value must absorb ||B1||:
    #   β * v ⊗ k = β * (x_write * ||B1||) ⊗ normalize(B1)
    b1_norm = B1[:, :, 0, :].norm(dim=-1, keepdim=True)  # (batch, L, 1)
    b1_norm = b1_norm.unsqueeze(2)  # (batch, L, 1, 1) — broadcast over nheads, headdim
    V_euler = x_write[:, :, :, :, 0] * b1_norm  # (batch, L, nheads, headdim)

    if use_trapezoidal:
        b1_prev_norm = B1_prev[:, :, 0, :].norm(dim=-1, keepdim=True).unsqueeze(2)
        V_prev = x_write_prev[:, :, :, :, 0] * b1_prev_norm
        # lam: (batch, L, 1, 1) — broadcast over nheads, headdim
        lam_v = lam.squeeze(-1)  # (batch, L, 1)
        if lam_v.dim() == 3:
            lam_v = lam_v.unsqueeze(-1)  # (batch, L, 1, 1)
        V = lam_v * V_euler + (1.0 - lam_v) * V_prev
    else:
        V = V_euler

    # Scalar decay: mean across d_state channels per chunk
    # alpha: (batch, L, nheads, d_state) → scalar per (batch, L, nheads)
    alpha_scalar = alpha.mean(dim=-1)  # (batch, L, nheads)

    # =====================================================================
    # Reshape into chunks: (batch, n_chunks, Cs, ...)
    # Then permute to put nheads before n_chunks for batched ops
    # =====================================================================

    def to_chunks(t, name=""):
        """Reshape (batch, L, ...) to (batch, nheads, n_chunks, Cs, ...) or
           (batch, n_chunks, Cs, ...) depending on shape."""
        return t.reshape(batch, n_chunks, Cs, *t.shape[2:])

    K_c = to_chunks(K)                    # (batch, n_chunks, Cs, d_state)
    Q_c = to_chunks(Q)                    # (batch, n_chunks, Cs, d_state)
    V_c = to_chunks(V)                    # (batch, n_chunks, Cs, nheads, headdim)
    beta_c = to_chunks(beta_flat)          # (batch, n_chunks, Cs, nheads)
    alpha_c = to_chunks(alpha_scalar)      # (batch, n_chunks, Cs, nheads)

    # For the WY algorithm, we need K, V, Q, beta all in the shape:
    # (batch, nheads, n_chunks, Cs, feature_dim)
    # K and Q are head-shared (B1, C are shared across heads in Naja)
    # Expand K, Q to have nheads dim
    K_c = K_c.unsqueeze(1).expand(-1, nheads, -1, -1, -1)  # (batch, nheads, n_chunks, Cs, d_state)
    Q_c = Q_c.unsqueeze(1).expand(-1, nheads, -1, -1, -1)  # same

    # V and beta need permutation from (batch, n_chunks, Cs, nheads, ...) to (batch, nheads, n_chunks, Cs, ...)
    V_c = V_c.permute(0, 3, 1, 2, 4)      # (batch, nheads, n_chunks, Cs, headdim)
    beta_c = beta_c.permute(0, 3, 1, 2)    # (batch, nheads, n_chunks, Cs)
    alpha_c = alpha_c.permute(0, 3, 1, 2)  # (batch, nheads, n_chunks, Cs)

    # =====================================================================
    # Precompute decay quantities (needed by both Step 1 and Step 4)
    # =====================================================================
    log_alpha_c = torch.log(alpha_c.clamp(min=1e-8))  # (batch, nheads, n_chunks, Cs)
    log_alpha_cumsum = torch.cumsum(log_alpha_c, dim=-1)  # (batch, nheads, n_chunks, Cs)
    log_gamma_c = log_alpha_c.sum(dim=-1)  # (batch, nheads, n_chunks)
    gamma_c = torch.exp(log_gamma_c)  # cumulative decay for each chunk

    # =====================================================================
    # Step 1: Intra-chunk WY transform
    # =====================================================================
    # Compute pseudo-keys W and pseudo-values U via UT transform
    W, U, _T = _prepare_wy_repr(K_c, V_c, beta_c, log_alpha_cumsum)
    # W: (batch, nheads, n_chunks, Cs, d_state) — pseudo-keys
    # U: (batch, nheads, n_chunks, Cs, headdim) — pseudo-values

    # =====================================================================
    # Step 2: Chunk state accumulation
    # =====================================================================
    # P_c = I - K_c^T · W_c — transition matrix per chunk, (d_state, d_state)
    # H_c = K_c^T · U_c — state contribution per chunk, (d_state, headdim)
    # Contract over position-in-chunk (t), keep feature dims (d, e):
    #   K_c: (b, n, c, t, d_state), W: (b, n, c, t, d_state) → P: (b, n, c, d_state, d_state)
    P = torch.eye(d_state, device=device, dtype=dtype) - \
        torch.einsum('bnctd, bncte -> bncde', K_c, W)
    # P: (batch, nheads, n_chunks, d_state, d_state)

    #   K_c: (b, n, c, t, d_state), U: (b, n, c, t, headdim) → H: (b, n, c, d_state, headdim)
    H = torch.einsum('bnctd, bncte -> bncde', K_c, U)
    # H: (batch, nheads, n_chunks, d_state, headdim)

    # =====================================================================
    # Step 3: Inter-chunk scan (sequential over n_chunks — small!)
    # =====================================================================
    # S_c = gamma_c * P_c @ S_{c-1} + H_c
    # S: (batch, nheads, d_state, headdim) — running hidden state

    states_list = []  # store S_{c-1} for each chunk (for Step 4)
    S = torch.zeros(batch, nheads, d_state, headdim, device=device, dtype=dtype)

    for c in range(n_chunks):
        states_list.append(S.clone())
        g = gamma_c[:, :, c].unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, 1, 1)
        S = g * torch.einsum('bnij, bnjk -> bnik', P[:, :, c], S) + H[:, :, c]

    # Stack: (batch, nheads, n_chunks, d_state, headdim)
    S_prev = torch.stack(states_list, dim=2)

    # =====================================================================
    # Step 4: Intra-chunk output
    # =====================================================================
    # O_c = Q_c @ S_{c-1} + tril(Q_c @ K_c^T) · (U_c - W_c @ S_{c-1})
    #
    # First term: each position reads from the inter-chunk state
    # Q_c: (batch, nheads, n_chunks, Cs, d_state)
    # S_prev: (batch, nheads, n_chunks, d_state, headdim)
    Y_off = torch.einsum('bncid, bncde -> bncie', Q_c, S_prev)
    # Y_off: (batch, nheads, n_chunks, Cs, headdim)

    # Second term: intra-chunk corrections
    # Q_c @ K_c^T: (batch, nheads, n_chunks, Cs, Cs) — attention-like matrix
    QKt = torch.einsum('bncid, bncjd -> bncij', Q_c, K_c)

    # Apply causal mask (lower triangular including diagonal)
    causal_mask = torch.tril(torch.ones(Cs, Cs, device=device, dtype=dtype))
    QKt = QKt * causal_mask

    # Apply per-position decay within chunk:
    # Position i reading from position j (j <= i) has cumulative decay
    # prod_{k=j+1}^{i} alpha_k applied.
    # log_alpha_cumsum already computed above.
    # decay_ij = exp(cumsum[i] - cumsum[j])
    decay_matrix = torch.exp(
        log_alpha_cumsum.unsqueeze(-1) - log_alpha_cumsum.unsqueeze(-2)
    )  # (batch, nheads, n_chunks, Cs, Cs)
    decay_matrix = decay_matrix * causal_mask  # zero out upper triangle
    QKt = QKt * decay_matrix

    # W_c @ S_{c-1}: correction for inter-chunk state
    WS = torch.einsum('bncid, bncde -> bncie', W, S_prev)
    # WS: (batch, nheads, n_chunks, Cs, headdim)

    # Intra-chunk correction: (U - WS) represents the delta-corrected values
    intra_correction = U - WS  # (batch, nheads, n_chunks, Cs, headdim)

    Y_diag = torch.einsum('bncij, bncje -> bncie', QKt, intra_correction)
    # Y_diag: (batch, nheads, n_chunks, Cs, headdim)

    # Also apply decay to the off-diagonal term (state read decays from chunk boundary)
    # decay from chunk start to each position within chunk
    decay_from_start = torch.exp(log_alpha_cumsum)  # (batch, nheads, n_chunks, Cs)
    Y_off = Y_off * decay_from_start.unsqueeze(-1)  # broadcast over headdim

    # Combine
    Y = Y_diag + Y_off
    # Y: (batch, nheads, n_chunks, Cs, headdim)

    # =====================================================================
    # Reshape back to (batch, seqlen, nheads, headdim, r)
    # =====================================================================
    # Y: (batch, nheads, n_chunks, Cs, headdim) → (batch, L, nheads, headdim)
    Y = Y.permute(0, 2, 3, 1, 4)  # (batch, n_chunks, Cs, nheads, headdim)
    Y = Y.reshape(batch, L, nheads, headdim)

    # Remove padding
    if pad_len > 0:
        Y = Y[:, :seqlen]

    # Add MIMO rank dimension (r=1 for Phase 5a)
    Y = Y.unsqueeze(-1)  # (batch, seqlen, nheads, headdim, 1)

    # For r > 1, replicate the output (Phase 5b will handle proper MIMO)
    if r > 1:
        Y = Y.expand(-1, -1, -1, -1, r)

    return Y


# ---------------------------------------------------------------------------
# Phase 6: KL Divergence Surprise (Inference-Time)
# ---------------------------------------------------------------------------

class KLSurpriseTracker(nn.Module):
    """Tracks EMA of predictive distribution and computes KL divergence.

    During inference, surprise = KL(p_t || p̄_t) where:
    - p_t = current softmax prediction
    - p̄_t = exponential moving average of past predictions

    Only the top-k logits are used to keep computation tractable.
    """

    def __init__(self, vocab_size: int, ema_decay: float = 0.99, top_k: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.ema_decay = ema_decay
        self.top_k = min(top_k, vocab_size)
        # EMA of full distribution (updated lazily)
        self.register_buffer(
            'ema_probs',
            torch.ones(vocab_size) / vocab_size,
        )

    @torch.no_grad()
    def update_ema(self, probs: Tensor):
        """Update EMA with batch-averaged distribution.

        Args:
            probs: (batch, vocab_size) softmax probabilities.
        """
        mean_probs = probs.mean(dim=0)  # (vocab_size,)
        self.ema_probs.mul_(self.ema_decay).add_(mean_probs, alpha=1.0 - self.ema_decay)

    def forward(self, logits: Tensor) -> Tensor:
        """Compute per-token KL surprise from logits.

        Args:
            logits: (batch, seqlen, vocab_size) raw logits.

        Returns:
            surprise: (batch, seqlen) KL divergence at each position.
        """
        batch, seqlen, V = logits.shape
        # Top-k for efficiency
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
        p = F.softmax(topk_logits, dim=-1)  # (batch, seqlen, k)

        # Gather EMA probs at top-k indices
        ema_expanded = self.ema_probs.unsqueeze(0).unsqueeze(0).expand(batch, seqlen, -1)
        q = ema_expanded.gather(-1, topk_idx)  # (batch, seqlen, k)
        q = q.clamp(min=1e-8)  # numerical stability

        # KL(p || q) = sum p * log(p/q)
        kl = (p * (p.clamp(min=1e-8).log() - q.log())).sum(dim=-1)  # (batch, seqlen)

        # Update EMA with current batch (no grad)
        with torch.no_grad():
            full_probs = F.softmax(logits.detach().reshape(-1, V), dim=-1)
            self.update_ema(full_probs)

        return kl


# ---------------------------------------------------------------------------
# Naja Mixer
# ---------------------------------------------------------------------------

class NajaMixer(nn.Module):
    """Naja mixer block.

    Combines:
    - PoPE (content/position decoupling)
    - Delta rule (targeted erase+write)
    - PoPE orthogonal pair (rotation via n_h=2 Householder)
    - Per-channel decay (KDA-style multi-scale)
    - MIMO (rank-r B, C, X projections)
    - Trapezoidal discretization (retained from Mamba3)
    - Surprise-modulated beta gates (Phase 4 placeholder)
    """

    def __init__(self, config: NajaConfig):
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

        # --- MIMO: create rank-r write input ---
        if r > 1:
            x_flat = x.reshape(batch, seqlen, cfg.d_inner)
            x_write = self.mimo_x_proj(x_flat)  # (batch, seqlen, d_inner * r)
            x_write = x_write.reshape(batch, seqlen, cfg.nheads, cfg.headdim, r)
        else:
            x_write = x.unsqueeze(-1)  # (batch, seqlen, nheads, headdim, 1)

        # --- Trapezoidal: create shifted versions ---
        if cfg.use_trapezoidal:
            # x_write is 5D: pad seqlen dim (dim -4 from the end)
            x_write_prev = F.pad(x_write[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
            B1_prev = F.pad(B1[:, :-1], (0, 0, 0, 0, 1, 0))
        else:
            x_write_prev = torch.zeros_like(x_write)
            B1_prev = torch.zeros_like(B1)

        # --- Run recurrence ---
        if cfg.use_wy_chunkwise:
            recurrence_fn = delta_recurrence_wy
        elif cfg.use_chunkwise:
            recurrence_fn = delta_recurrence_chunkwise
        else:
            recurrence_fn = delta_recurrence
        recurrence_kwargs = dict(
            x_write=x_write,
            x_write_prev=x_write_prev,
            B1=B1,
            B1_prev=B1_prev,
            B2=B2,
            C=C,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            lam=lam.unsqueeze(-1),  # (batch, seqlen, 1, 1)
            use_trapezoidal=cfg.use_trapezoidal,
            use_delta=cfg.use_delta_rule,
        )
        if cfg.use_chunkwise or cfg.use_wy_chunkwise:
            recurrence_kwargs['chunk_size'] = cfg.chunk_size
        y = recurrence_fn(**recurrence_kwargs)
        # y: (batch, seqlen, nheads, headdim, r)

        # --- MIMO contraction + skip connection ---
        if r > 1:
            # Learned contraction: (batch, seqlen, d_inner * r) -> (batch, seqlen, d_inner)
            y = y.reshape(batch, seqlen, cfg.d_inner * r)
            y = self.mimo_out_proj(y)
            y = y.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        else:
            y = y.squeeze(-1)  # (batch, seqlen, nheads, headdim)

        # Skip connection: D scales per-head passthrough
        y = y + x * self.D[None, None, :, None]

        # --- Gated output ---
        y = y.reshape(batch, seqlen, cfg.d_inner)
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
# Full Naja Block and Model
# ---------------------------------------------------------------------------

class NajaBlock(nn.Module):
    """One Naja layer: pre-norm Mixer + pre-norm MLP with residuals."""

    def __init__(self, config: NajaConfig):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer = NajaMixer(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

    def forward(self, x: Tensor, surprise: Optional[Tensor] = None) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x), surprise=surprise)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class NajaLM(nn.Module):
    """Complete Naja language model.

    Architecture:
        Embedding -> N x NajaBlock -> RMSNorm -> Linear(vocab)
    """

    def __init__(self, config: NajaConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            NajaBlock(config) for _ in range(config.n_layer)
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

    def forward_with_surprise(self, input_ids: Tensor) -> tuple:
        """Two-pass forward for Phase 4 surprise gating during training.

        Pass 1: Run without surprise to get logits, compute cross-entropy surprise.
        Pass 2: Run with the stop-gradiented surprise signal.

        Returns:
            (logits, surprise): logits from the second pass, surprise from the first.
        """
        # Pass 1: no surprise → get per-token cross-entropy
        with torch.no_grad():
            logits_p1 = self.forward(input_ids, surprise=None)
            # Surprise = -log p(x_t | x_{<t}) via next-step cross-entropy
            # logits_p1[:, :-1] predicts input_ids[:, 1:]
            ce = F.cross_entropy(
                logits_p1[:, :-1].reshape(-1, self.vocab_size),
                input_ids[:, 1:].reshape(-1),
                reduction='none',
            ).reshape(input_ids.shape[0], -1)  # (batch, seqlen-1)
            # Pad position 0 with mean surprise (no context yet)
            surprise = F.pad(ce, (1, 0), value=ce.mean().item())

        # Pass 2: run with stop-gradiented surprise
        logits = self.forward(input_ids, surprise=surprise.detach())
        return logits, surprise


def mamba3_base_config(**overrides) -> NajaConfig:
    """Create a NajaConfig equivalent to base Mamba3 (no delta features).

    Used as the ablation baseline: SISO, scalar decay, no delta rule,
    no surprise gating, no chunkwise.
    """
    defaults = dict(
        use_delta_rule=False,
        use_pope_perp=False,
        per_channel_decay=False,
        stable_reparam=False,
        use_surprise_gate=False,
        mimo_rank=1,
        use_chunkwise=False,
    )
    defaults.update(overrides)
    return NajaConfig(**defaults)
