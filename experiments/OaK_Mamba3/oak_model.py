"""
OaK-Mamba3: Options and Knowledge on Mamba3 backbone.

Architecture additions over base Mamba3 (SISO only, mimo_rank=1):
  - GridEncoder context: local depthwise conv + segment embeddings per grid
  - 2D-PoPE: factored row/col positional encoding for grid token segments
  - FastWeightHead: u_t -> additive (dB, dC) offsets for in-context adaptation
  - OptionRegister: learned option identity injected before SSM layers
  - GVFHead x n_gvfs: multi-timescale value prediction heads
  - TerminationHead, OptionValueHead, UncertaintyHead for full OaK machinery
  - BiOaKMixer: bidirectional wrapper (fwd + bwd OaKMixer) for MDLM training

Token vocabulary:
  0-9  : grid cell colors
  10   : SEP  (separator between grids)
  11   : QUERY (precedes test output)
  12   : PAD  (padding to multiple of chunk_size)
  13   : MASK (masked token for MDLM masked diffusion training)
  VOCAB_SIZE = 14

Segment IDs (for segment_embed):
  0 : input grid
  1 : output grid
  2 : query/test-output grid
"""

import math
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import SSM utilities from sibling Mamba3 directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Mamba3'))
from mamba3_block import (
    RMSNorm,
    apply_pope,
    stable_log_decay,
    ssd_trapz,
    SwiGLUMLP,
)

VOCAB_SIZE  = 14   # 0-9 colors + SEP(10) + QUERY(11) + PAD(12) + MASK(13)
NUM_COLORS  = 10
SEP_TOKEN   = 10
QUERY_TOKEN = 11
PAD_TOKEN   = 12
MASK_TOKEN  = 13


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OaKConfig:
    # SSM core (mirrors Mamba3Config, SISO only)
    d_model:    int   = 256
    d_state:    int   = 64      # SSM state dimension N
    expand:     int   = 2       # d_inner = expand * d_model
    headdim:    int   = 64      # head dimension P; nheads = d_inner // headdim
    chunk_size: int   = 64      # SSD chunk size (seqlen must be multiple)
    n_layer:    int   = 4
    mlp_expand: int   = 4
    stable_ssm: bool  = True    # StableSSM A reparameterization
    # Grid / encoding
    n_segments: int   = 3       # input=0, output=1, query=2
    # Options
    num_options: int  = 8
    d_option:   int   = 64
    # GVFs
    n_gvfs:     int   = 5

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads  = self.d_inner // self.headdim
        # With PoPE, B/C project to d_state//2 then expand to d_state
        self.d_bc    = self.d_state // 2


# ---------------------------------------------------------------------------
# Fast Weight Head
# ---------------------------------------------------------------------------

class FastWeightHead(nn.Module):
    """Maps mixer input u_t -> additive (dB, dC) offsets.

    B_eff_t = B_t + scale * dB(u_t)
    C_eff_t = C_t + scale * dC(u_t)

    Using u_t (the pre-norm residual stream, i.e. the previous layer's output)
    as the source gives causal, context-rich offsets without needing to access
    the internal SSM hidden state h_t.
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.to_dB = nn.Linear(d_model, d_state, bias=False)
        self.to_dC = nn.Linear(d_model, d_state, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)
        nn.init.normal_(self.to_dB.weight, std=0.02)
        nn.init.normal_(self.to_dC.weight, std=0.02)

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        # u: (B, T, d_model)
        # returns dB, dC each (B, T, d_state)
        return self.scale * self.to_dB(u), self.scale * self.to_dC(u)


# ---------------------------------------------------------------------------
# 2D-PoPE theta computation
# ---------------------------------------------------------------------------

def compute_theta_cs(
    theta: Tensor,
    grid_positions: List[Tuple[int, int, int]],
    d_bc: int,
) -> Tensor:
    """Compute theta cumsum with 2D override for grid segments.

    For non-grid positions: standard 1D cumulative sum over the sequence.
    For grid positions: factored 2D cumsum — first d_bc//2 features use
    cumsum along the row axis, last d_bc//2 features use cumsum along the
    col axis.  This encodes (row, col) position independently in different
    feature dimensions, analogous to multi-dimensional RoPE.

    Args:
        theta:          (batch, seqlen, d_bc) raw projected angles from in_proj
        grid_positions: list of (start_idx, H, W) for each grid segment
        d_bc:           full angle dimension (= d_state // 2)

    Returns:
        theta_cs: (batch, seqlen, d_bc) positional angles for apply_pope
    """
    if not grid_positions:
        return theta.cumsum(dim=1)

    batch, T, _ = theta.shape
    d_half = d_bc // 2

    # Build a sorted list of (start, end, is_grid, H, W) segments
    grid_positions_sorted = sorted(grid_positions, key=lambda x: x[0])
    parts = []
    prev = 0
    for start, H, W in grid_positions_sorted:
        end_g = start + H * W
        if prev < start:
            parts.append((prev, start, False, 0, 0))
        parts.append((start, end_g, True, H, W))
        prev = end_g
    if prev < T:
        parts.append((prev, T, False, 0, 0))

    # Accumulate 1D cumsum for non-grid positions, 2D for grid positions.
    # We maintain a running "carry" so that non-grid cumsum is continuous
    # across the sequence (excluding grid tokens from the accumulation).
    carry = torch.zeros(batch, d_bc, device=theta.device, dtype=theta.dtype)
    result_parts: List[Tensor] = []

    for start, end, is_grid, H, W in parts:
        if start >= end:
            continue
        seg = theta[:, start:end, :]  # (B, L, d_bc)

        if not is_grid:
            # Standard cumsum + running carry
            seg_cs = seg.cumsum(dim=1) + carry.unsqueeze(1)   # (B, L, d_bc)
            carry = seg_cs[:, -1, :]                           # update carry
            result_parts.append(seg_cs)
        else:
            # 2D factored cumsum — independent of sequence carry
            L = H * W
            th_raw = seg[:, :, :d_half].reshape(batch, H, W, d_half)   # row
            tw_raw = seg[:, :, d_half:].reshape(batch, H, W, d_half)   # col

            th_cs = th_raw.cumsum(dim=1).reshape(batch, L, d_half)
            tw_cs = tw_raw.cumsum(dim=2).reshape(batch, L, d_half)

            theta_2d = torch.cat([th_cs, tw_cs], dim=-1)   # (B, L, d_bc)
            result_parts.append(theta_2d)
            # Grid tokens do NOT update the carry (their position is spatial,
            # not sequential; non-grid carry continues from before the grid)

    return torch.cat(result_parts, dim=1)   # (B, T, d_bc)


# ---------------------------------------------------------------------------
# OaK Mixer (SISO Mamba3 + fast weights + 2D-PoPE)
# ---------------------------------------------------------------------------

def _pad_dim1(t: Tensor, pad_len: int) -> Tensor:
    """Pad tensor along dim=1 (sequence) with zeros."""
    if pad_len == 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[1] = pad_len
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=1)


class OaKMixer(nn.Module):
    """SISO Mamba3 mixer with fast weight head and 2D-PoPE support.

    Key differences from Mamba3Mixer:
      - SISO only (mimo_rank=1 always)
      - FastWeightHead adds u-conditioned offsets to projected B, C
      - compute_theta_cs handles 2D spatial cumsums for grid segments
      - PoPE only (no RoPE fallback), StableSSM only
    """

    def __init__(self, config: OaKConfig):
        super().__init__()
        self.config = config
        d  = config.d_inner
        n  = config.d_state
        dbc = config.d_bc   # = n // 2 (PoPE halves before polar expand)

        # Input projection: [z, x, B_raw, C_raw, dt, theta, lam]
        # z:        d_inner  (gate)
        # x:        d_inner  (SSM input)
        # B_raw:    d_bc     (input proj, SISO)
        # C_raw:    d_bc     (output proj, SISO)
        # dt:       nheads
        # theta:    d_bc     (positional angle, = d_state//2)
        # lam:      1        (trapezoidal mixing scalar)
        self._d_proj = 2 * d + 2 * dbc + config.nheads + dbc + 1
        self.in_proj  = nn.Linear(config.d_model, self._d_proj, bias=False)
        self.out_proj = nn.Linear(d, config.d_model, bias=False)

        # B/C normalization and bias (Mamba3 §3.4)
        self.B_norm  = RMSNorm(dbc)
        self.C_norm  = RMSNorm(dbc)
        self.B_bias  = nn.Parameter(torch.ones(dbc))
        self.C_bias  = nn.Parameter(torch.ones(dbc))

        # PoPE phase bias
        self.pope_delta_B = nn.Parameter(torch.zeros(dbc))
        self.pope_delta_C = nn.Parameter(torch.zeros(dbc))

        # SSM parameters (StableSSM)
        self.A_raw   = nn.Parameter(torch.empty(config.nheads))
        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        self.D       = nn.Parameter(torch.ones(config.nheads))

        # Fast weight head
        self.fast_weight = FastWeightHead(config.d_model, config.d_state)

        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.A_raw, -3.0, -0.5)
        dt_target = torch.exp(
            torch.empty(self.config.nheads).uniform_(math.log(0.001), math.log(0.1))
        )
        self.dt_bias.data.copy_(torch.log(torch.exp(dt_target) - 1))
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        u: Tensor,
        grid_positions: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tensor:
        """
        Args:
            u:              (batch, seqlen, d_model)
            grid_positions: list of (start_idx, H, W) for 2D-PoPE grid segments

        Returns:
            (batch, seqlen, d_model)
        """
        cfg = self.config
        batch, seqlen, _ = u.shape
        d, dbc = cfg.d_inner, cfg.d_bc

        # --- Project input ---
        proj = self.in_proj(u)
        z, x, B_raw, C_raw, dt, theta, lam_logit = torch.split(
            proj,
            [d, d, dbc, dbc, cfg.nheads, dbc, 1],
            dim=-1,
        )

        dt  = F.softplus(dt + self.dt_bias)   # (B, T, nheads)
        lam = torch.sigmoid(lam_logit)         # (B, T, 1)

        # --- StableSSM decay ---
        A_dt = stable_log_decay(self.A_raw * dt)  # (B, T, nheads)

        # --- Fast weight offsets from residual stream (u) ---
        dB, dC = self.fast_weight(u)   # each (B, T, d_state)

        # --- Positional angles: 2D for grid segments, 1D elsewhere ---
        theta_cs = compute_theta_cs(theta, grid_positions or [], dbc)
        # theta_cs: (B, T, d_bc)

        # --- QK-norm + bias, then PoPE ---
        B_raw = self.B_norm(B_raw) + self.B_bias
        C_raw = self.C_norm(C_raw) + self.C_bias

        # apply_pope: (B, T, d_bc), (B, T, d_bc), (d_bc,) -> (B, T, d_state)
        B = apply_pope(B_raw, theta_cs, self.pope_delta_B)   # (B, T, d_state)
        C = apply_pope(C_raw, theta_cs, self.pope_delta_C)   # (B, T, d_state)

        # --- Add fast weight offsets ---
        B = B + dB
        C = C + dC

        # --- Reshape x for multi-head ---
        x = x.reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        dt_exp = dt.unsqueeze(-1)   # (B, T, nheads, 1)
        x_dt   = x * dt_exp

        # --- SISO SSD (trapezoidal) with chunk-padding ---
        x_raw_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev     = x_raw_prev * dt_exp

        B_curr = B.unsqueeze(2)                              # (B, T, 1, d_state)
        B_prev = F.pad(B[:, :-1], (0, 0, 1, 0)).unsqueeze(2)
        C_curr = C.unsqueeze(2)
        lam_exp = lam.unsqueeze(-1)                          # (B, T, 1, 1)

        # Pad seqlen to multiple of chunk_size
        cs     = cfg.chunk_size
        pad_len = (cs - seqlen % cs) % cs
        if pad_len:
            x_dt    = _pad_dim1(x_dt,    pad_len)
            x_prev  = _pad_dim1(x_prev,  pad_len)
            A_dt    = _pad_dim1(A_dt,    pad_len)
            B_curr  = _pad_dim1(B_curr,  pad_len)
            B_prev  = _pad_dim1(B_prev,  pad_len)
            C_curr  = _pad_dim1(C_curr,  pad_len)
            lam_exp = _pad_dim1(lam_exp, pad_len)

        y, _ = ssd_trapz(x_dt, x_prev, A_dt, B_curr, B_prev, C_curr, lam_exp, cs)

        if pad_len:
            y = y[:, :seqlen]   # unpad

        # --- D skip connection + gated output ---
        y = y + x[:, :seqlen] * self.D.unsqueeze(-1)   # x already un-padded
        y = y.reshape(batch, seqlen, d)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Bidirectional helpers and BiOaKMixer
# ---------------------------------------------------------------------------

def _reverse_grid_positions(
    grid_positions: List[Tuple[int, int, int]],
    seqlen: int,
) -> List[Tuple[int, int, int]]:
    """Map forward grid positions to their reversed-sequence equivalents.

    For a grid segment (start, H, W) in a forward sequence of length seqlen,
    the corresponding start position in the reversed sequence is:
        new_start = seqlen - start - H * W

    Returns the list sorted by new_start ascending, so that downstream
    code (compute_theta_cs) can iterate segments in order.

    Args:
        grid_positions: list of (start, H, W) in forward sequence order
        seqlen:         total sequence length

    Returns:
        list of (new_start, H, W) sorted by new_start ascending
    """
    reversed_positions = []
    for start, H, W in grid_positions:
        new_start = seqlen - start - H * W
        reversed_positions.append((new_start, H, W))
    return sorted(reversed_positions, key=lambda x: x[0])


class BiOaKMixer(nn.Module):
    """Bidirectional OaK mixer wrapping two independent OaKMixer instances.

    Runs one causal OaKMixer in the forward direction and a second causal
    OaKMixer on the time-reversed sequence, then averages the two outputs.
    This gives each token access to both past and future context, which is
    appropriate for non-autoregressive (MDLM masked diffusion) training.

    The two mixers (self.fwd, self.bwd) are fully independent — they do not
    share weights — so each direction can learn its own temporal patterns.

    Args:
        config: OaKConfig shared by both sub-mixers

    Forward signature:
        u              (B, T, d_model)
        grid_positions list of (start, H, W) in the FORWARD sequence

    Returns:
        (B, T, d_model) — average of forward and backward mixer outputs
    """

    def __init__(self, config: OaKConfig):
        super().__init__()
        self.fwd = OaKMixer(config)
        self.bwd = OaKMixer(config)

    def forward(
        self,
        u: Tensor,
        grid_positions: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tensor:
        seqlen = u.shape[1]
        gp = grid_positions or []

        # Forward pass (standard causal direction)
        y_fwd = self.fwd(u, gp)

        # Backward pass: flip sequence, map grid positions, run mixer, flip back
        u_rev = u.flip(dims=[1])
        gp_rev = _reverse_grid_positions(gp, seqlen)
        y_bwd_rev = self.bwd(u_rev, gp_rev)
        y_bwd = y_bwd_rev.flip(dims=[1])

        return 0.5 * (y_fwd + y_bwd)


# ---------------------------------------------------------------------------
# OaK Block (Mixer + MLP)
# ---------------------------------------------------------------------------

class OaKBlock(nn.Module):
    """One OaK layer: pre-RMSNorm Mixer + pre-RMSNorm SwiGLU MLP."""

    def __init__(self, config: OaKConfig):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer      = BiOaKMixer(config)
        self.mlp_norm   = RMSNorm(config.d_model)
        self.mlp        = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

    def forward(
        self,
        x: Tensor,
        grid_positions: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x), grid_positions)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Option Register
# ---------------------------------------------------------------------------

class OptionRegister(nn.Module):
    """Inject current option identity into the residual stream.

    x_out = Linear(cat(x, option_emb(omega)))

    With num_options=1 and omega=0 everywhere (Phase 1), this reduces to a
    learned linear projection with a fixed-direction bias component.
    """

    def __init__(self, num_options: int, d_model: int, d_option: int):
        super().__init__()
        self.embed = nn.Embedding(num_options, d_option)
        self.proj  = nn.Linear(d_model + d_option, d_model, bias=False)

    def forward(self, x: Tensor, omega: Tensor) -> Tensor:
        # x: (B, T, d_model), omega: (B, T) long
        opt_emb = self.embed(omega)                      # (B, T, d_option)
        return self.proj(torch.cat([x, opt_emb], dim=-1))


# ---------------------------------------------------------------------------
# Output Heads
# ---------------------------------------------------------------------------

class GVFHead(nn.Module):
    def __init__(self, d_model: int, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.proj  = nn.Linear(d_model, 1)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h).squeeze(-1)   # (B, T)


class TerminationHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)   # (B, T, 1)


class OptionValueHead(nn.Module):
    def __init__(self, d_model: int, num_options: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_options)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)   # (B, T, num_options)


class UncertaintyHead(nn.Module):
    def __init__(self, d_model: int, n_gvfs: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_gvfs)

    def forward(self, h: Tensor) -> Tensor:
        return F.softplus(self.proj(h))   # (B, T, n_gvfs) positive variance


# ---------------------------------------------------------------------------
# OaK Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class OaKOutput:
    task_logits: Tensor   # (B, T, NUM_COLORS)
    gvf_vals:    Tensor   # (B, T, n_gvfs)
    term_prob:   Tensor   # (B, T, 1)
    opt_vals:    Tensor   # (B, T, num_options)
    unc_vals:    Tensor   # (B, T, n_gvfs)


# ---------------------------------------------------------------------------
# OaK Model
# ---------------------------------------------------------------------------

class OaKModel(nn.Module):
    """Full OaK-Mamba3 model.

    Forward call:
        tokens:         (B, T)     long  — 0-13 token ids
        grid_segments:  list of (start, H, W, seg_id) — grid locations
        omega:          (B, T)     long  — option indices (default: zeros)

    Returns OaKOutput with all head outputs at every position.

    Grid encoding pipeline:
        1. Embed all tokens via self.embed (B, T, d_model)
        2. For each grid segment:
               a. Reshape to (B, H, W, d_model), apply depthwise conv2d
               b. Add segment embedding (input/output/query)
               c. Write back into the sequence tensor
        3. Inject option identity via OptionRegister
        4. Pass through N OaKBlocks (2D-PoPE handled inside each BiOaKMixer)
        5. Apply final RMSNorm and all output heads
    """

    def __init__(self, config: OaKConfig):
        super().__init__()
        self.config = config

        # Token embedding (colors 0-9, SEP=10, QUERY=11, PAD=12, MASK=13)
        self.embed = nn.Embedding(VOCAB_SIZE, config.d_model)

        # Local spatial context (depthwise conv applied per grid, 2D)
        self.local_ctx = nn.Conv2d(
            config.d_model, config.d_model,
            kernel_size=3, padding=1, groups=config.d_model,
        )

        # Segment embeddings: input(0), output(1), query/test-out(2)
        self.segment_embed = nn.Embedding(config.n_segments, config.d_model)

        # Option register
        self.option_reg = OptionRegister(
            config.num_options, config.d_model, config.d_option
        )

        # SSM layers
        self.layers = nn.ModuleList([OaKBlock(config) for _ in range(config.n_layer)])
        self.norm   = RMSNorm(config.d_model)

        # Output heads
        self.task_head = nn.Linear(config.d_model, NUM_COLORS)

        gvf_gammas = [0.5, 0.9, 0.99, 0.7, 0.95]
        self.gvf_heads = nn.ModuleList([
            GVFHead(config.d_model, gvf_gammas[i]) for i in range(config.n_gvfs)
        ])
        self.term_head   = TerminationHead(config.d_model)
        self.opt_val_head = OptionValueHead(config.d_model, config.num_options)
        self.unc_head    = UncertaintyHead(config.d_model, config.n_gvfs)

    def _apply_grid_encoding(
        self,
        x: Tensor,
        grid_segments: List[Tuple[int, int, int, int]],
    ) -> Tensor:
        """Apply local conv + segment embed for each grid in the sequence.

        Modifies a clone of x in-place for each grid segment.
        """
        if not grid_segments:
            return x
        x = x.clone()   # single clone; subsequent slice-assigns are safe
        for start, H, W, seg_id in grid_segments:
            seg = x[:, start:start + H * W, :]          # (B, H*W, d_model)
            # Reshape to (B, d_model, H, W) for depthwise conv
            seg_2d = seg.reshape(x.shape[0], H, W, self.config.d_model)
            seg_2d = seg_2d.permute(0, 3, 1, 2).contiguous()   # (B, d_model, H, W)
            seg_2d = self.local_ctx(seg_2d)
            seg    = seg_2d.permute(0, 2, 3, 1).reshape(x.shape[0], H * W, -1)
            # Add segment embedding
            seg_id_t = torch.tensor(seg_id, device=x.device)
            seg = seg + self.segment_embed(seg_id_t)
            x[:, start:start + H * W, :] = seg
        return x

    def forward(
        self,
        tokens: Tensor,
        grid_segments: List[Tuple[int, int, int, int]],
        omega: Optional[Tensor] = None,
    ) -> 'OaKOutput':
        """
        Args:
            tokens:        (B, T) long
            grid_segments: list of (start, H, W, seg_id)
            omega:         (B, T) long option indices; defaults to zeros
        """
        B, T = tokens.shape
        device = tokens.device

        if omega is None:
            omega = torch.zeros(B, T, dtype=torch.long, device=device)

        # 1. Token embedding
        x = self.embed(tokens)   # (B, T, d_model)

        # 2. Grid-specific encoding (local conv + segment embed)
        x = self._apply_grid_encoding(x, grid_segments)

        # 3. Option register
        x = self.option_reg(x, omega)

        # 4. SSM layers — pass grid_positions (start, H, W only) for 2D-PoPE
        grid_positions = [(s, H, W) for s, H, W, _ in grid_segments]
        for layer in self.layers:
            x = layer(x, grid_positions)

        # 5. Final norm
        h = self.norm(x)   # (B, T, d_model)

        # 6. Output heads
        task_logits = self.task_head(h)                              # (B, T, 10)
        gvf_vals    = torch.stack(
            [head(h) for head in self.gvf_heads], dim=-1
        )                                                             # (B, T, n_gvfs)
        term_prob   = self.term_head(h)                              # (B, T, 1)
        opt_vals    = self.opt_val_head(h)                           # (B, T, num_options)
        unc_vals    = self.unc_head(h)                               # (B, T, n_gvfs)

        return OaKOutput(task_logits, gvf_vals, term_prob, opt_vals, unc_vals)
