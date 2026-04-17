"""
grokking_mamba3.py - GrokkingMamba3 model implementing three ideas:

  1. GrowableLinear: low-rank USV factorization that can grow in rank.
     Starts at rank r, adds new singular triplets when loss plateaus.
     Nuclear norm (S.abs().sum()) is the convex rank regularizer.

  2. AlignmentModule: shared T matrix across ALL layers enforces that
     the representation of shifted inputs is linearly predictable from
     unshifted inputs at every layer simultaneously. This is the key
     layer-coordination mechanism.

  3. GrokkingMamba3LM: full language model returning (logits, hidden_states)
     where hidden_states[l] is the residual stream after block l.
     Used by the training loop for alignment loss computation.

The base SSM is a simplified SISO Mamba3 (no fast weights, no 2D-PoPE,
no option register, no GVF heads) imported from the parent Mamba3 directory.
"""

from __future__ import annotations
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import SSM utilities from sibling Mamba3 directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'Mamba3'))
from mamba3_block import RMSNorm, apply_pope, stable_log_decay, ssd_trapz, SwiGLUMLP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GrokkingConfig:
    # Model
    vocab_size:    int   = 100     # p + 3 for p=97
    d_model:       int   = 128
    d_state:       int   = 64
    expand:        int   = 2
    headdim:       int   = 64
    chunk_size:    int   = 4       # must divide seq_len (4 for arithmetic)
    n_layer:       int   = 2
    mlp_expand:    int   = 4
    stable_ssm:    bool  = True
    # Grokking features
    use_growable:  bool  = False   # GrowableLinear in mixer in_proj/out_proj
    initial_rank:  int   = 16      # starting rank for GrowableLinear
    use_alignment: bool  = False   # alignment loss (requires passing hiddens)
    # Derived (set in __post_init__)
    d_inner:       int   = field(init=False)
    nheads:        int   = field(init=False)
    d_bc:          int   = field(init=False)

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.d_bc   = self.d_state // 2


# ---------------------------------------------------------------------------
# GrowableLinear
# ---------------------------------------------------------------------------

class GrowableLinear(nn.Module):
    """Linear layer W = U @ diag(S) @ V with growable rank.

    The weight matrix is stored as a rank-r outer-product sum:
        W_eff = (U * S) @ V    shape (out_features, in_features)

    This is mathematically equivalent to U @ diag(S) @ V but avoids
    materializing the full diagonal matrix explicitly.

    Nuclear norm approximation: S.abs().sum()
    (exact when U, V have orthonormal columns; a good proxy otherwise)

    Growth: adds n_new new columns to U and rows to V initialized at
    scale 'grow_scale' (default 1e-3). The small initialization ensures
    the new components begin as tiny perturbations of the current solution.

    IMPORTANT: after calling grow(), the optimizer must be updated to
    include the newly created parameters. Use get_new_params(model, opt)
    in the training loop.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        initial_rank: int   = 16,
        grow_scale:   float = 1e-3,
    ):
        super().__init__()
        self.in_f       = in_features
        self.out_f      = out_features
        self.grow_scale = grow_scale
        self.rank       = initial_rank
        self._growth_steps: List[int] = []   # training steps at which growth occurred

        r = initial_rank
        # Initialize so the effective weight W = (U*S)@V has the same
        # Frobenius scale as a xavier-uniform nn.Linear would have.
        # With U~N(0,1), S=s*ones, V~N(0,1):
        #   std(W_ij) = s * sqrt(r)
        # Target: s * sqrt(r) = sqrt(2 / (in + out))  [xavier]
        #   s = sqrt(2 / (r * (in + out)))
        s = math.sqrt(2.0 / (r * (in_features + out_features)))
        self.U = nn.Parameter(torch.randn(out_features, r))
        self.S = nn.Parameter(torch.ones(r) * s)
        self.V = nn.Parameter(torch.randn(r, in_features))

    def forward(self, x: Tensor) -> Tensor:
        # W_eff = (U * S.unsqueeze(0)) @ V  :  (out, in)
        W = (self.U * self.S.unsqueeze(0)) @ self.V
        return F.linear(x, W)

    @property
    def nuclear_norm(self) -> Tensor:
        """Approximate nuclear norm = sum of singular values."""
        return self.S.abs().sum()

    @property
    def effective_weight(self) -> Tensor:
        """Materialize the full (out, in) weight matrix."""
        return (self.U * self.S.unsqueeze(0)) @ self.V

    def grow(self, n_new: int = 1, step: int = -1) -> None:
        """Append n_new new singular triplets near zero.

        After calling this, the old U/S/V Parameters are replaced with
        new (larger) Parameters. The optimizer must be updated separately
        via get_new_params().
        """
        device = self.U.device
        gs = self.grow_scale
        new_U = torch.randn(self.out_f, n_new, device=device) * gs
        new_S = torch.ones(n_new, device=device) * gs
        new_V = torch.randn(n_new, self.in_f, device=device) * gs

        # Concatenate and re-register as new Parameters
        with torch.no_grad():
            full_U = torch.cat([self.U.data, new_U], dim=1)
            full_S = torch.cat([self.S.data, new_S])
            full_V = torch.cat([self.V.data, new_V], dim=0)

        # Remove old parameters from module registry
        del self._parameters['U']
        del self._parameters['S']
        del self._parameters['V']

        self.register_parameter('U', nn.Parameter(full_U))
        self.register_parameter('S', nn.Parameter(full_S))
        self.register_parameter('V', nn.Parameter(full_V))

        self.rank += n_new
        self._growth_steps.append(step)


def collect_growable(model: nn.Module) -> List[GrowableLinear]:
    """Return all GrowableLinear modules in model."""
    return [m for m in model.modules() if isinstance(m, GrowableLinear)]


def get_new_params(model: nn.Module, optimizer: torch.optim.Optimizer) -> List[Tensor]:
    """Find parameters in model that are not yet in any optimizer param group."""
    known = {id(p) for g in optimizer.param_groups for p in g['params']}
    return [p for p in model.parameters() if id(p) not in known]


def total_nuclear_norm(model: nn.Module) -> Tensor:
    """Sum of nuclear norms across all GrowableLinear modules."""
    growables = collect_growable(model)
    if not growables:
        return torch.tensor(0.0)
    device = next(model.parameters()).device
    return sum(g.nuclear_norm for g in growables).to(device)


# ---------------------------------------------------------------------------
# AlignmentModule
# ---------------------------------------------------------------------------

class AlignmentModule(nn.Module):
    """Shared T matrix enforcing linear predictability across all layers.

    For pairs (x_A, x_B) where x_B is a symmetry-shifted version of x_A,
    we enforce:
        h_l(x_B) ~ T @ h_l(x_A)   for all layers l simultaneously.

    T is the SAME matrix for all layers -- this is what forces layer
    coordination. Layer l's gradient of the alignment loss flows through
    h_l(x_A) and through T. Since T is shared, all layers receive a
    coherent signal about what the transformation should be.

    The target h_l(x_B) is detached (stop-gradient). This means the loss
    teaches h_l(x_A) to be predictable by the current T, without pulling
    h_l(x_B) to be identical (which would collapse the representation).
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Initialize T as identity -- the prior is that shifted inputs
        # are initially close to unshifted inputs (small shift).
        self.T = nn.Parameter(torch.eye(d_model))

    def forward(
        self,
        h_list_A: List[Tensor],   # [h_0(A), h_1(A), ..., h_L(A)], each (B, T_seq, d_model)
        h_list_B: List[Tensor],   # [h_0(B), h_1(B), ..., h_L(B)], each (B, T_seq, d_model)
    ) -> Tensor:
        """Compute alignment loss summed across layers.

        loss = (1/L) * sum_l || h_l(A) @ T.T - h_l(B).detach() ||^2
        """
        assert len(h_list_A) == len(h_list_B)
        total = torch.tensor(0.0, device=self.T.device)
        for h_A, h_B in zip(h_list_A, h_list_B):
            predicted = h_A @ self.T.T    # (B, T_seq, d_model)
            total = total + F.mse_loss(predicted, h_B.detach())
        return total / len(h_list_A)


# ---------------------------------------------------------------------------
# GrokkingMixer (simplified SISO Mamba3, no fast weights, 1D-PoPE only)
# ---------------------------------------------------------------------------

def _pad_dim1(t: Tensor, pad_len: int) -> Tensor:
    if pad_len == 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[1] = pad_len
    return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=1)


class GrokkingMixer(nn.Module):
    """SISO Mamba3 mixer.

    Identical to OaKMixer but without:
      - FastWeightHead (no B/C adaptation during the forward pass)
      - 2D-PoPE (uses standard 1D cumsum throughout)
      - OptionRegister (no option index input)

    When config.use_growable is True, in_proj and out_proj are
    GrowableLinear; otherwise they are standard nn.Linear.
    """

    def __init__(self, config: GrokkingConfig):
        super().__init__()
        d   = config.d_inner
        n   = config.d_state
        dbc = config.d_bc

        # [z, x, B_raw, C_raw, dt, theta, lam]
        d_proj = 2*d + 2*dbc + config.nheads + dbc + 1

        if config.use_growable:
            r = config.initial_rank
            self.in_proj  = GrowableLinear(config.d_model, d_proj, initial_rank=r)
            self.out_proj = GrowableLinear(d, config.d_model,       initial_rank=r)
        else:
            self.in_proj  = nn.Linear(config.d_model, d_proj, bias=False)
            self.out_proj = nn.Linear(d, config.d_model,       bias=False)

        self.B_norm = RMSNorm(dbc)
        self.C_norm = RMSNorm(dbc)
        self.B_bias = nn.Parameter(torch.ones(dbc))
        self.C_bias = nn.Parameter(torch.ones(dbc))

        self.pope_delta_B = nn.Parameter(torch.zeros(dbc))
        self.pope_delta_C = nn.Parameter(torch.zeros(dbc))

        self.A_raw   = nn.Parameter(torch.empty(config.nheads))
        self.dt_bias = nn.Parameter(torch.empty(config.nheads))
        self.D       = nn.Parameter(torch.ones(config.nheads))

        self._config = config
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.A_raw, -3.0, -0.5)
        dt_target = torch.exp(
            torch.empty(self._config.nheads).uniform_(math.log(0.001), math.log(0.1))
        )
        self.dt_bias.data.copy_(torch.log(torch.exp(dt_target) - 1))
        if isinstance(self.in_proj, nn.Linear):
            nn.init.xavier_uniform_(self.in_proj.weight)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, u: Tensor) -> Tensor:
        cfg   = self._config
        bsz, T, _ = u.shape          # bsz = batch size (avoid clash with SSM B matrix)
        d, dbc  = cfg.d_inner, cfg.d_bc

        proj = self.in_proj(u)
        z, x, B_raw, C_raw, dt, theta, lam_logit = torch.split(
            proj, [d, d, dbc, dbc, cfg.nheads, dbc, 1], dim=-1
        )

        dt  = F.softplus(dt + self.dt_bias)
        lam = torch.sigmoid(lam_logit)

        A_dt = stable_log_decay(self.A_raw * dt)

        # 1D cumsum PoPE (no 2D grid handling needed for arithmetic)
        theta_cs = theta.cumsum(dim=1)

        B_raw = self.B_norm(B_raw) + self.B_bias
        C_raw = self.C_norm(C_raw) + self.C_bias

        Bssm = apply_pope(B_raw, theta_cs, self.pope_delta_B)   # SSM B-matrix
        Cssm = apply_pope(C_raw, theta_cs, self.pope_delta_C)   # SSM C-matrix

        x   = x.reshape(bsz, T, cfg.nheads, cfg.headdim)
        dt_exp = dt.unsqueeze(-1)
        x_dt   = x * dt_exp

        x_raw_prev = F.pad(x[:, :-1], (0,0,0,0,1,0))
        x_prev     = x_raw_prev * dt_exp

        B_curr = Bssm.unsqueeze(2)
        B_prev = F.pad(Bssm[:, :-1], (0,0,1,0)).unsqueeze(2)
        C_curr = Cssm.unsqueeze(2)
        lam_exp = lam.unsqueeze(-1)

        cs = cfg.chunk_size
        pad_len = (cs - T % cs) % cs
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
            y = y[:, :T]

        y = y + x[:, :T] * self.D.unsqueeze(-1)
        y = y.reshape(bsz, T, d)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# GrokkingBlock and GrokkingMamba3LM
# ---------------------------------------------------------------------------

class GrokkingBlock(nn.Module):
    def __init__(self, config: GrokkingConfig):
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model)
        self.mixer      = GrokkingMixer(config)
        self.mlp_norm   = RMSNorm(config.d_model)
        self.mlp        = SwiGLUMLP(config.d_model, config.d_model * config.mlp_expand)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.mixer_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GrokkingMamba3LM(nn.Module):
    """Full language model.

    forward() returns:
        logits        : (B, seq_len, vocab_size)
        hidden_states : List[(B, seq_len, d_model)]  -- one per block

    hidden_states[l] is the residual stream AFTER block l (with residual
    added). These are used by the alignment module in the training loop.
    """

    def __init__(self, config: GrokkingConfig):
        super().__init__()
        self.config = config
        self.embed  = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([GrokkingBlock(config) for _ in range(config.n_layer)])
        self.norm   = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optional alignment module (initialized here so it is part of model.parameters())
        if config.use_alignment:
            self.align = AlignmentModule(config.d_model)
        else:
            self.align = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias if self.lm_head.bias is not None else
                        torch.empty(0))
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self, tokens: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            tokens: (B, seq_len) long

        Returns:
            logits        : (B, seq_len, vocab_size)
            hidden_states : list of (B, seq_len, d_model), len = n_layer
        """
        x = self.embed(tokens)         # (B, T, d_model)
        hidden_states: List[Tensor] = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)    # record after each block
        h = self.norm(x)
        logits = self.lm_head(h)
        return logits, hidden_states

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def rank_summary(self) -> str:
        """Print current rank of all GrowableLinear modules."""
        lines = []
        for name, m in self.named_modules():
            if isinstance(m, GrowableLinear):
                lines.append(f"  {name}: rank={m.rank}  nuclear_norm={m.nuclear_norm.item():.4f}")
        return "\n".join(lines) if lines else "  (no GrowableLinear modules)"

    def singular_values(self) -> dict:
        """Return singular values of all GrowableLinear modules (as numpy arrays)."""
        result = {}
        for name, m in self.named_modules():
            if isinstance(m, GrowableLinear):
                result[name] = m.S.detach().cpu().abs().numpy()
        return result
