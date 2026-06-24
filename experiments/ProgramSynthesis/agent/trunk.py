"""The fixed transformer trunk + policy/value heads.

Identical for every binding arm — only the positional scheme handed to attention
changes, so the encoder is the sole experimental variable. QK-norm is applied to
all arms uniformly (so magnitude is content-controlled); rotary phase is applied
only by the pope arms.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (
    ActionEmbed,
    PatchEmbed,
    build_patch_coords,
    make_scheme,
    patchify,
    NUM_ACTIONS,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # x: (B, H, N, hd); angles: (B, N, hd/2)
    a = angles.unsqueeze(1)                                  # (B, 1, N, hd/2)
    cos = a.cos().repeat_interleave(2, dim=-1)               # (B, 1, N, hd)
    sin = a.sin().repeat_interleave(2, dim=-1)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.qkv = nn.Linear(d, 3 * inner, bias=False)
        self.out = nn.Linear(inner, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(head_dim) ** 0.5))

    def forward(self, x: torch.Tensor, angles: Optional[torch.Tensor]) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))                       # (B, H, N, hd)
        q = F.normalize(q, dim=-1)                           # QK-norm (all arms)
        k = F.normalize(k, dim=-1)
        if angles is not None:
            q = apply_rotary(q, angles)
            k = apply_rotary(k, angles)
        # Memory-efficient attention: never materializes the N×N score matrix
        # (the cause of the 4 GB blow-up). The learnable temperature is folded
        # into q, so SDPA's own scaling is disabled.
        q = q * self.scale
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, head_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.attn = Attention(d, n_heads, head_dim)
        self.n2 = nn.RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d)
        )

    def forward(self, x: torch.Tensor, angles: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.n1(x), angles)
        x = x + self.mlp(self.n2(x))
        return x


class PolicyNet(nn.Module):
    """frames (+ inter-frame actions) -> action logits + value, under one binding."""

    def __init__(
        self,
        binding: str,
        *,
        d: int = 96,
        n_heads: int = 4,
        head_dim: int = 24,
        n_layers: int = 3,
        patch: int = 2,
        cell_dim: int = 8,
        crop: int = 16,
        window: int = 2,
    ):
        super().__init__()
        self.binding = binding
        self.patch = patch
        self.crop = crop                                     # LockPath boards fit 16×16
        self.window = window
        self.n_axis = crop // patch                          # patches per side
        self.patch_embed = PatchEmbed(patch, cell_dim, d)
        self.action_embed = ActionEmbed(d)
        self.type_embed = nn.Embedding(2, d)                 # 0 = patch, 1 = action
        self.scheme = make_scheme(
            binding, d=d, head_dim=head_dim, max_xy=self.n_axis, max_t=window
        )
        self.blocks = nn.ModuleList(
            Block(d, n_heads, head_dim) for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d)
        self.policy = nn.Linear(d, NUM_ACTIONS)
        self.value = nn.Linear(d, 1)
        # Buffer (not a fresh CPU tensor copied every forward) so it lives on the
        # model's device and doesn't split the CUDA graph under torch.compile.
        self.register_buffer(
            "_coords", build_patch_coords(self.n_axis, self.n_axis, window),
            persistent=False,
        )

    def forward(
        self, frames: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # frames: (B, W, grid, grid) long; actions: (B, W-1) long.
        b = frames.shape[0]
        frames = frames[..., : self.crop, : self.crop]       # boards live top-left
        cells = patchify(frames, self.patch)                 # (B, W, P², p*p)
        patch_tok = self.patch_embed(cells.flatten(1, 2))    # (B, W*P², d)
        coords = self._coords.unsqueeze(0).expand(b, -1, -1)

        act_tok = self.action_embed(actions)                 # (B, W-1, d)
        a = act_tok.shape[1]
        # Action-token coords: (px=0, py=0, t=transition index).
        t_idx = torch.arange(a, device=frames.device)
        act_coords = torch.zeros(b, a, 3, dtype=torch.long, device=frames.device)
        act_coords[..., 2] = t_idx

        tok = torch.cat([patch_tok, act_tok], dim=1)
        coords = torch.cat([coords, act_coords], dim=1)
        types = torch.zeros(tok.shape[1], dtype=torch.long, device=frames.device)
        types[patch_tok.shape[1]:] = 1
        tok = tok + self.type_embed(types)

        tok = self.scheme.add_pos(tok, coords)
        angles = self.scheme.angles(coords)
        for blk in self.blocks:
            tok = blk(tok, angles)
        pooled = self.norm(tok).mean(dim=1)                  # (B, d)
        return self.policy(pooled), self.value(pooled).squeeze(-1)


def build_model(binding: str, **cfg) -> PolicyNet:
    return PolicyNet(binding, **cfg)
