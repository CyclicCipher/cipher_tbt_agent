"""Binding-channel encoders — THE experimental knob.

Every arm maps a window of frames (+ the actions between them) to a token sequence
with content embeddings and per-token coordinates `(px, py, t)`. The arms differ
*only* in how those coordinates enter the model — the binding channel:

  none      coordinates unused — a bag of color tokens (the floor)
  content   learned absolute (px,py,t) embedding added to tokens (value-centric)
  pope2d    rotary phase over (px,py) inside attention; t not encoded
  pope2d1   rotary phase over (px,py,t) — change/motion becomes a phase primitive

`pope2d`/`pope2d1` are a pragmatic PoPE: multi-axis rotary + QK-norm (the trunk
applies QK-norm to all arms uniformly, so magnitude is content-controlled and the
rotary carries position). The full polar softplus-magnitude decoupling is deferred;
the binding contrast this experiment tests — phase vs content-token vs none — is
captured. See LEARNING_AGENT.md §3–4.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Cell color values are 0..15; actions are GameAction values 0..7 (+1 "none" slot).
NUM_COLORS = 16
NUM_ACTIONS = 8


# --- content embeddings -----------------------------------------------------


class PatchEmbed(nn.Module):
    """Embed a P×P patch of color cells into a d-vector."""

    def __init__(self, patch: int, cell_dim: int, d: int):
        super().__init__()
        self.patch = patch
        self.cell = nn.Embedding(NUM_COLORS, cell_dim)
        self.proj = nn.Linear(patch * patch * cell_dim, d)

    def forward(self, patch_cells: torch.Tensor) -> torch.Tensor:
        # patch_cells: (B, N, P*P) long  ->  (B, N, d)
        e = self.cell(patch_cells)                      # (B, N, P*P, cell_dim)
        e = e.flatten(start_dim=2)                      # (B, N, P*P*cell_dim)
        return self.proj(e)


class ActionEmbed(nn.Module):
    """Embed the action taken on a transition (-1 -> a learned 'none')."""

    def __init__(self, d: int):
        super().__init__()
        self.emb = nn.Embedding(NUM_ACTIONS + 1, d)     # index 0 reserved for 'none'

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.emb(actions.clamp(min=-1) + 1)      # (B, A) -> (B, A, d)


# --- positional / binding schemes -------------------------------------------


def _split_pairs(n_pairs: int, n_axes: int) -> List[int]:
    base, rem = divmod(n_pairs, n_axes)
    return [base + (1 if i < rem else 0) for i in range(n_axes)]


class NoneScheme(nn.Module):
    """No positional information at all."""

    uses_rotary = False

    def angles(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def add_pos(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return tokens


class ContentPosScheme(nn.Module):
    """Learned absolute (px,py,t) embedding added to token features (arm A)."""

    uses_rotary = False

    def __init__(self, d: int, max_xy: int, max_t: int):
        super().__init__()
        self.px = nn.Embedding(max_xy, d)
        self.py = nn.Embedding(max_xy, d)
        self.pt = nn.Embedding(max_t, d)

    def angles(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def add_pos(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        px, py, pt = coords[..., 0], coords[..., 1], coords[..., 2]
        return tokens + self.px(px) + self.py(py) + self.pt(pt)


class RotaryScheme(nn.Module):
    """Multi-axis rotary phase over the chosen coordinate axes (arms B / C).

    `axes` indexes into the (px,py,t) coordinate: [0,1] for 2D, [0,1,2] for 2D+1.
    Produces per-token rotary angles of width head_dim/2, consumed by attention.
    """

    uses_rotary = True

    def __init__(self, axes: List[int], head_dim: int, base: float = 10000.0):
        super().__init__()
        n_pairs = head_dim // 2
        counts = _split_pairs(n_pairs, len(axes))
        axis_of_pair: List[int] = []
        inv_freq: List[float] = []
        for axis, count in zip(axes, counts):
            for i in range(count):
                axis_of_pair.append(axis)
                inv_freq.append(base ** (-(i / max(count, 1))))
        self.register_buffer("axis_of_pair", torch.tensor(axis_of_pair, dtype=torch.long))
        self.register_buffer("inv_freq", torch.tensor(inv_freq, dtype=torch.float32))

    def angles(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N, 3) -> (B, N, head_dim/2)
        selected = coords.float()[..., self.axis_of_pair]   # (B, N, n_pairs)
        return selected * self.inv_freq

    def add_pos(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return tokens


def make_scheme(binding: str, *, d: int, head_dim: int, max_xy: int, max_t: int) -> nn.Module:
    if binding == "none":
        return NoneScheme()
    if binding == "content":
        return ContentPosScheme(d, max_xy, max_t)
    if binding == "pope2d":
        return RotaryScheme([0, 1], head_dim)
    if binding == "pope2d1":
        return RotaryScheme([0, 1, 2], head_dim)
    raise ValueError(f"unknown binding {binding!r} (none|content|pope2d|pope2d1)")


BINDINGS = ("none", "content", "pope2d", "pope2d1")


# --- patchify + coordinate construction -------------------------------------


def patchify(frames: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, W, H, W) long -> (B, W, n_patches, patch*patch) long."""
    b, w, hh, ww = frames.shape
    nph, npw = hh // patch, ww // patch
    f = frames.view(b, w, nph, patch, npw, patch)
    f = f.permute(0, 1, 2, 4, 3, 5).reshape(b, w, nph * npw, patch * patch)
    return f


def build_patch_coords(nph: int, npw: int, n_frames: int) -> torch.Tensor:
    """Coordinates (n_frames*nph*npw, 3) of (px, py, t) for each patch token."""
    rows = torch.arange(nph)
    cols = torch.arange(npw)
    py, px = torch.meshgrid(rows, cols, indexing="ij")
    base = torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1)   # (nph*npw, 2)
    out = []
    for t in range(n_frames):
        t_col = torch.full((base.shape[0], 1), t, dtype=torch.long)
        out.append(torch.cat([base, t_col], dim=-1))
    return torch.cat(out, dim=0)                                   # (n_frames*nph*npw, 3)
