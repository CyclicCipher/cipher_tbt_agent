"""
Environment 1: ARC-AGI-1 Analog — Rule Inference from Examples.

Procedurally generated episodes. Each episode contains K (input, output)
example pairs and one test input whose output the model must predict.
Rules are drawn from a compositional DSL sampled fresh each episode.

DSL primitives by difficulty:
  difficulty 1 (1-2 primitives, Category 1 only):
    translate, rotate, reflect, recolor, swap, symmetrize, frame, transpose
  difficulty 2 (2-3 primitives, Categories 1+2):
    + gravity, apply_to_color, apply_to_largest, isolate_largest, dilate, outline
  difficulty 3 (2-4 primitives, all categories):
    + fill_holes, color_by_size, count_to_bar, if_count, tile_h, tile_v

Grids: integer numpy arrays of shape (H, W), values 0-9.
Color 0 is background.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Grid = np.ndarray   # shape (H, W), dtype int64, values 0-9
Rule = Callable[[Grid], Grid]


# ---------------------------------------------------------------------------
# Helper: connected components (BFS, 4-connectivity)
# ---------------------------------------------------------------------------

def _connected_components(g: Grid) -> Tuple[np.ndarray, int]:
    """Simple 4-connectivity flood fill. Returns (label_map, n_components).
    Background (0) cells get label -1."""
    H, W = g.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    n = 0
    for r0 in range(H):
        for c0 in range(W):
            if g[r0, c0] != 0 and labels[r0, c0] == -1:
                # BFS
                queue = [(r0, c0)]
                labels[r0, c0] = n
                while queue:
                    r, c = queue.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if g[nr, nc] != 0 and labels[nr, nc] == -1:
                                labels[nr, nc] = n
                                queue.append((nr, nc))
                n += 1
    return labels, n


# ---------------------------------------------------------------------------
# Category 1 — Simple Transforms (Difficulty 1)
# ---------------------------------------------------------------------------

def prim_translate(dx: int, dy: int) -> Rule:
    """Shift all non-background cells by (dx cols, dy rows)."""
    def f(g: Grid) -> Grid:
        H, W = g.shape
        out = np.zeros_like(g)
        for r in range(H):
            for c in range(W):
                if g[r, c] != 0:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = g[r, c]
        return out.astype(np.int64)
    f.__name__ = f'translate({dx},{dy})'
    return f


def prim_rotate(k: int) -> Rule:
    """Rotate grid 90° counter-clockwise k times (k in 1-3)."""
    def f(g: Grid) -> Grid:
        return np.rot90(g, k).copy().astype(np.int64)
    f.__name__ = f'rotate({k})'
    return f


def prim_reflect(axis: str) -> Rule:
    """Reflect: 'h' = left-right flip, 'v' = up-down flip."""
    def f(g: Grid) -> Grid:
        if axis == 'h':
            return np.flip(g, axis=1).copy().astype(np.int64)
        else:
            return np.flip(g, axis=0).copy().astype(np.int64)
    f.__name__ = f'reflect({axis})'
    return f


def prim_recolor(src: int, tgt: int) -> Rule:
    """Replace all cells of color src with tgt."""
    def f(g: Grid) -> Grid:
        out = g.copy().astype(np.int64)
        out[g == src] = tgt
        return out
    f.__name__ = f'recolor({src}->{tgt})'
    return f


def prim_swap(c1: int, c2: int) -> Rule:
    """Swap colors c1 and c2 throughout the grid."""
    def f(g: Grid) -> Grid:
        out = g.copy().astype(np.int64)
        out[g == c1] = c2
        out[g == c2] = c1
        return out
    f.__name__ = f'swap({c1},{c2})'
    return f


def prim_symmetrize(mode: str) -> Rule:
    """Force symmetry.
    'h' = mirror left half to right half.
    'v' = mirror top half to bottom half.
    '4fold' = 4-way symmetry (copy top-left quadrant to all four quadrants).
    """
    def f(g: Grid) -> Grid:
        out = g.copy().astype(np.int64)
        H, W = g.shape
        if mode == 'h':
            for c in range(W // 2):
                out[:, W - 1 - c] = out[:, c]
        elif mode == 'v':
            for r in range(H // 2):
                out[H - 1 - r, :] = out[r, :]
        else:  # 4fold
            # First mirror left→right
            for c in range(W // 2):
                out[:, W - 1 - c] = out[:, c]
            # Then mirror top→bottom
            for r in range(H // 2):
                out[H - 1 - r, :] = out[r, :]
        return out
    f.__name__ = f'symmetrize({mode})'
    return f


def prim_frame(color: int) -> Rule:
    """Set border pixels (row 0, row H-1, col 0, col W-1) to color."""
    def f(g: Grid) -> Grid:
        out = g.copy().astype(np.int64)
        H, W = g.shape
        out[0, :] = color
        out[H - 1, :] = color
        out[:, 0] = color
        out[:, W - 1] = color
        return out
    f.__name__ = f'frame({color})'
    return f


def prim_transpose() -> Rule:
    """Matrix transpose (g.T.copy()); if not square, return g.copy() unchanged."""
    def f(g: Grid) -> Grid:
        H, W = g.shape
        if H != W:
            return g.copy().astype(np.int64)
        return g.T.copy().astype(np.int64)
    f.__name__ = 'transpose()'
    return f


# ---------------------------------------------------------------------------
# Category 2 — Object-Aware (Difficulty 2)
# ---------------------------------------------------------------------------

def prim_gravity(direction: int) -> Rule:
    """Slide each non-background pixel in direction as far as possible.
    direction: 0=down, 1=up, 2=left, 3=right.
    Process column-by-column for up/down, row-by-row for left/right.
    """
    def f(g: Grid) -> Grid:
        H, W = g.shape
        if g.max() == 0:
            return np.zeros_like(g, dtype=np.int64)
        out = np.zeros((H, W), dtype=np.int64)
        if direction == 0:  # down
            for c in range(W):
                col = g[:, c]
                vals = col[col != 0]
                out[H - len(vals):, c] = vals
        elif direction == 1:  # up
            for c in range(W):
                col = g[:, c]
                vals = col[col != 0]
                out[:len(vals), c] = vals
        elif direction == 2:  # left
            for r in range(H):
                row = g[r, :]
                vals = row[row != 0]
                out[r, :len(vals)] = vals
        else:  # right
            for r in range(H):
                row = g[r, :]
                vals = row[row != 0]
                out[r, W - len(vals):] = vals
        return out
    dir_names = {0: 'down', 1: 'up', 2: 'left', 3: 'right'}
    f.__name__ = f'gravity({dir_names.get(direction, direction)})'
    return f


def prim_apply_to_color(color: int, inner: Rule) -> Rule:
    """Apply inner rule, then keep original values where input was NOT target color."""
    def f(g: Grid) -> Grid:
        mask = g == color
        if not mask.any():
            return g.copy().astype(np.int64)
        transformed = inner(g)
        out = g.copy().astype(np.int64)
        out[mask] = transformed[mask]
        return out
    f.__name__ = f'apply_to_color({color},{inner.__name__})'
    return f


def prim_apply_to_largest(inner: Rule) -> Rule:
    """Apply inner rule only to pixels of the largest 4-connected non-background component."""
    def f(g: Grid) -> Grid:
        labels, n = _connected_components(g)
        if n == 0:
            return g.copy().astype(np.int64)
        sizes = [(labels == i).sum() for i in range(n)]
        largest = int(np.argmax(sizes))
        mask = labels == largest
        transformed = inner(g)
        out = g.copy().astype(np.int64)
        out[mask] = transformed[mask]
        return out
    f.__name__ = f'apply_to_largest({inner.__name__})'
    return f


def prim_isolate_largest() -> Rule:
    """Zero out everything except the largest connected component."""
    def f(g: Grid) -> Grid:
        labels, n = _connected_components(g)
        if n == 0:
            return np.zeros_like(g, dtype=np.int64)
        sizes = [(labels == i).sum() for i in range(n)]
        largest = int(np.argmax(sizes))
        out = np.zeros_like(g, dtype=np.int64)
        mask = labels == largest
        out[mask] = g[mask]
        return out
    f.__name__ = 'isolate_largest()'
    return f


def prim_dilate(n: int) -> Rule:
    """Morphological dilation by n steps.
    Each non-background pixel spreads its color to 4-connected background neighbors, n times.
    """
    def f(g: Grid) -> Grid:
        H, W = g.shape
        out = g.copy().astype(np.int64)
        for _ in range(n):
            new = out.copy()
            for r in range(H):
                for c in range(W):
                    if out[r, c] != 0:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and new[nr, nc] == 0:
                                new[nr, nc] = out[r, c]
            out = new
        return out
    f.__name__ = f'dilate({n})'
    return f


def prim_outline() -> Rule:
    """Keep only border pixels of each non-background connected component.
    A pixel is border if any of its 4 neighbors is background or out-of-bounds.
    Zero out interior pixels.
    """
    def f(g: Grid) -> Grid:
        H, W = g.shape
        out = g.copy().astype(np.int64)
        for r in range(H):
            for c in range(W):
                if g[r, c] != 0:
                    # Check if interior: all 4 neighbors are non-background and in-bounds
                    is_interior = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= H or nc < 0 or nc >= W:
                            is_interior = False
                            break
                        if g[nr, nc] == 0:
                            is_interior = False
                            break
                    if is_interior:
                        out[r, c] = 0
        return out
    f.__name__ = 'outline()'
    return f


# ---------------------------------------------------------------------------
# Category 3 — Relational / Counting (Difficulty 3)
# ---------------------------------------------------------------------------

def prim_fill_holes(fill_color: int) -> Rule:
    """Flood-fill from borders to find exterior background.
    Fill enclosed background (not reachable from border) with fill_color.
    """
    def f(g: Grid) -> Grid:
        H, W = g.shape
        # BFS from all border background cells to find exterior background
        visited = np.zeros((H, W), dtype=bool)
        queue = []
        for r in range(H):
            for c in [0, W - 1]:
                if g[r, c] == 0 and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        for c in range(W):
            for r in [0, H - 1]:
                if g[r, c] == 0 and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        while queue:
            r, c = queue.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and g[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        # Enclosed background: background cells not visited
        out = g.copy().astype(np.int64)
        enclosed = (g == 0) & ~visited
        out[enclosed] = fill_color
        return out
    f.__name__ = f'fill_holes({fill_color})'
    return f


def prim_color_by_size() -> Rule:
    """Find all non-background connected components; rank by size descending.
    Component at rank r gets color (r % 9) + 1. Background stays 0.
    """
    def f(g: Grid) -> Grid:
        labels, n = _connected_components(g)
        if n == 0:
            return np.zeros_like(g, dtype=np.int64)
        sizes = [(labels == i).sum() for i in range(n)]
        # Rank: largest first
        order = np.argsort(sizes)[::-1]  # indices sorted largest→smallest
        rank_of = np.empty(n, dtype=np.int64)
        for rank, comp_idx in enumerate(order):
            rank_of[comp_idx] = rank
        out = np.zeros_like(g, dtype=np.int64)
        for comp_idx in range(n):
            mask = labels == comp_idx
            color = int(rank_of[comp_idx] % 9) + 1
            out[mask] = color
        return out
    f.__name__ = 'color_by_size()'
    return f


def prim_count_to_bar(color: int) -> Rule:
    """Count cells of given color; output: a 1-row bar of that color in row 0,
    from col 0 to col count-1 (clamped to W). Everything else is background (0).
    """
    def f(g: Grid) -> Grid:
        H, W = g.shape
        count = int((g == color).sum())
        out = np.zeros((H, W), dtype=np.int64)
        fill = min(count, W)
        if fill > 0:
            out[0, :fill] = color
        return out
    f.__name__ = f'count_to_bar({color})'
    return f


def prim_if_count(color: int, n: int, then_rule: Rule, else_rule: Rule) -> Rule:
    """Apply then_rule if count(color) == n, else else_rule."""
    def f(g: Grid) -> Grid:
        count = int((g == color).sum())
        if count == n:
            return then_rule(g)
        else:
            return else_rule(g)
    f.__name__ = f'if_count({color}=={n},{then_rule.__name__},{else_rule.__name__})'
    return f


def prim_tile_h() -> Rule:
    """For each row, take the left half (cols 0..W//2-1), mirror it to the right half."""
    def f(g: Grid) -> Grid:
        H, W = g.shape
        out = g.copy().astype(np.int64)
        half = W // 2
        for c in range(half):
            out[:, W - 1 - c] = out[:, c]
        # If W is odd, center col (at index half) stays as-is
        return out
    f.__name__ = 'tile_h()'
    return f


def prim_tile_v() -> Rule:
    """Take top half rows, mirror to bottom half."""
    def f(g: Grid) -> Grid:
        H, W = g.shape
        out = g.copy().astype(np.int64)
        half = H // 2
        for r in range(half):
            out[H - 1 - r, :] = out[r, :]
        return out
    f.__name__ = 'tile_v()'
    return f


# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------

def gen_patches(H: int, W: int, rng: np.random.Generator, n_colors: int = 9) -> Grid:
    """1-4 colored rectangles on background."""
    grid = np.zeros((H, W), dtype=np.int64)
    n_patches = int(rng.integers(1, 5))
    for _ in range(n_patches):
        color = int(rng.integers(1, n_colors + 1))
        ph = int(rng.integers(1, max(2, H // 2 + 1)))
        pw = int(rng.integers(1, max(2, W // 2 + 1)))
        r = int(rng.integers(0, max(1, H - ph + 1)))
        c = int(rng.integers(0, max(1, W - pw + 1)))
        grid[r:r + ph, c:c + pw] = color
    return grid


def gen_scattered(H: int, W: int, rng: np.random.Generator,
                  n_colors: int = 9, density: Optional[float] = None) -> Grid:
    """Scattered pixels, density sampled from Uniform(0.1, 0.4) if not given."""
    grid = np.zeros((H, W), dtype=np.int64)
    if density is None:
        density = float(rng.uniform(0.1, 0.4))
    mask = rng.random((H, W)) < density
    colors = rng.integers(1, n_colors + 1, size=(H, W)).astype(np.int64)
    grid[mask] = colors[mask]
    return grid


def gen_two_objects(H: int, W: int, rng: np.random.Generator, n_colors: int = 9) -> Grid:
    """Exactly 2 differently-colored blobs in different quadrants."""
    grid = np.zeros((H, W), dtype=np.int64)
    # Pick 2 distinct colors
    c1 = int(rng.integers(1, n_colors + 1))
    c2 = int(rng.integers(1, n_colors + 1))
    while c2 == c1:
        c2 = int(rng.integers(1, n_colors + 1))

    half_H = max(1, H // 2)
    half_W = max(1, W // 2)

    # Quadrant assignments: pick 2 of 4 quadrants
    quadrants = [(0, 0), (0, half_W), (half_H, 0), (half_H, half_W)]
    idxs = rng.choice(4, size=2, replace=False)
    q1 = quadrants[idxs[0]]
    q2 = quadrants[idxs[1]]

    for color, (qr, qc) in [(c1, q1), (c2, q2)]:
        qH = H - qr if qr > 0 else half_H
        qW = W - qc if qc > 0 else half_W
        qH = max(1, qH)
        qW = max(1, qW)
        ph = int(rng.integers(1, max(2, qH // 2 + 1)))
        pw = int(rng.integers(1, max(2, qW // 2 + 1)))
        r = qr + int(rng.integers(0, max(1, qH - ph + 1)))
        c = qc + int(rng.integers(0, max(1, qW - pw + 1)))
        r = min(r, H - ph)
        c = min(c, W - pw)
        grid[r:r + ph, c:c + pw] = color

    return grid


def gen_ring(H: int, W: int, rng: np.random.Generator, n_colors: int = 9) -> Grid:
    """Hollow rectangle (ring shape) that creates an enclosed background region.
    If H < 3 or W < 3, fall back to gen_patches.
    """
    if H < 3 or W < 3:
        return gen_patches(H, W, rng, n_colors)
    grid = np.zeros((H, W), dtype=np.int64)
    color = int(rng.integers(1, n_colors + 1))
    # Ring spans the whole grid (border only)
    grid[0, :] = color
    grid[H - 1, :] = color
    grid[:, 0] = color
    grid[:, W - 1] = color
    # Optionally add a second ring / partial fill for variety
    if H >= 5 and W >= 5 and rng.random() < 0.4:
        c2 = int(rng.integers(1, n_colors + 1))
        while c2 == color:
            c2 = int(rng.integers(1, n_colors + 1))
        grid[1, 1:W - 1] = c2
        grid[H - 2, 1:W - 1] = c2
        grid[1:H - 1, 1] = c2
        grid[1:H - 1, W - 2] = c2
    return grid


def gen_geometric(H: int, W: int, rng: np.random.Generator, n_colors: int = 9) -> Grid:
    """A single geometric shape: cross, L-shape, T-shape, or diagonal bar."""
    grid = np.zeros((H, W), dtype=np.int64)
    color = int(rng.integers(1, n_colors + 1))
    shape = rng.choice(['cross', 'L', 'T', 'diag'])
    mid_r = H // 2
    mid_c = W // 2

    if shape == 'cross':
        grid[mid_r, :] = color
        grid[:, mid_c] = color
    elif shape == 'L':
        # Vertical bar on left + horizontal bar on bottom
        grid[:, 0] = color
        grid[H - 1, :] = color
    elif shape == 'T':
        # Top row + vertical center bar
        grid[0, :] = color
        grid[:, mid_c] = color
    else:  # diag
        length = min(H, W)
        for i in range(length):
            grid[i, i] = color

    return grid


def generate_grid(H: int, W: int, rng: np.random.Generator,
                  n_colors: int = 9, grid_type: str = 'random') -> Grid:
    """Generate a grid of the specified type.
    grid_type: 'patches', 'scattered', 'two_objects', 'ring', 'geometric', 'random'
    """
    if grid_type == 'random':
        grid_type = rng.choice(['patches', 'scattered', 'two_objects', 'ring', 'geometric'])
    if grid_type == 'patches':
        return gen_patches(H, W, rng, n_colors)
    elif grid_type == 'scattered':
        return gen_scattered(H, W, rng, n_colors)
    elif grid_type == 'two_objects':
        return gen_two_objects(H, W, rng, n_colors)
    elif grid_type == 'ring':
        return gen_ring(H, W, rng, n_colors)
    elif grid_type == 'geometric':
        return gen_geometric(H, W, rng, n_colors)
    else:
        raise ValueError(f'Unknown grid_type: {grid_type}')


# ---------------------------------------------------------------------------
# Rule sampling
# ---------------------------------------------------------------------------

_CAT1 = ['translate', 'rotate', 'reflect', 'recolor', 'swap',
         'symmetrize', 'frame', 'transpose']
_CAT2 = ['gravity', 'apply_to_color', 'apply_to_largest',
         'isolate_largest', 'dilate', 'outline']
_CAT3 = ['fill_holes', 'color_by_size', 'count_to_bar',
         'if_count', 'tile_h', 'tile_v']

_DIFFICULTY_1_TYPES = _CAT1
_DIFFICULTY_2_TYPES = _CAT1 + _CAT2
_DIFFICULTY_3_TYPES = _CAT1 + _CAT2 + _CAT3


def _sample_one_primitive(ptype: str, rng: np.random.Generator,
                           n_colors: int = 9,
                           inner_rule: Optional[Rule] = None) -> Rule:
    """Instantiate a single primitive of the given type."""
    if ptype == 'translate':
        dx = int(rng.integers(-3, 4))
        dy = int(rng.integers(-3, 4))
        if dx == 0 and dy == 0:
            dx = 1
        return prim_translate(dx, dy)

    elif ptype == 'rotate':
        k = int(rng.integers(1, 4))
        return prim_rotate(k)

    elif ptype == 'reflect':
        axis = rng.choice(['h', 'v'])
        return prim_reflect(axis)

    elif ptype == 'recolor':
        src = int(rng.integers(1, n_colors + 1))
        tgt = int(rng.integers(1, n_colors + 1))
        while tgt == src:
            tgt = int(rng.integers(1, n_colors + 1))
        return prim_recolor(src, tgt)

    elif ptype == 'swap':
        c1 = int(rng.integers(1, n_colors + 1))
        c2 = int(rng.integers(1, n_colors + 1))
        while c2 == c1:
            c2 = int(rng.integers(1, n_colors + 1))
        return prim_swap(c1, c2)

    elif ptype == 'symmetrize':
        mode = rng.choice(['h', 'v', '4fold'])
        return prim_symmetrize(mode)

    elif ptype == 'frame':
        color = int(rng.integers(1, n_colors + 1))
        return prim_frame(color)

    elif ptype == 'transpose':
        return prim_transpose()

    elif ptype == 'gravity':
        direction = int(rng.integers(0, 4))
        return prim_gravity(direction)

    elif ptype == 'apply_to_color':
        color = int(rng.integers(1, n_colors + 1))
        if inner_rule is None:
            inner_type = rng.choice(_CAT1)
            inner_rule = _sample_one_primitive(inner_type, rng, n_colors)
        return prim_apply_to_color(color, inner_rule)

    elif ptype == 'apply_to_largest':
        if inner_rule is None:
            inner_type = rng.choice(_CAT1)
            inner_rule = _sample_one_primitive(inner_type, rng, n_colors)
        return prim_apply_to_largest(inner_rule)

    elif ptype == 'isolate_largest':
        return prim_isolate_largest()

    elif ptype == 'dilate':
        n = int(rng.integers(1, 4))
        return prim_dilate(n)

    elif ptype == 'outline':
        return prim_outline()

    elif ptype == 'fill_holes':
        fill_color = int(rng.integers(1, n_colors + 1))
        return prim_fill_holes(fill_color)

    elif ptype == 'color_by_size':
        return prim_color_by_size()

    elif ptype == 'count_to_bar':
        color = int(rng.integers(1, n_colors + 1))
        return prim_count_to_bar(color)

    elif ptype == 'if_count':
        color = int(rng.integers(1, n_colors + 1))
        n = int(rng.integers(1, 6))
        then_type = rng.choice(_CAT1)
        else_type = rng.choice(_CAT1)
        then_rule = _sample_one_primitive(then_type, rng, n_colors)
        else_rule = _sample_one_primitive(else_type, rng, n_colors)
        return prim_if_count(color, n, then_rule, else_rule)

    elif ptype == 'tile_h':
        return prim_tile_h()

    elif ptype == 'tile_v':
        return prim_tile_v()

    else:
        raise ValueError(f'Unknown primitive type: {ptype}')


def compose(primitives: List[Rule]) -> Rule:
    """Apply list of rules left-to-right (first applied first)."""
    def f(g: Grid) -> Grid:
        result = g
        for p in primitives:
            result = p(result)
        return result
    f.__name__ = '+'.join(getattr(p, '__name__', '?') for p in primitives)
    return f


def sample_rule(difficulty: int, rng: np.random.Generator,
                n_colors: int = 9) -> Rule:
    """Sample a compositional rule of given difficulty.

    difficulty 1: 1-2 primitives from Category 1 only.
    difficulty 2: 2-3 primitives from Categories 1+2.
    difficulty 3: 2-4 primitives from all categories.
    """
    if difficulty == 1:
        n_prims = int(rng.integers(1, 3))
        prim_pool = _DIFFICULTY_1_TYPES
    elif difficulty == 2:
        n_prims = int(rng.integers(2, 4))
        prim_pool = _DIFFICULTY_2_TYPES
    else:
        n_prims = int(rng.integers(2, 5))
        prim_pool = _DIFFICULTY_3_TYPES

    chosen = [str(rng.choice(prim_pool)) for _ in range(n_prims)]
    prims = [_sample_one_primitive(t, rng, n_colors) for t in chosen]
    return compose(prims)


# ---------------------------------------------------------------------------
# Rule-aware grid generator selection
# ---------------------------------------------------------------------------

def _pick_grid_type(rule_name: str, rng: np.random.Generator) -> str:
    """Choose grid generator based on the first primitive in the rule name."""
    first_prim = rule_name.split('+')[0] if rule_name else ''

    if 'fill_holes' in first_prim:
        return 'ring'
    elif 'color_by_size' in first_prim or 'isolate_largest' in first_prim:
        return rng.choice(['two_objects', 'patches'])
    elif 'gravity' in first_prim:
        return rng.choice(['scattered', 'patches'])
    else:
        return rng.choice(['patches', 'scattered', 'two_objects', 'ring', 'geometric'])


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def sample_episode(
    H: int,
    W: int,
    K: int,
    difficulty: int,
    rng: np.random.Generator,
    n_colors: int = 9,
) -> Dict:
    """Sample one ARC-AGI-1 episode.

    Returns dict with keys:
      input_grids  (list of K Grid arrays)
      output_grids (list of K Grid arrays)
      test_input   (Grid)
      test_output  (Grid)
      rule_name    (str, for debugging)
      H            (int)
      W            (int)
    """
    MAX_ATTEMPTS = 10

    for _ in range(MAX_ATTEMPTS):
        rule = sample_rule(difficulty, rng, n_colors)
        rule_name = getattr(rule, '__name__', '?')
        grid_type = _pick_grid_type(rule_name, rng)

        all_inputs = [generate_grid(H, W, rng, n_colors, grid_type)
                      for _ in range(K + 1)]
        all_outputs = [rule(g) for g in all_inputs]

        # Ensure at least one cell differs in the test pair
        test_in = all_inputs[K]
        test_out = all_outputs[K]
        if not np.array_equal(test_in, test_out):
            return {
                'input_grids': all_inputs[:K],
                'output_grids': all_outputs[:K],
                'test_input': test_in,
                'test_output': test_out,
                'rule_name': rule_name,
                'H': H,
                'W': W,
            }

    # Fallback: use last sampled episode even if identical (shouldn't normally happen)
    rule = sample_rule(difficulty, rng, n_colors)
    rule_name = getattr(rule, '__name__', '?')
    grid_type = _pick_grid_type(rule_name, rng)
    all_inputs = [generate_grid(H, W, rng, n_colors, grid_type) for _ in range(K + 1)]
    all_outputs = [rule(g) for g in all_inputs]
    return {
        'input_grids': all_inputs[:K],
        'output_grids': all_outputs[:K],
        'test_input': all_inputs[K],
        'test_output': all_outputs[K],
        'rule_name': rule_name,
        'H': H,
        'W': W,
    }


# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(42)

    for diff in [1, 2, 3]:
        print(f'=== Difficulty {diff} ===')
        for trial in range(3):
            ep = sample_episode(H=6, W=6, K=3, difficulty=diff, rng=rng)
            print(f'  Trial {trial}: rule={ep["rule_name"]}')
            print(f'    input shape : {ep["test_input"].shape}')
            print(f'    output shape: {ep["test_output"].shape}')
            diff_cells = int((ep["test_input"] != ep["test_output"]).sum())
            print(f'    cells changed: {diff_cells}')
        print()
