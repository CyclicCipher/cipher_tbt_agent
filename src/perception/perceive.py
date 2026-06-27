"""Frame reading for the ColumnAgent (no semantic priors).

Only the cheap statistical fact (modal background) is kept. There is NO agency code here: agency is not a
perceptual inference to be discovered, it is the efference copy — the agent knows the action it issued —
so self-localization lives in the agent, on top of the L6 grid (see agent.py). This module just turns a
frame into cells and reports raw motion; which motion is *mine* is decided by the efference copy.
"""

from __future__ import annotations

from collections import Counter

_UNITS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

_BG_CACHE: dict = {}                                          # id(grid) -> (grid, bg), identity-checked + capped


def modal_background(grid):
    """The most common cell value — the (statistical, not semantic) background. Cached per grid OBJECT: a frame's
    grid is read several times per step (both perception passes + segmentation) and reused as the previous frame
    next step, so the Counter over the full 64x64 frame would otherwise run ~4x per frame redundantly."""
    key = id(grid)
    hit = _BG_CACHE.get(key)
    if hit is not None and hit[0] is grid:                    # identity check: a recycled id is not a false hit
        return hit[1]
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    if len(_BG_CACHE) > 16:                                   # keep only a small working set (the live frames)
        _BG_CACHE.clear()
    _BG_CACHE[key] = (grid, bg)                               # storing grid keeps it alive -> its id can't be reused
    return bg


def active_cells(grid, bg):
    """Non-background cells as {(x, y): colour} — the level inside the mostly-blank frame."""
    return {(x, y): v for y, row in enumerate(grid) for x, v in enumerate(row) if v != bg}


def bounding_box(cells):
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return min(xs), min(ys), max(xs), max(ys)


def detect_motion(prev_cells, cells):
    """Raw exafference+reafference: {colour: (dx, dy)} for every token that TRANSLATED by one cell. The
    agent later keeps the one its efference copy predicts (self) and treats the rest as environment."""
    moved = {}
    for (x, y), c in prev_cells.items():
        if cells.get((x, y)) == c:
            continue                                              # didn't leave — not a translation
        for dx, dy in _UNITS:
            p = (x + dx, y + dy)
            if cells.get(p) == c and prev_cells.get(p) != c:     # c newly appears one cell over
                moved[c] = (dx, dy)
                break
    return moved
