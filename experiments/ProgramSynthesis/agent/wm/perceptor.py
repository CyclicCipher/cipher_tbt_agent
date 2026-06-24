"""Perception: raw frame grid -> the bits the world-model needs.

Bitter-lesson clean: no LockPath knowledge. Background is the modal color; the
agent-object is whichever single-cell object *translates* when an action is taken
(discovered, not assumed); blockers and goals are inferred downstream from movement
and the score signal.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]
Pos = Tuple[int, int]


def modal_background(grid: Grid) -> int:
    counts: Counter = Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0]


def color_at(grid: Grid, x: int, y: int) -> Optional[int]:
    if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
        return grid[y][x]
    return None


def find_cells(grid: Grid, color: int) -> List[Pos]:
    return [(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == color]


def detect_move(
    old: Grid, new: Grid, bg: int, known_agent: Optional[int] = None
) -> Optional[Tuple[int, Pos, Pos]]:
    """Return (color, p_old, q_new) for the single-cell object that moved, else None."""
    gained: Dict[int, List[Pos]] = {}
    lost: Dict[int, List[Pos]] = {}
    for y in range(len(old)):
        for x in range(len(old[0])):
            o, n = old[y][x], new[y][x]
            if o == n:
                continue
            gained.setdefault(n, []).append((x, y))
            lost.setdefault(o, []).append((x, y))
    candidates = [
        c for c in gained
        if c != bg and len(gained.get(c, [])) == 1 and len(lost.get(c, [])) == 1
    ]
    if not candidates:
        return None
    c = known_agent if known_agent in candidates else candidates[0]
    return c, lost[c][0], gained[c][0]
