"""perceive.py — the peripheral RETINA (ARCHITECTURE §peripheral): transduce a game FRAME (a colour grid) into OBJECTS.

A frame is a 2-D grid of colour values (0 = background). This turns it into the agent's front-end: a set of OBJECTS, each a
same-colour 4-connected region — Core-Knowledge OBJECTNESS (the replica games use only Core-Knowledge priors, `tasks/game.py`).
It is the BOTTOM-UP proposal ONLY: the deeper object individuation — grouping by COMMON FATE (things that move together) and by
RECOGNITION (structure, not colour) — refines it (`reference_tbt_segmentation_and_grouping`; our `Column._common_fate_groups`),
and is where colour stops being load-bearing.

The 'self' is NOT a colour: the controllable ROOT is DISCOVERED as the object whose motion correlates with the action
(`SelfTracker`; `reference_l5_operator_kinds`, `feedback_bitter_lesson` — colour-as-self was the mistake). No semantics are read
from the frame — the agent infers the mechanics from colour + the score, exactly as the human in `play.py` does. Pure stdlib.
"""

from __future__ import annotations

from collections import Counter, namedtuple

Object = namedtuple("Object", ["color", "cells", "anchor"])   # colour, its 4-connected cells, top-left anchor


def segment(grid, background: int = 0) -> list:
    """Connected-component OBJECTNESS: every SAME-COLOUR 4-connected region of non-background cells is ONE object, anchored at
    its top-left cell (a stable point that tracks the object's motion). Returned sorted by anchor (deterministic)."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    objects = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or grid[y][x] == background:
                continue
            color = grid[y][x]
            cells, stack = set(), [(x, y)]
            seen[y][x] = True
            while stack:
                cx, cy = stack.pop()
                cells.add((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            objects.append(Object(color, frozenset(cells), min(cells)))
    objects.sort(key=lambda o: o.anchor)
    return objects


class SelfTracker:
    """Discover the CONTROLLABLE ROOT (the 'self'): the object whose motion CORRELATES with the action, learned over
    transitions — never colour-as-self (`feedback_bitter_lesson`; `reference_l5_operator_kinds`: the self is the object the
    action controls, DISCOVERED not assigned). `observe` counts, per colour, how often an object of that colour moved right
    after an action; `root` is the most-consistently-controlled colour — a colour that moves EVERY action (the self) outvotes
    one that moves only when pushed. Per-action displacement consistency + excluding autonomous movers is the refinement."""

    def __init__(self) -> None:
        self._moved: Counter = Counter()
        self._steps = 0

    def observe(self, before, after, action) -> None:
        """One transition: a colour that VACATED a cell and APPEARED at another moved. (`action` accepted for the deferred
        per-action correlation; the count-of-moves suffices to find the root in a movement game.)"""
        self._steps += 1
        b = {(o.color, o.anchor) for o in before}
        a = {(o.color, o.anchor) for o in after}
        out_colours = {c for c, _ in (b - a)}
        in_colours = {c for c, _ in (a - b)}
        for color in out_colours & in_colours:
            self._moved[color] += 1

    def root(self):
        """The colour of the discovered controllable root, or None until an object has been seen to move under action."""
        return self._moved.most_common(1)[0][0] if self._moved else None
