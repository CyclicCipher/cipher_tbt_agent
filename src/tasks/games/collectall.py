"""CollectAll (game_id "ca01") — step on every item to win. There is NO goal cell.

Two things here are outside LockPath's vocabulary, on purpose:
  - the win is a multi-target TOUR (visit every item), not "reach one goal cell";
  - each item is CONSUMED on contact (the cell the body steps on becomes background) — a contact effect that
    removes one cell, not a whole colour at once.

So this is the mechanic that tests whether the *value / dynamics* — not the agent code — can express a
collect-everything goal. The agent is never modified for it.

ASCII tiles:  #=wall  .=floor  A=agent  I=item
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..core import GRID_SIZE, Coordinates, Frame, Grid, GameAction
from ..game import Game

Pos = Tuple[int, int]

C_BG = 0
C_WALL = 1
C_AGENT = 2
C_ITEM = 4


class CollectAll(Game):
    game_id = "ca01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.walls: Set[Pos] = set()
        self.items: Set[Pos] = set()

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls, self.items = set(), set()
        rows = self._levels[level]
        self.height = len(rows)
        self.width = max(len(r) for r in rows)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                pos = (x, y)
                if ch == "#":
                    self.walls.add(pos)
                elif ch == "A":
                    self.agent = pos
                elif ch == "I":
                    self.items.add(pos)

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if not action.is_movement:
            return
        dx, dy = action.delta
        target = (self.agent[0] + dx, self.agent[1] + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        self.items.discard(target)                 # consume on contact
        self.agent = target

    def render(self) -> Frame:
        grid: Grid = [[C_BG for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        def put(pos: Pos, color: int) -> None:
            x, y = pos
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[y][x] = color

        for pos in self.walls:
            put(pos, C_WALL)
        for pos in self.items:
            put(pos, C_ITEM)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return not self.items

    def snapshot(self):
        return (self.agent, frozenset(self.items))

    def restore(self, snap) -> None:
        self.agent, items = snap
        self.items = set(items)

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# Few items per level (the oracle BFS is over (agent, remaining-items) = positions x 2^items).
_LEVELS: List[List[str]] = [
    # L0 — three items.
    [
        "########",
        "#A.I...#",
        "#......#",
        "#.I..I.#",
        "########",
    ],
    # L1 — four items.
    [
        "#########",
        "#A..I...#",
        "#.......#",
        "#.I...I.#",
        "#...I...#",
        "#########",
    ],
    # L2 — five items, a little more spread.
    [
        "##########",
        "#A...I...#",
        "#..I...I.#",
        "#........#",
        "#.I....I.#",
        "##########",
    ],
]
