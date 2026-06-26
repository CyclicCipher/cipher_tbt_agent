"""Sokoban (game_id "sk01") — push N blocks onto N pads, then reach the goal.

LockPath's block+pad mechanic scaled up: several blocks must each be pushed onto a pad before the goal counts.
It exists to test the agent's cover-pad loop AT SCALE — the factored navigation should push one block at a time
(agent × that block), never the joint state of all blocks. No keys/doors/hazards.

ASCII tiles:  #=wall  .=floor  A=agent  G=goal  B=block  P=pad
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..core import GRID_SIZE, Coordinates, Frame, Grid, GameAction
from ..game import Game

Pos = Tuple[int, int]

C_BG = 0
C_WALL = 1
C_AGENT = 2
C_GOAL = 3
C_BLOCK = 6
C_PAD = 7


class Sokoban(Game):
    game_id = "sk01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self.walls: Set[Pos] = set()
        self.blocks: Set[Pos] = set()
        self.pads: Set[Pos] = set()

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls, self.blocks, self.pads = set(), set(), set()
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
                elif ch == "G":
                    self.goal = pos
                elif ch == "B":
                    self.blocks.add(pos)
                elif ch == "P":
                    self.pads.add(pos)

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if not action.is_movement:
            return
        dx, dy = action.delta
        ax, ay = self.agent
        target = (ax + dx, ay + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        if target in self.blocks:
            beyond = (target[0] + dx, target[1] + dy)
            if not self._in_bounds(beyond) or beyond in self.walls or beyond in self.blocks:
                return
            self.blocks.discard(target)
            self.blocks.add(beyond)
        self.agent = target

    def render(self) -> Frame:
        grid: Grid = [[C_BG for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        def put(pos: Pos, color: int) -> None:
            x, y = pos
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[y][x] = color

        for pos in self.walls:
            put(pos, C_WALL)
        for pos in self.pads:
            put(pos, C_PAD)
        put(self.goal, C_GOAL)
        for pos in self.blocks:
            put(pos, C_BLOCK)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return self.agent == self.goal and self.pads.issubset(self.blocks)

    def snapshot(self):
        return (self.agent, frozenset(self.blocks))

    def restore(self, snap) -> None:
        self.agent, blocks = snap
        self.blocks = set(blocks)

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# Blocks sit in open floor (pushable in a straight line onto an aligned pad); the goal is reached last.
_LEVELS: List[List[str]] = [
    # L0 — one block (the LockPath block+pad case).
    [
        "##########",
        "#A.......#",
        "#.B...P..#",
        "#.......G#",
        "##########",
    ],
    # L1 — two blocks, two pads.
    [
        "###########",
        "#A........#",
        "#.B...P...#",
        "#.B...P..G#",
        "###########",
    ],
    # L2 — two blocks pushed in DIFFERENT directions (one right, one down), then the goal. (Movable count is
    # held at <=2: the agent navigates factored, so this still tests the multi-pad cover loop; the oracle teacher's
    # BFS is over the JOINT block state and blows up past 2 — a tooling cost, not an agent limit.)
    [
        "#########",
        "#A......#",
        "#.B...P.#",
        "#.......#",
        "#...B...#",
        "#...P..G#",
        "#########",
    ],
]
