"""LockPath (game_id "lp01") — an original ARC-AGI-3-style game.

A grid world that uses only Core Knowledge priors (objectness, geometry, basic
physics, agentness) and no language or symbolism, in the spirit of the real
benchmark. Each level introduces one new mechanic; the last level *composes* two
mechanics introduced separately — a deliberate in-environment analogue of the
RecurrentWorldModel A-then-B compositional generalization test.

Mechanics by level:
  L0  navigation        reach the goal (agentness, geometry)
  L1  key + door        a door blocks the goal; step on the key to open all doors
  L2  block + pad       push a block onto a pad to satisfy the win condition
  L3  composition       key+door AND block+pad together, with a hazard to avoid

The win condition is never stated; the agent infers it from the score signal.

Tiles in the ASCII level maps:
  #  wall        .  floor       A  agent start   G  goal
  K  key         D  door        B  block         P  pad (block target)
  X  hazard (stepping on it ends the game)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..core import GRID_SIZE, Coordinates, Frame, Grid, GameAction
from ..game import Game

Pos = Tuple[int, int]  # (x, y)

# Color palette (values 0..15).
C_BG = 0
C_WALL = 1
C_AGENT = 2
C_GOAL = 3
C_KEY = 4
C_DOOR = 5
C_BLOCK = 6
C_PAD = 7
C_HAZARD = 8

_LEVELS: List[List[str]] = [
    # L0 — pure navigation.
    [
        "########",
        "#A.....#",
        "#......#",
        "#......#",
        "#.....G#",
        "########",
    ],
    # L1 — key + door. A wall column splits the room; the only gap is a door.
    [
        "##########",
        "#A...#...#",
        "#....D...#",
        "#K...#...#",
        "#....#..G#",
        "##########",
    ],
    # L2 — block + pad. The win condition now also requires the pad covered.
    [
        "##########",
        "#A.......#",
        "#..B.....#",
        "#........#",
        "#.....P..#",
        "#.......G#",
        "##########",
    ],
    # L3 — composition: key+door AND block+pad, plus a hazard to avoid.
    [
        "############",
        "#A....#....#",
        "#..B..D....#",
        "#.....#.P..#",
        "#K..X.#...G#",
        "############",
    ],
]


class LockPath(Game):
    game_id = "lp01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        # `levels` lets a procedural generator (agent/layouts.py) supply its own
        # boards; default is the bundled 4-level demo curriculum.
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = 0
        self.height = 0
        self.agent: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self.walls: Set[Pos] = set()
        self.doors: Set[Pos] = set()
        self.keys: Set[Pos] = set()
        self.blocks: Set[Pos] = set()
        self.pads: Set[Pos] = set()
        self.hazards: Set[Pos] = set()
        self.has_key = False
        self._dead = False

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        # LockPath is pure navigation/push: only the four directional actions.
        return [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls, self.doors, self.keys = set(), set(), set()
        self.blocks, self.pads, self.hazards = set(), set(), set()
        self.has_key = False
        self._dead = False

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
                elif ch == "K":
                    self.keys.add(pos)
                elif ch == "D":
                    self.doors.add(pos)
                elif ch == "B":
                    self.blocks.add(pos)
                elif ch == "P":
                    self.pads.add(pos)
                elif ch == "X":
                    self.hazards.add(pos)

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if not action.is_movement:
            return
        dx, dy = action.delta
        ax, ay = self.agent
        nx, ny = ax + dx, ay + dy
        target = (nx, ny)

        if not self._in_bounds(target) or target in self.walls:
            return
        if target in self.doors and not self.has_key:
            return

        if target in self.blocks:
            beyond = (nx + dx, ny + dy)
            if not self._can_push_into(beyond):
                return
            self.blocks.discard(target)
            self.blocks.add(beyond)

        # Commit the move.
        self.agent = target
        if target in self.keys:
            self.keys.discard(target)
            self.has_key = True
        if target in self.hazards:
            self._dead = True

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
        for pos in self.hazards:
            put(pos, C_HAZARD)
        for pos in self.doors:
            if not self.has_key:           # an opened door renders as floor
                put(pos, C_DOOR)
        for pos in self.keys:
            put(pos, C_KEY)
        put(self.goal, C_GOAL)
        for pos in self.blocks:
            put(pos, C_BLOCK)
        put(self.agent, C_AGENT)           # agent drawn on top
        return [grid]

    def level_complete(self) -> bool:
        pads_covered = self.pads.issubset(self.blocks)
        return self.agent == self.goal and pads_covered

    def is_dead(self) -> bool:
        return self._dead

    # -- helpers ------------------------------------------------------------

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _can_push_into(self, pos: Pos) -> bool:
        if not self._in_bounds(pos) or pos in self.walls or pos in self.blocks:
            return False
        if pos in self.doors and not self.has_key:
            return False
        return True
