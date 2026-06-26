"""MultiKey (game_id "mk01") — an ARC-AGI-3-style game with TWO key→door mechanics.

Two differently-coloured keys, each opening its own colour of doors; the goal sits behind both. A direct
generalization of LockPath's single key+door: the agent must acquire BOTH keys, in the order the layout forces.
It exists to stress the agent's generality past LockPath's exactly-one-key world — the dynamics column should
discover TWO `color_gone` rules and the control loop should sequence two key-subgoals, not one.

ASCII tiles:  #=wall  .=floor  A=agent  G=goal  a=key_A  c=door_A  b=key_B  d=door_B  X=hazard
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
C_KEY_A = 4
C_DOOR_A = 5
C_HAZARD = 8
C_KEY_B = 9
C_DOOR_B = 10

_CHARS = {"a": "keys_a", "b": "keys_b", "c": "doors_a", "d": "doors_b", "X": "hazards"}


class MultiKey(Game):
    game_id = "mk01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self.walls: Set[Pos] = set()
        self.keys_a: Set[Pos] = set()
        self.keys_b: Set[Pos] = set()
        self.doors_a: Set[Pos] = set()
        self.doors_b: Set[Pos] = set()
        self.hazards: Set[Pos] = set()
        self.has_a = False
        self.has_b = False
        self._dead = False

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.keys_a, self.keys_b = set(), set()
        self.doors_a, self.doors_b = set(), set()
        self.walls, self.hazards = set(), set()
        self.has_a = self.has_b = False
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
                elif ch in _CHARS:
                    getattr(self, _CHARS[ch]).add(pos)

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if not action.is_movement:
            return
        dx, dy = action.delta
        ax, ay = self.agent
        target = (ax + dx, ay + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        if target in self.doors_a and not self.has_a:
            return
        if target in self.doors_b and not self.has_b:
            return
        self.agent = target
        if target in self.keys_a:
            self.keys_a.discard(target); self.has_a = True
        if target in self.keys_b:
            self.keys_b.discard(target); self.has_b = True
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
        for pos in self.hazards:
            put(pos, C_HAZARD)
        if not self.has_a:
            for pos in self.doors_a:
                put(pos, C_DOOR_A)
        if not self.has_b:
            for pos in self.doors_b:
                put(pos, C_DOOR_B)
        for pos in self.keys_a:
            put(pos, C_KEY_A)
        for pos in self.keys_b:
            put(pos, C_KEY_B)
        put(self.goal, C_GOAL)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return self.agent == self.goal

    def is_dead(self) -> bool:
        return self._dead

    def snapshot(self):
        return (self.agent, frozenset(self.keys_a), frozenset(self.keys_b),
                self.has_a, self.has_b, self._dead)

    def restore(self, snap) -> None:
        self.agent, ka, kb, self.has_a, self.has_b, self._dead = snap
        self.keys_a, self.keys_b = set(ka), set(kb)

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# A small bundled demo curriculum (the procedural generator supplies the real distribution). The walls seal
# each chamber so the doors are the ONLY passages — door_A behind chamber 1, door_B behind chamber 2 — forcing
# both keys, in series.
_LEVELS: List[List[str]] = [
    # L0 — one key (intro).
    [
        "##########",
        "#A.a#...G#",
        "#...c....#",
        "##########",
    ],
    # L1 — two keys in series: key_A opens door_A → chamber 2 has key_B → door_B → the goal.
    [
        "############",
        "#A.a#.b.#.G#",
        "#...c...d..#",
        "############",
    ],
]
