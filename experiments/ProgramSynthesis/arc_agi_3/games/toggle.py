"""Toggle (game_id "tg01") — a switch tile FLIPS a door open<->closed each time the body steps on it.

This is the adversarial pressure-test for the agent's hardcoded subgoal vocabulary. A press-ONCE toggle is
indistinguishable from a one-directional door (open it, walk through), so it would NOT break the agent. The
bite is REVERSIBILITY: here the door starts OPEN and the switch sits on the short path and CLOSES it. The naive
agent models a contact effect as a permanent `color_gone` and cannot even perceive a colour APPEARING, so it
walks onto the switch, slams the door it was about to use, and traps itself. The solution is the detour that
never touches the switch — and none of fire/cover/goal/collect expresses "avoid the tile that reverses an
effect." That has to come from a symmetric, learned dynamics + emergent (not enumerated) subgoals.

ASCII tiles:  #=wall  .=floor  A=agent  G=goal  S=switch  D=door (starts OPEN)
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
C_DOOR = 5
C_SWITCH = 8


class Toggle(Game):
    game_id = "tg01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self.switch: Pos = (0, 0)
        self.door: Pos = (0, 0)
        self.walls: Set[Pos] = set()
        self.door_open = True

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls = set()
        self.door_open = True                                  # the door begins OPEN (passable, rendered as floor)
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
                elif ch == "S":
                    self.switch = pos
                elif ch == "D":
                    self.door = pos

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if not action.is_movement:
            return
        dx, dy = action.delta
        target = (self.agent[0] + dx, self.agent[1] + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        if target == self.door and not self.door_open:        # a closed door blocks
            return
        self.agent = target
        if target == self.switch:                             # stepping on the switch FLIPS the door
            self.door_open = not self.door_open

    def render(self) -> Frame:
        grid: Grid = [[C_BG for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        def put(pos: Pos, color: int) -> None:
            x, y = pos
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[y][x] = color

        for pos in self.walls:
            put(pos, C_WALL)
        put(self.switch, C_SWITCH)
        if not self.door_open:                                # the door is only visible (and blocking) when CLOSED
            put(self.door, C_DOOR)
        put(self.goal, C_GOAL)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return self.agent == self.goal

    def snapshot(self):
        return (self.agent, self.door_open)

    def restore(self, snap) -> None:
        self.agent, self.door_open = snap

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# The door (col 5) starts open. The short path A->S->.->D crosses the switch (col 3), which slams the door shut
# before the body reaches it. The only solution is the lower detour, which never steps on the switch.
_LEVELS: List[List[str]] = [
    [
        "########",
        "#A.S.DG#",
        "#....###",
        "########",
    ],
]
