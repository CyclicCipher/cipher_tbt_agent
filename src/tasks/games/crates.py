"""Crates (game_id "cr01") — push the crate onto the pad. The fixture that both PAYS and PUSHES.

WHY THIS GAME EXISTS. The relation-SDR value (`Column.relation_code`) makes a configuration's worth generalise to
configurations never visited, and that was demonstrated in controlled measurement but never live, because no existing game
could exercise it: the games that PAY have no pushing (CollectAll, LockPath L0/L1 — so every rollout fork shares the current
configuration and there is nothing to generalise ACROSS), and the games that PUSH never pay (Sokoban pays nothing; LockPath
L2 is never reached, so `R` stays empty). This game closes exactly that gap and nothing else.

THE DESIGN IS ADVERSARIAL TO THE THING IT TESTS, on purpose. Every level puts the crate and the pad at DIFFERENT ABSOLUTE
POSITIONS, and the push runs in a different direction. So:
  * nothing POSITIONAL carries from one level to the next — a value learned at level 0's cells is worthless at level 1's,
    which is the standing limitation of the positional critic (`GoalMemory`'s docstring makes the same point for features);
  * what CAN carry is the RELATION "crate resting on pad", which is translation-invariant by construction and whose
    near-misses overlap by the grid code. If the relational value is doing nothing, later levels cost the same as a cold
    start, and the fixture says so.
That is the whole point: the game is built so that a null result is visible rather than hidden.

The mechanic is Core-Knowledge only (objectness, geometry, basic physics, agentness): walking into a crate pushes it one
cell if the far side is clear, and is blocked otherwise. No language, no symbols, nothing the agent is told.

ASCII tiles:  #=wall  .=floor  A=agent  B=crate  P=pad
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..core import GRID_SIZE, Coordinates, Frame, Grid, GameAction
from ..game import Game

Pos = Tuple[int, int]

C_BG = 0
C_WALL = 1
C_AGENT = 2
C_CRATE = 6
C_PAD = 7


class Crates(Game):
    game_id = "cr01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.crate: Pos = (0, 0)
        self.pad: Pos = (0, 0)
        self.walls: Set[Pos] = set()

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls = set()
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
                elif ch == "B":
                    self.crate = pos
                elif ch == "P":
                    self.pad = pos

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        """Walk, or push. Walking into the crate moves it one cell when the far side is clear; when it is not, the move is
        refused and NOTHING happens — the agent does not slide past a crate it failed to shift."""
        if not action.is_movement:
            return
        dx, dy = action.delta
        target = (self.agent[0] + dx, self.agent[1] + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        if target == self.crate:
            beyond = (target[0] + dx, target[1] + dy)
            if not self._in_bounds(beyond) or beyond in self.walls:
                return                                       # the crate cannot move, so neither does the agent
            self.crate = beyond
        self.agent = target

    def render(self) -> Frame:
        grid: Grid = [[C_BG for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        def put(pos: Pos, color: int) -> None:
            x, y = pos
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[y][x] = color

        for pos in self.walls:
            put(pos, C_WALL)
        put(self.pad, C_PAD)                                 # the crate is drawn OVER the pad once delivered
        put(self.crate, C_CRATE)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return self.crate == self.pad

    def is_dead(self) -> bool:
        return False                                         # no hazard: a crate shoved into a corner is a wasted level,
        #                                                      not a death — the cost shows up in actions, which is the score

    def snapshot(self):
        return (self.agent, self.crate)

    def restore(self, snap) -> None:
        self.agent, self.crate = snap

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# Every level moves the crate and pad to DIFFERENT CELLS and pushes in a DIFFERENT DIRECTION, so a value tied to positions
# transfers nothing and only the relation can. Small boards: the generic BFS oracle searches (agent, crate).
_LEVELS: List[List[str]] = [
    # L0 — push EAST, along the top.
    [
        "########",
        "#A.B..P#",
        "#......#",
        "#......#",
        "########",
    ],
    # L1 — push WEST, along the bottom, at the other end of the board.
    [
        "########",
        "#......#",
        "#......#",
        "#P..B.A#",
        "########",
    ],
    # L2 — push SOUTH, in a column: the same relation, a direction never yet used.
    [
        "########",
        "#..A...#",
        "#..B...#",
        "#......#",
        "#..P...#",
        "########",
    ],
    # L3 — push NORTH, and the agent starts away from the crate so it must go round to the far side first.
    [
        "########",
        "#....P.#",
        "#......#",
        "#....B.#",
        "#.A....#",
        "########",
    ],
]
