"""Warehouse (game_id "wh01") — every crate onto a pad. The win is a property of a SET, which a PAIR cannot express.

WHY THIS GAME EXISTS. `Crates` measured that a graded relational value is redundant wherever `goal_mem`'s `(mover, landmark)`
PAIR goal already names the win: it solved 4/4 at oracle with the relational term ablated, because "crate on pad" IS a pair
and a pair keyed on FEATURES transfers across levels for free. So the open question is what a pair CANNOT say. This game is
that case, and it is chosen to break the pair representation in two independent ways at once:

  * **The win is a SET property.** Not "a crate is on a pad" but "EVERY pad is covered". A pair naming one relation is true
    the moment the first crate lands, so it cannot distinguish half-done from done — and an agent that believes it has won
    stops.
  * **The instances are INDISTINGUISHABLE.** Every crate is one colour and every pad another, so `(6, 7)` names a COLOUR
    PAIR, not a crate and a pad. `Agent._positions` only reports unambiguously single-instance objects, so with two crates
    of one colour it reports neither — the pair goal has nothing to plan toward even in principle. The ASSIGNMENT (which
    crate to which pad) is free, which is the part no pair can carry.

That is the discrimination `test_crates` said was needed, built as adversarially as the fixture before it: if the relational
machinery contributes nothing HERE either, the fixture will say so plainly rather than hide it.

Core-Knowledge only (objectness, geometry, basic physics, agentness): walking into a crate shifts it one cell when the far
side is clear — including when the far side is another crate, which refuses. Nothing is told to the agent.

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


class Warehouse(Game):
    game_id = "wh01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.crates: Set[Pos] = set()
        self.pads: Set[Pos] = set()
        self.walls: Set[Pos] = set()

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls, self.crates, self.pads = set(), set(), set()
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
                    self.crates.add(pos)
                elif ch == "P":
                    self.pads.add(pos)

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        """Walk, or push ONE crate. A crate will not shift into a wall, off the board, or into another crate — so crates
        block each other, which is what makes the ORDER of the deliveries matter."""
        if not action.is_movement:
            return
        dx, dy = action.delta
        target = (self.agent[0] + dx, self.agent[1] + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        if target in self.crates:
            beyond = (target[0] + dx, target[1] + dy)
            if not self._in_bounds(beyond) or beyond in self.walls or beyond in self.crates:
                return
            self.crates.discard(target)
            self.crates.add(beyond)
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
        for pos in self.crates:                              # a delivered crate is drawn over its pad
            put(pos, C_CRATE)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        """EVERY pad covered — the set property. One crate on one pad is not a win, which is exactly what a pair goal
        cannot represent."""
        return self.pads <= self.crates

    def is_dead(self) -> bool:
        return False

    def snapshot(self):
        return (self.agent, frozenset(self.crates))

    def restore(self, snap) -> None:
        self.agent, crates = snap
        self.crates = set(crates)

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# Two crates and two pads per level, all alike; the crates and pads move to different cells and the push runs in a different
# direction each level, so nothing POSITIONAL can transfer and only the relation can. Boards stay small: the generic BFS
# oracle searches (agent, frozenset(crates)).
_LEVELS: List[List[str]] = [
    # L0 — both pushes EAST.
    [
        "########",
        "#A.B.P.#",
        "#......#",
        "#..B.P.#",
        "########",
    ],
    # L1 — both pushes WEST, at the other end of the board.
    [
        "########",
        "#.P.B.A#",
        "#......#",
        "#.P.B..#",
        "########",
    ],
    # L2 — both pushes SOUTH: the same set-goal, a direction not yet used.
    [
        "########",
        "#.A....#",
        "#.B.B..#",
        "#......#",
        "#.P.P..#",
        "########",
    ],
]
