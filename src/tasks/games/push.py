"""Push (game_id "pu01") — push the block onto the pad. Nothing else.

The PURE-PUSH fixture: LockPath/Sokoban's block+pad mechanic with the agent-goal conjunct STRIPPED OFF, so the win is a
single RELATION — a block co-located with a pad — and nothing about where the agent ends up. It isolates, for the agent, the
two things a nav level can't exercise: (1) an object whose motion the agent does NOT directly control (the block moves only
when the agent is behind it — a discovered push dynamic), and (2) a goal that is a RELATION between two objects (block on pad),
not a place the self reaches. The levels are shaped so a GREEDY move toward the pad pushes the block the WRONG way: the only
solution is to navigate to the FAR side of the block and push it toward the pad — the go-around a rollout finds and a one-step
value cannot. Single-cell blocks (multi-cell rigid pieces are Sokoban's job); no keys/doors/hazards/goal.

ASCII tiles:  #=wall  .=floor  A=agent  B=block  P=pad
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..core import GRID_SIZE, Coordinates, Frame, Grid, GameAction
from ..game import Game

Pos = Tuple[int, int]

C_BG = 0
C_WALL = 1
C_AGENT = 2
C_BLOCK = 6
C_PAD = 7


class Push(Game):
    game_id = "pu01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
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
        if target in self.blocks:                            # push the single block one cell along the move
            beyond = (target[0] + dx, target[1] + dy)
            if not self._in_bounds(beyond) or beyond in self.walls or beyond in self.blocks:
                return                                       # blocked by a wall / bound / another block
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
        for pos in self.blocks:
            put(pos, C_BLOCK)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return self.pads.issubset(self.blocks)               # every pad covered by a block — the ONLY win requirement

    def snapshot(self):
        return (self.agent, frozenset(self.blocks))

    def restore(self, snap) -> None:
        self.agent, blocks = snap
        self.blocks = set(blocks)

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height


# L0 is a CONSTRAINED slot: the block can be pushed only ONE way -- right, onto the pad -- so cold-start exploration reliably
# solves it (the block cannot be shoved anywhere useless), yet the agent still roams a small pocket that teaches all four nav
# moves plus the right-push, and discovers the goal (block-on-pad). L1 then reuses that learned push on an OPEN board where the
# agent must GO AROUND to the block's left side -- the rollout's strength, goal-directed by the transferred relation.
_LEVELS: List[List[str]] = [
    # L0 -- the block runs in a one-way rightward channel to the pad TWO cells away, so it takes two pushes: the first
    # (non-winning) is what registers the block as a mover and teaches the right-push; the second lands it on the pad and is
    # credited as the goal. The left pocket {(1,1),(2,1),(1,2),(1,3)} lets the agent learn up/down/left/right first.
    [
        "#######",
        "#A.####",
        "#.B.P##",
        "#.#####",
        "#######",
    ],
    # L1 -- the same rightward push, but the agent starts on the FAR side: it must navigate around to the block's left and push
    # it right onto the pad. Nothing positional carries over; only the discovered relation "block on pad" does.
    [
        "#######",
        "#....A#",
        "#.BP..#",
        "#######",
    ],
]
