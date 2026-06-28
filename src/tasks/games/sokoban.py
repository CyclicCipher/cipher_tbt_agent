"""Sokoban (game_id "sk01") — push N blocks onto N pads, then reach the goal.

LockPath's block+pad mechanic scaled up: several blocks must each be pushed onto a pad before the goal counts.
It exists to test the agent's cover-pad loop AT SCALE — the factored navigation should push one block at a time
(agent × that block), never the joint state of all blocks. No keys/doors/hazards.

A block is a RIGID connected component of `B` cells (4-connectivity), pushed as ONE unit — a single `B` is the
degenerate size-1 case (so the original single-cell levels are unchanged). The multi-cell levels
(`MULTICELL_LEVELS`) are the real-ARC bench: the controllable obstacles are multi-cell OBJECTS the agent's
perception must segment + recognise, and the planner must push by their whole footprint (not cell-by-cell).

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

_NBR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _components(cells: Set[Pos]) -> List[Set[Pos]]:
    """Partition `cells` into 4-connected components — each a rigid block. (A single cell is its own component,
    so single-`B` levels stay byte-identical to the original Sokoban.)"""
    cells = set(cells)
    seen: Set[Pos] = set()
    comps: List[Set[Pos]] = []
    for s in cells:
        if s in seen:
            continue
        comp, stack = set(), [s]
        seen.add(s)
        while stack:
            x, y = stack.pop()
            comp.add((x, y))
            for dx, dy in _NBR4:
                n = (x + dx, y + dy)
                if n in cells and n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


class Sokoban(Game):
    game_id = "sk01"

    def __init__(self, levels: Optional[List[List[str]]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.width = self.height = 0
        self.agent: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self.walls: Set[Pos] = set()
        self.pieces: List[Set[Pos]] = []                 # each piece = a rigid connected component of B cells
        self.pads: Set[Pos] = set()

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    def load_level(self, level: int) -> None:
        self._level = level
        self.walls, self.pads = set(), set()
        block_cells: Set[Pos] = set()
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
                    block_cells.add(pos)
                elif ch == "P":
                    self.pads.add(pos)
        self.pieces = _components(block_cells)            # adjacent B's = one rigid piece (size 1 = a single block)

    def block_cells(self) -> Set[Pos]:
        return set().union(*self.pieces) if self.pieces else set()

    def _piece_at(self, pos: Pos) -> int:
        for i, p in enumerate(self.pieces):
            if pos in p:
                return i
        return -1

    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if not action.is_movement:
            return
        dx, dy = action.delta
        ax, ay = self.agent
        target = (ax + dx, ay + dy)
        if not self._in_bounds(target) or target in self.walls:
            return
        i = self._piece_at(target)
        if i >= 0:                                        # push the WHOLE rigid piece by the move delta
            piece = self.pieces[i]
            shifted = {(x + dx, y + dy) for x, y in piece}
            newcells = shifted - piece                    # the cells the piece advances INTO
            others = set().union(*(p for j, p in enumerate(self.pieces) if j != i)) if len(self.pieces) > 1 else set()
            if any(not self._in_bounds(c) or c in self.walls or c in others for c in newcells):
                return                                    # blocked by a wall / bound / another piece
            self.pieces[i] = shifted
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
        for pos in self.block_cells():
            put(pos, C_BLOCK)
        put(self.agent, C_AGENT)
        return [grid]

    def level_complete(self) -> bool:
        return self.agent == self.goal and self.pads.issubset(self.block_cells())

    def snapshot(self):
        return (self.agent, tuple(frozenset(p) for p in self.pieces))

    def restore(self, snap) -> None:
        self.agent, pieces = snap
        self.pieces = [set(p) for p in pieces]

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
    # L1 — the two B's are 4-adjacent, so they are ONE rigid 2-cell block pushed onto the (adjacent) pad pair.
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


# The real-ARC bench: the controllable obstacles are MULTI-CELL rigid objects (adjacent B's), so perception must
# segment + recognise them and the planner must push by the whole footprint. M1 has two SAME-COLOUR pieces of
# DIFFERENT shapes — colour alone cannot tell them apart, so the scene is disambiguated by RECOGNISED object, the
# capability the single-cell colour-keyed replica never exercised.
MULTICELL_LEVELS: List[List[str]] = [
    # M0 — one horizontal DOMINO (2 cells) pushed right onto a pad, then the goal.
    [
        "##########",
        "#A.......#",
        "#.BB..P..#",
        "#........#",
        "#.......G#",
        "##########",
    ],
    # M1 — a DOMINO and an L-tromino (same colour, different shape), each onto its pad, then the goal.
    [
        "############",
        "#A.........#",
        "#.BB....P..#",
        "#..........#",
        "#.B........#",
        "#.BB...P...#",
        "#..........#",
        "#.........G#",
        "############",
    ],
]
