"""Tetris (game_id "tt01") — bounded FALLING Tetris: the multi-cell object-model stress test.

Unlike the push games, this stacks MULTI-CELL objects (the 7 tetrominoes — distinct shapes) under THREE mechanics
the static replica never had, each an R11 gap: (1) the controllable thing is a multi-cell PIECE, not a 1-cell
body; (2) ROTATION — the piece's pose includes ORIENTATION (Monty's pose = location + orientation); (3) AUTONOMOUS
GRAVITY — the piece falls one row every action, on its own, so the agent plans against a moving world. Clearing a
full row scores. It is deliberately "homework harder than the exam": the metric is how EFFICIENTLY a model learns
to reach a target score (lines cleared), and the current body-centric agent should score ~0 until it grows the
object/rotation/dynamics faculties.

Bounded for fast tests: a small well + a low target-lines win + top-out = GAME_OVER.

Actions:  ACTION3 left · ACTION4 right · ACTION5 rotate (CW) · ACTION2 soft-drop (an extra row this tick).
Colours:  0 bg · 1 wall (the well) · 2 the active piece · 3 the settled stack.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from ..core import GRID_SIZE, Coordinates, Frame, Grid, GameAction
from ..game import Game

Cell = Tuple[int, int]

C_BG = 0
C_WALL = 1
C_PIECE = 2
C_STACK = 3

# the 7 tetrominoes, rotation-0 cells (dx, dy) with dy DOWN; other rotations are derived (CW)
_BASE: Dict[str, List[Cell]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}


def _normalize(cells) -> frozenset:
    mnx = min(x for x, _ in cells)
    mny = min(y for _, y in cells)
    return frozenset((x - mnx, y - mny) for x, y in cells)


def _rotations(base) -> List[frozenset]:
    """The 4 CW rotations of a shape ((x, y) -> (y, -x), renormalized), duplicates kept (harmless)."""
    rots = [_normalize(base)]
    for _ in range(3):
        rots.append(_normalize([(y, -x) for x, y in rots[-1]]))
    return rots


_SHAPES: Dict[str, List[frozenset]] = {k: _rotations(v) for k, v in _BASE.items()}


def _bag(seed: int, n: int) -> List[str]:
    """A seeded 7-bag sequence (each bag is a shuffled permutation of the 7 pieces) — fair + deterministic."""
    rng = random.Random(seed)
    kinds = list(_BASE)
    seq: List[str] = []
    while len(seq) < n:
        b = kinds[:]
        rng.shuffle(b)
        seq += b
    return seq[:n]


class Tetris(Game):
    game_id = "tt01"

    def __init__(self, levels: Optional[List[dict]] = None) -> None:
        self._levels = levels if levels is not None else _LEVELS
        self._level = 0
        self.W = self.H = 0
        self.target = 1
        self.stack: Set[Cell] = set()                          # settled cells (interior coords: x in 1..W, y in 0..H-1)
        self.seq: List[str] = []
        self.idx = 0
        self.kind = "O"
        self.rot = 0
        self.ax = self.ay = 0                                  # piece anchor (top-left of its shape box)
        self.lines = 0
        self._dead = False

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def available_actions(self) -> List[GameAction]:
        return [GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]

    # ----- piece geometry -------------------------------------------------------------------------------
    def _cells(self, kind=None, rot=None, ax=None, ay=None) -> frozenset:
        kind = self.kind if kind is None else kind
        rot = self.rot if rot is None else rot
        ax = self.ax if ax is None else ax
        ay = self.ay if ay is None else ay
        return frozenset((ax + dx, ay + dy) for dx, dy in _SHAPES[kind][rot % 4])

    def _valid(self, cells) -> bool:
        return all(1 <= x <= self.W and 0 <= y < self.H and (x, y) not in self.stack for x, y in cells)

    def _spawn(self) -> None:
        self.kind = self.seq[self.idx % len(self.seq)]
        self.idx += 1
        self.rot = 0
        shape = _SHAPES[self.kind][0]
        w = max(x for x, _ in shape) + 1
        self.ax = 1 + (self.W - w) // 2                        # centred near the top
        self.ay = 0
        if not self._valid(self._cells()):                    # cannot place a new piece -> topped out
            self._dead = True

    def load_level(self, level: int) -> None:
        cfg = self._levels[level]
        self._level = level
        self.W, self.H, self.target = cfg["W"], cfg["H"], cfg.get("target", 1)
        self.stack = {tuple(c) for c in cfg.get("prefill", ())}
        self.seq = _bag(cfg.get("seed", level), cfg.get("pieces", 40))
        self.idx = 0
        self.lines = 0
        self._dead = False
        self._spawn()

    # ----- one tick: the agent's move, then gravity -----------------------------------------------------
    def apply(self, action: GameAction, coordinates: Optional[Coordinates]) -> None:
        if self._dead:
            return
        if action == GameAction.ACTION3:                      # left
            self._shift(-1, 0)
        elif action == GameAction.ACTION4:                    # right
            self._shift(1, 0)
        elif action == GameAction.ACTION5:                    # rotate CW
            cand = self._cells(rot=self.rot + 1)
            if self._valid(cand):
                self.rot = (self.rot + 1) % 4
        elif action == GameAction.ACTION2:                    # soft drop (an extra row before gravity)
            self._shift(0, 1)
        if not self._shift(0, 1):                             # GRAVITY: fall one row, or lock if it cannot
            self._lock()

    def _shift(self, dx: int, dy: int) -> bool:
        cand = self._cells(ax=self.ax + dx, ay=self.ay + dy)
        if self._valid(cand):
            self.ax += dx
            self.ay += dy
            return True
        return False

    def _lock(self) -> None:
        self.stack |= self._cells()                           # settle the piece
        full = [y for y in range(self.H) if all((x, y) in self.stack for x in range(1, self.W + 1))]
        if full:
            fset = set(full)
            kept = {(x, y) for (x, y) in self.stack if y not in fset}
            shifted: Set[Cell] = set()
            for (x, y) in kept:                               # rows above each cleared line drop down by the count below
                drop = sum(1 for fy in full if fy > y)
                shifted.add((x, y + drop))
            self.stack = shifted
            self.lines += len(full)
        self._spawn()

    def level_complete(self) -> bool:
        return self.lines >= self.target

    def is_dead(self) -> bool:
        return self._dead

    # ----- render / snapshot ----------------------------------------------------------------------------
    def render(self) -> Frame:
        grid: Grid = [[C_BG for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        def put(x: int, y: int, c: int) -> None:
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                grid[y][x] = c

        for y in range(self.H + 1):                           # left/right walls + the floor (top is open)
            put(0, y, C_WALL)
            put(self.W + 1, y, C_WALL)
        for x in range(self.W + 2):
            put(x, self.H, C_WALL)
        for (x, y) in self.stack:
            put(x, y, C_STACK)
        if not self._dead:
            for (x, y) in self._cells():
                put(x, y, C_PIECE)
        return [grid]

    def snapshot(self):
        return (frozenset(self.stack), self.kind, self.rot, self.ax, self.ay, self.idx, self.lines, self._dead)

    def restore(self, snap) -> None:
        stack, self.kind, self.rot, self.ax, self.ay, self.idx, self.lines, self._dead = snap
        self.stack = set(stack)


# Bounded levels: a small well so a few pieces can complete a row. Each level pre-fills the bottom row except a
# narrow GAP so a single well-placed (and, for L1/L2, rotated) piece clears a line — bounded + clearly solvable,
# while still exercising the falling piece + rotation + gravity. (An empty-board variant is the harder follow-on.)
def _row_except(W: int, y: int, gap) -> Set[Cell]:
    return {(x, y) for x in range(1, W + 1) if x not in gap}


_LEVELS: List[dict] = [
    # L0 — a 2-wide gap at the right; drop an O straight down into it.
    dict(W=6, H=10, target=1, seed=10, pieces=40, prefill=_row_except(6, 9, {5, 6})),
    # L1 — a 1-wide gap; an I rotated vertical drops in to clear the row.
    dict(W=6, H=10, target=1, seed=21, pieces=40, prefill=_row_except(6, 9, {3})),
    # L2 — clear TWO lines: the bottom two rows each miss the same 2-wide column, fillable by stacking pieces.
    dict(W=6, H=10, target=2, seed=32, pieces=40,
         prefill=_row_except(6, 9, {3, 4}) | _row_except(6, 8, {3, 4})),
]
