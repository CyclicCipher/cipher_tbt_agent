"""Perception — the ONLY task-format-aware code: turn an ARC frame into the planner's symbolic inputs.

The agent and the planner never see a raw grid, a colour, or a `GameAction`. They see:
  - a `WorldModel` (the one-time decode of the learned model into roles: body / pushable / blocking / the
    conditional effects / the goal), built once by `build_world`;
  - a per-frame `Scene` (the body cell, the colour->cells map, the static goal / required-absent cells);
  - a `Percept` (the scene plus the lifecycle flags new_level / terminal).
Perception also owns the action vocabulary (`DELTAS` = the move geometry handed to the spatial planner;
`to_action` / `reset_action` map the planner's move indices back to `GameAction`). Keeping ALL of this here is
what lets the agent stay a thin, reusable shell (see feedback_thin_shell_agent / REORG_PLAN.md). The control-loop
planner (`perception/control.py`) consumes the `Scene` + the `WorldModel` to build the forward model it rolls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from tasks import GameAction, GameState

from .perceive import modal_background, segment
from tbt.recognize import Recognizer

Cell = Tuple[int, int]

_MOVES = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
DELTAS: List[Cell] = [a.delta for a in _MOVES]          # move-index -> (dx, dy); the spatial planner's geometry
RESET_ACTION = GameAction.RESET


def to_action(move: int) -> GameAction:
    """Map a planner move-index (0..3) back to the environment's action."""
    return _MOVES[move]


class _ActionVocab:
    """The move geometry + the move-index<->action mapping the planner and the agent driver need, so neither
    has to know a `GameAction`. Mixed into every perception."""

    deltas = DELTAS
    reset_action = RESET_ACTION

    @staticmethod
    def to_action(move: int) -> GameAction:
        return _MOVES[move]


# ── the learned roles (decoded once) ─────────────────────────────────────────────────────────────────────
@dataclass
class WorldModel:
    """The learned OBJECT roles — body / pushable / blocking + the goal (its colour + the required-absent context).
    The DYNAMICS (what a contact changes, and death) are NOT here: they live in the column's learned faculty
    (`predict_effect`), read live by the planner (Step C1). This is purely the object-role decode (Step C 4.2
    will dissolve even these into per-object behaviours)."""

    body: int
    pushable: Set[int]
    blocking: Set[int]
    goal_colors: Set[int]
    required_absent: Set[int]


def build_world(objp, goal) -> WorldModel:
    """Decode the OBJECT learners into roles (the dynamics are the column's, not decoded here)."""
    # A PUSHABLE colour is never a 'required-absent' cover-target: you push it, you don't make it vanish, and a
    # mover's trail of vacated cells would otherwise become phantom (uncoverable) cover-goals. This also undoes
    # GoalModel's cross-level pooling artefact (a blockless level's wins make the block colour look absent-in-wins).
    required_absent = set(goal.required_absent()) - set(objp.pushable)
    return WorldModel(
        body=objp.body_color, pushable=set(objp.pushable), blocking=set(objp.blocking),
        goal_colors=set(goal.goal_colors), required_absent=required_absent,
    )


# ── the per-frame scene ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class Scene:
    """One fully-observed frame: the body cell, the colour->cells map, and the (accumulated, static-over-a-
    level) goal / required-absent cells. No raw grid leaves perception.

    `movers` = the pushable objects as RECOGNISED multi-cell components `(object_id, cells)`: a pushable colour
    is segmented into connected objects and each is identified pose-invariantly (`tbt.recognize`), so the planner
    plans over whole rigid OBJECTS (push the footprint), not independent colour-cells. A single-cell pushable is
    the degenerate size-1 object (id by colour) — so the colour-keyed single-cell replica is unchanged."""

    body_pos: Optional[Cell]
    by_color: Dict[int, Set[Cell]]
    goal_cells: Set[Cell]
    req_cells: Set[Cell]
    bg: int
    movers: List[Tuple[str, FrozenSet[Cell]]] = field(default_factory=list)


@dataclass
class Percept:
    """What the agent driver sees: a scene (or None on terminal) plus lifecycle flags."""

    scene: object                         # Scene | None
    new_level: bool
    terminal: bool


# ── full observation ─────────────────────────────────────────────────────────────────────────────────────
class Perception(_ActionVocab):
    """Full-observation perception: an ARC frame -> a Scene, remembering the (static) goal / required-absent
    cells over a level (they don't move; covering only removes them from view)."""

    def __init__(self, world: WorldModel):
        self.world = world
        self.rec = Recognizer()                               # pose-invariant object identity (learned online, label-free)
        self._level = -1
        self.new_level()

    def new_level(self):
        self.goal_cells: Set[Cell] = set()
        self.req_cells: Set[Cell] = set()

    def reset(self):                                           # new game
        self._level = -1
        self.new_level()

    def read(self, frame) -> Percept:
        if frame.state == GameState.GAME_OVER:
            return Percept(None, new_level=False, terminal=True)
        new_level = frame.level != self._level
        if new_level:
            self._level = frame.level
            self.new_level()
        return Percept(self._scene(frame.grid), new_level=new_level, terminal=False)

    def _scene(self, grid) -> Scene:
        bg = modal_background(grid)
        body_pos: Optional[Cell] = None
        by_color: Dict[int, Set[Cell]] = {}
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == bg:
                    continue
                if c == self.world.body:
                    body_pos = (x, y)
                else:
                    by_color.setdefault(c, set()).add((x, y))
        for c in self.world.goal_colors:
            self.goal_cells |= by_color.get(c, set())
        for c in self.world.required_absent:
            self.req_cells |= by_color.get(c, set())
        return Scene(body_pos=body_pos, by_color=by_color, goal_cells=set(self.goal_cells),
                     req_cells=set(self.req_cells), bg=bg, movers=self._movers(grid, bg))

    def _movers(self, grid, bg) -> List[Tuple[str, FrozenSet[Cell]]]:
        """Segment the pushable colours into connected OBJECTS and identify each pose-invariantly. A multi-cell
        object gets a recognised id (so a rotated re-encounter is the SAME object — object permanence); a single
        cell has no orientation, so it passes through with a colour id (the degenerate size-1 object)."""
        movers: List[Tuple[str, FrozenSet[Cell]]] = []
        for o in segment(grid, bg):
            if o.color not in self.world.pushable:
                continue
            cells = frozenset(o.cells)
            name = self.rec.recognize(sorted(cells))[0] if o.size >= 2 else f"pt{o.color}"
            movers.append((name, cells))
        return movers


# ── Tetris: object-based perception (Step 4 — the controllable is a MULTI-CELL object) ─────────────────────
_TETRIS_MOVES = [GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION2]  # left right rotate down


@dataclass
class TetrisScene:
    """One Tetris frame as objects: the controllable PIECE (multi-cell), the settled STACK, the WELL bounds."""
    piece: frozenset                     # the controllable object's cells (multi-cell)
    stack: frozenset                     # settled cells
    well: Tuple[int, int, int]           # (left, right, floor)


class TetrisPerception:
    """Object-based perception for Tetris: read the frame into the controllable PIECE (a MULTI-CELL object), the
    settled STACK, and the WELL. The controllable being multi-cell is the Step-4 point (not a 1-cell body). Colours
    are injected here for 4.1; learning the controllable by efference (and the well/stack generically) is 4.1b/4.2.
    Carries its own action vocabulary (left/right/rotate/down) so the agent driver stays game-agnostic."""

    reset_action = RESET_ACTION

    def __init__(self, piece_color: int, stack_color: int, wall_color: int):
        self.pc, self.sc, self.wc = piece_color, stack_color, wall_color
        self._level = -1

    def to_action(self, move: int) -> GameAction:
        return _TETRIS_MOVES[move]

    def reset(self):
        self._level = -1

    def new_level(self):
        pass

    def _cells(self, grid, color) -> frozenset:
        return frozenset((x, y) for y, row in enumerate(grid) for x, v in enumerate(row) if v == color)

    def read(self, frame) -> Percept:
        if frame.state == GameState.GAME_OVER:
            return Percept(None, new_level=False, terminal=True)
        new_level = frame.level != self._level
        self._level = frame.level
        grid = frame.grid
        walls = self._cells(grid, self.wc)
        left = min(x for x, _ in walls)
        right = max(x for x, _ in walls)
        floor = max(y for _, y in walls)
        return Percept(TetrisScene(self._cells(grid, self.pc), self._cells(grid, self.sc), (left, right, floor)),
                       new_level=new_level, terminal=False)
