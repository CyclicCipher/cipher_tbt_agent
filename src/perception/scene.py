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

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from tasks import GameAction, GameState

from .perceive import modal_background

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
    """The learned roles — the one-time decode of the three learners (dynamics + objects + goal) into the
    planner's vocabulary. The ONLY place the dynamics' string-encoded effects are parsed."""

    body: int
    pushable: Set[int]
    blocking: Set[int]
    death: Set[int]
    effects: Dict[int, Set[int]]          # contact colour -> {colours it REMOVES}  (a door opens)
    adds: Dict[int, Set[int]]             # contact colour -> {colours it ADDS}      (a door closes — symmetric)
    harmful: Set[int]                     # a trigger that ADDS a blocker (kept for B1; B3 makes it emerge)
    goal_colors: Set[int]
    required_absent: Set[int]

    @property
    def doors(self) -> Set[int]:
        """Colours some learned effect removes that are NOT pushable — removable BLOCKERS (a door), vs permanent
        walls or movers. Excluding pushables stops a pushed block's vacated cell ('colour 6 gone') from being
        misread as a door: a pushable is a mover, handled by the push, never a gated door."""
        return (set().union(*self.effects.values()) - self.pushable) if self.effects else set()


def build_world(dm, objp, goal) -> WorldModel:
    """Decode the three learners into the role vocabulary. The dynamics' rules are string-encoded
    ('color_5_gone' / 'color_5_appeared' / 'death'); this is the one place that string is read."""
    death: Set[int] = set()
    effects: Dict[int, Set[int]] = {}
    adds: Dict[int, Set[int]] = {}
    for _pred, desc, eff in dm.dyn_rules:
        m = re.search(r"c0==(\d+)", desc)
        if not m:
            continue
        v = int(m.group(1))
        if eff == "death":
            death.add(v)
        elif eff.startswith("color_") and eff.endswith("_gone"):
            effects.setdefault(v, set()).add(int(eff.split("_")[1]))
        elif eff.startswith("color_") and eff.endswith("_appeared"):
            adds.setdefault(v, set()).add(int(eff.split("_")[1]))
    door_colors: Set[int] = set()
    for s in effects.values():
        door_colors |= s
    for s in adds.values():
        door_colors |= s
    harmful = {t for t, added in adds.items() if added & door_colors}
    # A PUSHABLE colour is never a 'required-absent' cover-target: you push it, you don't make it vanish, and a
    # mover's trail of vacated cells would otherwise become phantom (uncoverable) cover-goals. This also undoes
    # GoalModel's cross-level pooling artefact (a blockless level's wins make the block colour look absent-in-wins).
    required_absent = set(goal.required_absent()) - set(objp.pushable)
    return WorldModel(
        body=objp.body_color, pushable=set(objp.pushable), blocking=set(objp.blocking),
        death=death, effects=effects, adds=adds, harmful=harmful,
        goal_colors=set(goal.goal_colors), required_absent=required_absent,
    )


# ── the per-frame scene ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class Scene:
    """One fully-observed frame: the body cell, the colour->cells map, and the (accumulated, static-over-a-
    level) goal / required-absent cells. No raw grid leaves perception."""

    body_pos: Optional[Cell]
    by_color: Dict[int, Set[Cell]]
    goal_cells: Set[Cell]
    req_cells: Set[Cell]
    bg: int


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
        return Scene(body_pos=body_pos, by_color=by_color,
                     goal_cells=set(self.goal_cells), req_cells=set(self.req_cells), bg=bg)


# ── Tetris: object-based perception (Step 4 — the controllable is a MULTI-CELL object) ─────────────────────
_TETRIS_MOVES = [GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION2]  # left right rotate down


def shape_of(cells) -> frozenset:
    """A cell-set's translation-invariant shape (cells relative to the bbox corner) — the object's pose-shape, the
    key on which the learned rotation operator is indexed (so a piece is recognised regardless of where it is)."""
    cells = frozenset(cells)
    if not cells:
        return frozenset()
    mnx, mny = min(x for x, _ in cells), min(y for _, y in cells)
    return frozenset((x - mnx, y - mny) for x, y in cells)


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


class TetrisLearner:
    """Learn the ROTATION operator from observation — the controllable object's orientation cycle (shape → next
    shape), i.e. the learned 'rotate' (Step 4 / increment 2b). Driven by watching rotate transitions; the planner
    reads `table` live (shared reference). Translation-invariant (`shape_of`), so it matches the game's rotation
    placed at the bbox-min anchor. (gravity (0,1) and the controllable colour are injected for 4.1.)"""

    def __init__(self, piece_color: int):
        self.pc = piece_color
        self.table: Dict[frozenset, frozenset] = {}

    def _shape(self, grid) -> frozenset:
        return shape_of((x, y) for y, row in enumerate(grid) for x, v in enumerate(row) if v == self.pc)

    def observe(self, prev_frame, action, frame):
        if action == GameAction.ACTION5:
            a, b = self._shape(prev_frame.grid), self._shape(frame.grid)
            if a and b and a != b:
                self.table[a] = b            # this shape rotates to that shape (the learned operator)

    def refresh(self):
        pass

    def new_level(self):
        pass

    def reset(self):
        pass                                 # keep the learned table across episodes
