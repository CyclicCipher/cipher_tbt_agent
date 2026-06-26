"""Perception — the ONLY task-format-aware code: turn an ARC frame into the planner's symbolic inputs.

The agent and the planner never see a raw grid, a colour, or a `GameAction`. They see:
  - a `WorldModel` (the one-time decode of the learned model into roles: body / pushable / blocking / the
    conditional effects / the goal), built once by `build_world`;
  - a per-frame `Scene` (the body cell, the colour->cells map, the static goal / required-absent cells);
  - a `Percept` (the scene plus the lifecycle flags new_level / terminal).
Perception also owns the action vocabulary (`DELTAS` = the move geometry handed to the spatial planner;
`to_action` / `reset_action` map the planner's move indices back to `GameAction`). Keeping ALL of this here is
what lets the agent stay a thin, reusable shell (see feedback_thin_shell_agent / REORG_PLAN.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from tasks import GameAction, GameState

from .objects import modal_background

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
        """Colours some learned effect removes — removable blockers (vs permanent walls)."""
        return set().union(*self.effects.values()) if self.effects else set()


def build_world(dm, objp, goal) -> WorldModel:
    """Decode the three learners into the role vocabulary. The dynamics' rules are string-encoded
    ('color_5_gone' / 'color_5_appeared' / 'death'); this is the one place that string is read."""
    death: Set[int] = set()
    effects: Dict[int, Set[int]] = {}
    adds: Dict[int, Set[int]] = {}
    for _pred, desc, eff in dm.rules:
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
    return WorldModel(
        body=objp.body_color, pushable=set(objp.pushable), blocking=set(objp.blocking),
        death=death, effects=effects, adds=adds, harmful=harmful,
        goal_colors=set(goal.goal_colors), required_absent=set(goal.required_absent()),
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
class EgoScene:
    """One egocentric observation: a {(ox,oy): colour} window re-centred on the body (the body excluded), plus
    the background. No absolute coordinates. The ego planner accumulates these into a self-frame map."""

    window: Optional[Dict[Cell, int]]
    bg: Optional[int]


@dataclass
class Percept:
    """What the agent driver sees: a scene (or None on terminal) plus lifecycle flags."""

    scene: object                         # Scene | EgoScene | None
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


# ── egocentric (partial) observation ─────────────────────────────────────────────────────────────────────
def egocentric(grid, radius, body_color):
    """Re-centre on the body: {(ox,oy): colour} for every in-bounds cell within `radius` (the body excluded),
    INCLUDING background (so the floor is perceived). No absolute coordinates leave this function."""
    pos = next(((x, y) for y, row in enumerate(grid) for x, v in enumerate(row) if v == body_color), None)
    if pos is None:
        return None, None
    bg = modal_background(grid)
    out: Dict[Cell, int] = {}
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            x, y = pos[0] + ox, pos[1] + oy
            if (ox, oy) != (0, 0) and 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                out[(ox, oy)] = grid[y][x]
    return out, bg


class EgoPerception(_ActionVocab):
    """Egocentric partial observation: a (2r+1)^2 window re-centred on the body colour. The body's absolute
    position is unknown — only the ego planner's path-integrated self-frame belief knows it."""

    def __init__(self, world: WorldModel, radius: int = 2, memory: bool = True):
        self.world = world
        self.radius = radius
        self.memory = memory
        self._level = -1

    def new_level(self):
        pass

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
        window, bg = egocentric(frame.grid, self.radius, self.world.body)
        return Percept(EgoScene(window, bg), new_level=new_level, terminal=False)
