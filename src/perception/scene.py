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
from tbt.column import CorticalColumn

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


class StateEncoder(_ActionVocab):
    """The ONE perception→planner bridge, for ANY game: a frame → the factored EGOCENTRIC state the planner plans
    over, + the traversability gates + F's emergent subgoal factors + the pragmatic goal-progress. The single
    place colours/roles touch planning; everything handed to the planner is OPAQUE (a coord tuple, a scalar in
    [0,1], (cell,bit) gates, factor cells) so the planner is domain-general — swap perception and it is unchanged.

    State (per frame):
      - the body's ABSOLUTE position, so POSITIONAL effects (a key opening a door at a fixed cell) are literals;
      - each movable object RELATIVE to the body, so RELATIONAL effects (a push) are literals;
      - a presence bit per learned door colour (the observable open/closed state).
    Multiple same-colour movers (Sokoban) are kept distinct by a STATEFUL common-fate tracker — each slot follows
    the nearest current mover cell across frames (one object moves per step), so G sees consistent per-object
    coordinates whether the game has 0, 1, or N of them. (1-cell movers for the replica; multi-cell objects ⇒
    segment + track the component — a later refinement.)"""

    def __init__(self, world: WorldModel):
        self.world = world
        self.doors = sorted(world.doors)
        self.reset()

    def reset(self):                                            # new episode/level
        self._slots = None                                     # tracked mover cells, stable order; set on 1st encode
        self._mapkey = None                                    # the spatial column is rebuilt when the walkable changes

    def column(self, scene):
        """The SR-frame column over the current walkable cells (the map). Built once per distinct walkable set
        from the cell adjacency under the action geometry; the planner reads place codes from it. Perception owns
        the spatial structure, so `tbt/value.py` never sees a cell or a delta — it gets the column."""
        walk = self.walkable(scene)
        key = frozenset(walk)
        if self._mapkey != key:
            cells = sorted(walk)
            cid = {c: i for i, c in enumerate(cells)}
            col = CorticalColumn(n_entities=max(1, len(cells)))
            for c in cells:
                for j, (dx, dy) in enumerate(self.deltas):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in cid:
                        col.observe(cid[c], j, cid[nb])
            col.consolidate()
            self._col, self._cid, self._mapkey = col, cid, key
        return self._col, self._cid

    @property
    def n_movers(self):
        return len(self._slots) if self._slots is not None else 0

    @property
    def n_doors(self):
        return len(self.doors)

    def _movers(self, scene):
        return sorted(c for col in sorted(self.world.pushable) for c in scene.by_color.get(col, set()))

    def encode(self, scene):
        """The factored state tuple (or None if the body is not visible)."""
        body = scene.body_pos
        if body is None:
            return None
        cur = self._movers(scene)
        if self._slots is None or len(cur) != len(self._slots):
            self._slots = list(cur)                            # (re)seed the slots
        else:                                                  # common-fate: GLOBAL closest-first assignment
            pairs = sorted((abs(self._slots[i][0] - c[0]) + abs(self._slots[i][1] - c[1]), i, c)
                           for i in range(len(self._slots)) for c in cur)
            new, slots_used, cells_used = list(self._slots), set(), set()
            for _, i, c in pairs:                              # claim the smallest displacements first (a stayed
                if i not in slots_used and c not in cells_used:  # block, dist 0, is matched before a moved one)
                    new[i] = c; slots_used.add(i); cells_used.add(c)
            self._slots = new
        co = [body[0], body[1]]
        for m in self._slots:
            co += [m[0] - body[0], m[1] - body[1]]
        for d in self.doors:
            co.append(1 if scene.by_color.get(d) else 0)
        return tuple(co)

    def gates(self, scene):
        """Traversability: {cell: bit-index} for each currently-shut door cell — the planner blocks a cell while
        the coord-bit at that index reads closed (1). General (cell, bit); the planner never knows it's a door."""
        gates, base = {}, 2 + 2 * self.n_movers
        for j, d in enumerate(self.doors):
            for cell in scene.by_color.get(d, set()):
                gates[cell] = base + j
        return gates

    def block_abs(self, coords):
        """The movers' ABSOLUTE cells recovered from a (possibly imagined) coord tuple — for pragmatic/factor
        checks and the value latent. Domain-general arithmetic on the opaque tuple."""
        ax, ay = coords[0], coords[1]
        return [(ax + coords[2 + 2 * i], ay + coords[2 + 2 * i + 1]) for i in range(self.n_movers)]

    def walkable(self, scene):
        """The map's walkable cells — the bounding box of content minus PERMANENT walls (blockers no learned
        effect removes; a removable door is walkable, gated separately). The planner builds its SR-frame column
        over these; perception owns 'which cells are walls'."""
        by = scene.by_color
        non_bg = {p for cells in by.values() for p in cells}
        if scene.body_pos is not None:
            non_bg.add(scene.body_pos)
        if not non_bg:
            return set()
        walls = {c for c in (self.world.blocking | self.world.death) if c not in self.world.doors}
        obstacles = {p for c in walls for p in by.get(c, set())}
        xs = [x for x, _ in non_bg]; ys = [y for _, y in non_bg]
        return {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
                if (x, y) not in obstacles}

    def factors(self, scene):
        """F's EMERGENT subgoal factors as (target_cell, kind): each required-absent cell must be COVERED, each
        goal cell REACHED. These are the win-condition's conjunctive terms read off F — not enumerated subgoal
        types — so the planner sequences them generically."""
        return ([(c, "cover") for c in sorted(scene.req_cells)]
                + [(c, "reach") for c in sorted(scene.goal_cells)])

    def satisfied(self, coords, factor):
        """Is `factor` met in this (possibly imagined) coordinate state? Opaque to the planner (it just gets a
        bool); here a 'cover' cell holds a mover, a 'reach' cell holds the agent."""
        cell, kind = factor
        if kind == "reach":
            return (coords[0], coords[1]) == cell
        return cell in set(self.block_abs(coords))

    def proximity(self, coords, factor):
        """A CONTINUOUS [0,1] progress toward satisfying `factor` (1 = satisfied) — the navigation gradient the
        task column routes toward: for 'reach', the agent nearing the cell; for 'cover', a mover nearing it.
        `1/(1+distance)`. Perception owns the metric, so the planner gets a smooth, opaque pull signal."""
        cell, kind = factor
        if kind == "reach":
            d = abs(coords[0] - cell[0]) + abs(coords[1] - cell[1])
        else:
            d = min((abs(b[0] - cell[0]) + abs(b[1] - cell[1]) for b in self.block_abs(coords)), default=99)
        return 1.0 / (1.0 + d)

    def route_proximity(self, coords, factor, occupied=frozenset()):
        """The agent's closeness to the OBJECT it must reach to satisfy `factor` — a non-reward EXPLORATION signal
        (so no local optimum, unlike `proximity` as a reward): for 'cover', the nearest mover NOT already on a done
        cell (the block to push); for 'reach', the goal. Heading exploration up this gradient brings the agent to
        the right object; the value-search then covers from there (the L0 case). 1/(1+agent→object)."""
        cell, kind = factor
        ax, ay = coords[0], coords[1]
        if kind == "reach":
            d = abs(ax - cell[0]) + abs(ay - cell[1])
        else:                                                  # cost = navigate to a free mover + push it to the
            free = [b for b in self.block_abs(coords) if b not in occupied]   # cell (both terms, so the bias does
            d = min(((abs(ax - b[0]) + abs(ay - b[1])) +                       # not just park the agent at a block)
                     (abs(b[0] - cell[0]) + abs(b[1] - cell[1])) for b in free), default=99)
        return 1.0 / (1.0 + d)

    def focus_mover(self, coords, factor, occupied=frozenset()):
        """The mover SLOT the value-search should FOCUS on for `factor` — the nearest free mover to push to a
        'cover' cell; None for 'reach' (no object). Lets the value FACTOR (agent × ONE object) instead of binding
        the JOINT of all movers, which doesn't converge (the 2^K problem)."""
        cell, kind = factor
        if kind == "reach":
            return None
        ax, ay = coords[0], coords[1]
        cand = [(abs(b[0] - cell[0]) + abs(b[1] - cell[1]) + abs(ax - b[0]) + abs(ay - b[1]), i)
                for i, b in enumerate(self.block_abs(coords)) if b not in occupied]
        return min(cand)[1] if cand else None


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
