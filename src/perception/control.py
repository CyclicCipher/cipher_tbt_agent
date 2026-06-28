"""The control-loop planner — build the MULTI-COLUMN forward model from a scene, then let the Neocortex roll it.

This is the perception-side ADAPTER satisfying the agent's `planner` contract (act / reset / new_level /
on_death). It does the one thing perception must do for Phase-2 planning: assemble the world model's forward
prediction by COMPOSING the active columns over the perceived scene, and hand that composition — a single opaque
`step(state, action) -> (next_state, signed_reward, done)` callable — to the domain-general rollout achiever in
`tbt/neocortex.py`. The PLANNING lives in the model; this file is the (role-aware) glue that builds the model's
forward function and the spine it plans over.

The forward model is the multi-column world model (the spine STAYS — never stripped; it is what makes this scale,
factor reference frames, and compose):
  - the SPATIAL MAP column (an SR-frame `CorticalColumn` over the navigable cells) — the agent-move prediction and
    the absolute-frame "is this a node" vote;
  - the FOCUS object-column — the egocentric push (the moved object advances in the heading) VOTED against the
    absolute map (the egocentric ⊗ absolute lateral vote); the basal ganglia GATES which object is the focus, so
    adding an object adds a column to gate among (additive), never a dimension of the rolled state (no 2^K);
  - the DYNAMICS — a contact's effect (a blocker removed / added), the column's learned conditional faculty,
    decoded here into the rolled door-state. (Role-based for Step B; Step C makes it the column's learned model.)
  - SIGNED value — an aversive contact is a terminal −1 (avoided by VALUE, not walled off), success a terminal +1.

The thalamus routes the active sub-goal's goal-state into the spatial frame (top-down CMP); the basal ganglia
gates the focus. Reach / cover / collect / the affordance / hazard-avoidance are NOT branches here — they EMERGE
from rolling this one model under signed value. The decode of which colour is a mover / a blocker / aversive / a
goal stays in perception (the WorldModel `self.world`, read LIVE so a cold-start's freshly-learned roles apply).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set, Tuple

from tbt.column import CorticalColumn
from tbt.neocortex import Neocortex

Cell = Tuple[int, int]
_EMPTY: FrozenSet[int] = frozenset()


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class _Layout:
    """The scene decoded into the forward model's geometry (the role-aware part, kept out of `tbt/`)."""
    frame: FrozenSet[Cell]                 # the spatial map column's reference frame (its navigable nodes)
    deaths: FrozenSet[Cell]                # aversive cells — steppable, terminal −1 (signed value, not walls)
    movers: FrozenSet[Cell]               # the relational objects (one is the focus; the rest are obstacles)
    door_at: Dict[Cell, int]               # a removable-blocker cell → its colour (shut until that colour is removed)
    opener: Dict[Cell, Set[int]]           # a contact cell → the colours it REMOVES (a blocker opens)
    closer: Dict[Cell, Set[int]]           # a contact cell → the colours it ADDS (a blocker closes — symmetric)
    flip: Dict[Cell, Set[int]]             # a contact cell → colours it BOTH adds and removes = a TOGGLE (flip state)
    removed0: FrozenSet[int]               # colours already removed at the start (a blocker seen open)


class NeocortexPlanner:
    """A `planner` for `tbt.agent.Agent`: scene → move index, planning by rolling the multi-column forward model."""

    def __init__(self, world, gamma: float = 0.95, seed: int = 0):
        self.world = world                                     # the learned roles (shared by reference; read live)
        self.neo = Neocortex(gamma=gamma, seed=seed)
        from .scene import DELTAS                              # the move geometry (perception owns it)
        self.deltas = DELTAS
        self.seed = seed
        self.reset()

    # ---- lifecycle ---------------------------------------------------------------------------------------
    def reset(self):                                           # new game
        self.rng = random.Random(self.seed)
        self.neo.reset()
        self._col = None
        self._mapkey = None
        self._bound = None
        self._focus: Optional[Cell] = None

    def new_level(self):                                       # the layout changed → rebuild map, rebind, drop focus
        self.neo.reset()
        self._mapkey = None
        self._bound = None
        self._focus = None

    def on_death(self):                                        # GAME_OVER: the env reloads THIS level → fresh focus
        self._bound = None
        self._focus = None

    # ---- the spatial map column (SR-frame over the navigable cells) --------------------------------------
    def _column(self, frame: FrozenSet[Cell]) -> CorticalColumn:
        """The spatial column over the current frame — the agent-move/where map. Cached per distinct frame (the
        eigh is the cost). The column's symbols are integers (its L4 codebook is integer-indexed), so we keep the
        cell↔symbol maps for the thalamus to read place codes by cell."""
        if self._mapkey != frame:
            cells = sorted(frame)
            cid = {c: i for i, c in enumerate(cells)}
            col = CorticalColumn(n_entities=max(1, len(cells)))
            for c in cells:
                for j, (dx, dy) in enumerate(self.deltas):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in cid:
                        col.observe(cid[c], j, cid[nb])
            col.consolidate()
            self._col, self._cid, self._mapkey = col, cid, frame
            self._sym2cell = {i: c for c, i in cid.items()}
        return self._col

    # ---- decode the scene into the forward model's geometry ----------------------------------------------
    def _layout(self, scene) -> _Layout:
        w, by = self.world, scene.by_color
        non_bg = {p for cells in by.values() for p in cells} | {scene.body_pos}
        xs = [x for x, _ in non_bg]
        ys = [y for _, y in non_bg]
        bbox = {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)}
        removable = ((set().union(*w.effects.values()) if w.effects else set()) |
                     (set().union(*w.adds.values()) if w.adds else set())) - w.pushable
        walls = {p for c in w.blocking if c not in removable for p in by.get(c, ())}
        opener = {p: set(rem) for tc, rem in w.effects.items() for p in by.get(tc, ())}
        closer = {p: set(add) for tc, add in w.adds.items() for p in by.get(tc, ())}
        # a contact that the learned dynamics says BOTH removes and adds a colour TOGGLES it (the switch flips the
        # door open↔closed) — honour that as a flip, not a one-way effect, so the planner can reason through it
        flip = {p: opener[p] & closer[p] for p in opener.keys() & closer.keys() if opener[p] & closer[p]}
        return _Layout(
            frame=frozenset(bbox - walls),                     # deaths + removable blockers stay IN the frame
            deaths=frozenset(p for c in w.death for p in by.get(c, ())),
            movers=frozenset(p for c in w.pushable for p in by.get(c, ())),
            door_at={p: c for c in removable for p in by.get(c, ())},
            opener=opener, closer=closer, flip=flip,
            removed0=frozenset(c for c in removable if not by.get(c)),
        )

    # ---- compose the active columns into one forward-model callable --------------------------------------
    def _forward_model(self, lay: _Layout, target: Cell, focus: Optional[Cell]):
        """`step(state, action) -> (next_state, signed_reward, done)` over the factored state
        `(agent, focus-object, removed-blockers)`. The composition: the spatial map column predicts the agent move
        and votes free; if the agent enters the focus object's cell, the focus column advances it (egocentric) iff
        the map votes the landing free (the lateral vote); the dynamics add/remove blockers; an aversive cell is a
        terminal −1 and the goal-state a terminal +1 (signed value drives avoidance + pursuit, no role branches)."""
        others = lay.movers - ({focus} if focus is not None else set())

        def free(t: Cell, removed: FrozenSet[int]) -> bool:
            return (t in lay.frame and t not in others
                    and not (t in lay.door_at and lay.door_at[t] not in removed))

        def step(state, a):
            ag, mv, removed = state
            dx, dy = self.deltas[a]
            t = (ag[0] + dx, ag[1] + dy)
            if not free(t, removed):                           # spatial map column: the agent-move vote
                return state, 0.0, False
            nmv = mv
            if mv is not None and t == mv:                     # focus object-column: the egocentric push…
                b = (t[0] + dx, t[1] + dy)
                if not free(b, removed):                        # …voted against the absolute map (lateral)
                    return state, 0.0, False
                nmv = b
            o, c, fl = lay.opener.get(t, _EMPTY), lay.closer.get(t, _EMPTY), lay.flip.get(t, _EMPTY)
            nrem = (((set(removed) | (o - fl)) - (c - fl)) ^ fl)   # dynamics: one-way effects, plus a toggle flip
            if t in lay.deaths:                                # signed value: aversive ⇒ terminal −1, never a wall
                return (t, nmv, frozenset(nrem)), -1.0, True
            done = (nmv == target) if focus is not None else (t == target)
            return (t, nmv, frozenset(nrem)), (1.0 if done else 0.0), done

        return step

    # ---- top-down CMP: bind the sub-goals, route the active goal-state -----------------------------------
    def _rebind(self, factors, col):
        key = (id(col), tuple(factors))
        if key != self._bound:
            subgoals = [(i, self._cid[cell]) for i, cell in enumerate(factors) if cell in self._cid]
            self._task_col, self._R, self._inv = self.neo.route_goals(subgoals, col)
            self._fidx = {cell: i for i, cell in enumerate(factors)}
            self._bound = key

    def _route(self, col, cell: Cell) -> Cell:
        i = self._fidx.get(cell)
        if i is None or cell not in self._cid:
            return cell
        node = self.neo.goal_node(self._R, self._task_col, col, self._inv, i, self._cid[cell])
        return self._sym2cell.get(node, cell)

    # ---- the planning step -------------------------------------------------------------------------------
    def act(self, scene, explore: float = 0.0):
        agent = scene.body_pos
        if agent is None:                                      # body not visible → can't plan; wander
            return self.rng.randrange(len(self.deltas))
        if explore and self.rng.random() < explore:           # keep discovery alive (the cold-start's epsilon)
            return self.rng.randrange(len(self.deltas))
        lay = self._layout(scene)
        col = self._column(lay.frame)
        factors = sorted(scene.req_cells) + sorted(scene.goal_cells)   # stable per level; the task column's sub-goals
        self._rebind(factors, col)
        n = len(self.deltas)
        # the active sub-goal = the first still-unsatisfied required cell, else the goal (terminal last)
        req_visible = {p for c in self.world.required_absent for p in scene.by_color.get(c, ())}
        pending = [c for c in sorted(scene.req_cells) if c in req_visible]
        if pending:
            C = pending[0]
            target = self._route(col, C)
            free_movers = [m for m in lay.movers if m not in scene.req_cells]
            if free_movers:                                    # COVER: the BG gates a focus object-column, push it
                if self._focus not in free_movers:
                    self._focus = self.neo.gate_focus(free_movers, [1.0 / (1.0 + _manh(m, C)) for m in free_movers])
                step = self._forward_model(lay, target, self._focus)
                return self.neo.achieve(step, (agent, self._focus, lay.removed0), n)
            step = self._forward_model(lay, target, None)      # COLLECT: no mover → the agent reaches it (emergent)
            return self.neo.achieve(step, (agent, None, lay.removed0), n)
        goals = sorted(scene.goal_cells)
        if not goals:
            return self.rng.randrange(n)
        target = self._route(col, goals[0])
        step = self._forward_model(lay, target, None)
        return self.neo.achieve(step, (agent, None, lay.removed0), n)
