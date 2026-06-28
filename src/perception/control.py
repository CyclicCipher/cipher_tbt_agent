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
  - the DYNAMICS — what a contact changes (a blocker removed / added) and death — come from the column's LEARNED
    `predict_effect` (the LM), keyed on the sensed (stepped-on colour, colour-presence) of the rolled state, so the
    toggle EMERGES from the context-conditioned rule (Step C1; no hand-coded opener / closer / flip).
  - SIGNED value — an aversive predicted state is a terminal −1 (avoided by VALUE, not walled off), goal a terminal +1.

The thalamus routes the active sub-goal's goal-state into the spatial frame (top-down CMP); the basal ganglia
gates the focus. Reach / cover / collect / the affordance / hazard-avoidance are NOT branches here — they EMERGE
from rolling this one model under signed value. The residual role decode (body / mover / goal) stays in perception
(`self.world`, read LIVE); the dynamics is the LEARNED column `self.dm` (its `predict_effect`), read live too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

from tbt.column import CorticalColumn
from tbt.neocortex import Neocortex

Cell = Tuple[int, int]


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class _Layout:
    """The scene decoded into the rollout's geometry. The DYNAMICS (which contact changes which colour, and death)
    come from the column's learned `predict_effect` (Step C1), not a decoded role schema; this holds only what the
    rollout needs to evaluate it — the navigable frame, the movers, the removable-blocker cells + their start state,
    and a static cell→colour map + colour-presence to reconstruct the sensed feature the dynamics keys on."""
    frame: FrozenSet[Cell]                 # the spatial map column's reference frame (its navigable nodes)
    movers: FrozenSet[Cell]               # the relational objects (one is the focus; the rest are obstacles)
    door_at: Dict[Cell, int]               # a removable-blocker cell → its colour (shut until that colour is removed)
    removed0: FrozenSet[int]               # colours already removed at the start (a blocker seen open)
    door_colours: FrozenSet[int]           # colours the learned dynamics can change (read off predict_effect's rules)
    colour_at: Dict[Cell, int]             # static cell → colour — the sensed 'stepped-on' feature
    static_present: FrozenSet[int]         # colours present in the scene (the presence-context; doors overridden live)
    bg: int                                # the background colour (a cell with no colour)


class NeocortexPlanner:
    """A `planner` for `tbt.agent.Agent`: scene → move index, planning by rolling the multi-column forward model."""

    def __init__(self, world, dynamics, gamma: float = 0.95, seed: int = 0):
        self.world = world                                     # the residual roles (body/pushable/blocking/goal; read live)
        self.dm = dynamics                                     # the dynamics column (the LM); predict_effect IS the forward model's dynamics
        self.neo = Neocortex(gamma=gamma, seed=seed)
        from .scene import DELTAS                              # the move geometry (perception owns it)
        self.deltas = DELTAS
        self.seed = seed
        self.reset()

    def _door_colours(self) -> set:
        """The colours the LEARNED dynamics can change — read off the column's rules (a `color_<c>_gone/appeared`
        effect), so WHICH colours are removable blockers comes from the learned model, not a role schema."""
        cols = set()
        for _pred, _desc, eff in self.dm.dyn_rules:
            if isinstance(eff, str) and eff.startswith("color_"):
                cols.add(int(eff.split("_")[1]))
        return cols

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

    # ---- decode the scene into the rollout's geometry ----------------------------------------------------
    def _layout(self, scene) -> _Layout:
        w, by = self.world, scene.by_color
        non_bg = {p for cells in by.values() for p in cells} | {scene.body_pos}
        xs = [x for x, _ in non_bg]
        ys = [y for _, y in non_bg]
        bbox = {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)}
        door_colours = self._door_colours() - set(w.pushable)   # removable blockers the LEARNED dynamics changes
        walls = {p for c in w.blocking if c not in door_colours for p in by.get(c, ())}
        return _Layout(
            frame=frozenset(bbox - walls),                     # removable blockers + aversive cells stay IN the frame
            movers=frozenset(p for c in w.pushable for p in by.get(c, ())),
            door_at={p: c for c in door_colours for p in by.get(c, ())},
            removed0=frozenset(c for c in door_colours if not by.get(c)),
            door_colours=frozenset(door_colours),
            colour_at={p: c for c, cells in by.items() for p in cells},
            static_present=frozenset(by.keys()),
            bg=scene.bg,
        )

    # ---- compose the active columns into one forward-model callable --------------------------------------
    def _forward_model(self, lay: _Layout, target: Cell, focus: Optional[Cell]):
        """`step(state, action) -> (next_state, signed_reward, done)` over `(agent, focus-object, removed-blockers)`.
        The spatial map column predicts the agent move + votes free; the focus object-column advances on a push iff
        the map votes the landing free (the egocentric ⊗ absolute lateral vote); the DYNAMICS — which colour
        vanishes/appears, and death — come from the column's learned `predict_effect`, keyed on the sensed
        (stepped-on colour, colour-presence) of the ROLLED state, so the toggle EMERGES from the context-conditioned
        rule (no hand-coded flip). Signed value: a predicted-death state is a terminal −1, the goal-state a +1."""
        others = lay.movers - ({focus} if focus is not None else set())

        def free(t: Cell, removed: FrozenSet[int]) -> bool:
            return (t in lay.frame and t not in others
                    and not (t in lay.door_at and lay.door_at[t] not in removed))

        def present_bits(removed):                             # colour-presence; the door colours set by the rolled state
            bits = [1 if c in lay.static_present else 0 for c in range(16)]
            for c in lay.door_colours:
                if 0 <= c < 16:
                    bits[c] = 0 if c in removed else 1
            return tuple(bits)

        memo: dict = {}                                        # predict_effect by (stepped-on, removed) — once per act, not per BFS step

        def effect_at(stepped_on, removed):
            key = (stepped_on, removed)
            if key not in memo:
                memo[key] = self.dm.predict_effect((stepped_on,) + present_bits(removed))
            return memo[key]

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
            eff = effect_at(lay.colour_at.get(t, lay.bg), removed)   # the column's learned dynamics
            nrem = removed
            if isinstance(eff, str) and eff.startswith("color_"):
                _, col, kind = eff.split("_")
                col = int(col)
                nrem = (removed | {col}) if kind == "gone" else (removed - {col})
            if eff == "death":                                 # signed value: aversive ⇒ terminal −1, never a wall
                return (t, nmv, nrem), -1.0, True
            done = (nmv == target) if focus is not None else (t == target)
            return (t, nmv, nrem), (1.0 if done else 0.0), done

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
