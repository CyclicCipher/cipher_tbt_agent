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

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

from tbt.column import CorticalColumn
from tbt.neocortex import Neocortex
from tbt.recognize import Recognizer
from tbt.reward import ValueLearner

Cell = Tuple[int, int]


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _anchor(cells: FrozenSet[Cell]) -> Cell:
    """The object's pose-anchor (bbox-min corner) — the translation part of its pose."""
    return (min(x for x, _ in cells), min(y for _, y in cells))


def _near(cells: FrozenSet[Cell], c: Cell) -> int:
    """Distance from an object (its nearest cell) to a target cell."""
    return min(_manh(p, c) for p in cells)


@dataclass(frozen=True)
class _Layout:
    """The scene decoded into the rollout's geometry. The DYNAMICS (which contact changes which colour, and death)
    come from the column's learned `predict_effect` (Step C1), not a decoded role schema; this holds only what the
    rollout needs to evaluate it — the navigable frame, the movers, the removable-blocker cells + their start state,
    and a static cell→colour map + colour-presence to reconstruct the sensed feature the dynamics keys on."""
    frame: FrozenSet[Cell]                 # the spatial map column's reference frame (its navigable nodes)
    movers: Tuple[Tuple[str, FrozenSet[Cell]], ...]   # recognised pushable OBJECTS (id, cells); one is the focus
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
        self._focus: Optional[str] = None                      # the recognised id of the focus object (BG-gated)
        self._visits: Dict[Cell, int] = {}                     # body-cell visit counts → novelty-directed exploration

    def new_level(self):                                       # the layout changed → rebuild map, rebind, drop focus
        self.neo.reset()
        self._mapkey = None
        self._bound = None
        self._focus = None
        self._visits = {}                                      # exploration is per-layout; the LEARNED model persists (in `world`)

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
            movers=tuple((mid, frozenset(cells)) for mid, cells in scene.movers),   # recognised pushable objects
            door_at={p: c for c in door_colours for p in by.get(c, ())},
            removed0=frozenset(c for c in door_colours if not by.get(c)),
            door_colours=frozenset(door_colours),
            colour_at={p: c for c, cells in by.items() for p in cells},
            static_present=frozenset(by.keys()),
            bg=scene.bg,
        )

    # ---- compose the active columns into one forward-model callable --------------------------------------
    def _forward_model(self, lay: _Layout, target: Cell, focus: Optional[FrozenSet[Cell]]):
        """`step(state, action) -> (next_state, signed_reward, done)` over `(agent, focus-anchor, removed-blockers)`.
        The spatial map column predicts the agent move + votes free; the focus object-column is a RIGID multi-cell
        body — a push advances its whole FOOTPRINT (`shape` translated to the anchor) iff every cell it advances
        into is voted free by the absolute map (the egocentric ⊗ absolute lateral vote); a single-cell object is
        the degenerate `shape={(0,0)}` case (byte-identical to the old cell-push). The DYNAMICS — which colour
        vanishes/appears, and death — come from the column's learned `predict_effect`, keyed on the sensed
        (stepped-on colour, colour-presence) of the ROLLED state, so the toggle EMERGES from the context-conditioned
        rule (no hand-coded flip). Signed value: a predicted-death state is a terminal −1, the goal-state a +1."""
        others = frozenset(p for _mid, cells in lay.movers if cells != focus for p in cells)
        shape = None
        if focus is not None:
            ax, ay = _anchor(focus)
            shape = frozenset((x - ax, y - ay) for x, y in focus)   # the object's cells relative to its anchor

        def footprint(anchor: Cell) -> FrozenSet[Cell]:
            return frozenset((anchor[0] + dx, anchor[1] + dy) for dx, dy in shape)

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
            ag, anchor, removed = state
            dx, dy = self.deltas[a]
            t = (ag[0] + dx, ag[1] + dy)
            if anchor is not None and t in footprint(anchor):  # focus object-column: the egocentric RIGID push…
                shifted = (anchor[0] + dx, anchor[1] + dy)
                if not all(free(c, removed) for c in footprint(shifted) - footprint(anchor)):
                    return state, 0.0, False                    # …every advanced-into cell voted against the absolute map
                n_anchor = shifted
            else:                                              # spatial map column: the agent-move vote
                if not free(t, removed):
                    return state, 0.0, False
                n_anchor = anchor
            eff = effect_at(lay.colour_at.get(t, lay.bg), removed)   # the column's learned dynamics
            nrem = removed
            if isinstance(eff, str) and eff.startswith("color_"):
                _, col, kind = eff.split("_")
                col = int(col)
                nrem = (removed | {col}) if kind == "gone" else (removed - {col})
            if eff == "death":                                 # signed value: aversive ⇒ terminal −1, never a wall
                return (t, n_anchor, nrem), -1.0, True
            done = (target in footprint(n_anchor)) if anchor is not None else (t == target)
            return (t, n_anchor, nrem), (1.0 if done else 0.0), done

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
        self._visits[agent] = self._visits.get(agent, 0) + 1   # coverage, for novelty-directed exploration
        if explore and self.rng.random() < explore:           # optional epsilon (off in the continuous online loop)
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
            # an object already covering a pad is PLACED (its cells overlap a required cell) — don't disturb it
            free = [(mid, cells) for mid, cells in lay.movers if not (cells & scene.req_cells)]
            if free:                                           # COVER: the BG gates a focus OBJECT (by recognised id)
                ids = [mid for mid, _ in free]
                if self._focus not in ids:                     # gate on IDENTITY → affinity persists across the push
                    self._focus = self.neo.gate_focus(ids, [1.0 / (1.0 + _near(cells, C)) for _, cells in free])
                focus = min((cells for mid, cells in free if mid == self._focus), key=lambda cs: _near(cs, C))
                step = self._forward_model(lay, target, focus)   # push the recognised object by its whole footprint
                return self.neo.achieve(step, (agent, _anchor(focus), lay.removed0), n)
            step = self._forward_model(lay, target, None)      # COLLECT: no mover → the agent reaches it (emergent)
            return self.neo.achieve(step, (agent, None, lay.removed0), n)
        goals = sorted(scene.goal_cells)
        if not goals:
            return self._explore(lay, agent, n)                # no goal learned yet → directed (novelty) exploration
        target = self._route(col, goals[0])
        step = self._forward_model(lay, target, None)
        return self.neo.achieve(step, (agent, None, lay.removed0), n)

    # ---- directed exploration: plan to the least-visited reachable cell (visit-novelty) ------------------
    def _reachable(self, lay: _Layout, start: Cell) -> set:
        """The cells reachable from `start` over the currently-free frame (walls/movers/shut-doors block), by BFS.
        It is what the agent can actually get to NOW, so exploration targets are always plannable."""
        others = frozenset(p for _mid, cells in lay.movers for p in cells)

        def free(t):
            return (t in lay.frame and t not in others
                    and not (t in lay.door_at and lay.door_at[t] not in lay.removed0))

        seen, q = {start}, deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in self.deltas:
                nb = (x + dx, y + dy)
                if nb not in seen and free(nb):
                    seen.add(nb)
                    q.append(nb)
        return seen

    def _explore(self, lay: _Layout, agent: Cell, n: int) -> int:
        """Novelty-directed exploration: roll the SAME forward model toward the nearest LEAST-VISITED reachable
        cell, so the agent covers the level to discover the body's effects and the goal (the sparse score) with far
        fewer actions than random ε. This is the epistemic half of active inference at the cold start; once the goal
        is learned the planner switches to exploiting it (above). Visit-novelty here; prediction-error next."""
        reachable = self._reachable(lay, agent) - {agent}
        if not reachable:
            return self.rng.randrange(n)
        target = min(reachable, key=lambda c: (self._visits.get(c, 0), _manh(agent, c)))
        step = self._forward_model(lay, target, None)
        return self.neo.achieve(step, (agent, None, lay.removed0), n)


# ── Tetris: object-based planner — the SAME achiever, a POSE-based object forward model, no game internals ──
class TetrisPlanner:
    """Object-based planner, TBT-faithful. Identify the controllable PIECE's OBJECT once per move by pose-invariant
    recognition (`tbt/recognize.py`), then roll a POSE state `(orientation, anchor)` whose rotation is the object's
    own ROTATION OPERATOR — the CW orbit of its canonical model (`cells_at`), exact by construction and matching the
    game's rotate with NO lookup table (so the table's wrong-entry bug class cannot exist). Translation and gravity
    shift the anchor; the absolute-map COLLISION VOTE (well + stack) gates every move. The SHARED bounded
    `Neocortex.achieve` plans one piece's placement (terminal on lock); the agent re-perceives per piece. The learned
    HORIZON value (EZ-V2) is bootstrapped at the lock so the planner can set up MULTI-piece clears. No game internals."""

    _SHIFT = {0: (-1, 0), 1: (1, 0), 3: (0, 1)}            # left, right, down (action 2 = rotate, via the operator)

    def __init__(self, recognizer: Recognizer, gravity=(0, 1), gamma: float = 0.95, seed: int = 0,
                 cap: int = 6000, value=None):
        self.rec = recognizer                              # the object library (learned by watching) + recognition
        self.gravity = gravity
        self.gamma = gamma
        self.cap = cap
        self.seed = seed
        # the learned HORIZON value (EZ-V2), bootstrapped at the lock so the planner sets up MULTI-PIECE clears
        # (the greedy one-piece rollout cannot). Trained online; shareable across pieces.
        self.value = value if value is not None else ValueLearner()
        self.neo = Neocortex(gamma=gamma, seed=seed)
        self.rng = random.Random(seed)
        self._orient_cache: Dict[str, list] = {}

    def reset(self):
        self.neo.reset()
        self.rng = random.Random(self.seed)

    def new_level(self):
        self.neo.reset()

    def on_death(self):
        pass

    def feats(self, stack, well):
        """The GAPS encoding the value learns over: empty interior cells in rows that have any stack — 'what's
        needed to complete a line'. It CHANGES with placement (unlike raw occupancy, which a constant pre-fill
        dominated — the Tetris-L2 lesson), so a generalising value can be learned over it."""
        left, right, floor = well
        rows = {y for _, y in stack}
        return frozenset((x, y) for y in rows for x in range(left + 1, right) if (x, y) not in stack)

    def _orientations(self, obj) -> list:
        """The object's 4 orientations as normalised integer cell-sets, generated by the rotation OPERATOR
        (`cells_at`, CW = the game's rotate direction) — not a learned table. Cached per object."""
        if obj.name not in self._orient_cache:
            outs = []
            for k in range(4):
                cells = obj.cells_at(-k * math.pi / 2, (0.0, 0.0))     # CW 90°·k (negative angle = clockwise)
                snap = [(int(round(float(x))), int(round(float(y)))) for x, y in cells]
                mnx, mny = min(x for x, _ in snap), min(y for _, y in snap)
                outs.append(frozenset((x - mnx, y - mny) for x, y in snap))
            self._orient_cache[obj.name] = outs
        return self._orient_cache[obj.name]

    def _pose(self, piece, orients):
        """The perceived piece's (orientation index, anchor): match its normalised shape to the operator's orbit."""
        mnx, mny = min(x for x, _ in piece), min(y for _, y in piece)
        norm = frozenset((x - mnx, y - mny) for x, y in piece)
        for k, o in enumerate(orients):
            if o == norm:
                return k, (mnx, mny)
        return None

    def _forward_model(self, orients, stack, well):
        """`step((orientation, ax, ay), action) -> (next_state, signed_reward, done)`. Rotation advances the
        orientation index through the operator's orbit; translate/gravity shift the anchor; collision vote = well +
        stack. Lock ⇒ terminal with reward = lines cleared now + γ·V(resulting stack) (the EZ-V2 horizon bootstrap)."""
        left, right, floor = well
        gx, gy = self.gravity

        def cells_of(state):
            k, ax, ay = state
            return [(ax + dx, ay + dy) for dx, dy in orients[k]]

        def valid(cells):
            return all(left < x < right and y < floor and (x, y) not in stack for x, y in cells)

        def step(state, a):
            k, ax, ay = state
            if a == 2:                                     # ROTATE — the operator advances the orientation (CW)
                moved = ((k + 1) % 4, ax, ay)
            else:
                dx, dy = self._SHIFT[a]
                moved = (k, ax + dx, ay + dy)
            if not valid(cells_of(moved)):                 # blocked (well/stack collision vote)
                moved = state
            fk, fax, fay = moved
            fell = (fk, fax + gx, fay + gy)
            if valid(cells_of(fell)):
                return fell, 0.0, False                    # gravity: keep falling
            new_stack = stack | set(cells_of(moved))       # cannot fall ⇒ LOCK
            rows = [y for y in range(floor) if all((x, y) in new_stack for x in range(left + 1, right))]
            v = self.value.value(self.feats(new_stack, (left, right, floor)))   # the learned HORIZON value (EZ-V2)
            return moved, float(len(rows)) + self.gamma * v, True   # lines now + the future value of the stack

        return step

    def act(self, scene, explore: float = 0.0):
        if not scene.piece:                                # body not visible yet → wait (let gravity bring a piece)
            return 3
        if explore and self.rng.random() < explore:
            return self.rng.randrange(4)
        obj = self.rec.identify_model(scene.piece)         # WHICH object is the controllable piece (pose-invariant)
        if obj is None:                                    # not yet learned → wander (the cold-start)
            return self.rng.randrange(4)
        orients = self._orientations(obj)
        pose = self._pose(scene.piece, orients)
        if pose is None:                                   # perceived shape not in the orbit (shouldn't happen) → wander
            return self.rng.randrange(4)
        k0, (ax, ay) = pose
        step = self._forward_model(orients, scene.stack, scene.well)
        return self.neo.achieve(step, (k0, ax, ay), 4, max_states=self.cap)
