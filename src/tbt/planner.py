"""The cortical planning loop over a navigable scene — the agent's "brain", assembled from the canonical
machinery and nothing task-specific.

Given a `WorldModel` (the learned roles) + per-frame `Scene`s (the body cell + colour->cells + goal/req cells)
+ the move geometry (`deltas`), it:
  - builds the SR-frame MAP (a CorticalColumn over the walkable cells) and path-integrates the body over it
    (loc_reset/move/sense — the recurrence);
  - discovers the abstract subgoal-MDP from the learned effects, values it with `reward.py` (the MuZero critic),
    gates the active subgoal with the BasalGanglia, and routes it with the Thalamus;
  - navigates ONE subgoal at a time (agent positions, or agent x one object), never the 2^K joint.

It speaks only move INDICES (0..len(deltas)-1), never a `GameAction`; perception maps those back. It imports
only `tbt/` — so it carries no colour, grid, or game knowledge. (Subgoal types are still the fire/cover/goal
labels here; Part B3 replaces them with affordances discovered from the dynamics — see EMERGENT_PLAN.md.)
"""

from __future__ import annotations

import random
from collections import defaultdict, deque

from .column import CorticalColumn
from .thalamus import Thalamus
from .basal_ganglia import BasalGanglia
from .reward import RewardModel

# Per-ACTION discount for subgoal selection. The critic discounts gamma ONCE per subgoal, so it minimises the
# NUMBER of subgoals to WIN; discounting each subgoal's value by _TOUR_GAMMA^(its action cost) restores
# per-action discounting, so the cost-optimal TOUR emerges from the same critic (no hand-coded TSP).
_TOUR_GAMMA = 0.99


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bfs_reach(walk, start, target, blocked, deltas):
    """Navigate the body to `target` over agent positions only (N², not the joint). Returns move indices."""
    q, seen = deque([(start, [])]), {start}
    while q:
        p, path = q.popleft()
        if p == target:
            return path
        for i, (dx, dy) in enumerate(deltas):
            nb = (p[0] + dx, p[1] + dy)
            if nb in walk and nb not in blocked and nb not in seen:
                seen.add(nb); q.append((nb, path + [i]))
    return []


def _bfs_push(walk, start, block, target, blocked, other_blocks, deltas):
    """Push ONE object onto `target` — BFS over (agent, object), not the joint of all objects."""
    q, seen = deque([((start, block), [])]), {(start, block)}
    while q:
        (a, b), path = q.popleft()
        if b == target:
            return path
        for i, (dx, dy) in enumerate(deltas):
            na = (a[0] + dx, a[1] + dy)
            if na not in walk or na in blocked or na in other_blocks:
                continue
            if na == b:
                nb = (b[0] + dx, b[1] + dy)
                if nb not in walk or nb in blocked or nb in other_blocks:
                    continue
                state = (na, nb)
            else:
                state = (na, b)
            if state not in seen:
                seen.add(state); q.append((state, path + [i]))
    return []


class Planner:
    """Full-observation planning over the spatial scene."""

    def __init__(self, world, deltas, seed=0):
        self.world = world
        self.deltas = deltas
        self.thal = Thalamus()
        self.bg = BasalGanglia(n_columns=1, seed=seed)
        self.seed = seed
        self.reset()

    def reset(self):                                           # new game
        self.rng = random.Random(self.seed)
        self.blocking = set(self.world.blocking)               # mutable: online obstacle (door-bump) learning
        self.goal_cells = set()
        self.req_cells = set()
        self.new_level()

    def new_level(self):
        self.plan = []
        self._intent = None
        self._cache = None
        self._located_key = None
        self.goal_cells = set()                                # the ego planner accumulates these in the self-
        self.req_cells = set()                                 # frame (reset per level); full-obs re-syncs each act
        self.bg.gate_reset()                                   # subgoal-gate affinity is within-level only

    def on_death(self):
        """GAME_OVER -> the env will reload the CURRENT level: drop the stale plan + intent, but keep the map
        and the gate affinity (the layout is unchanged; loc_sense re-anchors the belief next frame)."""
        self.plan = []
        self._intent = None

    # ---- the spatial column (SR-frame map) ----------------------------------------------------------------
    def _walls(self):
        """Permanent blockers — blocking colours no learned effect removes."""
        targets = self.world.doors
        return {c for c in self.blocking if c not in targets}

    def _walkable(self, by_color):
        non_bg = {p for cells in by_color.values() for p in cells}
        if not non_bg:
            return set()
        xs = [x for x, _ in non_bg]; ys = [y for _, y in non_bg]
        obstacles = {p for c in self._walls() for p in by_color.get(c, ())}
        return {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)
                if (x, y) not in obstacles}

    def _spatial(self, by_color):
        walk = self._walkable(by_color)
        key = frozenset(walk)
        if self._cache and self._cache["key"] == key:
            return self._cache
        cells = sorted(walk)
        cid = {c: i for i, c in enumerate(cells)}
        col = CorticalColumn(n_entities=max(1, len(cells)))
        for c in cells:
            for j, (dx, dy) in enumerate(self.deltas):
                nb = (c[0] + dx, c[1] + dy)
                if nb in walk:
                    col.observe(cid[c], j, cid[nb])
        col.consolidate()
        self._cache = dict(key=key, walk=walk, cells=cells, cid=cid, col=col)
        return self._cache

    # ---- the recurrent body belief over the map -----------------------------------------------------------
    def loc_reset(self, cell):
        c = self._cache
        if c and cell in c["cid"]:
            try:
                c["col"].loc_reset(c["cid"][cell])
            except Exception:
                pass

    def loc_move(self, move):
        c = self._cache
        try:
            c["col"].loc_move(move)
        except Exception:
            pass

    def loc_sense(self, cell):
        c = self._cache
        if c and cell in c["cid"]:
            try:
                c["col"].loc_sense(c["cid"][cell])             # keep=None -> the LEARNED selective gate
            except Exception:
                pass

    def loc_where(self):
        c = self._cache
        try:
            return c["cells"][c["col"].loc_where()]
        except Exception:
            return None

    def _track(self, seen, by_color):
        """PREDICT (loc_move) then CORRECT (loc_sense); on a believed-built map, begin the belief at the body.
        A push that did not move the body reveals an obstacle -> learn it and re-plan."""
        if self._cache["key"] != self._located_key:
            if seen is not None:
                self.loc_reset(seen)
            self._located_key = self._cache["key"]
        elif self._intent is not None:
            prev_seen, tcolor, was_push, prev_move = self._intent
            if seen is not None and seen != prev_seen:
                self.loc_move(prev_move)                       # PREDICT: path-integrate the real move
            elif seen is not None and not was_push and tcolor is not None and tcolor not in self.blocking:
                self.blocking.add(tcolor); self.plan = []      # learn an obstacle -> re-plan
                self._spatial(by_color)
        if seen is not None:
            self.loc_sense(seen)                               # CORRECT toward the sighting (learned gate)
        lw = self.loc_where()
        return lw if (lw is not None and lw in self._cache["cid"]) else seen

    # ---- the high-level abstract subgoal-MDP, valued by reward.py -----------------------------------------
    def _abstract_state(self, by_color):
        removed = frozenset(c for c in self.world.doors if c not in by_color)
        present_req = {p for c in self.world.required_absent for p in by_color.get(c, ())}
        cleared = frozenset(self.req_cells - present_req)
        return removed, cleared

    def _component(self, body, by_color, removed):
        """The body's door-gated navigation-reachable region given `removed` obstacles + hazards."""
        doors = self.world.doors
        blocked = {p for c, cells in by_color.items()
                   if ((c in self.blocking or c in doors) and c not in removed) or c in self.world.death
                   for p in cells}
        blocked |= {p for t in self.world.harmful for p in by_color.get(t, ())}
        walk = self._cache["walk"]
        seen, q = {body}, deque([body])
        while q:
            p = q.popleft()
            for dx, dy in self.deltas:
                nb = (p[0] + dx, p[1] + dy)
                if nb in walk and nb not in blocked and nb not in seen:
                    seen.add(nb); q.append(nb)
        return seen

    def _subgoals(self, state, body, by_color, comp_memo):
        rem, clr = state
        comp = comp_memo.get(rem)
        if comp is None:
            comp = comp_memo[rem] = self._component(body, by_color, rem)
        subs = []
        for T, removes in self.world.effects.items():          # fire a reachable trigger -> opens the way
            cell = next((p for p in by_color.get(T, ()) if p in comp), None)
            if cell is not None:
                subs.append((("fire", T), cell, (frozenset(rem | removes), clr)))
        haspush = any(p in comp for c in self.world.pushable for p in by_color.get(c, ()))
        for R in self.req_cells - clr:                         # cover a reachable uncovered req-cell
            if R in comp and haspush:
                subs.append((("cover", R), R, (rem, frozenset(clr | {R}))))
        goal_cell = next((g for g in self.goal_cells if g in comp), None)
        if goal_cell is not None:                              # reach the goal -> WIN iff all req cleared
            won = clr == frozenset(self.req_cells)
            subs.append((("goal", goal_cell), goal_cell, "WIN" if won else state))
        return subs

    def _value_subgoals(self, body, by_color):
        """Build the abstract subgoal-MDP, value it with reward.py (WIN = the sparse reward), and return
        [(subgoal_key, target_cell, value)] for the subgoals available NOW."""
        s0 = self._abstract_state(by_color)
        T, preds, comp_memo = {"WIN": []}, defaultdict(list), {}
        frontier, seen = [s0], {s0}
        while frontier:
            s = frontier.pop()
            nexts = []
            for _key, _cell, ns in self._subgoals(s, body, by_color, comp_memo):
                nexts.append(ns); preds[ns].append(s)
                if ns not in seen and ns != "WIN":
                    seen.add(ns); frontier.append(ns)
            T[s] = nexts
            if len(seen) > 256:
                break
        rm = RewardModel(max(2, len(T)), beta=0.0, prioritized=False, optimistic=False)   # PLANNING: converge
        rm.R_ext["WIN"] = 1.0                                  # to the true return so a not-yet-winning goal
        rm.plan(T, preds, s0)                                  # can't outvalue real progress
        return [(key, cell, rm.V[ns]) for key, cell, ns in self._subgoals(s0, body, by_color, comp_memo)]

    # ---- thalamus routing of the active subgoal's goal-state into the spatial column ---------------------
    def _route(self, valued, active_i):
        c = self._cache
        direct = valued[active_i][1]
        items = [(i, c["cid"][v[1]]) for i, v in enumerate(valued) if v[1] in c["cid"]]
        if direct not in c["cid"] or not items:
            return direct
        task_col = CorticalColumn(n_entities=len(valued))
        R = self.thal.bind(task_col, c["col"], items)
        idx = self.thal.read_location(R, task_col, c["col"], active_i)
        inv = {j: s for s, j in c["col"].loc.items()}
        if idx is not None and idx in inv:
            return c["cells"][inv[idx]]
        return direct

    # ---- factored navigation of one subgoal ---------------------------------------------------------------
    def _blocked(self, by_color, removed):
        doors = self.world.doors
        blocked = {p for c, cells in by_color.items()
                   if ((c in self.blocking or c in doors) and c not in removed) or c in self.world.death
                   for p in cells}
        return blocked | {p for t in self.world.harmful for p in by_color.get(t, ())}

    def _pushables_now(self, by_color):
        return {p for c in self.world.pushable for p in by_color.get(c, ())}

    def _navigate(self, key, target, body, by_color, removed):
        walk = self._cache["walk"]
        blocked = self._blocked(by_color, removed)
        pushables = self._pushables_now(by_color)
        if key[0] == "cover":
            free = [b for b in pushables if b not in self.req_cells]
            if not free:
                return []
            block = min(free, key=lambda b: _manh(b, target))
            return _bfs_push(walk, body, block, target, blocked, pushables - {block}, self.deltas)
        return _bfs_reach(walk, body, target, blocked | pushables, self.deltas)

    def _plan(self, body, by_color):
        valued = self._value_subgoals(body, by_color)
        if not valued:
            return []
        removed = self._abstract_state(by_color)[0]
        plans = [self._navigate(k, c, body, by_color, removed) for k, c, _v in valued]
        cand = [(i, valued[i][2] * (_TOUR_GAMMA ** len(plans[i]))) for i in range(len(valued)) if plans[i]]
        if not cand:
            return []
        i = cand[self.bg.gate([valued[j][0] for j, _s in cand],
                              [s for _j, s in cand])][0]        # the basal ganglia gates the cost-aware values
        target = self._route(valued, i)                        # the thalamus PROPOSES a route for the chosen one
        routed = self._navigate(valued[i][0], target, body, by_color, removed)
        return min((p for p in (routed, plans[i]) if p), key=len, default=[])

    # ---- the planning step --------------------------------------------------------------------------------
    def act(self, scene):
        """One scene -> one move index. Builds/caches the map, tracks the body, plans, and records the intent
        the next recurrence step needs."""
        self.goal_cells = scene.goal_cells
        self.req_cells = scene.req_cells
        by_color = scene.by_color
        seen = scene.body_pos
        self._spatial(by_color)
        body = self._track(seen, by_color)
        if body is None:
            return self.rng.randrange(len(self.deltas))
        if not self.plan:
            self.plan = self._plan(body, by_color)
        move = self.plan.pop(0) if self.plan else self.rng.randrange(len(self.deltas))
        dx, dy = self.deltas[move]
        t = (body[0] + dx, body[1] + dy)
        tcolor = next((c for c, cells in by_color.items() if t in cells), None)
        self._intent = (seen, tcolor, t in self._pushables_now(by_color), move)
        return move


# ===== PARTIAL OBSERVABILITY: the same planning under an egocentric self-frame — recurrence is ESSENTIAL ====
# The body's position is known only by path-integrating in a SELF-FRAME; the map is the accumulated windows.
# The subgoal/value/navigation layer above is UNCHANGED — it runs over the remembered self-frame scene.

_SELF = 31                                                     # self-frame grid; the body starts at its centre
_O = _SELF // 2
_CID = {(c % _SELF - _O, c // _SELF - _O): c for c in range(_SELF * _SELF)}    # self-frame offset -> symbol
_CELLS = {c: (c % _SELF - _O, c // _SELF - _O) for c in range(_SELF * _SELF)}  # symbol -> self-frame offset
_FRAME_COL = None


def _frame_column(deltas):
    """The self-frame map column — built ONCE and shared (the consolidate eigh is the cost). Its SR frame + L5
    operators are identical for every episode; only the per-episode belief differs and is reset."""
    global _FRAME_COL
    if _FRAME_COL is None:
        col = CorticalColumn(n_entities=_SELF * _SELF, seed=0)
        for x in range(_SELF):
            for y in range(_SELF):
                for j, (dx, dy) in enumerate(deltas):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < _SELF and 0 <= ny < _SELF:
                        col.observe(y * _SELF + x, j, ny * _SELF + nx)
        col.consolidate()
        _FRAME_COL = col
    return _FRAME_COL


class EgoPlanner(Planner):
    """Egocentric partial observation: the body position comes from the recurrent self-frame belief, the map
    is the accumulated windows. Inherits the whole subgoal/value/navigate layer."""

    def __init__(self, world, deltas, radius=2, memory=True, seed=0):
        self.radius, self.memory = radius, memory
        self._fcol = _frame_column(deltas)                     # before super().__init__ (new_level uses it)
        super().__init__(world, deltas, seed)

    def new_level(self):
        super().new_level()
        self._map = {}                                         # self-frame offset -> colour (the memory)
        self._fcol.loc_reset(_O * _SELF + _O)                  # dead-reckoning origin (self-frame centre)

    def _pos(self):
        return _CELLS.get(self._fcol.loc_where(), (0, 0))      # the recurrent belief, as a self-frame offset

    def _accumulate(self, window, bg, pos):
        self._map[pos] = bg                                    # the body stands on floor
        for (ox, oy), color in window.items():
            self._map[(pos[0] + ox, pos[1] + oy)] = color

    def _scene(self, bg):
        walls = self._walls()
        by_color = {}
        for cell, color in self._map.items():
            if color != bg:
                by_color.setdefault(color, set()).add(cell)
        walk = {cell for cell, color in self._map.items() if color not in walls}
        for c in self.world.goal_colors:
            self.goal_cells |= by_color.get(c, set())
        for c in self.world.required_absent:
            self.req_cells |= by_color.get(c, set())
        self._cache = dict(key=frozenset(walk), walk=walk, cells=_CELLS, cid=_CID, col=self._fcol)
        return by_color, walk

    def _frontier(self, pos, walk):
        """Head toward the nearest explored cell with an unexplored neighbour — reveal more of the map."""
        front = {c for c in walk if any((c[0] + dx, c[1] + dy) not in self._map for dx, dy in self.deltas)}
        if not front:
            return None
        path = _bfs_reach(walk, pos, min(front, key=lambda c: _manh(c, pos)), set(), self.deltas)
        return path[0] if path else None

    def _reactive(self, window, bg):
        """memory=False ablation: no path integration, no map — step toward a goal/key colour IF in view."""
        for (ox, oy), color in window.items():
            if color in self.world.goal_colors or color in self.world.effects:
                move = min(range(len(self.deltas)), key=lambda i: _manh((ox, oy), self.deltas[i]))
                if self.deltas[move] == (ox, oy) or abs(ox) + abs(oy) > 1:
                    return move
        return self.rng.randrange(len(self.deltas))

    def act(self, ego):
        window, bg = ego.window, ego.bg
        if window is None:
            return self.rng.randrange(len(self.deltas))
        if not self.memory:
            return self._reactive(window, bg)
        pos = self._pos()                                      # recurrent belief BEFORE this move
        self._accumulate(window, bg, pos)                     # remember the window (sequence memory)
        by_color, walk = self._scene(bg)
        if not self.plan:
            self.plan = self._plan(pos, by_color)              # the base subgoal layer over the remembered map
            if not self.plan:
                f = self._frontier(pos, walk)                  # nothing relevant seen yet -> explore
                self.plan = [f] if f is not None else [self.rng.randrange(len(self.deltas))]
        move = self.plan.pop(0) if self.plan else self.rng.randrange(len(self.deltas))
        known_walls = {c for c, col in self._map.items() if col in self._walls()}
        if (pos[0] + self.deltas[move][0], pos[1] + self.deltas[move][1]) not in known_walls:
            self.loc_move(move)                                # believed-free -> integrate
        return move
