"""The per-sub-goal navigator + the factored task loop — the agent's planning core (S4/S5).

Given F's learned sub-goal factors, the `FactoredPlanner` sequences them (`control_loop`'s additive task column,
S5) and the `ValuePlanner` navigates each by BFS in the LEARNED forward model `G` — over the (agent, the ONE
relevant mover, state-bits) joint, the general form of a push/cover/reach search where the mechanic (a push, a
door bit, …) comes from G, never a hand-coded rule. K independent N²-navigations sequenced, never the joint 2^K.

Everything domain-specific is OPAQUE here:
  - `coords` — a factored state tuple (agent + each mover, egocentric, + state bits); indexed, never interpreted;
  - `cid` — the walkable cells (the map); `gates` — `{cell: bit-index}`, a move onto a gated cell is blocked while
    that coord-bit reads 1;
  - `factors`, `satisfied`, `focus_mover` — F's sub-goals + how to test / assign them, callables from perception.
There is no grid / colour / delta / door / push in this file — swap perception (even to a non-spatial game) and
this planner is unchanged.
"""

from __future__ import annotations

import random
from collections import deque


class ValuePlanner:
    """Navigate ONE sub-goal at a time by BFS over the (agent, focused-mover, state-bits) joint in the learned G.

    The propagated search over that joint (S4's value-as-reachability) is what an agent-only proximity could not
    express: getting a mover onto a cell depends on where the agent stands relative to it (the push geometry), so
    the search must be over (agent × mover), not the mover alone. Factoring to the ONE relevant mover keeps it
    N², never the 2^K joint of all movers."""

    def __init__(self, G, n_movers, n_bit, n_actions, seed=0):
        self.G = G
        self.n_movers, self.n_bit, self.n_actions = n_movers, n_bit, n_actions
        self.rng = random.Random(seed)
        self._slotc = {}                                       # coords -> [agent cell, mover cells…] (cache)
        self._occ = frozenset()                                # immovable cells (parked / non-focus movers)
        self._cid = None                                       # the walkable map (set by perception via set_map)

    def set_map(self, col, cid):
        self._cid = cid

    def reset(self):                                            # the navigator is stateless across episodes
        pass

    def _slots(self, coords):
        s = self._slotc.get(coords)
        if s is None:
            ax, ay = coords[0], coords[1]
            s = [(ax, ay)] + [(ax + coords[2 + 2 * i], ay + coords[2 + 2 * i + 1]) for i in range(self.n_movers)]
            self._slotc[coords] = s
        return s

    # ---- the learned model G, gated by traversability (the map, a closed gate, a parked mover) ------------
    def _traversable(self, cell, coords, gates):
        if cell not in self._cid or cell in self._occ:        # off-map, or a parked mover — immovable, so the
            return False                                      # agent can't push it off (undoing a done factor)
        b = gates.get(cell)
        return not (b is not None and coords[b] == 1)         # a closed gate blocks while its state-bit reads 1

    def _forward(self, coords, action, gates):
        nxt = self.G.predict(coords, action)
        slots = self._slots(nxt)
        if len(set(slots)) != len(slots):                      # two entities on one cell — physically invalid
            return coords                                      # (G mispredicts push-into-wall as overlap) ⇒ blocked
        for cell, was in zip(slots, self._slots(coords)):
            if cell != was and not self._traversable(cell, coords, gates):   # a MOVED entity (agent or pushed
                return coords                                  # mover) lands non-walkable / on an obstacle —
        return nxt                                             # the static obstacles (unmoved) are not re-checked

    # ---- the per-sub-goal navigator: BFS over (agent, focused-mover) in the LEARNED model -----------------
    def navigate(self, coords, focus, target, gates, max_expand=3000):
        """Shortest action PATH to satisfy one sub-goal, by BFS over the (agent, focused-mover) joint in the
        LEARNED forward model G — the general form of `planner._bfs_push`, but the push (and any mechanic) comes
        from G, not a hand-coded `nb = b + delta` rule. The OTHER movers are immovable obstacles (`self._occ`, so
        the agent can't disturb a placed one), so the search is over N² (agent × the one mover), never the 2^K
        joint. `focus = (slot,)` pushes that mover onto `target`; `focus = ()` walks the agent there (a reach).
        Returns the whole action path (the caller follows it, re-planning only on deviation — a full BFS per step
        is what made the deep levels crawl); `[]` if already satisfied, unreachable, or the search bound is hit."""
        cells = self._slots(coords)                            # [agent, mover0, mover1, …]
        fset = set(focus)
        self._occ = frozenset(cells[1 + j] for j in range(self.n_movers) if j not in fset)   # non-focus = walls
        nb = 2 * self.n_movers

        def key(c):                                            # project to (agent, focused-mover, state-bits)
            k = (c[0], c[1])
            for i in focus:
                k += (c[2 + 2 * i], c[3 + 2 * i])
            return k + tuple(c[2 + nb:])

        def done(c):
            if focus:
                i = focus[0]
                return (c[0] + c[2 + 2 * i], c[1] + c[3 + 2 * i]) == target
            return (c[0], c[1]) == target

        if done(coords):
            return []
        q, parent, expand = deque([coords]), {key(coords): None}, 0
        while q and expand < max_expand:
            c = q.popleft(); expand += 1
            for a in range(self.n_actions):
                c2 = self._forward(c, a, gates)
                k = key(c2)
                if k in parent:
                    continue
                if done(c2):                                   # reconstruct the path (first → last) via parents
                    path, pk = [a], key(c)
                    while parent[pk] is not None:
                        ppk, pa = parent[pk]; path.append(pa); pk = ppk
                    path.reverse()
                    return path
                parent[k] = (key(c), a)
                q.append(c2)
        return []


class FactoredPlanner:
    """The task column — `control_loop`'s additive loop (S5): sequence F's factors, navigate each in G (S4/S5).

    A conjunctive goal (`cover₁ ∧ cover₂ ∧ … ∧ reach`) is sequenced, not solved jointly: REVEAL the factors one
    at a time (perception orders them, terminal `reach` last); navigate the current one to satisfaction (the BFS
    over the agent × the ONE relevant mover, in the learned G); when the revealed prefix is all satisfied, reveal
    the next. Already-covered cells are parked-mover obstacles the navigator routes around and cannot disturb, so
    done factors are PRESERVED. K independent N²-navigations sequenced, never the joint 2^K.

    Domain-general: `factors`, `satisfied(coords, factor)->bool`, and `focus_mover` are OPAQUE (from perception).
    For any game the loop sequences whatever sub-conditions F learned — pads to cover, or contradictions to
    resolve — and the navigator reaches each through the learned dynamics."""

    def __init__(self, vp: "ValuePlanner", satisfied, route_proximity, focus_mover):
        self.vp, self.satisfied, self.focus_mover = vp, satisfied, focus_mover
        self._k = 1
        self._focus = None                                     # the mover COMMITTED to the current factor
        self._plan = []                                        # the cached action path for the current factor
        self._last = None                                      # previous coords (a no-op ⇒ the plan deviated from G)
        self._cool = 0                                         # random-walk cooldown when a factor is unreachable
        self._tries = 0                                        # re-plans on the current factor without progress

    def reset(self):
        self.vp.reset()
        self._k, self._focus, self._plan, self._last, self._cool, self._tries = 1, None, [], None, 0, 0

    def set_map(self, col, cid):
        self.vp.set_map(col, cid)

    def act(self, coords, gates, factors):
        if coords is None or not factors:
            return self.vp.rng.randrange(self.vp.n_actions)
        prev_k = self._k
        while self._k < len(factors) and all(self.satisfied(coords, f) for f in factors[:self._k]):
            self._k += 1                                       # the revealed prefix is met → reveal the next factor
        if self._k != prev_k:                                  # a factor was satisfied → fresh focus + plan + counters
            self._focus, self._plan, self._cool, self._tries = None, [], 0, 0
        if self._cool > 0:                                     # unreachable for now → random-walk before re-searching
            self._cool -= 1; self._last = coords               # (a full BFS per step on a stuck factor is the slow path)
            return self.vp.rng.randrange(self.vp.n_actions)
        if self._plan and coords != self._last:                # a plan in flight + the last move took effect → FOLLOW
            self._last = coords
            return self._plan.pop(0)
        if self._last is not None and coords == self._last:    # the last action was a no-op (G mispredicted) → ESCAPE
            self._plan, self._last = [], coords                # with a random move (re-planning would repeat the no-op)
            return self.vp.rng.randrange(self.vp.n_actions)
        target, kind = factors[self._k - 1]                    # else (fresh state): plan the current factor
        movers = set(self.vp._slots(coords)[1:])               # a satisfied factor leaves an obstacle iff a MOVER
        occ = frozenset(f[0] for f in factors[:self._k - 1] if f[0] in movers)   # (not the agent) parks on its cell
        if self._focus is None:                                # COMMIT one mover per factor (re-picking flips movers)
            self._focus = self.focus_mover(coords, (target, kind), occ)
        self._tries += 1
        if self._tries > 6:                                    # this factor keeps re-planning without satisfying (G
            self._cool, self._tries = 30, 0                    # can't reach it) → back off, don't keep paying search
            return self.vp.rng.randrange(self.vp.n_actions)
        self._plan = self.vp.navigate(coords, (self._focus,) if self._focus is not None else (), target, gates)
        self._last = coords
        if not self._plan:                                     # unreachable within the search bound → random-walk +
            self._cool = 15                                    # a cooldown, instead of re-searching every step
            return self.vp.rng.randrange(self.vp.n_actions)
        return self._plan.pop(0)

    def learn(self, reward, done):                              # the navigator is exact (no value to learn)
        pass

    def flush(self):
        pass
