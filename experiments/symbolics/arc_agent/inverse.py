"""Planning by inverting the learned model (docs/AGENT.md part 29): parts 17 and 19's inverse model and viability,
rebuilt from what the columns learn, without roles.

- nav: breadth-first search over one column's own states, each step predicted by its own learned model with every
  other column held still (Brain.model_step on that column alone), never entering another column's cells.
- contact: what moved column B by d. The brain records, whenever a planned translation column moved, the action, each
  other planned translation column and its position relative to B (Brain.contact). To move B: Dijkstra over B's
  positions; a step takes the most frequent mover A to a recorded position relative to B (by nav, B in place), then
  the recorded action; B's and A's next states come from their own models with A placed there.
- plans: for the most probable single-column goal hypotheses, a nav plan for a column that keys move directly, a
  contact plan for a column moved only by others. They compete with the search's plans in Executive.decide.
- viability (part 19): the goal mass still attainable from a joint state, for the columns moved only by others (where
  a push can make a goal unreachable): by contact moves, relaxed (whether the mover can get behind is ignored).

Caches hold within one decision (the models change between decisions).
"""
import heapq
from collections import deque

import numpy as np

from arc_agent.frames import H, W

NAV_STATES, PUSH_STATES, REACH_STATES, GOALS_K = 600, 300, 300, 4


class Inverse:
    def __init__(self, brain, goals):
        self.brain, self.goals = brain, goals
        self.begin(None)

    def begin(self, static):
        """A new decision: forget the cached predictions."""
        self.static, self.one_cache, self.pair_cache, self.reach_cache = static, {}, {}, {}
        self.by_cid = {c.cid: c for c in self.brain.columns}

    def seen(self, cid, s):
        """s, if column cid would be on the screen there; else None (off the screen is not a state to plan through)."""
        c = self.by_cid.get(cid)
        return s if s is not None and (c is None or c.frame.inside(s)) else None

    # ---- the columns' own models ------------------------------------------------------------------------------
    def cols(self):
        return {c.cid: c for c in self.brain.planning() if c.frame.kind == "translation"}

    def driven(self, c):
        """Is this column self-propelled: has it moved in a step where no other planned column moved (common fate)?"""
        return self.brain.solo.get(c.cid, 0) > 0

    def one(self, cid, s, action):
        """cid's most probable next state under action, alone (no other column), from its own model."""
        key = (cid, s, action)
        if key not in self.one_cache:
            new, _ = self.brain.model_step({cid: s}, action, self.static, argmax=True)
            self.one_cache[key] = self.seen(cid, new[cid])
        return self.one_cache[key]

    def pair(self, a, sa, b, sb, action):
        """(A's, B's) most probable next states under action with A at sa and B at sb, from their models."""
        key = (a, sa, b, sb, action)
        if key not in self.pair_cache:
            new, _ = self.brain.model_step({a: sa, b: sb}, action, self.static, argmax=True)
            na, nb = self.seen(a, new[a]), self.seen(b, new[b])
            self.pair_cache[key] = None if na is None or nb is None else (na, nb)
        return self.pair_cache[key]

    def cells(self, c, s):
        return set(map(tuple, c.frame.place(s).tolist()))

    # ---- navigation and contact -------------------------------------------------------------------------------
    def nav(self, c, start, goal_ok, keys, avoid=frozenset(), limit=NAV_STATES):
        """Actions taking column c from start to a state satisfying goal_ok, or None."""
        if goal_ok(start):
            return []
        prev, q = {start: None}, deque([start])
        while q and len(prev) < limit:
            s = q.popleft()
            for a in keys:
                ns = self.one(c.cid, s, a)
                if ns is None or ns == s or ns in prev or (avoid and avoid & self.cells(c, ns)):
                    continue
                prev[ns] = (s, a)
                if goal_ok(ns):
                    path, cur = [], ns
                    while prev[cur] is not None:
                        cur, act = prev[cur]
                        path.append(act)
                    return path[::-1]
                q.append(ns)
        return None

    def movers(self, b):
        """B's recorded contacts: [(action, mover A, A's position relative to B)], the most frequent mover's only."""
        table = self.brain.contact.get(b, {})
        count = {}
        for cnt in table.values():
            for (action, a, rel), n in cnt.items():
                count[a] = count.get(a, 0) + n
        if not count:
            return None, []
        a = max(count, key=count.get)
        entries = sorted({(action, rel) for cnt in table.values() for (action, m, rel) in cnt if m == a})
        return a, entries

    def push(self, cb, b0, ca, a0, goal_ok, keys):
        """Actions that bring column B to a state satisfying goal_ok through the recorded contacts, or None."""
        _, entries = self.movers(cb.cid)
        heap, best, tie, n = [(0, 0, b0, a0, [])], {(b0, a0): 0}, 0, 0
        while heap:
            cost, _, b, a, path = heapq.heappop(heap)
            if path and goal_ok(b):
                return path
            n += 1
            if n > PUSH_STATES:
                return None
            for action, rel in entries:
                spot = self.spot(ca, a, cb, b, rel)
                route = self.nav(ca, a, lambda s: s == spot, keys, avoid=self.cells(cb, b))
                if route is None:
                    continue
                at = self.walk(ca, a, route)
                nxt = self.pair(ca.cid, at, cb.cid, b, action)
                if nxt is None or nxt[1] == b:
                    continue
                c2 = cost + len(route) + 1
                if c2 < best.get((nxt[1], nxt[0]), float("inf")):
                    best[(nxt[1], nxt[0])] = c2
                    tie += 1
                    heapq.heappush(heap, (c2, tie, nxt[1], nxt[0], path + route + [action]))
        return None

    def spot(self, ca, a, cb, b, rel):
        """A's state (keeping its turn) that puts its first cell at rel from B's first cell, B at state b."""
        target = cb.frame.place(b)[0] + np.array(rel)
        p0 = ca.frame.place((0, 0) + tuple(a[2:]))[0]
        return (int(target[0] - p0[0]), int(target[1] - p0[1])) + tuple(a[2:])

    def walk(self, c, s, route):
        for a in route:
            s = self.one(c.cid, s, a)
        return s

    # ---- goals ------------------------------------------------------------------------------------------------
    def holds(self, h, c, s):
        if h[0] == "at":
            return s == h[2]
        p = c.frame.place(s)
        if not np.all((p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)):
            return False
        return set(int(v) for v in self.static[p[:, 1], p[:, 0]]) == {h[2]}

    def plans(self, start, keys, held):
        """Candidate plans [(path, P(WIN), info)] for the most probable single-column goal hypotheses. held(states)
        names what holds in a joint state (Executive.held); the WIN probability is the goal mass newly held."""
        cols, out = self.cols(), []
        h0 = held(start)
        for h, _ in self.goals.top(GOALS_K):
            for c in cols.values():
                if c.kind != h[1] or c.cid not in start:
                    continue
                ok = lambda s, h=h, c=c: self.holds(h, c, s)
                if self.driven(c):
                    others = set().union(*[self.cells(o, start[o.cid]) for o in cols.values()
                                           if o is not c and o.cid in start]) if len(cols) > 1 else set()
                    route = self.nav(c, start[c.cid], ok, keys, avoid=frozenset(others))
                    end = dict(start)
                    if route:
                        end[c.cid] = self.walk(c, start[c.cid], route)
                else:
                    a, _ = self.movers(c.cid)
                    if a is None or a not in cols or a not in start:
                        continue
                    route = self.push(c, start[c.cid], cols[a], start[a], ok, keys)
                    end = None
                    if route:
                        end = self.simulate(start, route)
                if route and end is not None:
                    m = self.goals.mass(held(end) - h0)
                    out.append(([(a, None) for a in route[:-1]] + [(route[-1], tuple(sorted(end.items())))], m, 0.0))
        return out

    def simulate(self, start, route):
        """The joint state after route, by the brain's model (each step's most probable outcome)."""
        s = dict(start)
        for a in route:
            s, _ = self.brain.model_step(s, a, self.static, argmax=True)
        return s

    # ---- viability ----------------------------------------------------------------------------------------------
    def reach(self, c, s):
        """The states column c (moved only by others) could reach by recorded contacts, relaxed."""
        key = (c.cid, s)
        if key not in self.reach_cache:
            a, entries = self.movers(c.cid)
            seen, q = {s}, deque([s])
            while q and a is not None and len(seen) < REACH_STATES:
                b = q.popleft()
                ca = self.cols().get(a)
                for action, rel in entries:
                    if ca is None:
                        break
                    nxt = self.pair(a, self.spot(ca, (0, 0, 0), c, b, rel), c.cid, b, action)
                    if nxt is not None and nxt[1] not in seen:
                        seen.add(nxt[1])
                        q.append(nxt[1])
            self.reach_cache[key] = seen
        return self.reach_cache[key]

    def viable(self, states):
        """V: the goal mass of the most probable single-column hypotheses about columns moved only by others that is
        still attainable from this joint state."""
        cols, v = self.cols(), 0.0
        for h, m in self.goals.top(GOALS_K):
            for c in cols.values():
                if c.kind == h[1] and c.cid in states and not self.driven(c) and self.brain.contact.get(c.cid):
                    if any(self.holds(h, c, s) for s in self.reach(c, states[c.cid])):
                        v += m
                        break
        return v
