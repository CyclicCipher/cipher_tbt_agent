"""The explorer: a general ARC-AGI-3 agent that maps each level's state graph and learns which actions do anything.

No per-game code. What it knows in advance: the interface only (a 64x64 colour grid, the available action ids, RESET,
levels completed). What it learns, online:
- **States:** the frame, with the cells of an ambient ticker masked out. A ticker is a region that changes on almost
  every step, a few cells at a time, and whose cells change once and never revert (a step or energy counter). It is
  found from the statistics of change, not from where it sits on screen.
- **Candidate actions in a state:** each simple action, plus one click per object (a connected same-colour region,
  the background excluded).
- **Which actions have an effect:** P(effect | action kind) for kind = a key, or a click on an object of a given colour
  and size, from a Beta(1, 1) prior updated on every tried action. Untried actions are tried in order of this
  probability.
- **The level's state graph:** every tried (state, action) and where it led. When the current state has nothing left
  worth trying, the agent walks the known graph to the nearest state that has, with RESET as an edge back to the
  level's start (so a dead end costs one action, not the rest of the level). The engine restarts the WHOLE game if
  RESET comes before any action in the level, so a RESET is only ever sent after the agent has acted in the level.

Single file, numpy only, so it can be spliced into the Kaggle notebook as is.
"""
import hashlib
from collections import Counter, deque

import numpy as np

RESET, CLICK = 0, 6
MIN_TICK_STEPS = 6          # steps of evidence before a region can be judged a ticker


def components(grid):
    """4-connected same-colour regions: [(colour, cells as (row, col) list)]."""
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col, comp, q = grid[r, c], [], [(r, c)]
            seen[r, c] = True
            while q:
                y, x = q.pop()
                comp.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append((int(col), comp))
    return out


def size_bucket(n):
    return 0 if n <= 2 else 1 if n <= 9 else 2 if n <= 36 else 3 if n <= 144 else 4


class Ticker:
    """Finds the ambient ticker cells of a level from the change statistics (see the module docstring)."""

    def __init__(self, shape, mask=None):
        self.changes = np.zeros(shape, int)            # times each cell changed
        self.steps = []                                # the change mask of each ordinary step
        self.mask = np.zeros(shape, bool) if mask is None else mask.copy()

    def observe(self, before, after):
        """Returns True if the mask grew (the caller's state keys are then stale)."""
        d = (before != after) & ~self.mask
        self.changes += d
        self.steps.append(d)
        return len(self.steps) >= MIN_TICK_STEPS and self._update(after)

    def _update(self, grid):
        ever = self.changes > 0
        grown = False
        for _, comp in components(ever.astype(np.int8)):
            ys, xs = zip(*comp)
            region = np.zeros_like(self.mask)
            region[ys, xs] = True
            if not ever[ys[0], xs[0]]:
                continue
            touched = [d[region].sum() for d in self.steps]
            hits = [t for t in touched if t]
            reverted = (self.changes[region] >= 2).sum()
            if (len(hits) >= 0.9 * len(touched) and np.mean(hits) <= 3
                    and reverted <= 0.1 * region.sum()):
                for _, cells in components(grid):      # the rest of the counter: what touches the changed cells
                    if len(cells) <= 300 and any(region[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any() for y, x in cells):
                        cy, cx = zip(*cells)
                        region[cy, cx] = True
                self.mask |= region
                self.changes[region] = 0
                grown = True
        return grown


class Explorer:
    def __init__(self, game_id=""):
        self.game_id = game_id
        self.level = None
        self.effect = {}                               # action kind -> [effects, tries]  (kept across levels)
        self.notes = Counter()

    # ---- the level's model ----------------------------------------------------------------------------------------
    def _new_level(self, grid):
        old = getattr(self, "ticker", None)            # the counter usually carries over; revised if not
        self.ticker = Ticker(grid.shape, old.mask if old is not None else None)
        self.nodes = {}                                # key -> dict(actions=[(kind, id, data)], out={i: key})
        self.start = None
        self.prev = None                               # (key, action index, grid, was a reset) of the last step
        self.acted = 0                                 # actions in this level since its start or the last RESET

    def key(self, grid):
        g = grid.copy()
        g[self.ticker.mask] = 0
        return hashlib.blake2b(g.tobytes(), digest_size=12).digest()

    def _node(self, k, grid, available):
        if k in self.nodes:
            return self.nodes[k]
        acts = [(("key", a), a, None) for a in available if a not in (RESET, CLICK)]
        if CLICK in available:
            comps = components(grid)
            bg = max(comps, key=lambda c: len(c[1]))[0]
            for col, cells in comps:
                if col == bg and len(cells) > 64:
                    continue
                if self.ticker.mask[cells[0]]:
                    continue
                ys, xs = zip(*cells)
                cy, cx = np.mean(ys), np.mean(xs)
                y, x = min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
                acts.append((("click", col, size_bucket(len(cells))), CLICK, {"x": int(x), "y": int(y)}))
        node = dict(actions=acts, out={})
        self.nodes[k] = node
        return node

    def p_effect(self, kind):
        e, n = self.effect.get(kind, (0, 0))
        return (e + 1) / (n + 2)

    # ---- acting ---------------------------------------------------------------------------------------------------
    def act(self, fd):
        state = getattr(fd.state, "name", str(fd.state))
        # a copy: the local engine reuses its frame array, so a kept reference would become the next frame
        grid = np.array(fd.frame[-1], dtype=np.int8, copy=True) if len(fd.frame) else None
        if state in ("NOT_PLAYED", "GAME_OVER") or grid is None:
            self.prev, self.acted = None, 0
            self.notes["reset (game over)" if state == "GAME_OVER" else "reset (start)"] += 1
            return RESET, None
        if self.level != fd.levels_completed:
            self.level = fd.levels_completed
            self._new_level(grid)
        if self.prev is not None and not self.prev[3]:  # learn from the last step
            if self.ticker.observe(self.prev[2], grid):
                self.nodes, self.prev, self.start = {}, None, None
                self.notes["ticker found"] += 1
        k = self.key(grid)
        node = self._node(k, grid, fd.available_actions)
        if self.start is None or (self.prev is not None and self.prev[3]):
            self.start = k
        if self.prev is not None:
            pk, pi, pgrid, was_reset = self.prev
            if not was_reset and pk in self.nodes:
                self.nodes[pk]["out"][pi] = k
                kind = self.nodes[pk]["actions"][pi][0]
                e, n = self.effect.get(kind, (0, 0))
                self.effect[kind] = (e + (k != pk), n + 1)
        i = self._choose(k)
        if i is None and (k == self.start or not self.acted):   # nothing left anywhere: act at random
            self.notes["random (exhausted)"] += 1
            i = int(np.random.randint(len(node["actions"]))) if node["actions"] else None
        if i is None:                                  # walk back to the start
            self.prev, self.acted = (k, None, grid, True), 0
            self.notes["reset (exhausted)"] += 1
            return RESET, None
        _, a, data = node["actions"][i]
        self.prev = (k, i, grid, False)
        self.acted += 1
        return a, data

    def _choose(self, k):
        """The best untried action here; else the first step towards the nearest state with one; else None (RESET)."""
        def best_untried(key):
            node = self.nodes[key]
            cands = [(self.p_effect(node["actions"][i][0]), -i, i) for i in range(len(node["actions"])) if i not in node["out"]]
            return max(cands)[2] if cands else None

        i = best_untried(k)
        if i is not None:
            return i
        prev, q = {k: None}, deque([k])                # breadth-first over the known graph
        while q:
            cur = q.popleft()
            if cur != k and best_untried(cur) is not None:
                step = cur
                while prev[step][0] != k:
                    step = prev[step][0]
                return prev[step][1]
            for j, nxt in self.nodes[cur]["out"].items():
                if nxt not in prev and nxt in self.nodes:
                    prev[nxt] = (cur, j)
                    q.append(nxt)
        return None
