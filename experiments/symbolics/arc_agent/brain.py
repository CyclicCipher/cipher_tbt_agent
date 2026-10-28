"""The brain (ARCHITECTURE.md sections 2 and 5): built from a blueprint; one perception-and-learning step per action.

Areas: V1 (innate retina: the scene, the ground colour, and a Spotlight lattice of what lies under things), M1 (innate
motor map: keys, and click proposals aligned with V1, ranked by what has responded), assoc (learned columns, each a
model over its own frame), HC (episodes in a Spotlight lattice addressed by the joint state).

One step: every assoc column is looked for where its model says it went; what each action did to it is learned, in
the context the column read in its own frame before the action (part 20: conditions by attention, no rule says what
blocks); the change no column explains is split into sites, and frame types compete to explain each (translation by a
Hough vote over displacements, recolour by a consistent colour map, bar by extending a counter); a free slot takes the
winner.
"""
import math
from collections import Counter

import numpy as np

from arc_agent.blueprint import DEFAULT
from arc_agent.column import PMIN, Column
from arc_agent.frames import H, W, Bar, Recolour, Translation
from arc_agent.hidden import Hidden
from arc_agent.pfc import PFC
from arc_agent.spotlight import Spotlight

RADIUS = 16                       # the largest displacement looked for in one step
FRAC = 0.75                       # a column is still itself when this share of its cells match (structure persists)
MAX_SITES = 24                    # more change sites than this in one step is a scene change, not motion


def components(mask, conn8=True):
    seen = np.zeros(mask.shape, bool)
    out = []
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if conn8 else [])
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        comp, q = [], [(y, x)]
        seen[y, x] = True
        while q:
            cy, cx = q.pop()
            comp.append((cx, cy))
            for dy, dx in nb:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(comp)
    return out


def regions(grid):
    """4-connected same-colour regions: [(colour, [(x, y)])]."""
    out = []
    seen = np.zeros(grid.shape, bool)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if seen[y, x]:
                continue
            c, comp, q = grid[y, x], [], [(y, x)]
            seen[y, x] = True
            while q:
                cy, cx = q.pop()
                comp.append((cx, cy))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and not seen[ny, nx] and grid[ny, nx] == c:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append((int(c), comp))
    return out


def region_size(grid, x, y, cap=200):
    """How many cells of one colour are connected to (x, y), up to cap."""
    c, seen, q = grid[y, x], {(x, y)}, [(x, y)]
    while q and len(seen) < cap:
        px, py = q.pop()
        for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
            if (nx, ny) not in seen and 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] == c:
                seen.add((nx, ny))
                q.append((nx, ny))
    return len(seen)


def size_bucket(n):
    return 0 if n <= 2 else 1 if n <= 9 else 2 if n <= 36 else 3 if n <= 144 else 4


class LazyContext:
    """A column's context computed on demand (for planning): the same features as Brain.context, but each is read only
    when an expert asks for it. Conjunction features ("and", f, v, g) are present when f's value is v."""

    def __init__(self, brain, col, state, scene, others, under):
        self.brain = brain
        self.col, self.state, self.scene, self.others, self.under = col, state, scene, others, under
        self.window = brain.window_set(col)
        self.cells = col.frame.place(state)
        self.anchor = self.cells[0]
        self.own, self.vals = None, {}

    def __contains__(self, f):
        if f[0] == "at":
            return (f[1], f[2]) in self.window
        if f[0] == "self":
            return True
        if f[0] == "col":
            if f[1] not in self.others or f[1] == self.col.cid:
                return False
            h = self.brain.hidden
            return h.has(f) if (h is not None and h.kind(f[1]) is not None) else len(f) == 2
        if f[0] == "and":
            return f[1] in self and f[3] in self and self.get(f[1]) == f[2]
        if f[0] in ("mod", "num"):
            return self.col.frame.kind == "translation"
        return False

    def get(self, f, default=None):
        if f is None:                                   # the unconditional expert reads nothing
            return default
        if f in self.vals:
            return self.vals[f]
        if f[0] == "at":
            x, y = int(self.anchor[0]) + f[1], int(self.anchor[1]) + f[2]
            if not (0 <= x < W and 0 <= y < H):
                v = -1
            else:
                if self.own is None:
                    self.own = set(map(tuple, self.cells.tolist()))
                v = self.under(x, y) if (x, y) in self.own else int(self.scene[y, x])
        elif f[0] == "self":
            v = self.state
        elif f[0] == "col":
            h = self.brain.hidden
            if h is not None and h.kind(f[1]) is not None:    # a hidden-state column, by one of its readings
                v = h.reading(f[1], self.others[f[1]], f[2:]) if f[1] in self.others else default
            else:
                v = self.others.get(f[1], default)
        elif f[0] == "and":
            v = self.get(f[3], default)
        elif f[0] == "mod":
            v = self.brain.quotients(self.col, self.state)[f]
        elif f[0] == "num":
            v = self.brain.numeric(self.col, self.state, self.scene, self.under)
        else:
            v = default
        self.vals[f] = v
        return v


class V1:
    def __init__(self):
        self.scene, self.ground = None, 0
        self.under = Spotlight()                        # (x, y) -> "under": the colour beneath a column

    def start(self, grid, new_level):
        self.scene = grid.copy()
        if new_level:
            v, n = np.unique(grid, return_counts=True)
            self.ground = int(v[np.argmax(n)])
            self.under.forget()

    def under_at(self, x, y):
        return self.under.value((x, y), "under", self.ground)


class M1:
    """Click proposals: one point per V1 region (a first guess, demoted by evidence), and P(an action of this kind
    changes anything), Beta(1, 1), for keys and for clicks by the clicked region's signature (colour, size)."""

    def __init__(self):
        self.effect = {}
        self.done = set()                               # click addresses tried in this level

    def tried(self, action):
        return action[0] == 6 and (action[1], action[2]) in self.done

    def p_effect(self, kind):
        e, n = self.effect.get(kind, (0, 0))
        return (e + 1) / (n + 2)

    def record(self, kind, effect):
        e, n = self.effect.get(kind, (0, 0))
        self.effect[kind] = (e + bool(effect), n + 1)

    def proposals(self, grid, ground, claimed):
        out = []
        for c, cells in regions(grid):
            if c == ground and len(cells) > 64:
                continue
            if all(claimed[y, x] for x, y in cells):
                continue
            xs, ys = zip(*cells)
            cx, cy = np.mean(xs), np.mean(ys)
            x, y = min(cells, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            out.append((int(x), int(y), ("click", c, size_bucket(len(cells)))))
        return out


class Brain:
    def __init__(self, blueprint=DEFAULT, conditional=True, kind_prior=None, library=None, pfc=None, hidden=True):
        self.bp, self.conditional = blueprint, conditional
        self.pfc = pfc if pfc is not None else PFC()    # part 24: the prefrontal column, shared with whoever passed it
        self.hidden = Hidden(conditional, kind_prior) if hidden else None   # part 27: columns not on the screen
        self.level, self.last_act = 0, "key"
        self.kind_prior, self.library = kind_prior or {}, library or []
        self.slots = blueprint.area("assoc").columns
        self.v1, self.m1, self.hc = V1(), M1(), Spotlight()
        self.columns, self.next_id, self.t = [], 0, 0
        self.last_kind = None
        self.cache = {}                                 # planning: contexts per joint state
        self.last_evidence, self.last_cells = {}, {}     # part 24: the last step's same-cell evidence per candidate
        self.hc_n, self.hc_calls, self.hc_hits = {}, 0, 0   # part 28: writes per hippocampal cell; planner recalls                           # M1's kind of the last action

    # ---- the joint state and the columns used for planning ------------------------------------------------------
    def active(self):
        return [c for c in self.columns if c.active]

    def planning(self):
        """The columns planned with: action-driven, not a counter, and, if they compete with others to explain the same
        change, the one the PFC holds responsible (part 24)."""
        return [c for c in self.columns if c.active and c.frame.kind != "bar" and c.driver() == "action"
                and self.pfc.responsible(c.cid, self.last_act)]

    def clocks(self):
        return [c for c in self.columns if c.active and (c.frame.kind == "bar" or c.driver() == "clock")]

    def joint(self):
        """The state the planner searches: the planning columns', and the hidden-state columns' that some planned
        column relies on (part 28)."""
        out = [(c.cid, c.state) for c in self.planning()]
        if self.hidden is not None:
            hs = self.hidden.states()
            out += [(h, hs[h]) for h in sorted(self.relevant_hidden())]
        return tuple(out)

    def relevant_hidden(self):
        """Hidden-state columns that some planned column's action model leads with (its situation head) or reads (its
        attended key); a delay brings the delays before it, which it copies."""
        out = set()
        for c in self.planning():
            for b in c.keys.values():
                if not b.conditional:
                    continue
                for f in (b.situation(), b.attended()):
                    if f is not None and f[0] == "col" and self.hidden.kind(f[1]) is not None:
                        out.add(f[1])
        for d in list(out):
            if self.hidden.kind(d).startswith("delay"):
                out |= set(range(d, -10))
        return out

    def others(self, exclude=None, states=None):
        """The other columns' states a column reads as context: the planning columns' (or the given joint states), and
        the hidden-state columns' (part 27); in a given joint state, its own hidden states take precedence."""
        out = dict(self.hidden.states()) if self.hidden is not None else {}
        if states is None:
            out.update({c.cid: c.state for c in self.planning() if c is not exclude})
        else:
            out.update(states)
        return out

    def col_features(self, cid, state):
        """What another column's state gives a context: ("col", cid), or a hidden-state column's readings."""
        if self.hidden is not None and self.hidden.kind(cid) is not None:
            return self.hidden.features(cid, state)
        return {("col", cid): state}

    def event(self, action, scene=None):
        """An action as an event: a key, or a click named by its target (as M1 names it)."""
        aid, x, y = action
        if aid != 6:
            return ("key", aid)
        g = self.v1.scene if scene is None else scene
        return ("click", int(g[y, x]), size_bucket(region_size(g, x, y)))

    def claimed(self):
        m = np.zeros((H, W), bool)
        for c in self.clocks():
            p = c.frame.place(c.state)
            ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
            m[p[ok, 1], p[ok, 0]] = True
        return m

    def modelled(self, action):
        """Some planning column has seen this action (so its effect is the columns' to predict, not M1's)."""
        return any(c.belief(c.akey(action)).n > 0 for c in self.planning())

    # ---- context, read by a column in its own frame (part 20) ----------------------------------------------------
    def window(self, col):
        """Positions around the column, relative to its anchor (its first cell), out to its reach."""
        key = (len(col.frame.cells0), col.reach)
        if getattr(col, "_window", (None,))[0] != key:
            rel = col.frame.cells0 - col.frame.cells0[0]
            (x0, y0), (x1, y1) = rel.min(0), rel.max(0)
            m = col.reach
            area = (x1 - x0 + 2 * m + 1) * (y1 - y0 + 2 * m + 1)
            step = max(1, int(math.ceil(math.sqrt(area / 400))))
            xs, ys = np.arange(x0 - m, x1 + m + 1, step), np.arange(y0 - m, y1 + m + 1, step)
            col._window = (key, np.array([(x, y) for y in ys for x in xs], int))
        return col._window[1]

    def window_set(self, col):
        offs = self.window(col)
        if getattr(col, "_window_set", (None,))[0] is not col._window:
            col._window_set = (col._window, set(map(tuple, offs.tolist())))
        return col._window_set[1]

    def context(self, col, state, scene, others, under):
        """The column's features: ("at", dx, dy) the content at that position relative to its anchor (beneath the
        column: what lies under it; off the screen: -1); ("self",) its own state; ("col", cid) another column's
        state."""
        offs = self.window(col)
        cells = col.frame.place(state)
        pts = offs + cells[0]
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        vals = np.full(len(pts), -1, int)
        vals[ok] = scene[pts[ok, 1], pts[ok, 0]]
        own = set(map(tuple, cells.tolist()))
        for i, (x, y) in enumerate(pts.tolist()):
            if ok[i] and (x, y) in own:
                vals[i] = under(x, y)
        ctx = {("at", int(dx), int(dy)): int(v) for (dx, dy), v in zip(offs.tolist(), vals.tolist())}
        ctx[("self",)] = state
        for cid, st in others.items():
            if cid != col.cid:
                ctx.update(self.col_features(cid, st))
        if col.frame.kind == "translation":
            ctx.update(self.quotients(col, state))
            ctx[("num",)] = self.numeric(col, state, scene, under)
        return ctx

    def quotients(self, col, state):
        """Part 10: content regularity is a quotient of the geometry. The column's position modulo k, in units of its
        own moves."""
        u = col.unit or 1
        return {("mod", k): ((state[0] // u) % k, (state[1] // u) % k) for k in range(2, 7)}

    def numeric(self, col, state, scene, under):
        """The numeric features (part 23): along each of four rays from the column's middle, the distance to the
        first cell unlike what lies beneath the column (or to the screen's edge); then its own position."""
        cells = col.frame.place(state)
        (x0, y0), (x1, y1) = cells.min(0), cells.max(0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        base = under(int(cells[0][0]), int(cells[0][1])) if 0 <= cells[0][0] < W and 0 <= cells[0][1] < H else -1
        out = []
        for (dx, dy), (sx, sy) in (((0, -1), (cx, y0)), ((0, 1), (cx, y1)), ((-1, 0), (x0, cy)), ((1, 0), (x1, cy))):
            d, x, y = 0, sx + dx, sy + dy
            while 0 <= x < W and 0 <= y < H and scene[y, x] == base:
                d, x, y = d + 1, x + dx, y + dy
            out.append(d)
        return tuple(out) + (int(state[0]), int(state[1]))

    # ---- a new level, or the same level restarted ---------------------------------------------------------------
    def start(self, grid, new_level):
        self.v1.start(grid, new_level)
        if self.hidden is not None:
            self.hidden.start(new_level)
        if new_level:
            self.m1.done.clear()
            self.level += 1
        for c in self.columns:
            f = c.frame
            if f.kind == "translation":
                hits = f.find_all(grid)
                c.active = bool(hits)
                if hits:
                    c.state = min(hits, key=lambda s: abs(s[0] - c.state[0]) + abs(s[1] - c.state[1]))
            elif f.kind == "recolour":
                c.state, c.active = f.read(grid), True
            else:
                c.state = f.read(grid)

    # ---- one step ---------------------------------------------------------------------------------------------
    def observe(self, before, action, after, kind=None):
        """Learn from one transition. Returns a report: which columns moved, how many cells stayed unexplained, and
        which columns were recruited."""
        self.t += 1
        j0 = self.joint()                               # part 28: for the hippocampus
        self.last_kind = kind
        self.last_act = "key" if action[0] != 6 else "click"
        ground = self.v1.ground
        report = dict(moved=[], lost=[], recruited=[], unexplained=0)
        render = before.copy()
        others = self.others()
        learned = []                                    # part 27: what each visible column saw, for the hidden-state area
        planned = set(others)                           # (only the models in use: the PFC's rivals are not evidence)
        ctxs = {c.cid: self.context(c, c.state, before, others, self.v1.under_at) for c in self.active()}
        painted, claimed = {}, {}                       # part 24: each grouped column's prediction of its cells
        for c in self.active():
            if c.cid in self.pfc.member:
                painted[c.cid], claimed[c.cid] = self._paint(c, action, ctxs[c.cid])
        for c in self.active():
            f, old = c.frame, c.state
            akey = c.akey(action, old)
            ctx = ctxs[c.cid]
            if f.kind == "translation":
                p, _ = c.dist(akey, ctx)
                guess = f.apply(old, max(p, key=p.get))
                new = guess if f.matches(after, guess) else f.find(after, old, RADIUS)
                if new is None:                         # part of its content changed: the best partial match nearby
                    new = guess if f.score(after, guess) >= FRAC else f.find_near(after, old, RADIUS, FRAC)
                    if new is not None:                 # rewrite the changed cells in its content memory
                        p = f.place(new)
                        seen = after[p[:, 1], p[:, 0]]
                        for i in np.nonzero(seen != f.colours)[0]:
                            f.colours[i] = seen[i]
                            c.content.write(int(i), "colour", int(seen[i]))
                if new is None:                         # not nearby: anywhere on the screen
                    hits = f.find_all(after)
                    new = min(hits, key=lambda h: abs(h[0] - old[0]) + abs(h[1] - old[1])) if hits else None
                if new is None:
                    c.active = False
                    report["lost"].append(c.cid)
                    continue
                op = f.op_between(old, new)
                c.learn(akey, ctx, op, self.t)
                learned.append(((c.cid, akey), c.belief(akey)))
                cells_old = f.place(old)
                selfset = set(map(tuple, cells_old.tolist()))
                for x, y in cells_old.tolist():
                    if 0 <= x < W and 0 <= y < H:
                        render[y, x] = after[y, x]       # vacated cells show what was beneath
                newset = set(map(tuple, f.place(new).tolist()))
                for x, y in newset - selfset:            # newly covered: remember what was beneath (permanence)
                    if 0 <= x < W and 0 <= y < H:
                        self.v1.under.write((x, y), "under", int(before[y, x]))
                self.v1.under.forget(selfset - newset)
                f.draw(render, new)
                c.state = new
            else:
                new = f.read(after)
                op = f.op_between(old, new)
                if op is None:
                    c.active = False
                    report["lost"].append(c.cid)
                    continue
                c.learn(akey, ctx, op, self.t)
                learned.append(((c.cid, akey), c.belief(akey)))
                f.draw(render, new)
                c.state = new
            if op != f.identity:
                report["moved"].append(c.cid)
        if self.hidden is not None and self.hidden.observe(self.t, action[0], kind if kind is not None
                                                           else ("key", action[0]),
                                                           [(m, b) for m, b in learned
                                                            if b.conditional and b.T and m[0] in planned]):
            self.restate()
        residual = render != after
        report["unexplained"] = int(residual.sum())
        self.pfc.observe(self._group_evidence(painted, claimed, before, after), self.last_act)   # part 24
        if residual.any():
            self._recruit(before, after, residual, action, report)
        # M1: did this action change anything beyond the clocks?
        clockset = set(c.cid for c in self.clocks())
        effect = bool([m for m in report["moved"] if m not in clockset]) or bool(report["recruited"])
        if self.last_kind is not None:
            self.m1.record(self.last_kind, effect)
        if action[0] == 6:
            self.m1.done.add((action[1], action[2]))
        self.v1.scene = after.copy()
        addr = (self.level, tuple(sorted(j0)))          # part 28: the hippocampus remembers the transition; the cell
        n = self.hc_n.get((addr, action), 0)            # is written with step 1/(n+1): the outcomes' frequencies
        self.hc.write(addr, action, tuple(sorted(self.joint())), beta=1.0 / (n + 1))
        self.hc_n[(addr, action)] = n + 1
        return report

    def restate(self):
        """After the hidden graph and latent are refitted: rewrite their states in every stored episode that reads them,
        and forget the hippocampal transitions that name them (their labels may mean something else now)."""
        for hid, value_at in self.hidden.restated():
            for c in self.columns + self.hidden.columns():
                for b in c.keys.values():
                    b.restate(("self",) if c.cid == hid else ("col", hid), value_at)
        gone = {hid for hid, _ in self.hidden.restated()}
        stale = [a for a in self.hc.cells if any(cid in gone for cid, _ in a[1])]
        self.hc.forget(stale)
        for k in [k for k in self.hc_n if k[0] in set(stale)]:
            del self.hc_n[k]

    # ---- recruitment: frame types compete to explain what is left -----------------------------------------------
    def _new(self, frame, state):
        if len([c for c in self.columns if c.active]) >= self.slots:
            idle = [c for c in self.columns if not c.active]
            if not idle:
                return None
            self.columns.remove(min(idle, key=lambda c: c.uses))
        c = Column(self.next_id, frame, state, self.t, self.conditional, self.kind_prior, self.library)
        self.next_id += 1
        self.columns.append(c)
        return c

    def _recruit(self, before, after, residual, action, report):
        ground = self.v1.ground
        sites = components(residual)
        if len(sites) > MAX_SITES:
            return
        original = residual.copy()
        rigid = []                                      # (column, the cells it explains)
        for _ in range(3):                              # rigid explanations: common fate, translation or turn
            found = self._translation(before, after, residual, ground)
            turned = self._rotation(before, after, residual, ground)
            if turned is not None and (found is None or turned[2] > self._explained(found[0], found[1], residual)):
                found = turned[:2]
            if found is None:
                break
            cells, d = found
            d = tuple(d) + (0,) * (3 - len(d))
            frame = Translation(cells, [before[y, x] for x, y in cells])
            col = self._new(frame, (0, 0, 0))
            if col is None:
                return
            akey = col.akey(action, (0, 0, 0))
            others = self.others(col)
            col.learn(akey, self.context(col, (0, 0, 0), before, others, self.v1.under_at), d)   # evidence (part 15)
            col.state = d
            covered = set(map(tuple, cells))
            for x, y in cells:
                residual[y, x] = False
            for x, y in frame.place(d).tolist():
                if 0 <= x < W and 0 <= y < H:
                    residual[y, x] = False
                    covered.add((x, y))
                    if (x, y) not in set(map(tuple, cells)):
                        self.v1.under.write((x, y), "under", int(before[y, x]))
            rigid.append((col, covered))
            report["recruited"].append(col.cid)
        # every change with one colour map is also explained as a recolour (part 24: no fixed order between frames).
        # A rigid explanation's rival is ONE recolour over all the changed cells it covers (both strips of a toggle,
        # not each strip alone), so the two are compared on the same change; the PFC decides between them.
        sites = components(original)
        used = set()
        for r, cov in rigid:
            idx = [k for k, site in enumerate(sites) if cov & set(site)]
            if not idx:
                continue
            used |= set(idx)
            union = [cell for k in idx for cell in sites[k]]
            col = self._recolour(union, before, after, action, report)
            if col is not None:
                region = set(map(tuple, r.frame.cells0.tolist())) | set(union)
                self.pfc.group([r.cid, col.cid], [r.frame.kind, "recolour"], self.region_context(region, before, after))
        for k, site in enumerate(sites):                # the rest: a recolour, or one more cell of a counter
            if k in used:
                continue
            pairs = [(int(before[y, x]), int(after[y, x])) for x, y in site]
            m = {}
            if any(m.setdefault(a, b) != b for a, b in pairs) or self._extend_bar(site, pairs):
                continue
            self._recolour(site, before, after, action, report)

    def _recolour(self, cells, before, after, action, report):
        """A recolour column over these cells, if one colour map explains their change; else None."""
        pairs = [(int(before[y, x]), int(after[y, x])) for x, y in cells]
        m = {}
        if any(m.setdefault(a, b) != b for a, b in pairs):
            return None
        frame = Recolour(cells, [a for a, _ in pairs])
        col = self._new(frame, tuple(a for a, _ in pairs))
        if col is None:
            return None
        op = frame.op_between(col.state, tuple(b for _, b in pairs))
        others = self.others(col)
        col.learn(col.akey(action, col.state), self.context(col, col.state, before, others, self.v1.under_at), op)
        col.state = tuple(b for _, b in pairs)
        report["recruited"].append(col.cid)
        return col

    def _paint(self, c, action, ctx):
        """What column c predicts for the cells it claims this step: {cell: colour}, and the cells it claims (where it
        is now and where it predicts it will be)."""
        f, s = c.frame, c.state
        p, _ = c.dist(c.akey(action, s), ctx)
        ns = f.apply(s, max(p, key=p.get))
        out = {}
        if f.kind == "translation":
            for x, y in f.place(s).tolist():
                out[(x, y)] = self.v1.under_at(x, y)     # what it leaves shows what lay beneath
            for (x, y), v in zip(f.place(ns).tolist(), f.colours.tolist()):
                out[(x, y)] = int(v)
        elif f.kind == "recolour":
            for (x, y), v in zip(f.place(ns).tolist(), ns):
                out[(x, y)] = int(v)
        return out, set(out)

    def _group_evidence(self, painted, claimed, before, after):
        """Part 24: the rivals in a group are scored on the SAME cells, every cell any of them claims. Each predicts
        the cells it doesn't model as unchanged. Returns {cid: log predictive probability of what was seen}."""
        lp, hit, miss = {}, math.log(0.99), math.log(0.01)
        self.last_cells = {}
        for g in self.pfc.groups.values():
            members = [c for c in g["members"] if c in painted]
            cells = set()
            for c in members:
                cells |= claimed[c]
            cells = [(x, y) for x, y in cells if 0 <= x < W and 0 <= y < H]
            for c in members:
                n = sum(1 for x, y in cells if painted[c].get((x, y), int(before[y, x])) == int(after[y, x]))
                lp[c] = n * hit + (len(cells) - n) * miss
                self.last_cells[c] = len(cells)
        self.last_evidence = lp
        return lp

    def region_context(self, region, before, after):
        """The PFC's game-agnostic context for a contested region (part 24)."""
        xs, ys = zip(*region)
        colours = {int(before[y, x]) for x, y in region} | {int(after[y, x]) for x, y in region}
        return {("size",): size_bucket(len(region)), ("colours",): min(4, len(colours)),
                ("thin",): min(max(xs) - min(xs), max(ys) - min(ys)) <= 2, ("level",): min(self.level, 2)}

    @staticmethod
    def _explained(cells, d, residual):
        f = Translation(cells, [0] * len(cells))
        d = tuple(d) + (0,) * (3 - len(d))
        pts = set(map(tuple, f.place((0, 0, 0)).tolist())) | set(map(tuple, f.place(d).tolist()))
        return sum(1 for x, y in pts if 0 <= x < W and 0 <= y < H and residual[y, x])

    def _rotation(self, before, after, residual, ground):
        """Part 12: a set of cells turned about its own pivot (and perhaps moved). Candidate sets: the connected
        non-ground regions of the old screen that the change touches. Returns (cells, (dx, dy, k), explained)."""
        mask = before != ground
        best = None
        for comp in components(mask):
            if not 3 <= len(comp) <= 300 or not any(residual[y, x] for x, y in comp):
                continue
            f = Translation(comp, [before[y, x] for x, y in comp])
            for k in (1, 2, 3):
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        s = (dx, dy, k)
                        if f.matches(after, s):
                            e = self._explained(comp, s, residual)
                            if best is None or e > best[2]:
                                best = (comp, s, e)
        return best

    def harvest(self):
        """What the columns learned that carries to other games (part 23): the kinds of their attended experts, and
        the formulas the numeric experts fitted."""
        kinds, formulas = [], []
        for c in self.columns:
            kinds += c.harvest()
            for b in c.keys.values():
                if b.numeric is not None:
                    formulas += [(r, len(b.numeric.X)) for r in b.numeric.formulas()]
        return kinds, formulas

    def _extend_bar(self, site, pairs):
        """A small change with one colour map next to a column that just made the same change: a counter's next cell.
        A one-site recolour column becomes a bar; a bar grows."""
        if len(site) > 16 or len({p for p in pairs}) != 1:
            return False
        a, b = pairs[0]
        for c in self.active():
            f = c.frame
            if f.kind == "bar" and (f.empty, f.full) == (a, b):
                near = f.cells0
            elif f.kind == "recolour" and len(set(f.colours.tolist())) == 1 and int(f.colours[0]) == a and \
                    c.state and all(v == b for v in c.state):
                near = f.cells0
            else:
                continue
            if min(max(abs(x - nx), abs(y - ny)) for x, y in site for nx, ny in near.tolist()) > 3:
                continue
            if f.kind == "recolour":                      # becomes a bar: its cells, then the new ones
                bar = Bar(list(map(tuple, f.cells0.tolist())) + list(site), a, b)
                c.frame, c.state = bar, len(f.cells0) + len(site)
                c.keys, c.ev = {}, {"action": 0.0, "clock": 0.0}
                c.clock = type(c.clock)(bar.K, bar.identity, 0.5)
            else:
                for cell in site:
                    f.extend(cell)
                c.state += len(site)
            return True
        return False

    def _translation(self, before, after, residual, ground):
        """The displacement most changed cells agree on (old colour at p, the same colour at p + d), and the rigid set
        that moved by it: grown from the changed cells through same-coloured neighbours that also moved by d, minus
        unchanged cells nothing covers after the move."""
        ys, xs = np.nonzero(residual)
        old = [(x, y, before[y, x]) for x, y in zip(xs, ys) if before[y, x] != ground]
        new = {}
        for x, y in zip(xs, ys):
            if after[y, x] != ground:
                new.setdefault(int(after[y, x]), []).append((x, y))
        if not old or not new:
            return None
        votes, voters = Counter(), {}
        by_colour = {}
        for x, y, c in old:
            by_colour.setdefault(int(c), []).append((x, y))
        for c, pts in by_colour.items():
            if c not in new:
                continue
            a = np.array(pts)
            b = np.array(new[c])
            d = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
            d = d[(np.abs(d[:, 0]) <= RADIUS) & (np.abs(d[:, 1]) <= RADIUS) & ((d[:, 0] != 0) | (d[:, 1] != 0))]
            for k, n in zip(*np.unique(d, axis=0, return_counts=True)):
                votes[(int(k[0]), int(k[1]))] += int(n)
                voters.setdefault((int(k[0]), int(k[1])), set()).add(c)
        if not votes:
            return None
        best = None
        for d, v in sorted(votes.items(), key=lambda kv: (-kv[1], abs(kv[0][0]) + abs(kv[0][1])))[:3]:
            if v < 2:
                break
            M = self._moved_set(before, after, residual, ground, [o for o in old if int(o[2]) in voters[d]], d)
            if not M:
                continue
            moved_to = {(x + d[0], y + d[1]) for x, y in M}
            explained = sum(1 for x, y in M | moved_to if 0 <= x < W and 0 <= y < H and residual[y, x])
            # figure and ground: a boundary drawn through one colour costs (a uniform region's motion is not
            # identifiable), so "a floor patch moved" loses to "the thing on the floor moved"
            cut = sum(1 for x, y in M for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                      if (nx, ny) not in M and 0 <= nx < W and 0 <= ny < H and before[ny, nx] == before[y, x])
            score = explained - cut
            if best is None or score > best[0]:
                best = (score, M, d)
        if best is None or best[0] <= 0:
            return None
        cells = sorted(best[1], key=lambda p: (p[1], p[0]))
        return cells, best[2]

    def _moved_set(self, before, after, residual, ground, old, d):
        """The rigid set that moved by d: grown from the changed cells through same-coloured neighbours that also
        moved by d, minus unchanged cells that nothing covers after the move."""
        def moved(x, y):
            nx, ny = x + d[0], y + d[1]
            return 0 <= nx < W and 0 <= ny < H and before[y, x] != ground and after[ny, nx] == before[y, x]

        seed = [(x, y) for x, y, _ in old if moved(x, y)]
        cols = {int(before[y, x]) for x, y in seed}
        M, q = set(seed), list(seed)
        while q and len(M) < 600:
            x, y = q.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) not in M and 0 <= nx < W and 0 <= ny < H and int(before[ny, nx]) in cols and moved(nx, ny):
                    M.add((nx, ny))
                    q.append((nx, ny))
        changed = True
        while changed:                                  # an unchanged cell must be covered by the moved set
            changed = False
            for x, y in list(M):
                if not residual[y, x] and before[y, x] == after[y, x]:
                    src = (x - d[0], y - d[1])
                    if src not in M or before[src[1], src[0]] != after[y, x]:
                        M.discard((x, y))
                        changed = True
        return M

    # ---- the model, for planning ------------------------------------------------------------------------------
    def static_scene(self):
        """The scene with every planning column taken out: what lies beneath them, or the ground."""
        g = self.v1.scene.copy()
        for c in self.planning():
            for x, y in c.frame.place(c.state).tolist():
                if 0 <= x < W and 0 <= y < H:
                    g[y, x] = self.v1.under_at(x, y)
        return g

    def model_step(self, states, action, static):
        """Predicted next states of the planning columns ({cid: state}) and the bits an uncertain outcome is worth
        (part 18: the entropy of the predicted OUTCOME). bits > 0 means the outcome here is not known."""
        key = tuple(sorted(states.items()))
        self.hc_calls += 1                              # part 28: recall where every visit here agreed
        w = self.hc.read((self.level, key), action)
        if w:
            v, q = max(w.items(), key=lambda kv: kv[1])
            if q >= 1 - PMIN and len(v) == len(states) and all(cid in states for cid, _ in v):
                self.hc_hits += 1
                return dict(v), 0.0
        cols = {c.cid: c for c in self.planning()}
        if key not in self.cache:                       # the scene and every column's context at this joint state
            scene = static.copy()
            for cid, st in states.items():
                if cid in cols:
                    cols[cid].frame.draw(scene, st)
            others = self.others(states=states)
            self.cache[key] = {cid: LazyContext(self, cols[cid], st, scene, others, lambda x, y: int(static[y, x]))
                               for cid, st in states.items() if cid in cols}
            self.cache[("others", key)] = others
        ctxs = self.cache[key]
        new, bits = dict(states), 0.0
        for cid, s in states.items():
            if cid not in cols:                         # part 28: a hidden-state column, by its own learned model
                ns, b = self.hidden.predict(cid, s, self.cache[("others", key)], action[0], self.event(action))
                if b > 0:
                    bits = max(bits, b)
                else:
                    new[cid] = ns
                continue
            c = cols[cid]
            akey = c.akey(action, s)
            if akey[0] == "at":                         # a click elsewhere: unconditional, so cached per search
                ukey = ("u", cid, akey)
                if ukey not in self.cache:
                    self.cache[ukey] = c.dist(akey, {})[0]
                p = self.cache[ukey]
                if p.get(c.frame.identity, 0.0) >= 1 - PMIN:
                    continue
            else:
                p, _ = c.dist(akey, ctxs[cid])
            outcomes = {}
            for op, q in p.items():
                ns = c.frame.apply(s, op)
                outcomes[ns] = outcomes.get(ns, 0.0) + q
            rest = max(0.0, 1.0 - sum(p.values()))
            if rest > 1e-9:
                outcomes[("unknown", cid)] = rest       # an operator never seen: an outcome not yet imagined
            ns, q = max(outcomes.items(), key=lambda kv: kv[1])
            if q < 1 - PMIN:
                bits = max(bits, min(1.0, -sum(x * math.log2(x) for x in outcomes.values() if x > 0)))
                continue
            new[cid] = ns
        return new, bits
