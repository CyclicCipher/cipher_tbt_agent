"""Learning a frame from replayed episodes (ARCHITECTURE.md sections 0 and 4; docs/LAB.md parts 3 and 4).

**Input:** consolidation replays everything the hippocampus remembers.
- **The transitions** (place, moving input, next place) form a tree of the episodes.
- **The contents** say what came next at each place, by sensor, sensed inputs included.

**Output:** a Frame: places, the moves between them (seen and inferred), the content at each place, and coordinates
where the moves allow them.

**The rules: the column's innate prior, and nothing else.**
- **P3, same future, same place.**
  - **The strict form (part 3):** places whose observed futures are identical are one place: the same contents, and
    the same inputs leading to places that are again one place (the Myhill–Nerode test).
  - **The full form (part 4):** two places whose futures agree and never disagree are one place. They never
    disagree when, over every input sequence both have seen, their content never differs where both are determined
    (one input ever seen next). They agree on a content that identifies places: one that was never the determined
    content of two places shown to differ (amendment 2).
- **P4, regularity: a move is determined by where it takes one place.**
  - From any place u, going where input y leads instead of where x leads is a displacement d(x, y), the same at
    every place until refuted. So each ordered pair of inputs names one displacement, whose instances are collected
    from every place where both inputs were seen.
  - Two pairs whose displacements take some place to the same place are the same displacement, so their instances are
    pooled.
  - A pooled displacement fills in moves never seen: from u, input y leads to d(x, y) applied to where x leads.
  - A displacement must stay a one-to-one map. An instance that would send one place to two places (or two places to
    one) refutes its pair, which is then not used. A pooling that would do so is not made.
- **Coordinates (part 4; COORDINATES.md parts 1, 5 and 6).**
  - **The generators:** the moves and displacements that permute a set of places. The places they connect form
    orbits.
  - **The read-out "abelian":** on an orbit whose generators commute, the common eigenvectors of their permutation
    matrices are the group's characters. Normalized at the orbit's base (its first place met), each has an order n
    and a phase at every place.
  - **The module chain:** characters are kept in increasing order of n, each only if it splits places the kept ones
    leave together. This is the group's own chain of quotients (Z/105: 3, 5, 7; Z/81: 3, 9, 27, 81).
  - **The code:** each kept character becomes a module of a `Modules` code, and each generating input's move becomes
    one fixed shift of the code.
  - **Where it declines:** an orbit whose moves don't commute keeps its table, as does a frame whose moves permute
    nothing.

**Nothing here knows what an input is.** Inputs and places are SDRs, compared only for identity (they are replayed
exact copies). There are no values, no order, and no assumed cycle or commutativity: commutativity is read off the
moves.

**Implementation:** inside consolidation, places and inputs get integer row addresses (indices into the frame's move
table, in the order first met), so the rules run as array operations. The frame comes out as SDRs again.
- **P4's pooling** is connected components over the graph that links each pair of inputs to the place pairs its
  instances map.
- **Each component is checked for one-to-one.** A component with a conflict runs the sequential rule, instance by
  instance, so the result is the sequential algorithm's wherever the data is consistent.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from arc_agent_v2.HDClibrary import Modules


@dataclass
class Orbit:
    """Coordinates of one orbit: its module code, each place's code, and the shift of each input that permutes it."""
    modules: object
    codes: dict                         # frame place -> its code (an SDR of `modules`)
    shifts: dict                        # input -> the shift it makes (a code of `modules`)
    periods: tuple
    base: object                        # the place whose code is the identity
    size: int


@dataclass
class Frame:
    start: object                       # the frame's start place (an SDR)
    moves: list                         # [(place, input, next place)], seen and inferred
    content: list                       # [(place, sensor, input, times seen)]: what came next at each place
    places: dict                        # hippocampal place -> frame place
    inferred: int = 0                   # moves filled in by P4
    displacements: list = field(default_factory=list)   # each a {place: place} map, not refuted
    coords: list = field(default_factory=list)          # [Orbit]
    declined: list = field(default_factory=list)        # [(orbit size, why no coordinates)]


def consolidate(transitions, contents, start, regularity=True, coordinates=None):
    """A Frame from the hippocampus's replay: transitions [(place, input, next place, times seen)] and contents
    [(place, sensor, input, times seen)], of episodes that start at start. coordinates = (d, w, seed), or None."""
    places = same_future(transitions, contents)
    pid, P, xid, X = {}, [], {}, []

    def place(q):
        if q not in pid:
            pid[q] = len(P)
            P.append(q)
        return pid[q]

    def inp(x):
        if x not in xid:
            xid[x] = len(X)
            X.append(x)
        return xid[x]

    U, XS, V = [], [], []
    for h, x, nxt, _ in transitions:
        U.append(place(places[h]))
        XS.append(inp(x))
        V.append(place(places[nxt]))
    cont = defaultdict(lambda: defaultdict(Counter))
    for h, s, x, n in contents:
        cont[place(places.get(h, h))][s][inp(x)] += n
    move = np.full((len(P), len(X)), -1, np.int64)      # frame place, input -> frame place (-1: unknown)
    if U:
        move[U, XS] = V                                 # places merged by P3 agree on where each input leads
    cont = [dict(cont[u]) for u in range(len(P))]
    root = same_place(move, cont)                       # P3's full form
    if any(r != u for u, r in enumerate(root)):
        move, cont, P, places = _relabel(root, move, cont, P, places)
    inferred, maps = regular(move) if regularity and U else (0, [])
    us, xs = np.nonzero(move >= 0)
    moves = [(P[u], X[x], P[move[u, x]]) for u, x in zip(us.tolist(), xs.tolist())]
    content = [(P[u], s, X[x], int(n)) for u in range(len(P)) for s, c in cont[u].items() for x, n in c.items()]
    disp = [{P[a]: P[b] for a, b in m} for m in maps]
    coords, declined = coordinate(move, maps, P, X, coordinates) if coordinates and U else ([], [])
    return Frame(places.get(start, start), moves, content, places, inferred, disp, coords, declined)


def same_future(transitions, contents=()):
    """P3, strict: {hippocampal place: frame place}. Places with identical observed futures (the same contents, and the
    same inputs leading to places with identical futures) share one frame place, the first of them met."""
    children, order, seen = defaultdict(dict), [], set()
    sensed = defaultdict(set)
    for h, x, nxt, _ in transitions:
        children[h][x] = nxt
        for q in (h, nxt):
            if q not in seen:
                seen.add(q)
                order.append(q)
    for h, s, x, _ in contents:
        sensed[h].add((s, x))
        if h not in seen:
            seen.add(h)
            order.append(h)
    sig, intern = {}, {}
    for root in order:                                  # signatures bottom-up (iterative post-order)
        stack, on = [root], {root}
        while stack:
            h = stack[-1]
            pending = [c for c in children.get(h, {}).values() if c not in sig and c not in on]
            if pending:
                stack.extend(pending)
                on.update(pending)
                continue
            stack.pop()
            on.discard(h)
            if h not in sig:
                key = (frozenset((x, sig.get(c, -1)) for x, c in children.get(h, {}).items()),
                       frozenset(sensed.get(h, ())))
                sig[h] = intern.setdefault(key, len(intern))
    rep = {}
    return {h: rep.setdefault(sig[h], h) for h in order}


def _mismatch(keys, vals):
    """True where another entry has the same key but a different value."""
    if keys.size < 2:
        return np.zeros(keys.size, bool)
    o = np.lexsort((vals, keys))
    k, v = keys[o], vals[o]
    clash = k[1:][(k[1:] == k[:-1]) & (v[1:] != v[:-1])]
    return np.isin(keys, clash)


class Pool:
    """P4's bookkeeping over integer place ids. Each ordered pair of inputs (x, z), as the integer x * nX + z, belongs
    to a displacement class. A class holds its map in both directions (fwd, bwd: -1 where unknown) and is refuted
    (bad) when it would stop being one-to-one. owner[s, t] names a class that takes s to t, so a second class claiming
    the same place pair is pooled with it. Pooling relabels in place, so the class ids in `pair` and `owner` are
    always live."""

    def __init__(self, nP, nX):
        self.nP, self.nX, self.n = nP, nX, 0
        self.pair = np.full(nX * nX, -1, np.int64)
        self.owner = np.full((nP, nP), -1, np.int64)
        self.fwd = np.full((0, nP), -1, np.int64)
        self.bwd = np.full((0, nP), -1, np.int64)
        self.bad = np.zeros(0, bool)
        self.to = np.zeros(0, np.int64)                 # a pooled class -> the class it went into

    def new(self, k):
        if self.n + k > self.fwd.shape[0]:
            cap = max(2 * self.fwd.shape[0], self.n + k, 64)
            grow = cap - self.fwd.shape[0]
            self.fwd = np.vstack([self.fwd, np.full((grow, self.nP), -1, np.int64)])
            self.bwd = np.vstack([self.bwd, np.full((grow, self.nP), -1, np.int64)])
            self.bad = np.concatenate([self.bad, np.zeros(grow, bool)])
            self.to = np.concatenate([self.to, np.zeros(grow, np.int64)])
        ids = np.arange(self.n, self.n + k)
        self.to[ids] = ids
        self.n += k
        return ids

    def find(self, c):
        while self.to[c] != c:
            c = self.to[c]
        return c

    def seed(self, pairs, S, T):
        """The instances found before any filling: classes are the connected components of pairs and place pairs; a
        component that is not one-to-one runs the sequential rule, instance by instance, in order."""
        nP = self.nP
        up, pinv = np.unique(pairs, return_inverse=True)
        ust, sinv = np.unique(S * nP + T, return_inverse=True)
        n1, n = up.size, up.size + ust.size
        g = coo_matrix((np.ones(pinv.size, np.int8), (pinv, n1 + sinv)), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        comp = lab[pinv]
        trip = np.unique((comp * nP + S) * nP + T)
        tc, ts, tt = trip // (nP * nP), (trip // nP) % nP, trip % nP
        badc = np.zeros(lab.max() + 1, bool)
        for keys in (tc * nP + ts, tc * nP + tt):       # a place sent to two places, or two places to one
            u, cnt = np.unique(keys, return_counts=True)
            badc[u[cnt > 1] // nP] = True
        good = ~badc[tc]
        comps = np.unique(tc[good])
        cid = np.full(badc.size, -1, np.int64)
        cid[comps] = self.new(comps.size)
        c = cid[tc[good]]
        self.fwd[c, ts[good]] = tt[good]
        self.bwd[c, tt[good]] = ts[good]
        self.owner[ts[good], tt[good]] = c
        pc = lab[:n1]
        ok = ~badc[pc]
        self.pair[up[ok]] = cid[pc[ok]]
        for i in np.flatnonzero(badc[comp]):
            self.add(pairs[i:i + 1], S[i:i + 1], T[i:i + 1])

    def add(self, pairs, S, T):
        """Instances (pair: s -> t), taken together. A class they would make not one-to-one is refuted."""
        c = self.pair[pairs]
        if (c < 0).any():                               # pairs never seen: one new class each
            new = np.unique(pairs[c < 0])
            self.pair[new] = self.new(new.size)
            c = self.pair[pairs]
        keep = ~self.bad[c]
        c, S, T = c[keep], S[keep], T[keep]
        if c.size == 0:
            return
        f, b = self.fwd[c, S], self.bwd[c, T]
        clash = ((f >= 0) & (f != T)) | ((b >= 0) & (b != S))
        clash |= _mismatch(c * self.nP + S, T) | _mismatch(c * self.nP + T, S)
        if clash.any():
            self.bad[np.unique(c[clash])] = True
            keep = ~self.bad[c]
            c, S, T = c[keep], S[keep], T[keep]
        self.fwd[c, S] = T
        self.bwd[c, T] = S
        o = self.owner[S, T]
        free = np.flatnonzero(o < 0)
        if free.size:                                   # the first claim of a place pair owns it
            _, first = np.unique(S[free] * self.nP + T[free], return_index=True)
            fi = free[first]
            self.owner[S[fi], T[fi]] = c[fi]
            o = self.owner[S, T]
        for a, b in set(zip(c[o != c].tolist(), o[o != c].tolist())):
            self.pool(a, b)

    def pool(self, a, b):
        """Make two classes one, unless that would break one-to-one (then they are different displacements)."""
        a, b = self.find(a), self.find(b)
        if a == b or self.bad[a] or self.bad[b]:
            return
        fa, fb, ba, bb = self.fwd[a], self.fwd[b], self.bwd[a], self.bwd[b]
        if ((fa >= 0) & (fb >= 0) & (fa != fb)).any() or ((ba >= 0) & (bb >= 0) & (ba != bb)).any():
            return
        self.fwd[a] = np.where(fa >= 0, fa, fb)
        self.bwd[a] = np.where(ba >= 0, ba, bb)
        self.to[b] = a
        self.pair[self.pair == b] = a
        self.owner[self.owner == b] = a

    def apply(self, c, s):
        """Where class c takes place s, or -1 if unknown or refuted."""
        return -1 if c < 0 or self.bad[c] else int(self.fwd[c, s])

    def maps(self):
        out = []
        for c in range(self.n):
            if self.to[c] == c and not self.bad[c]:
                s = np.flatnonzero(self.fwd[c] >= 0)
                if s.size:
                    out.append(list(zip(s.tolist(), self.fwd[c, s].tolist())))
        return out


def regular(move):
    """P4 on the frame's move table (place, input -> place; -1 unknown), in place: fill in the moves the pooled
    displacements imply, until nothing changes. Returns (moves inferred, the displacements' maps as (s, t) lists)."""
    nP, nX = move.shape
    pool = Pool(nP, nX)
    pairs, S, T = [], [], []
    for u in range(nP):                                 # every ordered pair of inputs seen at a place
        xs = np.flatnonzero(move[u] >= 0)
        if xs.size < 2:
            continue
        i, j = np.nonzero(~np.eye(xs.size, dtype=bool))
        pairs.append(xs[i] * nX + xs[j])
        S.append(move[u, xs[i]])
        T.append(move[u, xs[j]])
    if pairs:
        pool.seed(np.concatenate(pairs), np.concatenate(S), np.concatenate(T))
    inferred, changed, ar = 0, True, np.arange(nX)
    while changed:
        changed = False
        for u in range(nP):
            xs = np.flatnonzero(move[u] >= 0)
            if xs.size == 0:
                continue
            cand = np.flatnonzero((pool.pair[xs[:, None] * nX + ar[None, :]] >= 0).any(0) & (move[u] < 0))
            for y in cand.tolist():
                t = -1
                for c, s in zip(pool.pair[xs * nX + y].tolist(), move[u, xs].tolist()):
                    t = pool.apply(c, s)
                    if t >= 0:
                        break
                if t < 0:
                    continue
                move[u, y] = t
                inferred += 1
                changed = True
                tt = np.full(xs.size, t, np.int64)
                pool.add(xs * nX + y, move[u, xs], tt)    # the new move's pairs with the moves known here
                pool.add(y * nX + xs, tt, move[u, xs])
                xs = np.append(xs, y)
    return inferred, pool.maps()


# ---- P3's full form ------------------------------------------------------------------------------------------------
def _fold(a, b, find, kids, cont):
    """Try to make places a and b one, folding their children together. Returns (ok, the changes) without applying
    them; not ok if some pair of places so merged both have a determined content of one sensor, and they differ."""
    over, okids, ocont = {}, {}, {}

    def f(u):
        u = find(u)
        while u in over:
            u = over[u]
        return u

    q = [(a, b)]
    while q:
        p, r = q.pop()
        p, r = f(p), f(r)
        if p == r:
            continue
        cp, cr = ocont.get(p, cont[p]), ocont.get(r, cont[r])
        for s in cp.keys() & cr.keys():
            if len(cp[s]) == 1 and len(cr[s]) == 1 and next(iter(cp[s])) != next(iter(cr[s])):
                return False, None
        mc = {s: cp.get(s, Counter()) + cr.get(s, Counter()) for s in cp.keys() | cr.keys()}
        mk = dict(okids.get(p, kids[p]))
        for x, v in okids.get(r, kids[r]).items():
            if x in mk:
                q.append((mk[x], v))
            else:
                mk[x] = v
        over[r], ocont[p], okids[p] = p, mc, mk
    return True, (over, ocont, okids)


def same_place(move, cont):
    """P3's full form, on the strictly merged frame. Returns each place's merged place (a root id).
    1. A determined content identifies places if no two places that have it conflict (checked before any merge).
    2. The places that share an identifying content are made one, unless a merge would conflict."""
    nP = move.shape[0]
    kids = [{int(x): int(move[u, x]) for x in np.flatnonzero(move[u] >= 0)} for u in range(nP)]
    cont = [dict(c) for c in cont]
    parent = list(range(nP))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    groups = defaultdict(list)
    for u in range(nP):
        for s, c in cont[u].items():
            if len(c) == 1:
                groups[(s, next(iter(c)))].append(u)
    ident = [g for g in groups.values()
             if len(g) > 1 and all(_fold(a, b, find, kids, cont)[0] for i, a in enumerate(g) for b in g[i + 1:])]
    for g in ident:
        for b in g[1:]:
            ok, changes = _fold(g[0], b, find, kids, cont)
            if ok:
                over, ocont, okids = changes
                for r, p in over.items():
                    parent[r] = p
                for p in ocont:
                    cont[p], kids[p] = ocont[p], okids[p]
    return [find(u) for u in range(nP)]


def _relabel(root, move, cont, P, places):
    """The frame after P3's full form: one place per root, in the order first met."""
    roots = sorted(set(root))
    nid = {r: i for i, r in enumerate(roots)}
    new = np.full((len(roots), move.shape[1]), -1, np.int64)
    us, xs = np.nonzero(move >= 0)
    new[[nid[root[u]] for u in us], xs] = [nid[root[v]] for v in move[us, xs]]
    merged = [defaultdict(Counter) for _ in roots]
    for u, c in enumerate(cont):
        for s, cnt in c.items():
            merged[nid[root[u]]][s].update(cnt)
    rep = {P[u]: P[root[u]] for u in range(len(P))}
    return new, [dict(m) for m in merged], [P[r] for r in roots], {h: rep[q] for h, q in places.items()}


# ---- coordinates ---------------------------------------------------------------------------------------------------
def coordinate(move, maps, P, X, cfg):
    """Coordinates for every orbit whose moves commute (module codes), and the orbits declined, with why."""
    d, w, seed = cfg
    nP, nX = move.shape
    gens = []                                           # (input id or None, permutation as a full-length array)
    for x in range(nX):
        col = move[:, x]
        dom = np.flatnonzero(col >= 0)
        if dom.size and np.array_equal(np.sort(col[dom]), dom):
            gens.append((x, col))
    for m in maps:
        s, t = np.array([a for a, _ in m]), np.array([b for _, b in m])
        if np.array_equal(np.sort(s), np.sort(t)):
            arr = np.full(nP, -1, np.int64)
            arr[s] = t
            gens.append((None, arr))
    if not gens:
        return [], [(0, "no move permutes a set of places")]
    rows, cols = [], []
    for _, arr in gens:
        dom = np.flatnonzero(arr >= 0)
        rows.append(dom)
        cols.append(arr[dom])
    g = coo_matrix((np.ones(sum(r.size for r in rows), np.int8), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nP, nP))
    _, lab = connected_components(g, directed=False)
    out, declined = [], []
    rng = np.random.default_rng((seed, 606))
    for o, label in enumerate(sorted(set(lab.tolist()), key=lambda l: int(np.flatnonzero(lab == l)[0]))):
        O = np.flatnonzero(lab == label)
        n = O.size
        if n < 2:
            continue
        loc = np.full(nP, -1, np.int64)
        loc[O] = np.arange(n)
        G, gin = [], []
        for x, arr in gens:
            img = arr[O]
            if (img >= 0).all() and (loc[img] >= 0).all():
                G.append(loc[img])
                gin.append(x)
        why = _check(G, n)
        if why:
            declined.append((n, why))
            continue
        chars = _characters(np.stack(G), n, rng)
        if chars is None:
            declined.append((n, "its moves' eigenvectors are not characters"))
            continue
        kept = _chain(*chars, n)
        if kept is None:
            declined.append((n, "the characters don't separate its places"))
            continue
        periods = tuple(int(k) for k, _ in kept)
        R = max(1, min(d // sum(periods), -(-w // len(periods))))
        modules = Modules(d, periods, R, seed + o)
        phases = np.stack([ph for _, ph in kept], 1)
        code = [modules.code(phases[i]) for i in range(n)]
        shifts = {}
        for x, gl in zip(gin, G):
            if x is None:
                continue
            delta = modules.unbind(code[gl[0]], code[0])
            if all(modules.bind(code[i], delta) == code[gl[i]] for i in range(n)):
                shifts[X[x]] = delta
        out.append(Orbit(modules, {P[O[i]]: code[i] for i in range(n)}, shifts, periods, P[O[0]], n))
    return out, declined


def _check(G, n):
    """Why an orbit gets no coordinates, or None: no generator, not one orbit under them, or they don't commute."""
    if not G:
        return "no move permutes it"
    Gm = np.stack(G)
    seen, frontier = np.zeros(n, bool), np.zeros(n, bool)
    seen[0] = frontier[0] = True
    while frontier.any():
        nxt = np.zeros(n, bool)
        nxt[Gm[:, frontier].ravel()] = True
        frontier = nxt & ~seen
        seen |= nxt
    if not seen.all():
        return "its moves don't connect its places"
    comp = Gm[:, Gm]                                    # comp[i, j] = g_i after g_j
    if not (comp == comp.transpose(1, 0, 2)).all():
        return "not abelian: its moves don't commute"
    return None


def _characters(Gm, n, rng, tries=5):
    """The common eigenvectors of commuting permutations, normalized at place 0: (phase indices j (n x n), the
    characters' orders). None if they are not characters (not a regular abelian action)."""
    for _ in range(tries):
        c = rng.standard_normal(len(Gm)) + 1j * rng.standard_normal(len(Gm))
        M = np.zeros((n, n), complex)
        for ci, gi in zip(c, Gm):
            M[gi, np.arange(n)] += ci
        vals, vecs = np.linalg.eig(M)
        gap = np.abs(vals[:, None] - vals[None, :]) + np.eye(n)
        if gap.min() > 1e-6:
            break
    else:
        return None
    chi = vecs / vecs[0:1, :]
    j = np.rint(np.angle(chi) / (2 * np.pi) * n).astype(np.int64) % n
    if np.abs(chi - np.exp(2j * np.pi * j / n)).max() > 1e-6:
        return None
    orders = n // np.gcd.reduce(np.vstack([j, np.full((1, n), n)]), axis=0)
    return j, orders


def _chain(j, orders, n):
    """The module chain: characters in increasing order, each kept only if it splits places the kept ones leave
    together. Returns [(order, phase per place)], or None if they never separate every place."""
    labels, kept, distinct = np.zeros(n, np.int64), [], 1
    for k in np.argsort(orders, kind="stable"):
        if orders[k] == 1:
            continue
        ph = j[:, k] * orders[k] // n
        _, new = np.unique(labels * orders[k] + ph, return_inverse=True)
        if new.max() + 1 > distinct:
            labels, distinct = new, new.max() + 1
            kept.append((orders[k], ph))
            if distinct == n:
                return kept
    return None
