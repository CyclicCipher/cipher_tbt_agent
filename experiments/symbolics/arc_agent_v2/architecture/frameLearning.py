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
- **Generalize, then split (part 6).**
  - **Per context:** once a pair is refuted, every pair is taken per context, a place's context being the input that
    led into it. Across contexts, two displacements pool only where they agree on two places.
  - **Constant moves:** an input that led from every place of a context to one place (the absorbing "× 0") leads there
    from all of them, and names no displacement.
  - **Shift classes:** where one shift per input contradicts the moves, phases are grouped into classes, each with its
    own shifts. A phase that fits no class is left out of the coordinate.
  - **The relation:** two classes are related by an automorphism x -> k x of the coordinate group, found from the
    inputs both have seen and applied to the rest. For "+" and "−", k = −1.
- **Counts for the column's union (part 9).** Each phase keeps what came next at its places (the likelihoods), and
  each input the phases it is made from (the prior a burst falls back to). Part 7's phase merge is gone: the column's
  union of hypotheses does its job (`column.py`).
- **One coordinate for every place, and phases (part 5).** Each moving input is one shift of a coordinate shared
  by every place, wherever it is made (P4's regularity across the frame); the start's coordinate is found. Naming (the
  determined symbol that comes next names the place's position) is tested per input and accepted only if consistent;
  where the data leave shifts free, the solution with the fewest inputs that shift (MDL). The coordinate group's orbits
  on the places are the phases; a block of inputs leads from a phase to one phase.
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
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from arc_agent_v2.architecture.HDClibrary import Modules


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
    factored: object = None                             # part 5: a Factored, or None
    unfactored: str = None                              # why not, if not
    split: int = 0                                      # part 6: the pairs P4 took per context (0: none refuted)


def consolidate(transitions, contents, start, regularity=True, coordinates=None, factoring=True, end=None,
                relate=True):
    """A Frame from the hippocampus's replay: transitions [(place, input, next place, times seen)] and contents
    [(place, sensor, input, times seen)], of episodes that start at start. coordinates = (d, w, seed), or None;
    factoring: part 5's (phase, coordinate) factorization, where there are coordinates; end: the hippocampus's SDR for
    an episode's end, as content; relate: part 6's relation between shift classes."""
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
    ctx = contexts(move)
    inferred, maps, split = regular(move, ctx) if regularity and U else (0, [], frozenset())
    if regularity and U and split:                      # part 6: absorbing elements, where contexts differ
        inferred += constant(move, ctx)
    us, xs = np.nonzero(move >= 0)
    moves = [(P[u], X[x], P[move[u, x]]) for u, x in zip(us.tolist(), xs.tolist())]
    content = [(P[u], s, X[x], int(n)) for u in range(len(P)) for s, c in cont[u].items() for x, n in c.items()]
    disp = [{P[a]: P[b] for a, b in m} for _, m in maps]
    coords, declined, systems = coordinate(move, maps, P, X, coordinates) if coordinates and U else ([], [], [])
    fac, why = None, None
    if factoring and systems and places.get(start, start) in pid:
        fac, why = factor(move, cont, pid[places.get(start, start)], P, X, systems, xid.get(end, -1), relate)
    return Frame(places.get(start, start), moves, content, places, inferred, disp, coords, declined, fac, why,
                 len(split))


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
    the same place pair is pooled with it. Pooling links a class to the one it went into (`to`); ids in `pair` and
    `owner` are resolved through those links when read (`live`), so a pooling costs nothing however many labels there
    are."""

    def __init__(self, nP, nX, nlabels=None):
        self.nP, self.nX, self.n = nP, nX, 0
        self.cctx = {}                                  # class -> the contexts of its labels (part 6)
        self.pair = np.full(nlabels or nX * nX, -1, np.int64)   # a label (a pair, maybe in a context) -> its class
        self.owner = np.full((nP, nP), -1, np.int64)
        self.fwd = np.full((0, nP), -1, np.int64)
        self.bwd = np.full((0, nP), -1, np.int64)
        self.bad = np.zeros(0, bool)
        self.to = np.zeros(0, np.int64)                 # a pooled class -> the class it went into
        self.refused = set()                            # live classes whose maps conflict (maps only grow: for good)

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

    def live(self, c):
        """The live class of each id in c (-1 stays -1)."""
        c = np.asarray(c, np.int64)
        out = c.copy()
        ok = out >= 0
        if not ok.any():                                # no class (there may be none yet)
            return out
        while True:
            nxt = np.where(ok, self.to[np.where(ok, out, 0)], -1)
            if np.array_equal(nxt, out):
                return out
            out = nxt

    def context(self, labels):
        """The context of each label: -2 for a pair taken in every context, else the context it was split by."""
        base = self.nX * self.nX
        return np.where(labels < base, -2, labels // base - 2)

    def seed(self, pairs, S, T):
        """The instances found before any filling, one context at a time: classes are the connected components of
        pairs and place pairs, and a component that is not one-to-one runs the sequential rule, instance by instance.
        Then classes of different contexts that claim the same place pair are offered for pooling."""
        ctx = self.context(pairs)
        clashes = []
        for k in np.unique(ctx).tolist():
            m = ctx == k
            clashes += self._seed(pairs[m], S[m], T[m], k)
        for a, b in clashes:
            self.pool(a, b)

    def _seed(self, pairs, S, T, k):
        nP = self.nP
        up, pinv = np.unique(pairs, return_inverse=True)
        ust, sinv = np.unique(S * nP + T, return_inverse=True)
        n1, n = up.size, up.size + ust.size
        moving = S != T                                 # a fixed point (s -> s) identifies no displacement (part 6)
        g = coo_matrix((np.ones(int(moving.sum()), np.int8), (pinv[moving], n1 + sinv[moving])), shape=(n, n))
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
        for c in cid[comps].tolist():
            self.cctx[c] = frozenset([k])
        c = cid[tc[good]]
        self.fwd[c, ts[good]] = tt[good]
        self.bwd[c, tt[good]] = ts[good]
        mv = ts[good] != tt[good]
        o = self.live(self.owner[ts[good][mv], tt[good][mv]])
        mine = o < 0                                    # a place pair another context's class already claims: a clash
        self.owner[ts[good][mv][mine], tt[good][mv][mine]] = c[mv][mine]
        clashes = list(set(zip(c[mv][~mine].tolist(), o[~mine].tolist())))
        pc = lab[:n1]
        ok = ~badc[pc]
        self.pair[up[ok]] = cid[pc[ok]]
        for i in np.flatnonzero(badc[comp]):
            self.add(pairs[i:i + 1], S[i:i + 1], T[i:i + 1])
        return clashes

    def add(self, pairs, S, T):
        """Instances (pair: s -> t), taken together. A class they would make not one-to-one is refuted."""
        c = self.live(self.pair[pairs])
        if (c < 0).any():                               # pairs never seen: one new class each
            new = np.unique(pairs[c < 0])
            ids = self.new(new.size)
            self.pair[new] = ids
            for i, k in zip(ids.tolist(), self.context(new).tolist()):
                self.cctx[i] = frozenset([k])
            c = self.live(self.pair[pairs])
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
        mv = S != T                                     # fixed points are part of the map, not evidence of identity
        c, S, T = c[mv], S[mv], T[mv]
        if c.size == 0:
            return
        o = self.live(self.owner[S, T])
        free = np.flatnonzero(o < 0)
        if free.size:                                   # the first claim of a place pair owns it
            _, first = np.unique(S[free] * self.nP + T[free], return_index=True)
            fi = free[first]
            self.owner[S[fi], T[fi]] = c[fi]
            o = self.live(self.owner[S, T])
        for a, b in set(zip(c[o != c].tolist(), o[o != c].tolist())):
            self.pool(a, b)

    def pool(self, a, b):
        """Make two classes one, unless that would break one-to-one (then they are different displacements). Within a
        context one shared place pair suffices (P4: one place determines a displacement). Across contexts, which may
        carry different groups, two displacements are pooled only where they agree on at least two places: one
        agreement can be a coincidence (a translation and a multiplication meet at one place)."""
        a, b = self.find(a), self.find(b)
        if a == b or self.bad[a] or self.bad[b] or (min(a, b), max(a, b)) in self.refused:
            return
        fa, fb, ba, bb = self.fwd[a], self.fwd[b], self.bwd[a], self.bwd[b]
        if ((fa >= 0) & (fb >= 0) & (fa != fb)).any() or ((ba >= 0) & (bb >= 0) & (ba != bb)).any():
            self.refused.add((min(a, b), max(a, b)))
            return
        ca, cb = self.cctx.get(a, frozenset()), self.cctx.get(b, frozenset())
        if not (ca & cb) and ((fa >= 0) & (fb >= 0) & (fa == fb)).sum() < 2:
            return
        self.cctx[a] = ca | cb
        self.fwd[a] = np.where(fa >= 0, fa, fb)
        self.bwd[a] = np.where(ba >= 0, ba, bb)
        self.to[b] = a

    def apply(self, c, s):
        """Where class c takes place s, or -1 if unknown or refuted."""
        if c < 0:
            return -1
        c = self.find(c)
        return -1 if self.bad[c] else int(self.fwd[c, s])

    def maps(self, family=lambda label: None):
        """Each live class's map as (s, t) pairs, with its family: the set of contexts its labels carry."""
        fam = defaultdict(set)
        labels = np.flatnonzero(self.pair >= 0)
        for label, c in zip(labels.tolist(), self.live(self.pair[labels]).tolist()):
            fam[c].add(family(label))
        out = []
        for c in range(self.n):
            if self.to[c] == c and not self.bad[c]:
                s = np.flatnonzero(self.fwd[c] >= 0)
                if s.size:
                    out.append((frozenset(fam[c]), list(zip(s.tolist(), self.fwd[c, s].tolist()))))
        return out


def contexts(move):
    """Each place's context: the one input that leads into it, or -1 (none, or several)."""
    ctx = np.full(move.shape[0], -1, np.int64)
    us, xs = np.nonzero(move >= 0)
    seen = {}
    for x, v in zip(xs.tolist(), move[us, xs].tolist()):
        seen.setdefault(v, set()).add(x)
    for v, ins in seen.items():
        if len(ins) == 1:
            ctx[v] = ins.pop()
    return ctx


def regular(move, ctx=None):
    """P4 on the frame's move table (place, input -> place; -1 unknown), in place: fill in the moves the pooled
    displacements imply, until nothing changes. Generalize, then split (part 6): a pair refuted as one displacement
    is tried again as one displacement per context (the input that led to the place). The refutation says the
    context matters, so every pair is then taken per context; displacements that agree across contexts still pool
    through the place pairs they share. Once split, an input whose move is constant in a context (`constants`)
    names no displacement there: where it leads is one place whatever the place it was made from, so a pair with it
    would map one place to many, or many to one. Returns (moves inferred, the displacements' maps as (family, (s, t)
    pairs), the pairs split)."""
    nX = move.shape[1]
    pairs, S, T = _instances(move)
    const = constants(move, ctx) if ctx is not None else None
    if ctx is not None and pairs.size and (_mismatch(pairs * move.shape[0] + S, T) |
                                            _mismatch(pairs * move.shape[0] + T, S)).any():
        # a pair whose own instances are not one-to-one is refuted whatever else is pooled: split at once
        inferred, pool, split = _regular(move, ctx, frozenset(np.unique(pairs).tolist()), const)
    else:
        before = move.copy()
        inferred, pool, split = _regular(move, ctx, frozenset())
        labels = np.flatnonzero(pool.pair >= 0)
        if ctx is not None and pool.bad[pool.live(pool.pair[labels])].any():   # every pair is tried per context
            move[:] = before
            inferred, pool, split = _regular(move, ctx, frozenset(labels.tolist()), const)
    family = lambda label: None if label < nX * nX else int(label // (nX * nX)) - 2
    return inferred, pool.maps(family), split


def _instances(move, where=False):
    """Every ordered pair of inputs seen at a place, as (pair x * nX + z, where x leads, where z leads[, the place])."""
    nX = move.shape[1]
    us, xs = np.nonzero(move >= 0)
    cnt = np.bincount(us, minlength=move.shape[0])
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    out = [], [], [], []
    for k in np.unique(cnt[cnt >= 2]).tolist():         # places with k inputs seen, all at once
        rows = np.flatnonzero(cnt == k)
        idx = first[rows][:, None] + np.arange(k)[None, :]
        X = xs[idx]
        i, j = np.nonzero(~np.eye(k, dtype=bool))
        out[0].append((X[:, i] * nX + X[:, j]).ravel())
        out[1].append(move[rows[:, None], X[:, i]].ravel())
        out[2].append(move[rows[:, None], X[:, j]].ravel())
        out[3].append(np.repeat(rows, i.size))
    if not out[0]:
        e = np.zeros(0, np.int64)
        return (e, e, e, e) if where else (e, e, e)
    res = tuple(np.concatenate(a) for a in out)
    o = np.argsort(res[3], kind="stable")               # place by place, in the order a place lists them
    res = tuple(a[o] for a in res)
    return res if where else res[:3]


def _regular(move, ctx, split, const=None):
    nP, nX = move.shape
    base = nX * nX
    sp = np.array(sorted(split), np.int64)
    is_split = np.zeros(base, bool)                     # pair -> taken per context (a lookup, not a search)
    is_split[sp] = True
    free = np.ones((nX + 1, nX), bool) if const is None else ~const   # [context + 1, input]: names displacements

    def lab(u, pairs):                                  # a pair's label at place u: itself, or itself in u's context
        if not sp.size:
            return pairs
        return np.where(is_split[pairs], base * (2 + ctx[u]) + pairs, pairs)

    pool = Pool(nP, nX, base * (nX + 2) if sp.size else base)
    pairs, S, T, U = _instances(move, True)
    if const is not None and pairs.size:
        k = ctx[U] + 1
        keep = free[k, pairs // nX] & free[k, pairs % nX]
        pairs, S, T, U = pairs[keep], S[keep], T[keep], U[keep]
    if pairs.size:
        pool.seed(np.where(is_split[pairs], base * (2 + ctx[U]) + pairs, pairs) if sp.size else pairs, S, T)
    inferred, changed, ar = 0, True, np.arange(nX)
    while changed:
        changed = False
        for u in range(nP):
            ok = free[ctx[u] + 1] if ctx is not None else free[0]
            xs = np.flatnonzero((move[u] >= 0) & ok)
            if xs.size == 0:
                continue
            cand = np.flatnonzero((pool.pair[lab(u, xs[:, None] * nX + ar[None, :])] >= 0).any(0) & (move[u] < 0) & ok)
            for y in cand.tolist():
                t = -1
                for c, s in zip(pool.live(pool.pair[lab(u, xs * nX + y)]).tolist(), move[u, xs].tolist()):
                    t = pool.apply(c, s)
                    if t >= 0:
                        break
                if t < 0:
                    continue
                move[u, y] = t
                inferred += 1
                changed = True
                tt = np.full(xs.size, t, np.int64)
                pool.add(lab(u, xs * nX + y), move[u, xs], tt)    # the new move's pairs with the moves known here
                pool.add(lab(u, y * nX + xs), tt, move[u, xs])
                xs = np.append(xs, y)
    return inferred, pool, split


def constants(move, ctx):
    """[context + 1, input]: True where the input's move is constant in the context (part 6): from every place of the
    context where it was seen (at least 2), it led to the same place. Context -1 (none, or several) has none."""
    nX = move.shape[1]
    out = np.zeros((nX + 1, nX), bool)
    for k in sorted(set(ctx.tolist()) - {-1}):
        M = move[ctx == k]
        seen = M >= 0
        hi = M.max(0)
        lo = np.where(seen, M, np.iinfo(np.int64).max).min(0)
        out[k + 1] = (seen.sum(0) >= 2) & (lo == hi)
    return out


def constant(move, ctx):
    """Constant moves (part 6): an input that, from every place of one context where it was seen (at least 2), led to
    the same place, leads there from every place of that context. Returns the moves filled in."""
    filled = 0
    const = constants(move, ctx)
    for k, y in zip(*np.nonzero(const)):
        U = np.flatnonzero(ctx == k - 1)
        seen = move[U, y]
        missing = U[seen < 0]
        move[missing, y] = seen[seen >= 0][0]
        filled += missing.size
    return filled


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
def _cyclic(col):
    """A map (place -> place, -1 where undefined) restricted to the places on its complete cycles, where it is a
    permutation; -1 elsewhere (part 5: a map can permute one set of places while being partial elsewhere)."""
    n = col.size
    out, state = np.full(n, -1, np.int64), np.zeros(n, np.int8)
    for s in range(n):
        if col[s] < 0 or state[s]:
            continue
        path, u = [], s
        while u >= 0 and state[u] == 0:
            state[u] = 1
            path.append(u)
            u = int(col[u])
        if u >= 0 and state[u] == 1:                    # closed a cycle (a fixed point carries no structure)
            cyc = path[path.index(u):]
            if len(cyc) > 1:
                out[cyc] = col[cyc]
        state[path] = 2
    return out


def coordinate(move, maps, P, X, cfg):
    """Coordinates for every orbit whose moves commute (module codes), the orbits declined with why, and the systems
    for factor(): [(orbit's place ids, phases (n x modules), Modules)]. Orbits acted on by the same generators share
    one code: the first one's characters, carried to the others by its generators' steps."""
    d, w, seed = cfg
    nP, nX = move.shape
    unsplit = frozenset([None])
    gens = []                                           # (input id or None, permutation on its complete cycles, family)
    for x in range(nX):
        arr = _cyclic(move[:, x])
        if (arr >= 0).any():
            gens.append((x, arr, unsplit))
    for fam, m in maps:
        arr = np.full(nP, -1, np.int64)
        arr[[a for a, _ in m]] = [b for _, b in m]
        arr = _cyclic(arr)
        if (arr >= 0).any():
            gens.append((None, arr, fam))
    if not gens:
        return [], [(0, "no move permutes a set of places")], []
    orbits = []                                         # (family, its place ids), per family's own components
    for fam in sorted({f for _, _, f in gens}, key=lambda f: (f != unsplit, sorted(map(str, f)))):
        mine = [arr for _, arr, f in gens if f == fam]
        rows = np.concatenate([np.flatnonzero(arr >= 0) for arr in mine])
        cols = np.concatenate([arr[arr >= 0] for arr in mine])
        g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(nP, nP))
        _, lab = connected_components(g, directed=False)
        for label in sorted(set(lab[rows].tolist()), key=lambda l: int(np.flatnonzero(lab == l)[0])):
            orbits.append((fam, np.flatnonzero(lab == label)))
    out, declined, systems, shared = [], [], [], {}
    rng = np.random.default_rng((seed, 606))
    for o, (fam, O) in enumerate(orbits):
        n = O.size
        if n < 2:
            continue
        loc = np.full(nP, -1, np.int64)
        loc[O] = np.arange(n)
        G, gin, key = [], [], []
        for i, (x, arr, f) in enumerate(gens):
            if f != fam:
                continue
            img = arr[O]
            if (img >= 0).all() and (loc[img] >= 0).all():
                G.append(loc[img])
                gin.append(x)
                key.append(i)
        why = _check(G, n)
        if why:
            declined.append((n, why))
            continue
        key = tuple(key)
        phases = _carry(shared[key], G, n) if key in shared else None
        if phases is not None:
            modules = shared[key][0]
        else:
            chars = _characters(np.stack(G), n, rng)
            kept = None if chars is None else _chain(*chars, n)
            if kept is None:
                declined.append((n, "its moves' eigenvectors are not characters" if chars is None
                                 else "the characters don't separate its places"))
                continue
            periods = tuple(int(k) for k, _ in kept)
            R = max(1, min(d // sum(periods), -(-w // len(periods))))
            modules = Modules(d, periods, R, seed + o)
            phases = np.stack([ph for _, ph in kept], 1)
            per = np.array(periods, np.int64)
            shared.setdefault(key, (modules, [(phases[gl[0]] - phases[0]) % per for gl in G]))
        code = [modules.code(phases[i]) for i in range(n)]
        shifts = {}
        for x, gl in zip(gin, G):
            if x is None:
                continue
            delta = modules.unbind(code[gl[0]], code[0])
            if all(modules.bind(code[i], delta) == code[gl[i]] for i in range(n)):
                shifts[X[x]] = delta
        out.append(Orbit(modules, {P[O[i]]: code[i] for i in range(n)}, shifts, modules.periods, P[O[0]], n))
        systems.append((O, phases, modules))
    return out, declined, systems


def _carry(system, G, n):
    """An orbit's phases in an earlier orbit's code: from its base, each generator adds the step it adds there. None if
    that is inconsistent."""
    modules, steps = system
    per = np.array(modules.periods, np.int64)
    ph = np.full((n, len(per)), -1, np.int64)
    ph[0] = 0
    queue = [0]
    while queue:
        u = queue.pop()
        for gl, st in zip(G, steps):
            v, want = gl[u], (ph[u] + st) % per
            if ph[v, 0] < 0:
                ph[v] = want
                queue.append(v)
            elif not np.array_equal(ph[v], want):
                return None
    return ph if (ph >= 0).all() else None


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


# ---- one coordinate for every place, and phases (part 5) -------------------------------------------------------------
@dataclass
class Factored:
    """A frame factored into (phase, coordinate). Codes are of `modules`; a phase is named by its first place met."""
    modules: object
    phase: dict                         # frame place -> its phase
    start: tuple                        # (phase, coordinate code) where every episode starts
    trans: dict                         # (phase, input) -> the phase it leads to
    shift: dict                         # (shift class, input) -> its shift (a code)
    coord: dict                         # frame place -> its coordinate (a code)
    content: dict                       # (phase, sensor) -> ("const", symbol) or ("named", None)
    named: dict                         # coordinate code -> the symbol whose own position it is
    naming: list                        # the inputs whose naming was accepted
    identities: list                    # the inputs whose shift is the identity in every class they shift in
    phases: int
    cls: dict = field(default_factory=dict)          # part 6: phase -> its shift class (none: left out)
    relation: dict = field(default_factory=dict)     # (class, class) -> k per module: x -> k x takes one to the other
    left_out: list = field(default_factory=list)     # the phases whose moves contradict every class
    related: list = field(default_factory=list)      # [(class, input)]: shifts given by a relation, never seen
    counts: dict = field(default_factory=dict)       # part 9: (phase, sensor) -> {input: times it came next there}
    origins: dict = field(default_factory=dict)      # part 9: input -> {phase: places of it the input was made from}
    vocab: int = 0                                   # part 9: K, the number of inputs the frame knows
    cells: dict = field(default_factory=dict)        # part 9: (phase, coordinate) -> {sensor: {input: n}}, a place's


class _Forms:
    """Affine forms over free parameters in a group of vectors mod `periods`: (constant, {parameter: coefficient})."""

    def __init__(self, periods):
        self.per = tuple(int(n) for n in periods)

    def const(self, v):
        return (tuple(int(x) % n for x, n in zip(v, self.per)), {})

    def param(self, i):
        return ((0,) * len(self.per), {i: 1})

    def add(self, a, b, s=1):
        c = tuple((x + s * y) % n for x, y, n in zip(a[0], b[0], self.per))
        k = dict(a[1])
        for p, v in b[1].items():
            k[p] = k.get(p, 0) + s * v
            if all(k[p] % n == 0 for n in self.per):
                del k[p]
        return (c, k)

    def zero(self, f):
        return not f[1] and not any(f[0])

    def subst(self, f, i, g):
        """f with parameter i replaced by the form g."""
        if i not in f[1]:
            return f
        h = f[1][i]
        k = dict(f[1])
        del k[i]
        return self.add((f[0], k), (tuple(h * x for x in g[0]), {p: h * v for p, v in g[1].items()}))


def _solve(alg, E, N, C0, inputs, check=False):
    """One coordinate per place and one shift per input with c(v) = c(u) + d(x) for every move (u, x, v) in E, and
    c(v) = c(w) for every naming pair (v, w) in N. C0: the coordinates known (an orbit's places, as forms). Where the
    data leave something free, the solution with the most identity shifts (MDL). Returns (C, D) as vectors, or None if
    the moves contradict each other; with check, only whether they do (True or None).
    **Propagation:** a move is looked at again whenever one of its three unknowns becomes known, so each move costs a
    few form additions however many there are."""
    C, D, cons = dict(C0), {}, []
    nparam = max([p for f in C0.values() for p in f[1]], default=-1) + 1
    at, of = defaultdict(list), defaultdict(list)       # place -> its moves (i) and naming pairs (~j); input -> its moves
    for i, (u, x, v) in enumerate(E):
        at[u].append(i)
        at[v].append(i)
        of[x].append(i)
    for j, (v, w) in enumerate(N):
        at[v].append(~j)
        at[w].append(~j)
    done, ndone = bytearray(len(E)), bytearray(len(N))  # a move (a pair) whose constraint is recorded
    queue = deque(list(range(len(E))) + [~j for j in range(len(N))])
    rank = {x: i for i, x in enumerate(inputs)}

    def settle(f):
        if not alg.zero(f):
            cons.append(f)

    while True:
        while queue:
            i = queue.popleft()
            if i >= 0:
                u, x, v = E[i]
                cu, dx, cv = C.get(u), D.get(x), C.get(v)
                if cu is not None and dx is not None:
                    f = alg.add(cu, dx)
                    if cv is None:
                        C[v] = f
                        queue.extend(at[v])
                    elif not done[i]:
                        done[i] = 1
                        settle(alg.add(cv, f, -1))
                elif cu is not None and cv is not None:
                    D[x] = alg.add(cv, cu, -1)
                    queue.extend(of[x])
                elif dx is not None and cv is not None:
                    C[u] = alg.add(cv, dx, -1)
                    queue.extend(at[u])
            else:
                v, w = N[~i]
                cv, cw = C.get(v), C.get(w)
                if cv is not None and cw is None:
                    C[w] = cv
                    queue.extend(at[w])
                elif cw is not None and cv is None:
                    C[v] = cw
                    queue.extend(at[v])
                elif cv is not None and not ndone[~i]:
                    ndone[~i] = 1
                    settle(alg.add(cv, cw, -1))
        count = Counter(x for u, x, v in E if x not in D and (u in C or v in C))
        if not count:
            break
        x = max(count, key=lambda x: (count[x], -rank[x]))   # a free shift
        D[x] = alg.param(nparam)
        nparam += 1
        queue.extend(of[x])
    for f in cons:                                      # any contradiction among the moves
        if not f[1] and any(f[0]):
            return None
    while True:                                         # eliminate parameters with a +-1 coefficient
        pick = next(((f, i) for f in cons for i, k in f[1].items()
                     if all(k % n == 1 for n in alg.per) or all(k % n == n - 1 for n in alg.per)), None)
        if pick is None:
            break
        f, i = pick
        s = 1 if all(f[1][i] % n == 1 for n in alg.per) else -1
        rest = (f[0], {p: v for p, v in f[1].items() if p != i})
        g = (tuple(-s * x for x in rest[0]), {p: -s * v for p, v in rest[1].items()})
        C = {u: alg.subst(h, i, g) for u, h in C.items()}
        D = {x: alg.subst(h, i, g) for x, h in D.items()}
        cons = [h for h in (alg.subst(h, i, g) for h in cons) if not alg.zero(h)]
        if any(not h[1] and any(h[0]) for h in cons):
            return None
    theta = _choose(alg, {} if check else D, cons)
    if theta is None:
        return None
    if check:
        return True
    ev = lambda f: tuple((c + sum(k * theta.get(p, (0,) * len(alg.per))[m] for p, k in f[1].items())) % n
                         for m, (c, n) in enumerate(zip(f[0], alg.per)))
    return {u: ev(f) for u, f in C.items()}, {x: ev(f) for x, f in D.items()}


def _choose(alg, D, cons, limit=2000000):
    """Values for the free parameters: those satisfying the remaining constraints with the most inputs whose shift is
    the identity, by exhaustive search over each group of parameters that occur together (MDL). None if no values
    satisfy the constraints. The grid has one axis per parameter; a form is evaluated only on the axes of the
    parameters it uses, and the scores are summed once (ties: the first assignment in grid order)."""
    params = sorted({p for f in list(D.values()) + cons for p in f[1]})
    if not params:
        return {}
    parent = {p: p for p in params}

    def find(p):
        while parent[p] != p:
            p = parent[p]
        return p

    for f in list(D.values()) + cons:
        ps = list(f[1])
        for p in ps[1:]:
            parent[find(p)] = find(ps[0])
    groups = defaultdict(list)
    for p in params:
        groups[find(p)].append(p)
    elems = np.stack(np.meshgrid(*[np.arange(n) for n in alg.per], indexing="ij"), -1).reshape(-1, len(alg.per))
    per = np.array(alg.per, np.int64)
    ne, nm = len(elems), len(alg.per)
    theta = {}
    for ps in groups.values():
        m = len(ps)
        if ne ** m > limit:
            return None
        axis = {p: i for i, p in enumerate(ps)}
        inside = lambda f: f[1] and set(f[1]) <= set(ps)

        def zero(f):                                    # where f is zero, on the axes of its parameters
            v = np.array(f[0], np.int64).reshape((1,) * m + (nm,))
            for p, k in f[1].items():
                sh = [1] * m + [nm]
                sh[axis[p]] = ne
                v = v + k * elems.reshape(sh)
            return ((v % per) == 0).all(-1)

        ok = np.ones((ne,) * m, bool)
        for f in cons:
            if inside(f):
                ok = ok & zero(f)
        if not ok.any():
            return None
        parts = {}                                      # the scores of the forms on the same axes, summed
        for f in D.values():
            if inside(f):
                key = tuple(sorted(axis[p] for p in f[1]))
                parts[key] = parts.get(key, 0) + zero(f).astype(np.int64)
        score = np.zeros((ne,) * m, np.int64)
        for part in parts.values():
            score = score + part
        best = np.unravel_index(int(np.argmax(np.where(ok, score, -1))), (ne,) * m)
        theta.update({p: tuple(int(x) for x in elems[best[i]]) for i, p in enumerate(ps)})
    return theta


def factor(move, cont, start, P, X, systems, end=-1, relate=True):
    """Part 5: one coordinate for every place, and phases. Part 6: where one shift per input contradicts the moves,
    shift classes, and (relate) the relation between them. Part 9: each phase's counts and each input's origins, for the
    column's union of hypotheses. end: the input
    id of an episode's end, if any. Returns a Factored, or None (with why) if the frame can't be factored."""
    if not systems:
        return None, "no coordinates"
    systems = sorted(systems, key=lambda s: -len(s[0]))  # the coordinate: the system that covers the most places
    O0, ph0, modules = systems[0]
    alg = _Forms(modules.periods)
    C0 = {}
    for k, (O, ph, m) in enumerate(systems):
        if m is not modules:
            continue
        for i, u in enumerate(O.tolist()):              # each orbit of the first system, with a free offset
            C0[u] = alg.const(ph[i]) if k == 0 else alg.add(alg.const(ph[i]), alg.param(k - 1))
    us, xs = np.nonzero(move >= 0)
    E = [(int(u), int(x), int(move[u, x])) for u, x in zip(us, xs)]
    out_of = set(us.tolist())                           # a place whose only observed future is the episode's end has
    ahead = lambda v: any(x != end for c in cont[v].values() for x in c)   # nothing for a coordinate to predict
    E = [(u, x, v) for u, x, v in E if v in out_of or ahead(v)]           # left out
    inputs = sorted({x for _, x, _ in E})
    det = {u: {s: next(iter(c)) for s, c in cont[u].items() if len(c) == 1} for u in range(len(cont))}
    own = {x: int(move[start, x]) for x in inputs if move[start, x] >= 0}
    names = {}                                          # input y -> naming pairs (place after y, the content's own place)
    for u, y, v in E:
        for s, sym in det.get(v, {}).items():
            if sym in own and own[sym] != v:
                names.setdefault(y, set()).add((v, own[sym]))
    phase, block = _phases(E, len(P), systems, modules)
    nX = len(X)
    cls = {ph: 0 for ph in set(phase)}                  # step 3: one class, one shift per input everywhere
    if _solve(alg, E, [], C0, inputs, check=True) is None:
        cls = _classes(alg, E, phase, C0, nX)           # part 6: generalize, then split by phase
    keyed = [(u, cls[phase[u]] * nX + x, v) for u, x, v in E if phase[u] in cls]
    keys = sorted({k for _, k, _ in keyed})
    tried = {y: _solve(alg, keyed, sorted(pairs), C0, keys) for y, pairs in names.items() if len(pairs) >= 2}
    accepted = [y for y, sol in tried.items() if sol]
    sol = tried[accepted[0]] if len(accepted) == 1 else \
        _solve(alg, keyed, sorted(p for y in accepted for p in names[y]), C0, keys)   # one naming: its solve is it
    if sol is None:
        accepted, sol = [], _solve(alg, keyed, [], C0, keys)
    if sol is None:
        return None, "the moves are not one shift each"
    Cv, Dv = sol
    relation, related = _relate(Dv, nX, modules.periods) if relate else ({}, [])
    kind = lambda us, s: _kind(us, s, det, cont, accepted, own, Cv)
    rep = {}
    for u in range(len(P)):
        rep.setdefault(phase[u], u)
    ph_sdr = {u: P[rep[phase[u]]] for u in range(len(P))}
    trans, seen = {}, {}
    for u, x, v in E:
        seen.setdefault((phase[u], block[x]), set()).add(phase[v])
    for (pu, b), targets in seen.items():
        if len(targets) == 1:
            t = targets.pop()
            for x in inputs:
                if block[x] == b:
                    trans[(P[rep[pu]], X[x])] = P[rep[t]]
    code = lambda v: modules.code(np.array(v, np.int64))
    coord = {P[u]: code(v) for u, v in Cv.items()}
    named = {}
    for x, u in own.items():
        if u in Cv:
            named.setdefault(code(Cv[u]), []).append(X[x])
    named = {c: xs[0] for c, xs in named.items() if len(xs) == 1}
    content = {}
    for ph in set(phase):
        members = [u for u in range(len(P)) if phase[u] == ph]
        for s in {s for u in members for s in det.get(u, {})}:
            k = kind(members, s)
            if k is not None:
                content[(P[rep[ph]], s)] = k if k[0] == "named" else ("const", X[k[1]])
    if start not in Cv:
        return None, "the start has no coordinate"
    zero = tuple(0 for _ in modules.periods)
    shifts = {(k // nX, X[k % nX]): code(v) for k, v in Dv.items()}
    nonzero = {k % nX for k, v in Dv.items() if v != zero}
    identities = [X[x] for x in dict.fromkeys(k % nX for k in Dv) if x not in nonzero]
    left = sorted({rep[phase[u]] for u, _, _ in E if phase[u] not in cls})
    counts, origins = defaultdict(Counter), defaultdict(Counter)
    for u in range(len(P)):                             # what came next at each phase's places: the likelihoods
        for s, c in cont[u].items():
            for x, n in c.items():
                counts[(ph_sdr[u], s)][X[x]] += n
    for u, x, v in E:                                   # the phases each input is made from: a burst's prior
        origins[X[x]][ph_sdr[u]] += 1
    cells = defaultdict(lambda: defaultdict(Counter))   # a coordinated place's content, by its (phase, coordinate)
    for u, v in Cv.items():
        for s, c in cont[u].items():
            for x, n in c.items():
                cells[(ph_sdr[u], code(v))][s][X[x]] += n
    return Factored(modules, {P[u]: ph_sdr[u] for u in range(len(P))}, (ph_sdr[start], code(Cv[start])), trans,
                    shifts, coord, content, named, [X[y] for y in accepted], identities, len(set(phase)),
                    {P[rep[ph]]: k for ph, k in cls.items()}, relation, [P[u] for u in left],
                    [(k // nX, X[k % nX]) for k in related],
                    {k: dict(c) for k, c in counts.items()}, {k: dict(c) for k, c in origins.items()}, len(X),
                    {k: {s: dict(c) for s, c in v.items()} for k, v in cells.items()}), None


def _kind(us, s, det, cont, accepted, own, Cv):
    """What a set of places predicts of a sensor: ("const", input) if every place with content there has the same one,
    ("named", None) if each place's content is the symbol whose own position is its coordinate (where naming was
    accepted), else None."""
    syms = [(u, det[u][s]) for u in us if s in det.get(u, {})]
    if not syms:
        return None
    if len({sym for _, sym in syms}) == 1 and len(syms) == sum(1 for u in us if s in cont[u]):
        return ("const", syms[0][1])
    if accepted and all(sym in own and own[sym] in Cv and u in Cv and Cv[own[sym]] == Cv[u] for u, sym in syms):
        return ("named", None)
    return None


def _classes(alg, E, phase, C0, nX):
    """Shift classes (part 6): the phases, largest first (by the moves made from them), each joining the first class
    whose shifts it shares, or else starting a class of its own; a phase whose moves contradict even a class of its
    own is left out of the coordinate. Returns {phase: class}."""
    moves = defaultdict(list)
    for m in E:
        moves[phase[m[0]]].append(m)
    cls, kept, n = {}, [], 0
    for ph in sorted(moves, key=lambda ph: (-len(moves[ph]), ph)):
        for k in range(n + 1):
            trial = kept + [(u, k * nX + x, v) for u, x, v in moves[ph]]
            if _solve(alg, trial, [], C0, sorted({x for _, x, _ in trial}), check=True):
                cls[ph], kept, n = k, trial, max(n, k + 1)
                break
    return cls


def _relate(D, nX, periods):
    """The relation between two shift classes (part 6): one automorphism of the coordinate group, x -> k x in each
    module (k a unit), that takes every input's shift in one class to its shift in the other, for every input seen in
    both. Found from those, and applied to the inputs seen in only one (in place, in D: class * nX + input -> shift).
    Where the shared inputs leave k open, an image is given only if every k left gives the same one. Returns
    ({(class, class): k per module, where one k per module is left}, the keys it gave)."""
    classes = sorted({k // nX for k in D})
    relation, given = {}, []
    for i in classes:
        for j in classes:
            if j <= i:
                continue
            shared = [x for x in range(nX) if i * nX + x in D and j * nX + x in D]
            if not shared:
                continue
            ks = []
            for m, n in enumerate(periods):
                units = np.array([k for k in range(1, n) if np.gcd(k, n) == 1], np.int64)
                a = np.array([D[i * nX + x][m] for x in shared], np.int64)
                b = np.array([D[j * nX + x][m] for x in shared], np.int64)
                ks.append(units[((units[:, None] * a[None, :] - b[None, :]) % n == 0).all(1)])
            if any(k.size == 0 for k in ks):
                continue                                # no one automorphism: not related
            if all(k.size == 1 for k in ks):
                relation[(i, j)] = tuple(int(k[0]) for k in ks)
            for src, dst, inverse in ((i, j, False), (j, i, True)):
                for x in range(nX):
                    if src * nX + x not in D or dst * nX + x in D:
                        continue
                    img = []
                    for m, (n, k) in enumerate(zip(periods, ks)):
                        kk = np.array([pow(int(q), -1, n) for q in k], np.int64) if inverse else k
                        vals = np.unique(kk * D[src * nX + x][m] % n)
                        if vals.size != 1:              # the data leave the image open
                            break
                        img.append(int(vals[0]))
                    else:
                        D[dst * nX + x] = tuple(img)
                        given.append(dst * nX + x)
    return relation, given


def _phases(E, nP, systems, modules):
    """Phases: each orbit of the coordinate system is one; then, until nothing changes, inputs that lead from one
    place to one phase are one block, and the places a block leads to from one phase are one phase."""
    parent = list(range(nP))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for O, ph, m in systems:
        if m is modules:
            for u in O[1:].tolist():
                parent[find(u)] = find(int(O[0]))
    xs = sorted({x for _, x, _ in E})
    bpar = {x: x for x in xs}

    def bfind(x):
        while bpar[x] != x:
            x = bpar[x]
        return x

    changed = True
    while changed:
        changed = False
        by_place = defaultdict(dict)
        for u, x, v in E:
            by_place[u].setdefault(find(v), []).append(x)
        for targets in by_place.values():
            for group in targets.values():
                for x in group[1:]:
                    a, b = bfind(group[0]), bfind(x)
                    if a != b:
                        bpar[b], changed = a, True
        lead = defaultdict(list)
        for u, x, v in E:
            lead[(find(u), bfind(x))].append(v)
        for vs in lead.values():
            for v in vs[1:]:
                a, b = find(vs[0]), find(v)
                if a != b:
                    parent[b], changed = a, True
    return [find(u) for u in range(nP)], {x: bfind(x) for x in xs}
