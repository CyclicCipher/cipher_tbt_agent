"""Learning a frame from replayed episodes (ARCHITECTURE.md sections 0 and 4; docs/LAB.md part 3).

**Input:** consolidation replays every transition the hippocampus remembers, (place, input, next place). Together they
form a tree of the episodes.

**Output:** a Frame: places, the moves between them (seen and inferred), and the content at each place.

**Two rules from the column's innate prior, and nothing else:**
- **P3, same future, same place.** Places whose observed futures are identical are one place: the same inputs lead
  from them to places that are again one place, down to the episodes' ends (the Myhill–Nerode test). This is the
  strict form, identical futures. A looser form (merging places whose futures merely never disagree) is not needed
  yet.
- **P4, regularity: a move is determined by where it takes one place.**
  - From any place u, going where input y leads instead of where x leads is a displacement d(x, y), the same at
    every place until refuted. So each ordered pair of inputs names one displacement, whose instances are collected
    from every place where both inputs were seen.
  - Two pairs whose displacements take some place to the same place are the same displacement, so their instances are
    pooled.
  - A pooled displacement fills in moves never seen: from u, input y leads to d(x, y) applied to where x leads.
  - A displacement must stay a one-to-one map. An instance that would send one place to two places (or two places to
    one) refutes its pair, which is then not used. A pooling that would do so is not made.

**Nothing here knows what an input is.** Inputs and places are SDRs, compared only for identity (they are replayed
exact copies). There are no values, no order, no assumed cycle, and no assumed commutativity.

**Implementation:** inside consolidation, places and inputs get integer row addresses (indices into the frame's move
table, in the order first met), so the rules run as array operations. The frame comes out as SDRs again.
- **P4's pooling** is connected components over the graph that links each pair of inputs to the place pairs its
  instances map.
- **Each component is checked for one-to-one.** A component with a conflict runs the sequential rule, instance by
  instance, so the result is the sequential algorithm's wherever the data is consistent.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Frame:
    start: object                       # the frame's start place (an SDR)
    moves: list                         # [(place, input, next place)], seen and inferred
    content: list                       # [(place, input, times seen)]: what followed at each place
    places: dict                        # hippocampal place -> frame place
    inferred: int = 0                   # moves filled in by P4
    displacements: list = field(default_factory=list)   # each a {place: place} map, not refuted


def consolidate(transitions, start, regularity=True):
    """A Frame from the hippocampus's replay [(place, input, next place, times seen)], whose episodes start at start."""
    places = same_future(transitions)
    pid, P, xid, X = {}, [], {}, []
    U, XS, V, N = [], [], [], []
    for h, x, nxt, n in transitions:
        for q in (places[h], places[nxt]):
            if q not in pid:
                pid[q] = len(P)
                P.append(q)
        if x not in xid:
            xid[x] = len(X)
            X.append(x)
        U.append(pid[places[h]])
        XS.append(xid[x])
        V.append(pid[places[nxt]])
        N.append(n)
    move = np.full((len(P), len(X)), -1, np.int64)      # frame place, input -> frame place (-1: unknown)
    count = np.zeros((len(P), len(X)), np.int64)
    if U:
        move[U, XS] = V                                 # places merged by P3 agree on where each input leads
        np.add.at(count, (U, XS), N)
    inferred, maps = regular(move) if regularity and U else (0, [])
    us, xs = np.nonzero(move >= 0)
    moves = [(P[u], X[x], P[move[u, x]]) for u, x in zip(us.tolist(), xs.tolist())]
    cu, cx = np.nonzero(count)
    content = [(P[u], X[x], int(count[u, x])) for u, x in zip(cu.tolist(), cx.tolist())]
    disp = [{P[a]: P[b] for a, b in m} for m in maps]
    return Frame(places.get(start, start), moves, content, places, inferred, disp)


def same_future(transitions):
    """P3: {hippocampal place: frame place}. Places with identical observed futures share one frame place (the first
    of them met)."""
    children, order, seen = defaultdict(dict), [], set()
    for h, x, nxt, _ in transitions:
        children[h][x] = nxt
        for q in (h, nxt):
            if q not in seen:
                seen.add(q)
                order.append(q)
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
                key = frozenset((x, sig.get(c, -1)) for x, c in children.get(h, {}).items())
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
