"""An assoc column (ARCHITECTURE.md section 3): a reusable model over one reference frame.

- frame: the coordinate system and its operators (frames.py);
- state: where the column is in its frame now;
- content: its template, in a Spotlight lattice addressed by the frame's own coordinates (the cell index), kept apart
  from the structure (the offsets);
- what each action does to it, conditioned on context by attention (parts 20 and 25): keys are context features read
  in the column's own frame (the content at one position relative to it; another column's state; quotients; numeric
  features), each holding a table from its value to the operator seen; the query is the situation; the scores are
  learned by the transformer's own gradient written as a local rule (docs/ATTENTION.md). Nothing says what blocks a
  move: it is whatever the attention learns to read;
- drivers competing by evidence (part 13): "action" (the operator depends on which action) against "clock" (one
  operator on every action, as counters and timers behave).

An action, seen from a column: a key ("key", id); a click on the column's own cells ("self",); a click elsewhere
("at", x, y), whose effect on this column is a remote binding.
"""
import math
import zlib
from collections import Counter

import numpy as np

from arc_agent.spotlight import Spotlight

A = 0.02                          # pseudo-count: one consistent observation makes a deterministic outcome ~0.98
PMIN = 0.02
TOP = 8                           # (head, key) pairs read for a prediction (the most attended)
HEADS, DQ, ETA = 2, 64, 0.5       # attention (part 25): heads; hashed query dimensions; the local rule's rate
KEY_KINDS = ("u", "at", "col", "self", "mod", "num", "other")
QUERY_ONLY = ("self",)            # the column's own state is never an item to read
QUERY_KINDS = ("col", "mod", "self")  # the situation (the query): other columns' states, quotients, the own state,
                                       # and what lies beneath
DK = len(KEY_KINDS) + 15 + 5      # kind; (along, across) bins for positions; a quotient's modulus
NUM_MIN, NUM_EVERY = 6, 8         # the numeric expert: observations before its first fit; new ones between refits


def feature_kind(f, op=None):
    """What kind of context an expert reads, relative to the operator when it is a move (for priors across games)."""
    if f is None:
        return "unconditional"
    if f[0] == "at":
        if isinstance(op, tuple) and len(op) == 3 and all(isinstance(v, int) for v in op) and (op[0] or op[1]):
            return "ahead" if f[1] * op[0] + f[2] * op[1] > 0 else "elsewhere"
        return "elsewhere"
    return {"col": "other column", "self": "own state", "mod": "quotient", "num": "numeric",
            "and": "conjunction"}.get(f[0], "other")


class Numeric:
    """The numeric expert (part 23): a translation's displacement as a function of the column's numeric features
    (distances along four rays, its own position), fitted by the function finder of experiments A-B3
    (regression/claude/symreg2.py: residual-greedy, affine atoms, BIC). Refitted when its own prediction and the
    column's attention-weighted prediction both fail (part 25: compute is spent where error remains), less often
    after each refit that leaves the recent data unexplained."""

    def __init__(self, library=None):
        self.X, self.Y, self.fit, self.last, self.library = [], [], None, -10 ** 9, library or []
        self.every = NUM_EVERY                          # doubles after each refit that doesn't explain the data

    def predict(self, x):
        if self.fit is None:
            return None
        out = []
        for f in self.fit:
            if isinstance(f, int):
                out.append(f)
                continue
            with np.errstate(all="ignore"):
                v = float(f.predict(np.array([x], float))[0])
            if not math.isfinite(v):                    # outside where the formula is defined: no prediction
                return None
            out.append(int(round(v)))
        return tuple(out)

    def add(self, x, op, missed=True):
        self.X.append(tuple(float(v) for v in x))
        self.Y.append(op)
        if missed and len(self.X) >= NUM_MIN and len(self.X) - self.last >= self.every and self.predict(x) != op:
            self.refit()
            if any(self.predict(xx) != yy for xx, yy in zip(self.X[-NUM_MIN:], self.Y[-NUM_MIN:])):
                self.every *= 2                         # compute is spent where a formula can pay off

    def refit(self):
        from regression.claude.symreg2 import Finder
        X = np.array(self.X, float)
        fit = []
        for c in range(len(self.Y[0])):
            y = np.array([o[c] for o in self.Y], float)
            if np.all(y == y[0]) or c == 2:
                fit.append(int(Counter(int(v) for v in y).most_common(1)[0][0]))
            else:
                fit.append(Finder([f"n{i}" for i in range(X.shape[1])], library=self.library, affine=True,
                                  k_max=2).fit(X, y))
        self.fit, self.last = fit, len(self.X)

    def formulas(self):
        return [f.best for f in (self.fit or []) if not isinstance(f, int)]
P_IDENTITY = {"key": 0.5, "self": 0.5, "at": 0.99,  # innate: a click elsewhere rarely changes me (locality)
              "pfc": 0.5}                         # the PFC column: staying with the same model, before any evidence


class ConditionalOps:
    """What one action does to one column, by attention (part 25; docs/ATTENTION.md): the transformer's mechanism,
    with its learning written as a local rule.

    Keys are the context features (and the unconditional expert, key None); each has a key vector (its kind; where a
    position lies relative to the action's usual direction of motion; a quotient's modulus) and a bias per head. The
    query is the situation: a hashed one-hot of the non-positional features' values and of what lies beneath the
    column. Score s(h, key) = bias(h, key) + q . W_h . k; attention a = softmax over every (head, key); the prediction is
    the attention-weighted mixture of each attended (head, key)'s table from the key's value to the operator.

    Learning, from the log-likelihood's own gradient: the responsibilities r = a P(observed) / sum; W_h += ETA sum (r - a)
    q k^T and bias += ETA (r - a) (query x key x responsibility error: a three-factor Hebbian rule); each (head, key)'s
    table counts the observation with weight r (EM), so heads specialise in the situations they are attended in.
    Unseen values back off to the unconditional counts. A key starts at the description-length prior (part 23)."""

    def __init__(self, K, identity, p0, conditional=True, kind_prior=None, library=None):
        self.K, self.identity, self.p0, self.conditional = K, identity, p0, conditional
        self.kind_prior = kind_prior or {}              # kind -> log head start, learned on other games (part 23)
        self.numeric = Numeric(library) if conditional else None
        self.n0 = Counter()                             # every observation, unweighted: the backoff
        self.n = 0
        self.keys, self.index = [None], {None: 0}       # key list; key -> row
        self.bias = np.zeros((HEADS, 1))
        self.W = np.random.default_rng(7).normal(0, 0.01, (HEADS, DQ + 1, DK))
        self.tables = {}                                # (head, key) -> {value: [Counter(op), total weight]}
        self.direction, self._K = (0, 0), None          # the action's usual move; the key matrix for it
        self.last = {}                                  # key -> attention (summed over heads) at the last observation
        self.qfeats = []                                # the situation's features: the query (never keys)

    # ---- vectors ----------------------------------------------------------------------------------------------
    def _key_vector(self, f):
        v = np.zeros(DK)
        kind = "u" if f is None else f[0] if f[0] in KEY_KINDS else "other"
        v[KEY_KINDS.index(kind)] = 1.0
        if kind == "at":
            ux, uy = self.direction
            m = max(abs(ux), abs(uy), 1)
            along = (f[1] * ux + f[2] * uy) / m / m if (ux or uy) else 0.0
            across = abs(-f[1] * uy + f[2] * ux) / m / m if (ux or uy) else abs(f[1]) + abs(f[2])
            a = 0 if along < 0 else 1 if along < 1 else 2 if along < 2 else 3 if along < 3 else 4
            c = 0 if across < 0.5 else 1 if across < 1.5 else 2
            v[len(KEY_KINDS) + 3 * a + c] = 1.0
        elif kind == "mod":
            v[len(KEY_KINDS) + 15 + (f[1] - 2) % 5] = 1.0
        return v

    def _key_matrix(self):
        if self._K is None or len(self._K) != len(self.keys):
            self._K = np.array([self._key_vector(f) for f in self.keys])
        return self._K

    def _query(self, ctx):
        """The situation: a hashed one-hot (stable hash) of the other columns' states, the column's quotients and own
        state, and what lies beneath it, plus a bias."""
        idx = [DQ]
        for f in self.qfeats:
            if f in ctx:
                idx.append(zlib.crc32(repr((f, ctx.get(f))).encode()) % DQ)
        if ("at", 0, 0) in ctx:
            idx.append(zlib.crc32(repr(("under", ctx.get(("at", 0, 0)))).encode()) % DQ)
        return idx

    def _scores(self, ctx):
        q = self._query(ctx)
        qW = self.W[:, q, :].sum(1)                      # (heads, DK)
        return self.bias + qW @ self._key_matrix().T     # (heads, keys)

    # ---- the values ---------------------------------------------------------------------------------------------
    def _base(self, op):
        return self.p0 if op == self.identity else (1 - self.p0) / (self.K - 1)

    def _p0(self):
        """The backoff: named operators, and the probability of each operator never seen."""
        N = sum(self.n0.values())
        named = set(self.n0) | {self.identity}
        p = {op: (self.n0[op] + A * self._base(op)) / (N + A) for op in named}
        rest = A * (1 - self.p0) / (self.K - 1) / (N + A)
        return p, rest

    def _expert(self, h, f, v, p0, rest):
        if f is not None and f[0] == "num":            # the fitted formula, shared by both heads
            op = self.numeric.predict(v) if v is not None else None
            if op is None:
                return p0, rest
            p = {o: 0.01 * x for o, x in p0.items()}
            p[op] = p.get(op, 0.0) + 0.99
            return p, 0.01 * rest
        cell = self.tables.get((h, f), {}).get(v)
        if not cell or cell[1] <= 0:
            return p0, rest
        cnt, N = cell
        p = {op: (cnt[op] + A * p0.get(op, rest)) / (N + A) for op in set(p0) | set(cnt)}
        return p, A * rest / (N + A)

    # ---- prediction -------------------------------------------------------------------------------------------
    def dist(self, ctx):
        """The attention-weighted prediction: ({operator: p}, probability of each operator never seen). ctx is a dict
        of every feature, or a lazy context (brain.LazyContext) that computes only the features asked for. Only the
        TOP most attended (head, key) pairs that this context can feed are read."""
        p0, rest = self._p0()
        if not self.conditional:
            return p0, rest
        s = self._scores(ctx)
        order = np.argsort(-s, axis=None)
        picked = []
        for flat in order:
            h, k = divmod(int(flat), s.shape[1])
            f = self.keys[k]
            if f is None or (f in ctx and f[0] not in QUERY_ONLY):
                picked.append((h, k, s[h, k]))
                if len(picked) == TOP:
                    break
        m = max(x for _, _, x in picked)
        w = [math.exp(x - m) for _, _, x in picked]
        z = sum(w)
        out, r = Counter(), 0.0
        for (h, k, _), wi in zip(picked, w):
            f = self.keys[k]
            p, pr = self._expert(h, f, ctx.get(f) if f is not None else None, p0, rest)
            for op, x in p.items():
                out[op] += wi / z * x
            r += wi / z * pr
        return dict(out), r

    def best(self, ctx):
        p, _ = self.dist(ctx)
        op = max(p, key=p.get)
        return op, p[op]

    def predictive(self, ctx, op):
        p, rest = self.dist(ctx)
        return p.get(op, rest)

    # ---- learning ---------------------------------------------------------------------------------------------
    def _ensure(self, ctx, op):
        for f in ctx:                                   # the situation's own features form the query
            if f[0] in QUERY_KINDS and f not in self.qfeats:
                self.qfeats.append(f)
        ctx = {f: v for f, v in ctx.items() if f[0] not in QUERY_ONLY}
        kinds = Counter(f[0] for f in ctx)
        new = [f for f in ctx if f not in self.index]
        for f in new:                                   # description length: log K behind, K the keys of its kind;
            self.index[f] = len(self.keys)              # plus the head start its kind earned on other games
            self.keys.append(f)
            b = -math.log(kinds[f[0]]) + self.kind_prior.get(feature_kind(f, op), 0.0)
            self.bias = np.hstack([self.bias, np.full((HEADS, 1), b)])
        if isinstance(op, tuple) and len(op) == 3 and all(isinstance(v, int) for v in op) and (op[0] or op[1]):
            moves = Counter({o: c for o, c in self.n0.items() if isinstance(o, tuple) and len(o) == 3
                             and all(isinstance(v, int) for v in o) and (o[0] or o[1])})
            d = max(moves, key=moves.get)[:2] if moves else op[:2]
            if d != self.direction:
                self.direction, self._K = d, None
        if new:
            self._K = None

    def _learn(self, live, q, vals, op):
        """One step of the local rule on one observation: responsibilities from the current scores and tables, then
        W_h += eta (r - a) q k^T and bias += eta (r - a). vals: each live key's value.
        Returns (attention, responsibilities, the mixture's probability of op)."""
        p0, rest = self._p0()
        s = self.bias[:, live] + self.W[:, q, :].sum(1) @ self._key_matrix()[live].T
        a = np.exp(s - s.max())
        a /= a.sum()
        P = np.zeros_like(a)
        for j, (k, v) in enumerate(zip(live, vals)):
            for h in range(HEADS):
                p, pr = self._expert(h, self.keys[k], v, p0, rest)
                P[h, j] = p.get(op, pr)
        r = a * P
        pmix = float(r.sum())                           # the column's probability of what happened
        r /= max(pmix, 1e-300)
        err = r - a
        K = self._key_matrix()[live]
        for h in range(HEADS):                          # W_h += eta (r - a) q k^T ;  bias += eta (r - a)
            self.W[h, q, :] += ETA * (err[h] @ K)[None, :]
            self.bias[h, live] += ETA * err[h]
        return a, r, pmix

    def observe(self, ctx, op):
        """Responsibilities, then the local rule for the scores and EM for the tables."""
        if not self.conditional:
            self.n0[op] += 1
            self.n += 1
            return
        self._ensure(ctx, op)
        self._key_matrix()
        live = [k for k, f in enumerate(self.keys) if f is None or (f in ctx and f[0] not in QUERY_ONLY)]
        vals = [ctx.get(self.keys[k]) if self.keys[k] is not None else None for k in live]
        a, r, pmix = self._learn(live, self._query(ctx), vals, op)
        for j, (k, v) in enumerate(zip(live, vals)):    # EM: each (head, key) learns where it is responsible
            f = self.keys[k]
            if f is not None and f[0] == "num":
                continue
            for h in range(HEADS):
                cell = self.tables.setdefault((h, f), {}).setdefault(v, [Counter(), 0.0])
                cell[0][op] += r[h, j]
                cell[1] += r[h, j]
        self.last = {self.keys[k]: float(a[:, j].sum()) for j, k in enumerate(live)}
        self.n0[op] += 1
        if ("num",) in ctx and isinstance(op, tuple) and len(op) == 3:
            self.numeric.add(ctx[("num",)], op, missed=pmix < 0.5)
        self.n += 1

    def attended(self):
        return max(self.last, key=self.last.get) if self.last else None


class Column:
    def __init__(self, cid, frame, state, born, conditional=True, kind_prior=None, library=None):
        self.cid, self.frame, self.state, self.born = cid, frame, state, born
        self.conditional, self.kind_prior, self.library = conditional, kind_prior, library
        self.unit = 0                                   # the gcd of the moves seen: the lattice quotients are taken in
        self.content = Spotlight()                      # the template, addressed by cell index in the frame
        for i, c in enumerate(getattr(frame, "colours", [])):
            self.content.write(i, "colour", int(c))
        self.keys = {}                                  # akey -> ConditionalOps (the action driver)
        self.clock = ConditionalOps(frame.K, frame.identity, 0.5, conditional=False)
        self.ev = {"action": 0.0, "clock": 0.0}
        self.active, self.uses = True, 0
        self.reach = 4                                  # the context window's margin: grows with the largest move seen

    @property
    def kind(self):
        """What carries across levels: the frame type and the content's shape and colours."""
        f = self.frame
        c = f.cells0 - f.cells0.min(0)
        return (f.kind, tuple(map(tuple, c.tolist())), tuple(int(v) for v in getattr(f, "colours", [])))

    def akey(self, action, state=None):
        aid, x, y = action
        if aid != 6:
            return ("key", aid)
        s = self.state if state is None else state
        if getattr(self, "_cellset", (None,))[0] != s:  # cached per state: the click test runs inside the search
            self._cellset = (s, set(map(tuple, self.frame.place(s).tolist())))
        return ("self",) if (x, y) in self._cellset[1] else ("at", x, y)

    def belief(self, akey):
        if akey not in self.keys:
            self.keys[akey] = ConditionalOps(self.frame.K, self.frame.identity, P_IDENTITY[akey[0]],
                                             conditional=self.conditional and akey[0] != "at",
                                             kind_prior=self.kind_prior, library=self.library)
        return self.keys[akey]

    def driver(self):
        return "clock" if self.ev["clock"] > self.ev["action"] else "action"

    def dist(self, akey, ctx):
        b = self.clock if self.driver() == "clock" else self.belief(akey)
        return b.dist({} if b is self.clock else ctx)

    def learn(self, akey, ctx, op):
        """Prequential evidence for each driver, then both learn the observed operator."""
        b = self.belief(akey)
        lp_a = math.log(max(b.predictive(ctx, op), 1e-12))
        lp_c = math.log(max(self.clock.predictive({}, op), 1e-12))
        self.last_lp = lp_c if self.driver() == "clock" else lp_a   # this step's evidence, for the PFC (part 24)
        self.ev["action"] += lp_a
        self.ev["clock"] += lp_c
        b.observe(ctx, op)
        self.clock.observe({}, op)
        self.uses += 1
        if isinstance(op, tuple) and len(op) == 3 and all(isinstance(v, int) for v in op):
            self.reach = min(16, max(self.reach, abs(op[0]) + 1, abs(op[1]) + 1))   # local context only
            for v in op[:2]:
                if v:
                    self.unit = math.gcd(self.unit, abs(v))

    def harvest(self):
        """What this column learned that carries to other games: for each key, the kind of its most-attended expert,
        relative to the key's usual operator."""
        out = []
        for akey, b in self.keys.items():
            if akey[0] != "key" or b.n < 3:
                continue
            op = b.n0.most_common(1)[0][0]
            out.append(feature_kind(b.attended(), op))
        return out
