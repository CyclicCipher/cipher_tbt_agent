"""An assoc column (ARCHITECTURE.md section 3): a reusable model over one reference frame.

- frame: the coordinate system and its operators (frames.py);
- state: where the column is in its frame now;
- content: its template, in a Spotlight lattice addressed by the frame's own coordinates (the cell index), kept apart
  from the structure (the offsets);
- what each action does to it, conditioned on context by in-context attention (part 26; docs/ATTENTION.md section 7):
  every observation is kept as an episode; keys are context features read in the column's own frame (the content at
  one position relative to it; another column's state; quotients; numeric features); what a key's value means, and
  how much a key is trusted, are read from the episodes of similar situations, and only the similarity is learned.
  Nothing says what blocks a move: it is whatever the attention learns to read;
- drivers competing by evidence (part 13): "action" (the operator depends on which action) against "clock" (one
  operator on every action, as counters and timers behave).

An action, seen from a column: a key ("key", id); a click on the column's own cells ("self",); a click elsewhere
("at", x, y), whose effect on this column is a remote binding.
"""
import math
from collections import Counter

import numpy as np

from arc_agent.spotlight import Spotlight

A = 0.02                          # pseudo-count: one consistent observation makes a deterministic outcome ~0.98
PMIN = 0.02
TOP = 8                           # keys read for a prediction (the most attended)
ETA, MEMORY = 0.1, 1000           # in-context attention (part 26): the similarity's step; episodes kept per action
QUERY_ONLY = ("self",)            # the column's own state is never an item to read
QUERY_KINDS = ("col", "self")     # the situation (with what lies beneath): other columns' states, the own state.
                                  # Not quotients: five codes of one position, whose similarity weights add up
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
    """What one action does to one column, by in-context attention (part 26; docs/ATTENTION.md section 7).

    Every observation is kept as an episode: the situation (the other columns' states, what lies beneath the column,
    its own state), every key's value, the operator seen, and each key's prequential log probability of
    that operator (what it predicted before seeing it). A key is a context feature read in the column's own frame (the
    content at one position relative to it; another column's state; a quotient; the numeric expert), or the
    unconditional expert (key None).

    For the current situation q, a stored episode t counts kappa_t = exp(-sum_f theta_f [q_f != q_t,f]):
    - a key's value means the kappa-weighted counts of the operators seen with that value (unseen values back off to
      the kappa-weighted counts of every episode);
    - a key's score is its prior (description length, part 23, plus the warm-up head start) plus its evidence in
      similar situations, sum_t kappa_t log P_key,t; the TOP most attended keys are read.
    Only theta is learned, by the gradient of each new operator's log probability (RMSprop). theta = 0 is the track
    record of part 20; a large theta_f gives part 21's conjunction cases on feature f."""

    def __init__(self, K, identity, p0, conditional=True, kind_prior=None, library=None):
        self.K, self.identity, self.p0, self.conditional = K, identity, p0, conditional
        self.kind_prior = kind_prior or {}              # kind -> log head start, learned on other games (part 23)
        self.numeric = Numeric(library) if conditional else None
        self.n0 = Counter()                             # every observation, unweighted
        self.n = 0
        self.keys, self.index, self.prior = [None], {None: 0}, [0.0]
        self.sfeats, self.sindex = [], {}               # the situation's features (the query)
        self.theta, self.rms = np.zeros(0), np.zeros(0)
        self.codes, self.ops, self.opindex = {}, [], {}
        self.T = 0                                      # the memory: situation codes S, key value codes V (-1: the
        self.S = np.full((16, 0), -1, np.int32)         # key was absent), prequential log probabilities L, operators Y
        self.V = np.full((16, 1), -1, np.int32)
        self.L = np.zeros((16, 1))
        self.Y = np.zeros(16, np.int32)
        self.last, self.cache = {}, {}                  # attention at the last observation; per-situation reads

    # ---- codes ------------------------------------------------------------------------------------------------
    def _code(self, v, add=False):
        try:
            hash(v)
        except TypeError:
            v = repr(v)
        c = self.codes.get(v)
        if c is None and add:
            c = self.codes[v] = len(self.codes)
        return -2 if c is None else c                   # -2: a value never stored, so it matches no episode

    def _situation(self, ctx, add=False):
        q = np.full(len(self.sfeats), -1, np.int32)
        for i, f in enumerate(self.sfeats):
            if f == ("under",):
                if ("at", 0, 0) in ctx:
                    q[i] = self._code(("under", ctx.get(("at", 0, 0))), add)
            elif f in ctx:
                q[i] = self._code(ctx.get(f), add)
        return q

    # ---- the backoff ------------------------------------------------------------------------------------------
    def _base(self, op):
        return self.p0 if op == self.identity else (1 - self.p0) / (self.K - 1)

    def _p0(self):
        """Unweighted: the clock driver and clicks elsewhere, which read no context."""
        N = sum(self.n0.values())
        named = set(self.n0) | {self.identity}
        p = {op: (self.n0[op] + A * self._base(op)) / (N + A) for op in named}
        return p, A * (1 - self.p0) / (self.K - 1) / (N + A)

    # ---- reading ----------------------------------------------------------------------------------------------
    def _read(self, q):
        """For situation q: each episode's weight, the keys in order of score, the scores, and the backoff."""
        live = self.theta > 0                           # a feature with theta 0 changes no weight
        ck = tuple(q[live].tolist())
        hit = self.cache.get(ck)
        if hit is None:
            T = self.T
            kap = np.exp(-((self.S[:T, live] != q[live][None, :]).astype(float) @ self.theta[live]))
            s = np.array(self.prior) + kap @ self.L[:T, :len(self.keys)]
            cnt = np.bincount(self.Y[:T], weights=kap, minlength=len(self.ops))
            Z = kap.sum()
            p0 = {op: (cnt[i] + A * self._base(op)) / (Z + A) for i, op in enumerate(self.ops)}
            if self.identity not in p0:
                p0[self.identity] = A * self._base(self.identity) / (Z + A)
            rest = A * (1 - self.p0) / (self.K - 1) / (Z + A)
            hit = self.cache[ck] = (kap, np.argsort(-s), s, p0, rest, {})
        return hit

    def _table(self, hit, k, v):
        """What key k's value v means here: ({operator: p}, probability of each operator never seen)."""
        kap, _, _, p0, rest, tables = hit
        c = self._code(v)
        if (k, c) not in tables:
            m = self.V[:self.T, k] == c
            N = float(kap[m].sum())
            if N <= 0:
                tables[(k, c)] = (p0, rest)
            else:
                cnt = np.bincount(self.Y[:self.T][m], weights=kap[m], minlength=len(self.ops))
                p = {op: (cnt[i] + A * p0[op]) / (N + A) for i, op in enumerate(self.ops)}
                for op, x in p0.items():
                    if op not in p:
                        p[op] = A * x / (N + A)
                tables[(k, c)] = (p, A * rest / (N + A))
        return tables[(k, c)]

    def _expert(self, hit, k, ctx):
        f = self.keys[k]
        if f is None:
            return hit[3], hit[4]
        if f[0] == "num":                               # the fitted formula
            op = self.numeric.predict(ctx.get(f))
            if op is None:
                return hit[3], hit[4]
            p = {o: 0.01 * x for o, x in hit[3].items()}
            p[op] = p.get(op, 0.0) + 0.99
            return p, 0.01 * hit[4]
        return self._table(hit, k, ctx.get(f))

    def dist(self, ctx):
        """The prediction: ({operator: p}, probability of each operator never seen). ctx is a dict of every feature, or
        a lazy context (brain.LazyContext) that computes only the features asked for. Only the TOP most attended keys
        that this context can feed are read."""
        if not self.conditional or self.T == 0:
            return self._p0()
        hit = self._read(self._situation(ctx))
        picked = []
        for k in hit[1]:
            f = self.keys[k]
            if f is None or (f in ctx and f[0] not in QUERY_ONLY):
                picked.append(k)
                if len(picked) == TOP:
                    break
        s = hit[2][picked]
        w = np.exp(s - s.max())
        w /= w.sum()
        out, r = Counter(), 0.0
        for k, wi in zip(picked, w):
            p, pr = self._expert(hit, k, ctx)
            for op, x in p.items():
                out[op] += wi * x
            r += wi * pr
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
        for f in list(ctx) + [("under",)]:              # the situation's features
            if (f[0] in QUERY_KINDS or f == ("under",)) and f not in self.sindex:
                self.sindex[f] = len(self.sfeats)
                self.sfeats.append(f)
                self.theta, self.rms = np.append(self.theta, 0.0), np.append(self.rms, 0.0)
                self.S = np.hstack([self.S, np.full((len(self.S), 1), -1, np.int32)])
        feats = [f for f in ctx if f[0] not in QUERY_ONLY]
        kinds = Counter(f[0] for f in feats)
        new = [f for f in feats if f not in self.index]
        if new:                                         # the new keys' past: absent, so they backed off
            self.V = np.hstack([self.V, np.full((len(self.V), len(new)), -1, np.int32)])
            self.L = np.hstack([self.L, np.repeat(self.L[:, :1], len(new), 1)])
        for f in new:                                   # description length: log K behind, K the keys of its kind;
            self.index[f] = len(self.keys)              # plus the head start its kind earned on other games
            self.keys.append(f)
            self.prior.append(-math.log(kinds[f[0]]) + self.kind_prior.get(feature_kind(f, op), 0.0))
        if op not in self.opindex:
            self.opindex[op] = len(self.ops)
            self.ops.append(op)
        if self.T == len(self.Y):                       # grow the memory
            self.S = np.vstack([self.S, np.full_like(self.S, -1)])
            self.V = np.vstack([self.V, np.full_like(self.V, -1)])
            self.L = np.vstack([self.L, np.zeros_like(self.L)])
            self.Y = np.append(self.Y, np.zeros_like(self.Y))

    def _evaluate(self, ctx, op):
        """For a new observation (its keys and operator already coded by _ensure): each key's probability of op, the
        attention, the mixture's probability, and d log p / d theta. Nothing is stored."""
        T, y = self.T, self.opindex[op]
        live = np.array([k for k, f in enumerate(self.keys) if f is None or (f in ctx and f[0] not in QUERY_ONLY)])
        vals = np.array([self._code(ctx.get(self.keys[k]), True) if k else -1 for k in live], np.int32)
        q = self._situation(ctx, True)
        M = (self.S[:T] != q[None, :]).astype(float)
        kap = np.exp(-M @ self.theta)
        yv = (self.Y[:T] == y).astype(float)
        Z = kap.sum()
        p0y = (kap @ yv + A * self._base(op)) / (Z + A)
        B = (self.V[:T, live] == vals[None, :]).astype(float)
        C, N = (kap * yv) @ B, kap @ B
        P = (C + A * p0y) / (N + A)
        table = np.ones(len(live), bool)                # keys whose prediction is a kappa-weighted table
        coef = np.zeros(len(live))                      # the others: d log P / d p0
        P[0], table[0], coef[0] = p0y, False, 1.0 / p0y  # live[0] is the unconditional expert
        for j, k in enumerate(live):
            f = self.keys[k]
            if f is not None and f[0] == "num":
                pred = self.numeric.predict(ctx.get(f))
                P[j] = p0y if pred is None else 0.01 * p0y + 0.99 * (pred == op)
                table[j], coef[j] = False, (1.0 if pred is None else 0.01) / P[j]
        Lv = self.L[:T, live]
        s = np.array(self.prior)[live] + kap @ Lv
        a = np.exp(s - s.max())
        a /= a.sum()
        pmix = float(a @ P)                             # the column's probability of what happened
        r = a * P / max(pmix, 1e-300)
        g = None
        if T and len(self.theta):                       # d log p / d theta: through the evidence, then the values
            dp0 = -M.T @ (kap * (yv - p0y)) / (Z + A)
            u, w = np.where(table, r / (C + A * p0y), 0.0), np.where(table, r / (N + A), 0.0)
            g = (-M.T @ (kap * (Lv @ (r - a)))                         # which key is read
                 - M.T @ (kap * yv * (B @ u)) + A * u.sum() * dp0 + M.T @ (kap * (B @ w))   # what values mean
                 + float(r @ coef) * dp0)
        return live, vals, q, P, p0y, a, pmix, g

    def observe(self, ctx, op):
        """The operator's log probability from every key, the gradient step on theta, then the episode is stored."""
        if not self.conditional:
            self.n0[op] += 1
            self.n += 1
            return
        self._ensure(ctx, op)
        live, vals, q, P, p0y, a, pmix, g = self._evaluate(ctx, op)
        T, y = self.T, self.opindex[op]
        if g is not None:
            self.rms = 0.9 * self.rms + 0.1 * g ** 2
            self.theta = np.clip(self.theta + ETA * g / (np.sqrt(self.rms) + 1e-8), 0.0, 20.0)
        self.S[T] = q
        self.V[T] = -1
        self.V[T, live] = vals
        self.L[T] = math.log(max(p0y, 1e-300))          # an absent key backs off
        self.L[T, live] = np.log(np.maximum(P, 1e-300))
        self.Y[T] = y
        self.T += 1
        if self.T >= MEMORY:                            # forget the oldest quarter
            d = MEMORY // 4
            for arr in (self.S, self.V, self.L, self.Y):
                arr[:self.T - d] = arr[d:self.T]
            self.T -= d
        self.cache = {}
        self.last = {self.keys[k]: float(a[j]) for j, k in enumerate(live)}
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
