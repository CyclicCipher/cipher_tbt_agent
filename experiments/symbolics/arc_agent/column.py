"""An assoc column (ARCHITECTURE.md section 3): a reusable model over one reference frame.

- frame: the coordinate system and its operators (frames.py);
- state: where the column is in its frame now;
- content: its template, in a Spotlight lattice addressed by the frame's own coordinates (the cell index), kept apart
  from the structure (the offsets);
- what each action does to it, conditioned on context by attention (part 20): experts compete by track record. One
  expert is unconditional ("this action applies operator X", the efference copy). Every other expert reads one context
  feature, in the column's own frame (the content at one position relative to it; the state of another column) and
  holds a table from that feature's value to the operator seen. Nothing says what blocks a move: it is whichever
  feature predicts best;
- drivers competing by evidence (part 13): "action" (the operator depends on which action) against "clock" (one
  operator on every action, as counters and timers behave).

An action, seen from a column: a key ("key", id); a click on the column's own cells ("self",); a click elsewhere
("at", x, y), whose effect on this column is a remote binding.
"""
import math
from collections import Counter

from arc_agent.spotlight import Spotlight

A = 0.02                          # pseudo-count: one consistent observation makes a deterministic outcome ~0.98
PMIN = 0.02
TOP = 8                           # experts used for a prediction (the most attended)
P_IDENTITY = {"key": 0.5, "self": 0.5, "at": 0.99}  # innate: a click elsewhere rarely changes me (locality)


class ConditionalOps:
    """What one action does to one column: attention over experts by track record (accumulated log predictive
    probability). Unseen values of a feature back off to the unconditional expert."""

    def __init__(self, K, identity, p0, conditional=True):
        self.K, self.identity, self.p0, self.conditional = K, identity, p0, conditional
        self.n0 = Counter()                             # the unconditional expert's counts
        self.tables = {}                                # feature -> {value: Counter(op)}
        self.ev = {None: 0.0}                           # expert -> log evidence (None = unconditional)
        self.n = 0

    def _base(self, op):
        return self.p0 if op == self.identity else (1 - self.p0) / (self.K - 1)

    def _p0(self):
        """The unconditional expert: named operators, and the probability of each operator never seen."""
        N = sum(self.n0.values())
        named = set(self.n0) | {self.identity}
        p = {op: (self.n0[op] + A * self._base(op)) / (N + A) for op in named}
        rest = A * (1 - self.p0) / (self.K - 1) / (N + A)
        return p, rest

    def _expert(self, f, v, p0, rest):
        if f is None:
            return p0, rest
        cnt = self.tables.get(f, {}).get(v)
        if not cnt:
            return p0, rest
        N = sum(cnt.values())
        p = {op: (cnt[op] + A * p0.get(op, rest)) / (N + A) for op in set(p0) | set(cnt)}
        return p, A * rest / (N + A)

    def _weights(self, ctx):
        feats = [None] + ([f for f in ctx if f in self.ev] if self.conditional else [])
        top = sorted(feats, key=lambda f: -self.ev[f])[:TOP]
        m = max(self.ev[f] for f in top)
        w = {f: math.exp(self.ev[f] - m) for f in top}
        z = sum(w.values())
        return {f: x / z for f, x in w.items()}

    def dist(self, ctx):
        """The attention-weighted prediction: ({operator: p}, probability of each operator never seen)."""
        p0, rest = self._p0()
        out, r = Counter(), 0.0
        for f, w in self._weights(ctx).items():
            p, pr = self._expert(f, ctx.get(f), p0, rest)
            for op, x in p.items():
                out[op] += w * x
            r += w * pr
        return dict(out), r

    def best(self, ctx):
        p, _ = self.dist(ctx)
        op = max(p, key=p.get)
        return op, p[op]

    def predictive(self, ctx, op):
        p, rest = self.dist(ctx)
        return p.get(op, rest)

    def observe(self, ctx, op):
        """Each expert is scored on its prediction of this outcome, then every expert learns it."""
        p0, rest = self._p0()
        if self.conditional:
            for f in ctx:
                if f not in self.ev:                    # a new feature: no track record, starts level with the prior
                    self.ev[f] = self.ev[None]
        for f in list(self.ev):
            if f is not None and f not in ctx:
                continue
            p, pr = self._expert(f, ctx.get(f), p0, rest)
            self.ev[f] += math.log(max(p.get(op, pr), 1e-12))
        self.n0[op] += 1
        if self.conditional:
            for f, v in ctx.items():
                self.tables.setdefault(f, {}).setdefault(v, Counter())[op] += 1
        self.n += 1

    def attended(self):
        return max(self.ev, key=self.ev.get)


class Column:
    def __init__(self, cid, frame, state, born, conditional=True):
        self.cid, self.frame, self.state, self.born = cid, frame, state, born
        self.conditional = conditional
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
        cells = self.frame.place(self.state if state is None else state)
        if any(cx == x and cy == y for cx, cy in cells.tolist()):
            return ("self",)
        return ("at", x, y)

    def belief(self, akey):
        if akey not in self.keys:
            self.keys[akey] = ConditionalOps(self.frame.K, self.frame.identity, P_IDENTITY[akey[0]],
                                             conditional=self.conditional and akey[0] != "at")
        return self.keys[akey]

    def driver(self):
        return "clock" if self.ev["clock"] > self.ev["action"] else "action"

    def dist(self, akey, ctx):
        b = self.clock if self.driver() == "clock" else self.belief(akey)
        return b.dist({} if b is self.clock else ctx)

    def learn(self, akey, ctx, op):
        """Prequential evidence for each driver, then both learn the observed operator."""
        b = self.belief(akey)
        self.ev["action"] += math.log(max(b.predictive(ctx, op), 1e-12))
        self.ev["clock"] += math.log(max(self.clock.predictive({}, op), 1e-12))
        b.observe(ctx, op)
        self.clock.observe({}, op)
        self.uses += 1
        if isinstance(op, tuple) and len(op) == 2 and all(isinstance(v, int) for v in op):
            self.reach = min(16, max(self.reach, abs(op[0]) + 1, abs(op[1]) + 1))   # local context only
