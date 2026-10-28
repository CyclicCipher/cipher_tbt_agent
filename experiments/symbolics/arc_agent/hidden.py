"""The hidden-state area (docs/HIDDEN_STATE.md sections 7-8; docs/AGENT.md part 27): columns whose state is not on the screen.

- Delay columns: the event 1, 2, ..., LAGS steps ago. Each is an ordinary Column over a Choice frame, and what an event
  does to it is learned like any column's (lag k copies lag k - 1): nothing but the wiring makes it a delay line.
  Similar columns specialise by lag, as time cells do by delay.
- A graph column: an ordinary Column over a Choice frame of KG states whose state is inferred, not read. An HMM with
  event-conditioned transitions, fitted by Baum-Welch on the stored history (cut into CHUNK-step pieces), whose
  evidence is what the visible columns saw: read from their own memories at fit time (no second copy), each episode's
  operator given the value of that column's most trusted key now. Starts keep their state (a state persists until an
  event changes it). After each fit the labels are matched to the old ones (the assignment that agrees most) and every
  stored episode's graph state is rewritten, wherever it is read.

- Counters, timers and a clock (part 28; HIDDEN_STATE.md section 7's lines and circles): per event type, a counter
  (read by others as its count mod 2 and mod 3 and capped at CAP) and a timer (steps since the event, capped at TCAP);
  one clock, steps since the level began (read modulo PERIODS). Line frames; each learns its own dynamics like any column.
- A latent column (part 28; the user's JEPA + SIGReg): h = W c over the temporal context c of the counters, timers
  and clock, trained as LeJEPA (a linear predictor per event plus SIGReg on h); its state is the signs of h.

Their states reach the visible columns as ordinary features ("col", cid, *reading): Brain.others(), Brain.context().
The planner rolls forward those some planned column relies on (Brain.joint(), Brain.model_step()).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from arc_agent.column import PMIN, Column
from arc_agent.frames import Choice, Line

LAGS, KG = 4, 6
MAX_EVENTS, CAP, TCAP = 8, 8, 9                         # event types with a counter and a timer; caps on their readings
PERIODS = (2, 3, 4, 6, 8, 12, 16)                       # the clock's readings: periods spaced about geometrically, as grid
                                                        # modules' scales are (amended after W15, seed 0: 2-6 can't say 8)
DH, LAM, NDIR, LR, JEPA_ITERS = 4, 1.0, 16, 0.01, 40    # the latent: dimensions, SIGReg weight, directions, Adam, steps
TK = np.linspace(0, 4, 17)                              # Epps-Pulley: points of t, trapezoid x Gaussian weights
TW = np.exp(-TK ** 2 / 2) * np.r_[0.5, np.ones(15), 0.5] * (TK[1] - TK[0])
FIT_EVERY, CHUNK, STARTS, ITERS, STAY, PT, PE = 100, 50, 8, 40, 0.8, 0.01, 0.1
HISTORY = 3000                                          # steps kept for fitting


class Graph:
    """The HMM: pi (KG), Tr (events + 1, KG, KG; the last is the identity, for padding), Em (observers, values, operators,
    KG). Sizes grow with the codes; a warm start pads the old parameters."""

    def __init__(self, rng):
        self.rng, self.params = rng, None

    def random(self, nE, nM, nV, nY):
        r = self.rng
        Tr = r.dirichlet(np.ones(KG), size=(nE, KG))
        Tr = np.concatenate([STAY * np.eye(KG)[None] + (1 - STAY) * Tr, np.eye(KG)[None]], 0)
        Em = r.dirichlet(np.ones(nY), size=(nM, nV, KG)).transpose(0, 1, 3, 2)
        return {"pi": 0.5 * np.eye(KG)[0] + 0.5 * r.dirichlet(np.ones(KG)), "Tr": Tr, "Em": Em}

    def padded(self, nE, nM, nV, nY):
        """The last parameters, grown to the current sizes (new events keep the state; new emissions are uniform)."""
        p = self.params
        Tr = np.concatenate([p["Tr"][:-1], np.repeat(np.eye(KG)[None], nE - (len(p["Tr"]) - 1), 0),
                             np.eye(KG)[None]], 0)
        Em = np.full((nM, nV, nY, KG), 1.0 / nY)
        m, v, y = (min(a, b) for a, b in zip(p["Em"].shape[:3], (nM, nV, nY)))
        Em[:m, :v, :y] = p["Em"][:m, :v, :y]               # (a column's evidence key may have changed: sizes shrink)
        Em /= Em.sum(2, keepdims=True)
        return {"pi": p["pi"], "Tr": Tr, "Em": Em}

    @staticmethod
    def emissions(p, L, T, em):
        b = np.ones((L, T, KG))
        if len(em):
            lb = np.zeros((L, T, KG))
            np.add.at(lb, (em[:, 0], em[:, 1]), np.log(p["Em"][em[:, 2], em[:, 3], em[:, 4]]))
            b = np.exp(lb)
        return b

    def estep(self, p, e, em):
        L, T = e.shape
        b = self.emissions(p, L, T, em)
        alpha, c = np.zeros((L, T, KG)), np.zeros((L, T))
        a = p["pi"][None] * b[:, 0]
        for t in range(T):
            if t:
                a = np.einsum("lk,lkj->lj", alpha[:, t - 1], p["Tr"][e[:, t - 1]]) * b[:, t]
            c[:, t] = a.sum(1)
            alpha[:, t] = a / c[:, t:t + 1]
        beta = np.ones((L, T, KG))
        cTr = np.zeros_like(p["Tr"])
        for t in range(T - 2, -1, -1):
            w = b[:, t + 1] * beta[:, t + 1] / c[:, t + 1:t + 2]
            beta[:, t] = np.einsum("lkj,lj->lk", p["Tr"][e[:, t]], w)
            np.add.at(cTr, e[:, t], alpha[:, t][:, :, None] * p["Tr"][e[:, t]] * w[:, None, :])
        gam = alpha * beta
        gam /= gam.sum(-1, keepdims=True)
        cEm = np.zeros_like(p["Em"])
        if len(em):
            np.add.at(cEm, (em[:, 2], em[:, 3], em[:, 4]), gam[em[:, 0], em[:, 1]])
        return float(np.log(c).sum()), gam[:, 0].sum(0), cTr, cEm

    @staticmethod
    def mstep(cpi, cTr, cEm):
        Tr = cTr[:-1] + PT
        Tr = np.concatenate([Tr / Tr.sum(-1, keepdims=True), np.eye(KG)[None]], 0)
        Em = cEm + PE
        return {"pi": (cpi + PT) / (cpi + PT).sum(), "Tr": Tr, "Em": Em / Em.sum(2, keepdims=True)}

    def fit(self, e, em, sizes):
        best = None
        for s in range(STARTS):
            p = self.padded(*sizes) if (s == 0 and self.params is not None) else self.random(*sizes)
            for _ in range(ITERS):
                ll, cpi, cTr, cEm = self.estep(p, e, em)
                p = self.mstep(cpi, cTr, cEm)
            ll = self.estep(p, e, em)[0]
            if best is None or ll > best[0]:
                best = (ll, p)
        return best[1]


class Jepa:
    """The latent column's encoder (part 28; HIDDEN_STATE.md section 6, LeJEPA): h = W c over the temporal context c;
    a linear predictor per event, h' ~ P_e h + b_e; loss = mean ||P_e h + b_e - h'||^2 + LAM * SIGReg(h) (mean form of
    the Epps-Pulley statistic over NDIR random directions). Adam. The context grows with the event types: W is padded."""

    def __init__(self, rng):
        self.rng, self.params, self.m, self.v, self.k = rng, None, {}, {}, 0

    def ensure(self, D, nE):
        if self.params is None:
            self.params = {"W": self.rng.normal(0, 1 / np.sqrt(max(D, 1)), (DH, D)),
                           "P": np.tile(np.eye(DH), (nE, 1, 1)), "b": np.zeros((nE, DH))}
        p = self.params
        if p["W"].shape[1] < D:
            p["W"] = np.hstack([p["W"], self.rng.normal(0, 0.1, (DH, D - p["W"].shape[1]))])
        if len(p["P"]) < nE:
            n = nE - len(p["P"])
            p["P"] = np.concatenate([p["P"], np.tile(np.eye(DH), (n, 1, 1))])
            p["b"] = np.concatenate([p["b"], np.zeros((n, DH))])
        for k, val in p.items():                        # Adam's moments, grown to match
            if k not in self.m or self.m[k].shape != val.shape:
                self.m[k], self.v[k] = np.zeros_like(val), np.zeros_like(val)

    @staticmethod
    def sigreg(Hs, U):
        S = Hs @ U
        n, M = S.shape
        arg = TK[:, None, None] * S[None]
        cs, sn = np.cos(arg), np.sin(arg)
        Cm, Sm = cs.mean(1), sn.mean(1)
        g = np.exp(-TK ** 2 / 2)[:, None]
        val = float((TW[:, None] * ((Cm - g) ** 2 + Sm ** 2)).sum(0).mean())
        dS = ((TW[:, None, None] * (2 * (Cm - g)[:, None, :] * (-TK[:, None, None] * sn)
                                    + 2 * Sm[:, None, :] * (TK[:, None, None] * cs))).sum(0)) / n / M
        return val, dS @ U.T

    def grads(self, X, A, X2, Cs, U):
        p = self.params
        W, P, b = p["W"], p["P"], p["b"]
        H, H2 = X @ W.T, X2 @ W.T
        e = np.einsum("nij,nj->ni", P[A], H) + b[A] - H2
        gE = 2 * e / len(X)
        gP, gb = np.zeros_like(P), np.zeros_like(b)
        np.add.at(gP, A, gE[:, :, None] * H[:, None, :])
        np.add.at(gb, A, gE)
        gW = np.einsum("nji,nj->ni", P[A], gE).T @ X - gE.T @ X2
        sig, gHs = self.sigreg(Cs @ W.T, U)
        return {"W": gW + LAM * gHs.T @ Cs, "P": gP, "b": gb}, float((e ** 2).sum(1).mean()), sig

    def train(self, X, A, X2, allc, iters):
        for _ in range(iters):
            U = self.rng.normal(size=(DH, NDIR))
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            Cs = allc[self.rng.choice(len(allc), size=min(500, len(allc)), replace=False)]
            g, _, _ = self.grads(X, A, X2, Cs, U)
            self.k += 1
            for key, gk in g.items():
                self.m[key] = 0.9 * self.m[key] + 0.1 * gk
                self.v[key] = 0.999 * self.v[key] + 0.001 * gk ** 2
                mh, vh = self.m[key] / (1 - 0.9 ** self.k), self.v[key] / (1 - 0.999 ** self.k)
                self.params[key] = self.params[key] - LR * mh / (np.sqrt(vh) + 1e-8)

    def code(self, c):
        """The latent column's state: the signs of h = W c."""
        if self.params is None:
            return 0
        W = self.params["W"]
        h = W[:, :len(c)] @ c
        return int(sum(1 << d for d in range(DH) if h[d] > 0))


class Hidden:
    def __init__(self, conditional=True, kind_prior=None, seed=0):
        self.conditional, self.kind_prior = conditional, kind_prior
        self.delays = [Column(-10 - k, Choice(range(8)), None, 0, conditional, kind_prior) for k in range(1, LAGS + 1)]
        self.graph_col = Column(-20, Choice(range(KG)), 0, 0, conditional, kind_prior)
        self.clock = Column(-30, Line(), 0, 0, conditional, kind_prior)
        self.latent = Column(-40, Choice(range(2 ** DH)), 0, 0, conditional, kind_prior)
        self.events = []                                # the event types with a counter and a timer, in order seen
        self.counters, self.timers = {}, {}             # event -> Column
        self.kinds = {c.cid: "delay %d" % (-10 - c.cid) for c in self.delays}
        self.kinds.update({self.graph_col.cid: "graph", self.clock.cid: "clock", self.latent.cid: "latent"})
        self.graph = Graph(np.random.default_rng(seed))
        self.jepa = Jepa(np.random.default_rng(seed + 1))
        self.codes = {"e": {}, "m": {}}
        self.levels = [[]]                              # per level: (brain step, event code)
        self.sources = {}                               # observer (cid, akey) -> its ConditionalOps (the evidence)
        self.key = {}                                   # observer -> the key index its evidence is read with
        self.trans = []                                 # the latent's training data: (context, event code, next context)
        self.contexts = {}                              # brain step -> the temporal context before it
        self.belief = np.eye(KG)[0]
        self.steps = 0
        self.history_states, self.latent_states = {}, {}

    # ---- the columns and how others read them ---------------------------------------------------------------------
    def columns(self):
        return (self.delays + [self.graph_col, self.clock, self.latent] + [self.counters[e] for e in self.events]
                + [self.timers[e] for e in self.events])

    def by_cid(self, cid):
        for c in self.columns():
            if c.cid == cid:
                return c
        return None

    def states(self):
        return {c.cid: c.state for c in self.columns()}

    def kind(self, cid):
        return self.kinds.get(cid)

    def readings(self, cid, state):
        """How another column reads this one: {reading: value}. Counters by their residues and their capped count,
        the clock by its residues; the others by their state (reading ())."""
        k = self.kinds.get(cid, "")
        if k.startswith("counter"):
            return {("mod", 2): state % 2, ("mod", 3): state % 3, ("cap",): min(state, CAP)}
        if k == "clock":
            return {("mod", m): state % m for m in PERIODS}
        return {(): state}

    def features(self, cid, state):
        """The situation features and keys this column gives another column's context."""
        return {("col", cid) + r: v for r, v in self.readings(cid, state).items()}

    def reading(self, cid, state, r):
        return self.readings(cid, state).get(tuple(r))

    def has(self, f):
        """Does feature f = ("col", cid, *reading) name a reading of a hidden-state column?"""
        return f[1] in self.kinds and tuple(f[2:]) in self.readings(f[1], 0)

    def context(self, c, state, states, event):
        """A hidden-state column's own context: its state and the event; a delay also reads the delay before it."""
        ctx = {("self",): state, ("event",): event}
        k = self.kinds[c.cid]
        if k.startswith("delay") and c.cid != self.delays[0].cid:
            prev = c.cid + 1
            ctx[("col", prev)] = states.get(prev)
        return ctx

    # ---- one step -------------------------------------------------------------------------------------------------
    def code(self, kind, x, add=True):
        try:
            hash(x)
        except TypeError:
            x = repr(x)
        d = self.codes[kind]
        if x not in d and add:
            d[x] = len(d)
        return d.get(x)

    def ensure_event(self, event):
        if event in self.counters or len(self.events) >= MAX_EVENTS:
            return
        i = len(self.events)
        self.events.append(event)
        self.counters[event] = Column(-100 - i, Line(), 0, 0, self.conditional, self.kind_prior)
        self.timers[event] = Column(-200 - i, Line(cap=TCAP), TCAP, 0, self.conditional, self.kind_prior)
        self.kinds[-100 - i] = "counter %s" % (event,)
        self.kinds[-200 - i] = "timer %s" % (event,)

    def start(self, new_level):
        """A new level, or a restart: every hidden-state column starts again."""
        if self.levels[-1]:
            self.levels.append([])
        p = self.graph.params
        self.belief = p["pi"].copy() if p is not None else np.eye(KG)[0]
        for cid, s in self.reset_states().items():
            self.by_cid(cid).state = s

    def reset_states(self):
        out = {c.cid: None for c in self.delays}
        p = self.graph.params
        out[self.graph_col.cid] = int((p["pi"] if p is not None else np.eye(KG)[0]).argmax())
        out[self.clock.cid] = 0
        for e in self.events:
            out[self.counters[e].cid], out[self.timers[e].cid] = 0, TCAP
        out[self.latent.cid] = self.jepa.code(self.context_vector({**self.states(), **out}))
        return out

    def advance(self, states, event):
        """The innate dynamics of the delays, counters, timers and clock under an event (the graph and the latent are
        inferred, not given)."""
        new = {self.delays[0].cid: event}
        for k in range(1, LAGS):
            new[self.delays[k].cid] = states[self.delays[k - 1].cid]
        new[self.clock.cid] = states[self.clock.cid] + 1
        for e in self.events:
            n, s = states[self.counters[e].cid], states[self.timers[e].cid]
            new[self.counters[e].cid] = n + (e == event)
            new[self.timers[e].cid] = 0 if e == event else min(s + 1, TCAP)
        return new

    def context_vector(self, states):
        """The temporal context the latent is read from: phases and caps of each counter, traces of each timer, the
        clock's phases."""
        c = []
        for e in self.events:
            n, s = states[self.counters[e].cid], states[self.timers[e].cid]
            c += [np.cos(np.pi * n), np.cos(2 * np.pi * n / 3), np.sin(2 * np.pi * n / 3), min(n, CAP) / CAP,
                  np.exp(-s / 2), np.exp(-s / 4)]
        t = states[self.clock.cid]
        for m in PERIODS:
            c += [np.cos(2 * np.pi * t / m), np.sin(2 * np.pi * t / m)]
        return np.array(c)

    def emission(self, m, b, i):
        """Observer m's episode i, as (observer, value of its evidence key, operator) codes."""
        k = self.key.get(m)
        if k is None:
            k = self.key[m] = b.index.get(b.trusted(), 0)
        v = int(b.V[i, k]) + 2 if k < b.V.shape[1] else 0     # +2: the codes for "absent" and "never stored"
        return self.code("m", m), v, int(b.Y[i])

    def observe(self, t, aid, event, learned):
        """One step: what the visible columns just learned (learned: [(observer, its ConditionalOps)]) updates the
        graph's belief; the event moves every hidden-state column; each learns what the event did to it. Returns True
        after a fit (stored graph and latent states were rewritten: Brain.restate)."""
        try:
            hash(event)
        except TypeError:
            event = repr(event)
        self.ensure_event(event)
        ecode = self.code("e", event)
        for m, b in learned:
            self.sources[m] = b
        em = [self.emission(m, b, b.T - 1) for m, b in learned]
        p = self.graph.params
        if p is not None:                               # the belief: this step's evidence, then the event
            for m, v, y in em:
                if m < p["Em"].shape[0] and v < p["Em"].shape[1] and y < p["Em"].shape[2]:
                    self.belief = self.belief * p["Em"][m, v, y]
            self.belief /= self.belief.sum()
            if ecode < len(p["Tr"]) - 1:
                self.belief = self.belief @ p["Tr"][ecode]
        old = self.states()
        new = self.advance(old, event)
        new[self.graph_col.cid] = int(self.belief.argmax())
        c0, c1 = self.context_vector(old), self.context_vector(new)
        new[self.latent.cid] = self.jepa.code(c1)
        self.contexts[t] = c0
        self.trans.append((c0, ecode, c1))
        akey = ("key", aid)
        for c in self.columns():                        # each learns what the event did to it, like any column
            ctx = self.context(c, old[c.cid], old, event)
            c.learn(akey, ctx, c.frame.op_between(old[c.cid], new[c.cid]), t)
            c.state = new[c.cid]
        self.levels[-1].append((t, ecode))
        self.steps += 1
        if self.steps % FIT_EVERY == 0:
            self.fit()
            return True
        return False

    def predict(self, cid, state, states, aid, event):
        """For the planner: this column's next state under an action, from its own learned model, and the outcome's
        uncertainty in bits (0: certain)."""
        c = self.by_cid(cid)
        p, rest = c.dist(("key", aid), self.context(c, state, states, event))
        op = max(p, key=p.get)
        if p[op] >= 1 - PMIN:
            return c.frame.apply(state, op), 0.0
        return c.frame.apply(state, op), min(1.0, -sum(x * np.log2(x) for x in p.values() if x > 0))

    # ---- fitting ------------------------------------------------------------------------------------------------
    def evidence(self):
        """Every visible column's stored episodes as emissions, read with each one's most trusted key now:
        {brain step: [(m, v, y)]}."""
        self.key = {m: b.index.get(b.trusted(), 0) for m, b in self.sources.items()}
        out = {}
        for m, b in self.sources.items():
            for i in range(b.T):
                out.setdefault(int(b.when[i]), []).append(self.emission(m, b, i))
        return out

    def pieces(self, ev):
        """The stored history cut into CHUNK-step pieces: events (L, T) and emissions [piece, step, m, v, y]."""
        steps = sum(([s for s in lev] for lev in self.levels), [])[-HISTORY:]
        chunks, cur, lev_starts = [], [], set()
        n = 0
        for lev in self.levels:
            lev_starts.add(n)
            n += len(lev)
        n0 = n - len(steps)
        for i, s in enumerate(steps):
            if cur and (len(cur) == CHUNK or (i + n0) in lev_starts):
                chunks.append(cur)
                cur = []
            cur.append(s)
        if cur:
            chunks.append(cur)
        nE = len(self.codes["e"])
        e = np.full((len(chunks), CHUNK), nE)
        em = []
        for li, ch in enumerate(chunks):
            for ti, (t, ec) in enumerate(ch):
                e[li, ti] = ec
                em += [(li, ti, m, v, y) for m, v, y in ev.get(t, [])]
        return e, np.array(em, int).reshape(-1, 5)

    def sizes(self, ev):
        allem = [x for ems in ev.values() for x in ems]
        return (len(self.codes["e"]), max(1, len(self.codes["m"])), max([v for _, v, _ in allem], default=0) + 1,
                max([y for _, _, y in allem], default=0) + 1)

    def filtered(self, p, ev, levels=None):
        """The most probable state before each stored step, filtered through each level ({brain step: state}),
        and the belief after the last step."""
        out, a = {}, p["pi"].copy()
        for lev in (levels or self.levels):
            a = p["pi"].copy()
            for t, ec in lev:
                out[t] = int(a.argmax())
                for m, v, y in ev.get(t, []):
                    a = a * p["Em"][m, v, y]
                a /= a.sum()
                a = a @ p["Tr"][ec]
        return out, a

    def fit(self):
        """Refit the graph (Baum-Welch, labels kept) and train the latent (LeJEPA); rewrite their stored states."""
        ev = self.evidence()
        e, em = self.pieces(ev)
        sizes = self.sizes(ev)
        p = self.graph.fit(e, em, sizes)
        if self.graph.params is not None:               # keep the labels: the assignment that agrees most
            old, new = self.filtered(self.graph.padded(*sizes), ev)[0], self.filtered(p, ev)[0]
            agree = np.zeros((KG, KG))
            for t, s in new.items():
                agree[s, old[t]] += 1
            rows, cols = linear_sum_assignment(-agree)
            perm = np.empty(KG, int)
            perm[cols] = rows                           # new label rows[i] becomes old label cols[i]
            p = {"pi": p["pi"][perm], "Tr": np.concatenate([p["Tr"][:-1][:, perm][:, :, perm], p["Tr"][-1:]], 0),
                 "Em": p["Em"][..., perm]}
        self.graph.params = p
        self.history_states, _ = self.filtered(p, ev)
        _, a = self.filtered(p, ev, [self.levels[-1]])   # the current level's belief, under the new model
        self.belief = a
        self.graph_col.state = int(a.argmax())
        trans = self.trans[-HISTORY:]                   # the latent: LeJEPA on the stored contexts
        D = max(len(c1) for _, _, c1 in trans)
        pad = lambda c: np.pad(c, (0, D - len(c)))
        X = np.array([pad(c0) for c0, _, _ in trans])
        X2 = np.array([pad(c1) for _, _, c1 in trans])
        A = np.array([ec for _, ec, _ in trans])
        self.jepa.ensure(D, len(self.codes["e"]))
        self.jepa.train(X, A, X2, np.vstack([X, X2[-1:]]), JEPA_ITERS)
        self.latent_states = {t: self.jepa.code(c) for t, c in self.contexts.items()}
        self.latent.state = self.jepa.code(self.context_vector(self.states()))

    def restated(self):
        """The hidden-state columns whose stored values a fit rewrote: [(cid, value at a brain step)]."""
        return [(self.graph_col.cid, self.history_states.get), (self.latent.cid, self.latent_states.get)]
