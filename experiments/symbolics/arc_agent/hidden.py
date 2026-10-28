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

Their states reach the visible columns as ordinary situation features ("col", cid): Brain.others().
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from arc_agent.column import Column
from arc_agent.frames import Choice

LAGS, KG = 4, 6
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


class Hidden:
    def __init__(self, conditional=True, kind_prior=None, seed=0):
        self.delays = [Column(-10 - k, Choice(range(8)), None, 0, conditional, kind_prior) for k in range(1, LAGS + 1)]
        self.graph_col = Column(-20, Choice(range(KG)), 0, 0, conditional, kind_prior)
        self.graph = Graph(np.random.default_rng(seed))
        self.codes = {"e": {}, "m": {}}
        self.levels = [[]]                              # per level: (brain step, event code)
        self.sources = {}                               # observer (cid, akey) -> its ConditionalOps (the evidence)
        self.key = {}                                   # observer -> the key index its evidence is read with
        self.belief = np.eye(KG)[0]
        self.steps = 0

    def columns(self):
        return self.delays + [self.graph_col]

    def states(self):
        return {c.cid: c.state for c in self.columns()}

    def code(self, kind, x, add=True):
        try:
            hash(x)
        except TypeError:
            x = repr(x)
        d = self.codes[kind]
        if x not in d and add:
            d[x] = len(d)
        return d.get(x)

    def start(self, new_level):
        """A new level, or a restart: the hidden machine starts again."""
        if self.levels[-1]:
            self.levels.append([])
        p = self.graph.params
        self.belief = p["pi"].copy() if p is not None else np.eye(KG)[0]
        for c in self.delays:
            c.state = None
        self.graph_col.state = int(self.belief.argmax())

    def emission(self, m, b, i):
        """Observer m's episode i, as (observer, value of its evidence key, operator) codes."""
        k = self.key.get(m)
        if k is None:
            k = self.key[m] = b.index.get(b.trusted(), 0)
        v = int(b.V[i, k]) + 2 if k < b.V.shape[1] else 0     # +2: the codes for "absent" and "never stored"
        return self.code("m", m), v, int(b.Y[i])

    def observe(self, t, aid, event, learned):
        """One step: what the visible columns just learned (learned: [(observer, its ConditionalOps)]) updates the
        graph's belief, the event moves every hidden-state column, and each learns what the event did to it. Returns True
        after a fit (the graph's stored states were rewritten: Brain.restate)."""
        try:
            hash(event)
        except TypeError:
            event = repr(event)
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
        new = {self.delays[0].cid: event}
        for k in range(1, LAGS):
            new[self.delays[k].cid] = old[self.delays[k - 1].cid]
        new[self.graph_col.cid] = int(self.belief.argmax())
        akey = ("key", aid)
        for c in self.columns():                        # each learns what the event did to it, like any column
            ctx = {("self",): old[c.cid], ("event",): event}
            ctx.update({("col", o): s for o, s in old.items() if o != c.cid})
            c.learn(akey, ctx, c.frame.op_between(old[c.cid], new[c.cid]), t)
            c.state = new[c.cid]
        self.levels[-1].append((t, ecode))
        self.steps += 1
        if self.steps % FIT_EVERY == 0:
            self.fit()
            return True
        return False

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

    def value_at(self, t):
        """The graph column's state before brain step t, under the current model (None if not stored)."""
        return getattr(self, "history_states", {}).get(t)
