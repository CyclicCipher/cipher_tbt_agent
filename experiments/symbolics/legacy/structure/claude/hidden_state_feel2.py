"""Second hidden-state feel-experiment (docs/HIDDEN_STATE.md section 8): rules that counters, timers and a clock cannot
express. R1 an ordered combination lock, R2 a mode machine, R3 replay (a follower repeats the move made 3 steps
earlier); H1 toggle and H4 clock as regression checks. Situation features of part 26's in-context memory:
  N none;  T lines and circles (counts, residues, steps since, clock);  TD  T + a delay line (the last 6 actions);
  TDG  TD + a learned graph (an HMM with event-conditioned transitions, Baum-Welch over the stored levels);  O oracle.
    python -m structure.claude.hidden_state_feel2      (from experiments/symbolics; parallel, a few minutes)
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import structure.claude.hidden_state_feel as H1
from structure.claude.hidden_state_feel import DIRS, POS, Memory

H1.PRIOR = 0.3                                          # build check: without it, 54+ features' thetas drift up together

STEPS, LEVEL, SEEDS, NE = 3000, 100, 5, 7
PROBS = [0.15] * 4 + [0.4 / 3] * 3                     # 4 moves, 3 buttons (actions 4, 5, 6 = buttons 1, 2, 3)
KS, LAGS, KG = (2, 3, 4, 5, 6), 6, 6
RULES = ("R1 lock", "R2 mode", "R3 replay", "H1 toggle", "H4 clock")
ARMS = ("N", "T", "TD", "TDG", "O")
FIT_EVERY, STARTS, ITERS, PT, PE, STAY = 250, 12, 60, 0.01, 0.1, 0.8   # build check: 12 starts, 60 iterations, and
                                                                       # starts that keep their state (STAY)


class Graph:
    """Hidden states with a transition matrix per event (index NE is the identity, for padding); emissions are the
    outcome given (visible memory, value of its most trusted key). Fitted by Baum-Welch, batched over levels."""

    def __init__(self, n_mem, n_v, n_y, rng):
        self.n_mem, self.n_v, self.n_y, self.rng = n_mem, n_v, n_y, rng
        self.params = self.random()

    def random(self):
        r = self.rng
        Tr = np.concatenate([r.dirichlet(np.ones(KG), size=(NE, KG)), np.eye(KG)[None]], 0)
        Tr[:NE] = STAY * np.eye(KG)[None] + (1 - STAY) * Tr[:NE]     # a state persists until an event changes it
        Em = r.dirichlet(np.ones(self.n_y), size=(self.n_mem, self.n_v, KG)).transpose(0, 1, 3, 2)
        pi = 0.5 * np.eye(KG)[0] + 0.5 * r.dirichlet(np.ones(KG))
        return {"pi": pi, "Tr": Tr, "Em": Em}                               # Em: (mem, v, y, state)

    @staticmethod
    def arrays(levels):
        L, T = len(levels), max(len(l) for l in levels)
        e = np.full((L, T), NE)
        m, v, y = np.full((L, T), -1), np.zeros((L, T), int), np.zeros((L, T), int)
        for i, lev in enumerate(levels):
            for t, (ei, mi, vi, yi) in enumerate(lev):
                e[i, t], m[i, t], v[i, t], y[i, t] = ei, mi, vi, yi
        return e, m, v, y

    @staticmethod
    def emissions(p, m, v, y):
        b = np.ones(m.shape + (KG,))
        mask = m >= 0
        b[mask] = p["Em"][m[mask], v[mask], y[mask]]
        return b

    def estep(self, p, e, m, v, y):
        L, T = e.shape
        b = self.emissions(p, m, v, y)
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
            xi = alpha[:, t][:, :, None] * p["Tr"][e[:, t]] * w[:, None, :]
            np.add.at(cTr, e[:, t], xi)
        gam = alpha * beta
        gam /= gam.sum(-1, keepdims=True)
        cEm = np.zeros_like(p["Em"])
        mask = m >= 0
        np.add.at(cEm, (m[mask], v[mask], y[mask]), gam[mask])
        return float(np.log(c).sum()), gam[:, 0].sum(0), cTr, cEm

    def mstep(self, cpi, cTr, cEm):
        Tr = cTr[:NE] + PT
        Tr = np.concatenate([Tr / Tr.sum(-1, keepdims=True), np.eye(KG)[None]], 0)
        Em = cEm + PE
        return {"pi": (cpi + PT) / (cpi + PT).sum(), "Tr": Tr, "Em": Em / Em.sum(2, keepdims=True)}

    def fit(self, levels, trace=None):
        e, m, v, y = self.arrays(levels)
        best = None
        for s in range(STARTS):
            p = self.params if s == 0 else self.random()
            for _ in range(ITERS):
                ll, cpi, cTr, cEm = self.estep(p, e, m, v, y)
                if trace is not None:
                    trace.append(ll)
                p = self.mstep(cpi, cTr, cEm)
            ll = self.estep(p, e, m, v, y)[0]
            if best is None or ll > best[0]:
                best = (ll, p)
        self.params = best[1]
        return best[0]

    def states(self, levels):
        """The most probable state before each step's outcome, filtered forward (what the agent could know then)."""
        p = self.params
        e, m, v, y = self.arrays(levels)
        b = self.emissions(p, m, v, y)
        L, T = e.shape
        out = np.zeros((L, T), int)
        a = np.repeat(p["pi"][None], L, 0)
        for t in range(T):
            out[:, t] = a.argmax(1)
            a = a * b[:, t]
            a /= a.sum(1, keepdims=True)
            a = np.einsum("lk,lkj->lj", a, p["Tr"][e[:, t]])
        return [out[i, :len(lev)].tolist() for i, lev in enumerate(levels)]


def t_features(counts, last, t):
    f = []
    for e in range(NE):
        f += [counts[e] % k for k in KS] + [min(counts[e], 8), 9 if last[e] is None else min(t - last[e], 9)]
    return f + [t % k for k in KS]


def run(job):
    rule, arm, seed = job
    rng = np.random.default_rng(seed)
    follower = rule.startswith("R3")
    n_mem, K = (1, 1) if follower else (4, 9)
    mems = [Memory(K, 0) for _ in range(n_mem)]
    F = {"N": 0, "T": 54, "TD": 60, "TDG": 61, "O": 1}[arm]
    S = np.zeros((n_mem, STEPS, F), int)                # per memory: the stored situations, and their global steps
    at, n = np.zeros((n_mem, STEPS), int), [0] * n_mem
    for mm in mems:
        mm.theta, mm.rms = np.zeros(F), np.zeros(F)
    graph = Graph(n_mem, 3, 5 if follower else 2, np.random.default_rng(1000 + seed)) if arm == "TDG" else None
    levels, gstate, cur_g, truth = [], {}, None, []
    hits = {"300": [], "3000": []}
    for step in range(STEPS):
        if step % LEVEL == 0:
            counts, last, hist, t, q, mode = [0] * NE, [None] * NE, [], 0, 0, 0
            levels.append([])
            if graph is not None:
                cur_g = graph.params["pi"].copy()
        lag3 = hist[-3] if len(hist) >= 3 else -1
        hidden = {"R1": q, "R2": mode, "R3": lag3, "H1": counts[4] % 2, "H4": int(t % 6 < 3)}[rule[:2]]
        truth.append(hidden)
        feats = []
        if arm in ("T", "TD", "TDG"):
            feats += t_features(counts, last, t)
        if arm in ("TD", "TDG"):
            feats += [hist[-k] if len(hist) >= k else -1 for k in range(1, LAGS + 1)]
        if arm == "TDG":
            feats.append(int(cur_g.argmax()))
        if arm == "O":
            feats.append(hidden)
        a = int(rng.choice(NE, p=PROBS))
        mi, vals, yv = -1, None, None
        if follower:
            mi, vals, yv = 0, [0], (lag3 if 0 <= lag3 < 4 else 4)
        elif a < 4:
            opened = {"R1": q == 3, "R2": mode == 2, "H1": counts[4] % 2 == 1, "H4": t % 6 < 3}[rule[:2]]
            cells = rng.choice(3, size=8, p=[0.6, 0.2, 0.2])
            ahead = cells[POS.index(DIRS[a])]
            mi, vals, yv = a, list(cells) + [0], int(ahead == 1 or (ahead == 2 and not opened))
        if mi >= 0:
            M = (S[mi, :n[mi]] != np.array(feats, int)[None, :]).astype(float)
            att = mems[mi].parts(M, vals, 0)[-1]
            pred = int(np.argmax([float(att @ mems[mi].parts(M, vals, c)[-2]) for c in range(5 if follower else 2)]))
            ok = pred == yv
            mems[mi].learn(M, vals, yv)
            S[mi, n[mi]], at[mi, n[mi]] = feats, step
            n[mi] += 1
            if 100 <= step < 300:
                hits["300"].append(ok)
            if step >= 2000:
                hits["3000"].append(ok)
        levels[-1].append((a, mi, vals, yv))
        if graph is not None:                           # the filter: this step's outcome, then the event
            if mi >= 0:
                best = int(np.argmax(mems[mi].L.sum(0)))
                cur_g = cur_g * graph.params["Em"][mi, vals[best], yv]
                cur_g /= cur_g.sum()
            cur_g = cur_g @ graph.params["Tr"][a]
        if a >= 4:                                      # the hidden machines
            b = a - 3
            if q != 3:
                q = 1 if b == 1 else 2 if (b == 2 and q == 1) else 3 if (b == 3 and q == 2) else 0
            mode = (mode + 1) % 3 if b == 1 else 0 if b == 2 else mode
        counts[a] += 1
        last[a] = t
        hist.append(a)
        t += 1
        if graph is not None and step % FIT_EVERY == FIT_EVERY - 1:
            graph.fit(coded(levels, mems))
            gs = graph.states(coded(levels, mems))
            flat = np.array(sum(gs, []))
            for mi2 in range(n_mem):                    # every stored episode's graph state, under the new model
                S[mi2, :n[mi2], -1] = flat[at[mi2, :n[mi2]]]
            cur_g = refilter(graph, coded(levels, mems)[-1])
    out = {"rule": rule, "arm": arm, "seed": seed, "300": float(np.mean(hits["300"])),
           "3000": float(np.mean(hits["3000"]))}
    if graph is not None:
        gs = sum(graph.states(coded(levels, mems)), [])
        tr = np.array(truth[:len(gs)])
        out["purity"] = float(sum(np.bincount(tr[np.array(gs) == g] + 1).max() for g in set(gs)) / len(gs))
    if arm in ("T", "TD", "TDG"):
        out["top"] = top_names(np.mean([m.theta for m in mems], 0), arm)
    return out


def top_names(th, arm):
    names = []
    for e in range(NE):
        names += [f"action {e} count mod {k}" for k in KS] + [f"action {e} count", f"steps since action {e}"]
    names += [f"clock mod {k}" for k in KS]
    names += [f"action {k} back" for k in range(1, LAGS + 1)]
    names += ["graph state"]
    order = np.argsort(-th)[:3]
    return [(names[i], round(float(th[i]), 1)) for i in order]


def coded(levels, mems):
    """The levels with each emission's key value taken from its memory's most trusted key now."""
    best = [int(np.argmax(m.L.sum(0))) if len(m.Y) else 0 for m in mems]
    return [[(a, mi, (vals[best[mi]] if mi >= 0 else 0), (yv if mi >= 0 else 0)) for a, mi, vals, yv in lev]
            for lev in levels]


def refilter(graph, level):
    p = graph.params
    a = p["pi"].copy()
    for e, mi, v, y in level:
        if mi >= 0:
            a = a * p["Em"][mi, v, y]
            a /= a.sum()
        a = a @ p["Tr"][e]
    return a


def check_em():
    """Apparatus: Baum-Welch's likelihood never decreases, and it recovers a known 3-state mode machine."""
    rng = np.random.default_rng(0)
    levels, truth = [], []
    for _ in range(20):
        mode, lev = 0, []
        for _ in range(LEVEL):
            a = int(rng.choice(NE, p=PROBS))
            mi, v, y = -1, 0, 0
            if a < 4:
                mi, v = a, int(rng.choice(3, p=[0.6, 0.2, 0.2]))
                y = int(v == 1 or (v == 2 and mode != 2))
            truth.append(mode)
            lev.append((a, mi, v, y))
            mode = (mode + 1) % 3 if a == 4 else 0 if a == 5 else mode
        levels.append(lev)
    g = Graph(4, 3, 2, np.random.default_rng(1))
    trace = []
    g.fit(levels, trace)
    drops = [trace[i] - trace[i + 1] for i in range(len(trace) - 1) if (i + 1) % ITERS and trace[i + 1] < trace[i] - 1e-6]
    gs = np.array(sum(g.states(levels), []))
    tr = np.array(truth)
    purity = sum(np.bincount(tr[gs == s]).max() for s in set(gs.tolist())) / len(gs)
    print(f"EM check: likelihood decreases within a start: {len(drops)}; purity on a known mode machine {purity:.3f}",
          flush=True)


def main():
    check_em()
    t0 = time.time()
    jobs = [(r, a, s) for r in RULES for a in ARMS for s in range(SEEDS)]
    jobs.sort(key=lambda j: j[1] != "TDG")             # the slow arm first
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(run, jobs))
    for rule in RULES:
        print(f"\n{rule}")
        for arm in ARMS:
            rs = [r for r in res if r["rule"] == rule and r["arm"] == arm]
            line = (f"  {arm:3s}: 300 steps {np.mean([r['300'] for r in rs]):.3f}  3000 steps "
                    f"{np.mean([r['3000'] for r in rs]):.3f}  per seed {[round(r['3000'], 3) for r in rs]}")
            if arm == "TDG":
                line += f" | graph purity {[round(r['purity'], 3) for r in rs]}"
            if arm in ("T", "TD", "TDG"):
                line += f" | largest theta (seed 0) {rs[0]['top']}"
            print(line, flush=True)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
