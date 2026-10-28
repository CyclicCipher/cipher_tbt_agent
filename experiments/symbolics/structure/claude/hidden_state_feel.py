"""Feel-experiment for docs/HIDDEN_STATE.md section 6: hidden doors (toggle, threshold, timer, clock) on the synthetic
grid of attention_feel.py, read through part 26's in-context memory with different situation features:
  N  none;  T  innate temporal frames (counter residues, capped count, steps since the last press, clock residues);
  J  a LeJEPA latent over a temporal context (predictor per action, SIGReg);  O  the true open/closed state.
    python -m structure.claude.hidden_state_feel      (from experiments/symbolics; parallel, about a minute)
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

A0, ETA = 0.02, 0.1
STEPS, LEVEL, SEEDS, PRESS = 3000, 50, 5, 0.2
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
POS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]
KS, TAUS = (2, 3, 4, 5, 6), (1, 2, 4, 8)
RULES = ("H1 toggle", "H2 threshold", "H3 timer", "H4 clock")
ARMS = ("N", "T", "J", "O")
T_NAMES = [f"presses mod {k}" for k in KS] + ["presses (cap 8)", "since press (cap 9)"] + [f"steps mod {k}" for k in KS]
DEFINING = {"H1": 0, "H2": 5, "H3": 6, "H4": 11}
D, DH, LAM, NDIR, LR = 25, 4, 1.0, 16, 0.01
TK = np.linspace(0, 4, 17)
TW = np.exp(-TK ** 2 / 2) * np.r_[0.5, np.ones(15), 0.5] * (TK[1] - TK[0])   # trapezoid x Gaussian weight


def is_open(rule, n, since, t):
    if rule.startswith("H1"):
        return n % 2 == 1
    if rule.startswith("H2"):
        return n >= 3
    if rule.startswith("H3"):
        return since is not None and 1 <= since <= 4
    return t % 6 < 3


def t_features(n, since, t):
    return [n % k for k in KS] + [min(n, 8), 9 if since is None else min(since, 9)] + [t % k for k in KS]


def context(n, since, t):
    c = []
    for k in KS:
        c += [np.cos(2 * np.pi * n / k), np.sin(2 * np.pi * n / k)]
    c.append(min(n, 8) / 8)
    c += [0.0 if since is None else np.exp(-since / tau) for tau in TAUS]
    for k in KS:
        c += [np.cos(2 * np.pi * t / k), np.sin(2 * np.pi * t / k)]
    return np.array(c)


class Memory:
    """Part 26's in-context memory for one action, with any non-negative per-feature mismatch M (T x F)."""

    def __init__(self, K, F):
        self.V, self.Y, self.L = np.zeros((0, K), int), np.zeros(0, int), np.zeros((0, K))
        self.theta, self.rms = np.zeros(F), np.zeros(F)

    def parts(self, M, vals, y):
        kap = np.exp(-M @ self.theta)
        yv = (self.Y == y).astype(float)
        p0 = (kap @ yv + A0 * 0.5) / (kap.sum() + A0)
        B = (self.V == np.asarray(vals)[None, :]).astype(float)
        C, N = (kap * yv) @ B, kap @ B
        P = (C + A0 * p0) / (N + A0)
        E = kap @ self.L
        a = np.exp(E - E.max())
        return kap, yv, p0, B, C, N, P, a / a.sum()

    def logp(self, M, vals, y):
        *_, P, a = self.parts(M, vals, y)
        return float(np.log(a @ P))

    def grad(self, M, vals, y):
        kap, yv, p0, B, C, N, P, a = self.parts(M, vals, y)
        r = a * P / (a @ P)
        dp0 = -M.T @ (kap * (yv - p0)) / (kap.sum() + A0)
        u, w = r / (C + A0 * p0), r / (N + A0)
        g = -M.T @ (kap * (self.L @ (r - a))) - M.T @ (kap * yv * (B @ u)) + A0 * u.sum() * dp0 + M.T @ (kap * (B @ w))
        return g, P, a

    def learn(self, M, vals, y):
        g, P, a = self.grad(M, vals, y)
        if len(self.Y) and len(self.theta):
            self.rms = 0.9 * self.rms + 0.1 * g ** 2
            self.theta = np.clip(self.theta + ETA * g / (np.sqrt(self.rms) + 1e-8), 0.0, 20.0)
        self.V = np.vstack([self.V, vals])
        self.Y = np.append(self.Y, y)
        self.L = np.vstack([self.L, np.log(P)])


class Jepa:
    """h = W c; predictor h' ~ P_a h + b_a; loss = mean ||P_a h + b_a - h'||^2 + LAM SIGReg(h). Adam."""

    def __init__(self, rng, nA=5):
        self.rng = rng
        self.params = {"W": rng.normal(0, 1 / np.sqrt(D), (DH, D)), "P": np.tile(np.eye(DH), (nA, 1, 1)),
                       "b": np.zeros((nA, DH))}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.k = 0
        self.X, self.A, self.X2, self.all = [], [], [], []

    @staticmethod
    def sigreg(Hs, U):
        """Mean over directions of the Epps-Pulley statistic (mean form) and its gradient with respect to Hs."""
        S = Hs @ U                                      # (n, M)
        n, M = S.shape
        arg = TK[:, None, None] * S[None]               # (K, n, M)
        cs, sn = np.cos(arg), np.sin(arg)
        Cm, Sm = cs.mean(1), sn.mean(1)                 # (K, M)
        g = np.exp(-TK ** 2 / 2)[:, None]
        val = float((TW[:, None] * ((Cm - g) ** 2 + Sm ** 2)).sum(0).mean())
        dS = ((TW[:, None, None] * (2 * (Cm - g)[:, None, :] * (-TK[:, None, None] * sn)
                                    + 2 * Sm[:, None, :] * (TK[:, None, None] * cs))).sum(0)) / n / M
        return val, dS @ U.T

    def loss(self, p, U, Cs):
        X, X2, A = np.array(self.X), np.array(self.X2), np.array(self.A)
        W, P, b = p["W"], p["P"], p["b"]
        H, H2 = X @ W.T, X2 @ W.T
        e = np.einsum("nij,nj->ni", P[A], H) + b[A] - H2
        Lp = float((e ** 2).sum(1).mean())
        gE = 2 * e / len(X)
        gP, gb = np.zeros_like(P), np.zeros_like(b)
        np.add.at(gP, A, gE[:, :, None] * H[:, None, :])
        np.add.at(gb, A, gE)
        gW = np.einsum("nji,nj->ni", P[A], gE).T @ X - gE.T @ X2
        sig, gHs = self.sigreg(Cs @ W.T, U)
        gW = gW + LAM * gHs.T @ Cs
        return Lp + LAM * sig, Lp, sig, {"W": gW, "P": gP, "b": gb}

    def batch(self):
        U = self.rng.normal(size=(DH, NDIR))
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        allc = np.array(self.all)
        Cs = allc[self.rng.choice(len(allc), size=min(500, len(allc)), replace=False)]
        return U, Cs

    def train(self, iters=10):
        if len(self.X) < 10:
            return
        for _ in range(iters):
            U, Cs = self.batch()
            _, _, _, grads = self.loss(self.params, U, Cs)
            self.k += 1
            for key, gk in grads.items():
                self.m[key] = 0.9 * self.m[key] + 0.1 * gk
                self.v[key] = 0.999 * self.v[key] + 0.001 * gk ** 2
                mh, vh = self.m[key] / (1 - 0.9 ** self.k), self.v[key] / (1 - 0.999 ** self.k)
                self.params[key] = self.params[key] - LR * mh / (np.sqrt(vh) + 1e-8)


def run(job):
    rule, arm, seed = job
    rng = np.random.default_rng(seed)
    F = {"N": 0, "T": 12, "J": DH, "O": 1}[arm]
    mems = [Memory(9, F) for _ in range(4)]
    store = [[] for _ in range(4)]                      # per action: the stored situations (codes, or contexts for J)
    jepa = Jepa(np.random.default_rng(100 + seed)) if arm == "J" else None
    hits = {"300": [], "3000": []}
    opened_log, c_log = [], []
    prev = None
    for step in range(STEPS):
        if step % LEVEL == 0:
            n, last, t, prev = 0, None, 0, None
        since = None if last is None else t - last
        c = context(n, since, t)
        opened = is_open(rule, n, since, t)
        if jepa is not None:
            if prev is not None:
                jepa.X.append(prev[0]), jepa.A.append(prev[1]), jepa.X2.append(c)
            jepa.all.append(c)
            opened_log.append(opened), c_log.append(c)
        a = 4 if rng.random() < PRESS else int(rng.integers(4))
        cells = rng.choice(3, size=8, p=[0.6, 0.2, 0.2])
        if a < 4:
            ahead = cells[POS.index(DIRS[a])]
            y = int(ahead == 1 or (ahead == 2 and not opened))
            vals = list(cells) + [0]
            if arm == "J":
                W = jepa.params["W"]
                Hs = np.array(store[a]).reshape(len(store[a]), D) @ W.T
                M = (Hs - W @ c) ** 2 / 2
                s = c
            else:
                s = {"N": [], "T": t_features(n, since, t), "O": [int(opened)]}[arm]
                M = (np.array(store[a]).reshape(len(store[a]), F) != np.array(s)[None, :]).astype(float)
            *_, P1, att = mems[a].parts(M, vals, 1)
            p = float(att @ P1)
            correct = (p > 0.5) == bool(y)
            mems[a].learn(M, vals, y)
            store[a].append(s)
            if 100 <= step < 300:
                hits["300"].append(correct)
            if step >= 2000:
                hits["3000"].append(correct)
        else:
            n, last = n + 1, t
        prev = (c, a)
        t += 1
        if jepa is not None and step % 25 == 24:
            jepa.train(10)
    out = {"rule": rule, "arm": arm, "seed": seed, "300": float(np.mean(hits["300"])),
           "3000": float(np.mean(hits["3000"]))}
    if arm == "T":
        out["theta"] = np.mean([m.theta for m in mems], 0).round(2).tolist()
    if arm == "J":
        H = np.array(c_log) @ jepa.params["W"].T
        cov = np.corrcoef(H.T)
        U, Cs = jepa.batch()
        _, Lp, sig, _ = jepa.loss(jepa.params, U, Cs)
        X1 = np.c_[H, np.ones(len(H))]
        beta = np.linalg.lstsq(X1, np.array(opened_log, float), rcond=None)[0]
        out.update(var=H.var(0).round(2).tolist(), maxcorr=float(np.abs(cov - np.eye(DH)).max()), Lp=Lp, sig=sig,
                   probe=float(np.mean((X1 @ beta > 0.5) == np.array(opened_log))),
                   base=float(max(np.mean(opened_log), 1 - np.mean(opened_log))))
    return out


def check_gradients():
    """Finite differences: the memory's theta gradient with a continuous mismatch, and JEPA + SIGReg's gradient."""
    rng = np.random.default_rng(3)
    mem = Memory(3, 2)
    for _ in range(80):
        mem.V = np.vstack([mem.V, rng.integers(3, size=3)])
        mem.Y = np.append(mem.Y, rng.integers(2))
        mem.L = np.vstack([mem.L, np.log(rng.uniform(0.1, 1, 3))])
    mem.theta = np.array([0.6, 0.3])
    M, vals = rng.uniform(0, 2, (80, 2)), [1, 0, 2]
    g = mem.grad(M, vals, 1)[0]
    num = []
    for f in range(2):
        e = np.zeros(2)
        e[f] = 1e-6
        mem.theta += e
        up = mem.logp(M, vals, 1)
        mem.theta -= 2 * e
        dn = mem.logp(M, vals, 1)
        mem.theta += e
        num.append((up - dn) / 2e-6)
    print("memory theta gradient: analytic", np.round(g, 6), "numeric", np.round(num, 6), flush=True)
    j = Jepa(np.random.default_rng(4))
    for i in range(60):
        j.X.append(rng.normal(size=D)), j.A.append(int(rng.integers(5))), j.X2.append(rng.normal(size=D))
        j.all.append(rng.normal(size=D))
    U, Cs = j.batch()
    _, _, _, gr = j.loss(j.params, U, Cs)
    for key, idx in (("W", (1, 3)), ("P", (2, 1, 0)), ("b", (4, 2))):
        p = {k: v.copy() for k, v in j.params.items()}
        p[key][idx] += 1e-6
        up = j.loss(p, U, Cs)[0]
        p[key][idx] -= 2e-6
        dn = j.loss(p, U, Cs)[0]
        print(f"JEPA + SIGReg gradient {key}{idx}: analytic {gr[key][idx]:.6f} numeric {(up - dn) / 2e-6:.6f}",
              flush=True)


def main():
    check_gradients()
    t0 = time.time()
    jobs = [(r, a, s) for r in RULES for a in ARMS for s in range(SEEDS)]
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(run, jobs))
    for rule in RULES:
        print(f"\n{rule}")
        for arm in ARMS:
            rs = [r for r in res if r["rule"] == rule and r["arm"] == arm]
            line = (f"  {arm}: 300 steps {np.mean([r['300'] for r in rs]):.3f}  3000 steps "
                    f"{np.mean([r['3000'] for r in rs]):.3f}  per seed {[round(r['3000'], 3) for r in rs]}")
            if arm == "T":
                tops = [int(np.argmax(r["theta"])) for r in rs]
                line += (f" | largest theta: {[T_NAMES[i] for i in tops]} (defining in "
                         f"{sum(i == DEFINING[rule[:2]] for i in tops)} of {len(rs)})")
            if arm == "J":
                line += (f" | var {rs[0]['var']}, max |corr| {np.mean([r['maxcorr'] for r in rs]):.2f}, prediction "
                         f"loss {np.mean([r['Lp'] for r in rs]):.4f}, SIGReg {np.mean([r['sig'] for r in rs]):.4f}, "
                         f"linear probe of the hidden state {np.mean([r['probe'] for r in rs]):.3f} (majority "
                         f"{np.mean([r['base'] for r in rs]):.3f})")
            print(line, flush=True)
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
