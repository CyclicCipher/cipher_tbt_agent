"""Feel-experiment for part 26 (docs/AGENT.md; docs/ATTENTION.md section 7): in-context attention on the synthetic
switch-door task of attention_feel.py, with a distractor situation feature (a random number 0-9 each step) beside the
switch.

One memory per direction (as the agent keeps one per (column, action)). Each step is stored as an episode: the
situation (switch, distractor), every key's value, the outcome, and each key's prequential log probability of it.
  similarity of an episode   kappa = exp(-sum_f theta_f [feature f differs])
  a key's value table        kappa-weighted counts of the outcomes seen with that value (backoff: all episodes)
  a key's score              its kappa-weighted prequential evidence; attention = softmax
  learning                   theta only, by the gradient of the new outcome's log probability (RMSprop, eta 0.1)
Arms: track record (A of attention_feel.py), part 25's gradient attention (C), in-context with theta learned, fixed at
0, and fixed at 20 on the switch only (part 21's conjunction cases).
    python -m structure.claude.attention_incontext_feel      (from experiments/symbolics)
"""
import numpy as np

import structure.claude.attention_feel as F

A0, ETA = 0.02, 0.1


class Memory:
    """The episodes of one (column, action): situation codes S (T x F), key values V (T x K), outcomes Y (T),
    prequential log probabilities L (T x K)."""

    def __init__(self, K, nF):
        self.S, self.V, self.Y, self.L = (np.zeros((0, nF)), np.zeros((0, K), int), np.zeros(0, int),
                                          np.zeros((0, K)))
        self.theta, self.v = np.zeros(nF), np.zeros(nF)

    def parts(self, sit, vals, y):
        """Everything the prediction and the gradient need, for outcome y."""
        M = (self.S != np.array(sit)[None, :]).astype(float)
        kap = np.exp(-M @ self.theta)
        yv = (self.Y == y).astype(float)
        p0 = (kap @ yv + A0 * 0.5) / (kap.sum() + A0)
        B = (self.V == np.array(vals)[None, :]).astype(float)
        C, N = (kap * yv) @ B, kap @ B
        P = (C + A0 * p0) / (N + A0)
        E = kap @ self.L
        a = np.exp(E - E.max())
        a /= a.sum()
        return M, kap, yv, p0, B, C, N, P, a

    def logp(self, sit, vals, y, theta=None):
        if theta is not None:
            old, self.theta = self.theta, theta
        _, _, _, _, _, _, _, P, a = self.parts(sit, vals, y)
        if theta is not None:
            self.theta = old
        return float(np.log(a @ P))

    def grad(self, sit, vals, y):
        M, kap, yv, p0, B, C, N, P, a = self.parts(sit, vals, y)
        pm = a @ P
        r = a * P / pm
        g_att = -M.T @ (kap * (self.L @ (r - a)))
        dp0 = -M.T @ (kap * (yv - p0)) / (kap.sum() + A0)
        u, w = r / (C + A0 * p0), r / (N + A0)
        g_val = -M.T @ (kap * yv * (B @ u)) + A0 * u.sum() * dp0 + M.T @ (kap * (B @ w))
        return g_att + g_val, P

    def store(self, sit, vals, y, ell):
        self.S = np.vstack([self.S, sit])
        self.V = np.vstack([self.V, vals])
        self.Y = np.append(self.Y, y)
        self.L = np.vstack([self.L, ell])


class InContext:
    def __init__(self, learn=True, theta=None):
        self.mem = [Memory(10, 2) for _ in range(4)]
        self.learn = learn
        if theta is not None:
            for m in self.mem:
                m.theta = np.array(theta, float)
        self.rng = np.random.default_rng(99)

    def step(self, d, switch, cells, y):
        m = self.mem[d]
        vals = F.values(cells, switch)
        sit = [switch, int(self.rng.integers(10))]
        _, _, _, _, _, _, _, P1, a = m.parts(sit, vals, 1)
        p = float(a @ P1)
        correct = (p > 0.5) == bool(y)
        g, P = m.grad(sit, vals, y)
        if self.learn:
            m.v = 0.9 * m.v + 0.1 * g ** 2
            m.theta = np.clip(m.theta + ETA * g / (np.sqrt(m.v) + 1e-8), 0.0, 20.0)
        m.store(sit, vals, y, np.log(P))
        return correct, -np.log(max(p if y else 1 - p, 1e-12))


def check_gradient():
    """Finite differences against the analytic gradient on a stored memory (the apparatus check)."""
    L = InContext(learn=False, theta=[0.7, 0.3])
    stream = list(F.world(np.random.default_rng(5)))
    for t in stream[:200]:
        L.step(*t)
    d, switch, cells, y = stream[200]
    m, vals, sit = L.mem[d], F.values(cells, switch), [switch, 3]
    g, _ = m.grad(sit, vals, y)
    num = []
    for f in range(2):
        e = np.zeros(2)
        e[f] = 1e-5
        num.append((m.logp(sit, vals, y, m.theta + e) - m.logp(sit, vals, y, m.theta - e)) / 2e-5)
    print("gradient check: analytic", np.round(g, 6), "numeric", np.round(num, 6), flush=True)


def main():
    check_gradient()
    for steps, last in ((300, 200), (3000, 1000)):
        F.STEPS = steps
        arms = (("A track record", lambda: F.Learner("A", 1)), ("C gradient (part 25)", lambda: F.Learner("C", 2)),
                ("in-context, theta learned", lambda: InContext()),
                ("in-context, theta = 0", lambda: InContext(False, [0, 0])),
                ("in-context, theta = (20, 0)", lambda: InContext(False, [20, 0])))
        for name, mk in arms:
            acc, th = [], []
            for seed in range(5):
                L = mk()
                res = [L.step(*t) for t in F.world(np.random.default_rng(seed))]
                acc.append(np.mean([c for c, _ in res[-last:]]))
                if isinstance(L, InContext):
                    th.append(np.round(np.mean([m.theta for m in L.mem], 0), 2).tolist())
            print(f"{steps} {name}: {np.mean(acc):.3f} {[round(float(x), 3) for x in acc]}"
                  + (f" | theta (switch, distractor) per seed {th}" if th else ""), flush=True)


if __name__ == "__main__":
    main()
