"""Part 25's variants of the attention feel-experiment (docs/AGENT.md part 25; docs/ATTENTION.md section 7): why the
local rule is too slow for a game, and what did and didn't speed it up. Same task as attention_feel.py.
  Nat     the natural-gradient (Bayes) step for W: delta s = eta log(r / a)
  Replay  C plus prioritised replay of stored steps through the same rule (tables counted once)
  Split   C with a learned bias; prior: the 8 position keys start behind (as in the agent: about -log 121);
          split: every key's tables count every step, split across heads by r(h, key) / sum_h r(h, key)
  BB      a bias learned by exact Bayes (the track record, tau = 1) plus W by the gradient
    python -m structure.claude.attention_feel_variants      (from experiments/symbolics; about 3 minutes)
"""
import numpy as np

import structure.claude.attention_feel as F


class Nat(F.Learner):
    """C with the natural-gradient (Bayes) step: delta s = eta log(r / a), spread over the query's active rows."""
    def __init__(self, heads, eta):
        super().__init__("C", heads)
        self.eta = eta
    def step(self, d, switch, cells, y):
        vals = F.values(cells, switch)
        P = self.experts(d, vals)
        a = self.weights(d, switch)
        p = float((a * P[..., 1]).sum())
        correct = (p > 0.5) == bool(y)
        py = P[..., y]
        pm = (a * py).sum()
        r = a * py / pm
        q = self.query(d, switch)
        self.W += self.eta * np.einsum("hk,q->hqk", np.log(py / pm), q) / q.sum()
        for h in range(self.H):
            for k in range(self.K):
                self.counts[d, k, h, vals[k], y] += r[h, k]
        return correct, -np.log(max(p if y else 1 - p, 1e-12))

class Replay(F.Learner):
    """C (gradient, 2 heads, EM tables) plus replay: after each step, R stored steps are replayed through the same
    gradient rule for W (tables are counted once, online). prio: sample in proportion to surprise at storage."""
    def __init__(self, heads, R, prio=False, tables=False):
        super().__init__("C", heads)
        self.R, self.prio, self.tables, self.buf, self.w = R, prio, tables, [], []
        self.rng = np.random.default_rng(1)
    def grad(self, d, switch, cells, y, count):
        vals = F.values(cells, switch)
        P = self.experts(d, vals)
        a = self.weights(d, switch)
        py = P[..., y]
        pm = (a * py).sum()
        r = a * py / max(pm, 1e-12)
        self.W += F.ETA * np.einsum("hk,q->hqk", r - a, self.query(d, switch))
        if count:
            for h in range(self.H):
                for k in range(self.K):
                    self.counts[d, k, h, vals[k], y] += r[h, k]
        return P, a, pm
    def step(self, d, switch, cells, y):
        vals = F.values(cells, switch)
        P = self.experts(d, vals)
        a = self.weights(d, switch)
        p = float((a * P[..., 1]).sum())
        correct = (p > 0.5) == bool(y)
        _, _, pm = self.grad(d, switch, cells, y, True)
        self.buf.append((d, switch, cells, y))
        self.w.append(1 - pm if self.prio else 1.0)
        w = np.array(self.w) + 1e-3
        for i in self.rng.choice(len(self.buf), size=self.R, p=w / w.sum()):
            self.grad(*self.buf[i], False)
        return correct, -np.log(max(p if y else 1 - p, 1e-12))

class Split(F.Learner):
    """C, but each key's tables count every step; the count is split across heads by r(h, key) / sum_h r(h, key).
    prior: the keys' start (bias); K_extra: irrelevant keys added (to mimic the agent's many position keys)."""
    def __init__(self, heads, split=True, prior=0.0):
        super().__init__("C", heads)
        self.split, self.prior = split, prior
        self.b = np.zeros((heads, self.K))
        self.b[:, :8] = prior                                    # the 8 position keys start behind
    def weights(self, d, switch):
        s = np.einsum("q,hqk->hk", self.query(d, switch), self.W) + self.b
        s = s - s.max()
        a = np.exp(s)
        return a / a.sum()
    def step(self, d, switch, cells, y):
        vals = F.values(cells, switch)
        P = self.experts(d, vals)
        a = self.weights(d, switch)
        p = float((a * P[..., 1]).sum())
        correct = (p > 0.5) == bool(y)
        py = P[..., y]
        r = a * py / max((a * py).sum(), 1e-300)
        self.W += F.ETA * np.einsum("hk,q->hqk", r - a, self.query(d, switch))
        self.b += F.ETA * (r - a)
        w = r / r.sum(0, keepdims=True) if self.split else r
        for h in range(self.H):
            for k in range(self.K):
                self.counts[d, k, h, vals[k], y] += w[h, k]
        return correct, -np.log(max(p if y else 1 - p, 1e-12))

class BB(F.Learner):
    """bias: tau * (log P - log p_mix) per step (tau = 1: exact Bayes, the track record); W: gradient. EM r tables."""
    def __init__(self, heads, tau, prior=0.0, wgrad=True):
        super().__init__("C", heads)
        self.tau, self.wgrad = tau, wgrad
        self.b = np.zeros((heads, self.K))
        self.b[:, :8] = prior
    def weights(self, d, switch):
        s = np.einsum("q,hqk->hk", self.query(d, switch), self.W) + self.b
        s = s - s.max()
        a = np.exp(s)
        return a / a.sum()
    def step(self, d, switch, cells, y):
        vals = F.values(cells, switch)
        P = self.experts(d, vals)
        a = self.weights(d, switch)
        p = float((a * P[..., 1]).sum())
        correct = (p > 0.5) == bool(y)
        py = P[..., y]
        pm = (a * py).sum()
        r = a * py / max(pm, 1e-300)
        if self.wgrad:
            self.W += F.ETA * np.einsum("hk,q->hqk", r - a, self.query(d, switch))
        self.b += self.tau * (np.log(py) - np.log(pm))
        for h in range(self.H):
            for k in range(self.K):
                self.counts[d, k, h, vals[k], y] += r[h, k]
        return correct, -np.log(max(p if y else 1 - p, 1e-12))


def run(name, mk, steps, last):
    F.STEPS = steps
    acc = []
    for seed in range(5):
        L = mk()
        res = [L.step(*t) for t in F.world(np.random.default_rng(seed))]
        acc.append(np.mean([c for c, _ in res[-last:]]))
    print(steps, name, round(float(np.mean(acc)), 3), [round(float(x), 3) for x in acc], flush=True)


def main():
    for steps, last in ((300, 200), (1000, 500), (3000, 1000)):
        run("A track record", lambda: F.Learner("A", 1), steps, last)
        run("C gradient, 2 heads", lambda: F.Learner("C", 2), steps, last)
        run("C + natural-gradient step (eta 0.5)", lambda: Nat(2, 0.5), steps, last)
        run("C + prioritised replay x4", lambda: Replay(2, 4, True), steps, last)
        for prior in (0.0, -4.8):
            run(f"C + bias, position prior {prior}", lambda: Split(2, False, prior), steps, last)
            run(f"C + bias, split counts, position prior {prior}", lambda: Split(2, True, prior), steps, last)
        run("Bayes bias + gradient W", lambda: BB(2, 1.0), steps, last)


if __name__ == "__main__":
    main()
