"""Feel-experiment for docs/ATTENTION.md section 6: does attention learned by the responsibility rule handle a
conjunction that a track-record mixture cannot?

The task (a synthetic W7): each step the agent moves in one of 4 directions. The 8 cells around it are floor, wall or
door; a switch is on or off (it flips with probability 0.1 per step). The move is blocked by a wall, or by a door while
the switch is off. Keys: the 8 neighbour cells and the switch, plus the unconditional expert. Each key's value is a
table from its value to P(blocked), one per (direction, key, head), learned by counts (weighted by responsibility in
arm C). Arms:
  A  track record: weights proportional to exp(each expert's accumulated log predictive probability), per direction
     (the arc_agent now);
  B  attention: s = q . W k, q = (direction, switch, bias) one-hot, k = the key's one-hot; W learned by the local rule
     delta W = eta (r - a) q k^T; tables counted from every step;
  C  as B with 2 heads, each head's tables counted with that head's responsibility (EM), so heads can specialise.
    python -m structure.claude.attention_feel      (from experiments/symbolics)
"""
import numpy as np

STEPS, SEEDS, ETA, A0 = 3000, 5, 0.5, 0.02
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
POS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]


def world(rng):
    switch = 0
    for _ in range(STEPS):
        if rng.random() < 0.1:
            switch = 1 - switch
        cells = rng.choice(3, size=8, p=[0.6, 0.2, 0.2])          # 0 floor, 1 wall, 2 door
        d = int(rng.integers(4))
        ahead = cells[POS.index(DIRS[d])]
        blocked = int(ahead == 1 or (ahead == 2 and switch == 0))
        yield d, switch, cells, blocked


def values(cells, switch):
    return list(cells) + [switch, 0]                              # key 9: the switch; key 10: unconditional (constant)


class Learner:
    def __init__(self, arm, heads=1):
        self.arm, self.H, self.K = arm, heads, 10
        self.counts = np.full((4, self.K, heads, 3, 2), A0)       # (dir, key, head, value, outcome)
        self.ev = np.zeros((4, self.K))                           # arm A: track record per (dir, key)
        self.W = np.random.default_rng(0).normal(0, 0.01, (heads, 7, self.K))   # arm B, C: query x key, per head

    def query(self, d, switch):
        q = np.zeros(7)
        q[d], q[4 + switch], q[6] = 1, 1, 1
        return q

    def experts(self, d, vals):
        """P(outcome) of every (head, key): shape (H, K, 2)."""
        c = self.counts[d, np.arange(self.K)[:, None], np.arange(self.H)[None, :], np.array(vals)[:, None]]
        c = np.transpose(c, (1, 0, 2))
        return c / c.sum(-1, keepdims=True)

    def weights(self, d, switch):
        if self.arm == "A":
            s = self.ev[d][None, :]
        else:
            s = np.einsum("q,hqk->hk", self.query(d, switch), self.W)
        s = s - s.max()
        a = np.exp(s)
        return a / a.sum()

    def step(self, d, switch, cells, y):
        vals = values(cells, switch)
        P = self.experts(d, vals)                                 # (H, K, 2)
        a = self.weights(d, switch)                               # (H, K)
        p = float((a * P[..., 1]).sum())
        correct = (p > 0.5) == bool(y)
        py = P[..., y]
        r = a * py / max((a * py).sum(), 1e-12)                   # responsibilities
        if self.arm == "A":
            self.ev[d] += np.log(py[0])
        else:
            self.W += ETA * np.einsum("hk,q->hqk", r - a, self.query(d, switch))
        w = r if self.arm == "C" else np.ones_like(r)             # EM: each head learns where it is responsible
        for h in range(self.H):
            for k in range(self.K):
                self.counts[d, k, h, vals[k], y] += w[h, k]
        return correct, -np.log(max(p if y else 1 - p, 1e-12))


def main():
    for arm, heads in (("A", 1), ("B", 1), ("C", 2), ("C", 4)):
        acc, loss = [], []
        for seed in range(SEEDS):
            L = Learner(arm, heads)
            res = [L.step(*t) for t in world(np.random.default_rng(seed))]
            acc.append(np.mean([c for c, _ in res[-1000:]]))
            loss.append(np.mean([x for _, x in res[-1000:]]))
        print(f"arm {arm} heads {heads}: accuracy on the last 1000 steps {np.mean(acc):.3f} "
              f"({', '.join(f'{x:.3f}' for x in acc)}), log loss {np.mean(loss):.3f}", flush=True)


if __name__ == "__main__":
    main()
