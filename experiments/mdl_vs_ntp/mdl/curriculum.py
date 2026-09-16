"""Family sampling distributions.

`gain`  — the epiplexity-gain curriculum: a prequential learning-progress estimator. Progress is zero both for structure
          a bounded learner cannot absorb (the keyed-hash "noisy-TV" families) and for structure it has already absorbed;
          a relearn term catches structure once absorbed and since forgotten.
`curio` — the identical pipeline with the gain replaced by the CURRENT prediction error, i.e. plain curiosity, which the
          noisy-TV families should capture.
`uniform` / `error` are the ntp/exit baseline and an alias respectively.

    progress = max(0, ℓ_f(r−1) − ℓ_f(r) − 0.01)
    relearn  = max(0, ℓ_f(r) − ℓ_f^min − 0.05)
    g_f      ← 0.7·g_f + 0.3·(progress + relearn)
    q        = stable_softmax(g / (0.5·std(g) + 1e-8));  q = water_fill_cap(q, 0.05);  p = 0.9·q + 0.1·uniform
"""
from __future__ import annotations

import numpy as np


def stable_softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def water_fill_cap(q, cap):
    """Iteratively cap entries at `cap`, redistributing the excess proportionally over the uncapped ones."""
    q = q.astype(np.float64).copy()
    capped = np.zeros(len(q), dtype=bool)
    for _ in range(len(q)):
        over = (q > cap + 1e-12) & ~capped
        if not over.any():
            break
        excess = float((q[over] - cap).sum())
        q[over] = cap
        capped |= over
        free = ~capped
        if free.sum() == 0:
            break
        w = q[free]
        q[free] = w + excess * (w / w.sum() if w.sum() > 0 else 1.0 / free.sum())
    return q / q.sum()


class Curriculum:
    def __init__(self, fids, mode="gain", cap=0.05, mix=0.1, ema=0.7, hyst=0.01, relearn_hyst=0.05):
        self.fids = list(fids)
        self.index = {f: i for i, f in enumerate(self.fids)}
        self.mode, self.cap, self.mix, self.ema = mode, cap, mix, ema
        self.hyst, self.relearn_hyst = hyst, relearn_hyst
        n = len(self.fids)
        self.g = np.zeros(n)
        self.l_prev = np.full(n, np.nan)
        self.l_min = np.full(n, np.inf)
        self.p = np.full(n, 1.0 / n)
        self.rounds = 0

    def update(self, losses):
        """`losses`: {fid: probe NLL (bits/token)} for every family. Returns the new sampling distribution."""
        l = np.array([losses[f] for f in self.fids], dtype=np.float64)
        if self.mode in ("uniform",):
            return self.p
        if self.mode == "gain":
            if self.rounds > 0:
                progress = np.maximum(0.0, self.l_prev - l - self.hyst)
                relearn = np.maximum(0.0, l - self.l_min - self.relearn_hyst)
                self.g = self.ema * self.g + (1 - self.ema) * (progress + relearn)
            score = self.g
        else:                                                  # curio / error: current prediction error
            score = l
        q = stable_softmax(score / (0.5 * score.std() + 1e-8))
        q = water_fill_cap(q, self.cap)
        self.p = (1 - self.mix) * q + self.mix / len(q)
        self.l_min = np.minimum(self.l_min, l)
        self.l_prev = l
        self.rounds += 1
        return self.p

    def sample(self, rng, n):
        return [self.fids[i] for i in rng.choice(len(self.fids), size=n, p=self.p)]

    def mass_by(self, categories):
        """Probability mass per category label, `categories`: {fid: label}."""
        out = {}
        for f, p in zip(self.fids, self.p):
            out[categories[f]] = out.get(categories[f], 0.0) + float(p)
        return out

    def state(self):
        return dict(g=self.g, l_prev=self.l_prev, l_min=self.l_min, p=self.p, rounds=self.rounds)

    def load_state(self, st):
        self.g, self.l_prev, self.l_min, self.p, self.rounds = st["g"], st["l_prev"], st["l_min"], st["p"], st["rounds"]
