"""The arithmetic world (milestones A1-A3, ARCHITECTURE.md section 6): written equations, shown as streams of SDRs.

An episode is one equation, such as "3 + 4 = 7", shown one symbol at a time. Every symbol, digits and signs alike,
enters as a random SDR (HDClibrary.CategoryEncoder). The brain is told nothing about any of them: not what a symbol
means, not which symbols are numbers or operators, not where the arguments or the result are (ARCHITECTURE.md section
0). The labels ("3", "+", "=") live only here, to build episodes and to score; the brain never sees them. The world
computes the answers with integer arithmetic: it is the ground truth, not part of the brain.

The operation "?" is a control, not arithmetic: a random table of answers (drawn once per world), with no structure
to find. A brain that scores well above chance on its held-out equations is reading something it shouldn't.

A1: addition mod p. The p * p equations are split at random: train_frac of them are shown in training, the rest are
held out. The test shows a held-out equation up to "=", asks for the brain's prediction of the next SDR, and decodes
that prediction to the nearest symbol (here, in the world).
"""
import numpy as np

from arc_agent_v2.HDClibrary import CategoryEncoder, ItemMemory

OPS = {"+": lambda a, b, p: (a + b) % p,
       "-": lambda a, b, p: (a - b) % p,
       "*": lambda a, b, p: (a * b) % p}


class ArithmeticWorld:
    def __init__(self, p=13, ops=("+",), train_frac=0.5, d=2048, w=40, seed=0):
        self.p, self.ops = p, tuple(ops)
        enc = CategoryEncoder(d, w, seed)
        self.symbols = [str(i) for i in range(p)] + list(self.ops) + ["="]
        self.sdr_of = {s: enc.encode(s) for s in self.symbols}
        self._memory = ItemMemory(d)                    # decoding, for scoring only
        for s in self.symbols:
            self._memory.add(self.sdr_of[s])
        self.table = np.random.default_rng((seed, 99)).integers(0, p, (p, p))   # the "?" control's answers
        eqs = [(op, a, b) for op in self.ops for a in range(p) for b in range(p)]
        order = np.random.default_rng(seed).permutation(len(eqs))
        n = int(round(train_frac * len(eqs)))
        self.train = [eqs[i] for i in order[:n]]
        self.held_out = [eqs[i] for i in order[n:]]

    def answer(self, op, a, b):
        return int(self.table[a, b]) if op == "?" else OPS[op](a, b, self.p)

    def written(self, op, a, b):
        """The equation's symbols, in the order written."""
        return [str(a), op, str(b), "=", str(self.answer(op, a, b))]

    def episode(self, op, a, b):
        """What the brain is shown: the equation as a list of SDRs."""
        return [self.sdr_of[s] for s in self.written(op, a, b)]

    def epoch(self, rng):
        """Every training equation once, in a random order, as episodes."""
        return [self.episode(*self.train[i]) for i in rng.permutation(len(self.train))]

    def queries(self, held_out=True):
        """[(the episode up to "=", the answer's SDR, the equation)]."""
        out = []
        for eq in (self.held_out if held_out else self.train):
            ep = self.episode(*eq)
            out.append((ep[:-1], ep[-1], eq))
        return out

    def decode(self, sdr):
        """The symbol nearest an SDR (None for no prediction)."""
        if sdr is None or sdr.w == 0:
            return None
        i, _ = self._memory.nearest(sdr)
        return self.symbols[i]

    def score(self, predict, held_out=True):
        """The fraction of equations whose answer is the symbol nearest predict(prefix)."""
        qs = self.queries(held_out)
        hits = sum(self.decode(predict(prefix)) == self.written(*eq)[-1] for prefix, _, eq in qs)
        return hits / len(qs)
