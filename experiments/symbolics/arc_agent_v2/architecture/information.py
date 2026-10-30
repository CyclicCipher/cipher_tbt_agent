"""Content and code lengths: what came next at a place, as a distribution over SDRs, and the cross-entropy of what came
under it (docs/LAB.md parts 9 and 12).

**Content by overlap (part 12).** What came next at a cell is a `Content`: the inputs seen there, and for each of the
d bits how many of them had it active (c_i of N).
- **The prior:** one pseudo-observation of an average SDR, each bit active with probability w/d. So a bit's
  probability is p_i = (c_i + w/d) / (N + 1).
- **An input's likelihood:** log L(x) = (1/2w) [Σ_{i∈x} log p_i + Σ_{i∉x} log(1 − p_i)].
  - **The 1/2w** (amendment 3): two SDRs that differ are told apart by 2w bits, x's own being on and the other's being
    off, and that difference is one event, one input rather than another. Counted bit by bit, every comparison would
    be 2w times too confident. This is tempered (generalized) Bayes, with learning rate 1/2w (Bissiri, Holmes & Walker
    2016).
  - **The identity limit:** for random SDRs (two share about 0.8 of 40 bits), a cell that saw one input N times gives it
    log((N + w/d) / (N + 1)) and any other about log(e^-2.5 / (N + 1)): the Dirichlet process's rule with α = 1 over a
    vocabulary of about 12. So for symbols a cell predicts what it saw, with one pseudo-observation of prior, not
    Laplace's K.
  - **Similar inputs** share bits, so each is likely where the other was seen.
- **A probability over inputs,** where one is needed (a prediction, a code length): L normalized over a set of inputs,
  the ones the region knows.

**The code length** has the form of Barron, Gnanapragasam & Tschannen 2026, "A fix for softmax and cross-entropy"
(arXiv 2610.31642).
- **The margin:** m = log p(target) − log p(everything else).
- **The loss:** softplus(−m), divided by ln 2 for bits.
- **The rest's mass is computed directly,** never as 1 − p(target).
- **Why:** done naively, a confident right answer's loss rounds to exactly zero once p(target) is within the float's
  spacing of 1 (the paper's "Softmax Collapse").
"""
import math

import numpy as np

LN2 = math.log(2.0)
NEG_INF = float("-inf")


def softplus(x):
    """log(1 + e^x), without overflow or loss of precision."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def logsumexp(xs):
    xs = list(xs)
    m = max(xs, default=NEG_INF)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


def bits(log_target, log_rest):
    """The code length of the target, in bits: softplus(−m) / ln 2 with the margin m = log_target − log_rest."""
    if log_rest == NEG_INF:
        return 0.0
    if log_target == NEG_INF:
        return float("inf")
    return softplus(log_rest - log_target) / LN2


class Content:
    """What came next at a cell: the inputs seen there ({input: n}, N in all), and per bit how many had it active.
    Likelihoods are memoized until the next `add`."""
    __slots__ = ("d", "w", "counts", "N", "_bits", "_c", "_T", "_ll", "_Z")

    def __init__(self, d, w, counts=None):
        self.d, self.w = d, w
        self.counts, self.N = {}, 0.0
        self._bits = self._c = self._T = None
        self._ll, self._Z = {}, {}
        for x, n in (counts or {}).items():
            self.counts[x] = self.counts.get(x, 0) + n
            self.N += n

    def add(self, x, n=1):
        self.counts[x] = self.counts.get(x, 0) + n
        self.N += n
        self._bits = self._c = self._T = None
        self._ll, self._Z = {}, {}

    def __len__(self):
        return len(self.counts)

    def _stats(self):
        """The active bits (sorted), their counts, and T = Σ over all d bits of log(1 − p_i)."""
        if self._bits is None:
            a, N1 = self.w / self.d, self.N + 1.0
            if self.counts:
                idx = np.concatenate([x.idx for x in self.counts]).astype(np.int64)
                wt = np.repeat(np.array(list(self.counts.values()), np.float64), [x.idx.size for x in self.counts])
                self._bits, inv = np.unique(idx, return_inverse=True)
                self._c = np.bincount(inv, weights=wt)
            else:
                self._bits, self._c = np.zeros(0, np.int64), np.zeros(0)
            zero = self.d - self._bits.size
            self._T = zero * math.log1p(-a / N1) + float(np.log1p(-(self._c + a) / N1).sum())
        return self._bits, self._c, self._T

    def loglik(self, x):
        """log L(x), tempered by 1/2w (the module docstring)."""
        v = self._ll.get(x)
        if v is None:
            bits_, c, T = self._stats()
            a, N1 = self.w / self.d, self.N + 1.0
            cx = np.zeros(x.idx.size)
            if bits_.size:
                pos = np.searchsorted(bits_, x.idx)
                pos[pos >= bits_.size] = 0
                hit = bits_[pos] == x.idx
                cx[hit] = c[pos[hit]]
            q = (cx + a) / N1
            v = self._ll[x] = (T + float((np.log(q) - np.log1p(-q)).sum())) / (2 * self.w)
        return v

    def logZ(self, inputs):
        """log Σ L(y) over these inputs (a tuple, the normalizer of a probability over them)."""
        z = self._Z.get(inputs)
        if z is None:
            z = self._Z[inputs] = logsumexp(self.loglik(y) for y in inputs)
        return z

    def best(self):
        """The input seen here that is most likely here, or None."""
        return max(self.counts, key=self.loglik) if self.counts else None


# ---- mixtures (the column's union of hypotheses, the thalamus's thoughts) ------------------------------------------
# A component is (log-weight, Content): one hypothesis's content.
def mixture_logp(components, x, prior):
    """log L(x) under the mixture, its weights normalized (a density, for Bayes' rule); the prior's if it has no
    component."""
    if not components:
        return prior.loglik(x)
    z = logsumexp(lw for lw, _ in components)
    return logsumexp(lw + c.loglik(x) for lw, c in components) - z


def _over(components, known, extra=()):
    """The inputs a probability is normalized over: those known, those any component has seen, and extra."""
    seen = (x for _, c in components for x in c.counts)
    return tuple(dict.fromkeys([*known, *seen, *extra]))


def mixture_scores(components, known=()):
    """The log-probability of each input some component has seen, each component normalized over the known inputs."""
    over = _over(components, known)
    cands = list(dict.fromkeys(x for _, c in components for x in c.counts))
    return {x: logsumexp(lw + c.loglik(x) - c.logZ(over) for lw, c in components) for x in cands}


def mixture_bits(components, target, known=()):
    """The code length of target under the mixture, normalized over the known inputs (and target): the target's and
    the rest's masses are summed separately, the rest's directly."""
    over = _over(components, known, (target,))
    rest = tuple(y for y in over if y != target)
    t = logsumexp(lw + c.loglik(target) - c.logZ(over) for lw, c in components)
    r = logsumexp(lw + c.logZ(rest) - c.logZ(over) for lw, c in components) if rest else NEG_INF
    return bits(t, r)
