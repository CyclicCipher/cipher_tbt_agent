"""Code lengths in bits: the cross-entropy of what came, under a predictive distribution (docs/LAB.md part 9).

**The form** is that of Barron, Gnanapragasam & Tschannen 2026, "A fix for softmax and cross-entropy" (arXiv
2610.31642).
- **The margin:** m = log p(target) − log p(everything else).
- **The loss:** softplus(−m), divided by ln 2 for bits.
- **The rest's mass is computed directly,** never as 1 − p(target).
- **Why:** done naively, a confident right answer's loss rounds to exactly zero once p(target) is within the float's
  spacing of 1 (the paper's "Softmax Collapse": past a margin of about 16.6 in fp32, 36.7 in fp64).

**Laplace's rule** gives a count-based distribution its probabilities, (n + 1) / (N + K). Here the rest's mass,
(N + K − n − 1) / (N + K), is exact from integers. A mixture of such distributions (the column's union of hypotheses)
adds the target's and the rest's masses separately, in log space.
"""
import math

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


def laplace(n, N, K):
    """(log P(target), log P(the rest)) by Laplace's rule: the target seen n times of N, among K possible inputs."""
    rest = N + K - n - 1
    return math.log(n + 1) - math.log(N + K), (math.log(rest) - math.log(N + K)) if rest > 0 else NEG_INF


def bits(log_target, log_rest):
    """The code length of the target, in bits: softplus(−m) / ln 2 with the margin m = log_target − log_rest."""
    if log_rest == NEG_INF:
        return 0.0
    if log_target == NEG_INF:
        return float("inf")
    return softplus(log_rest - log_target) / LN2


# ---- mixtures of count distributions (the column's union, the thalamus's hypotheses) -------------------------------
# A component is (log-weight, {input: n}, N): one hypothesis's counts, by Laplace's rule over K inputs.
def mixture_scores(components, K):
    """The (unnormalized) log-probability of each input some component has seen."""
    cands = list(dict.fromkeys(x for _, c, _ in components for x in c))
    return {x: logsumexp(lw + laplace(c.get(x, 0), N, K)[0] for lw, c, N in components) for x in cands}


def mixture_logp(components, x, K):
    """log P(x) under the mixture, its weights normalized; log(1/K) if it has no component."""
    if not components:
        return -math.log(K)
    z = logsumexp(lw for lw, _, _ in components)
    return logsumexp(lw + laplace(c.get(x, 0), N, K)[0] for lw, c, N in components) - z


def mixture_bits(components, target, K):
    """The code length of target under the mixture (the margin form is unchanged by the weights' total)."""
    parts = [laplace(c.get(target, 0), N, K) for _, c, N in components]
    return bits(logsumexp(lw + t for (lw, _, _), (t, _) in zip(components, parts)),
                logsumexp(lw + r for (lw, _, _), (_, r) in zip(components, parts)))
