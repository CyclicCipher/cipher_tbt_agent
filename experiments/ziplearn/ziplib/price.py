"""ziplib.price — the codes both scripts price with (DESIGN §21.2; the module spec's `price.py`). No model code.

Every function here is a MOVE, byte-for-byte the same arithmetic as its origin (the check script `_check_price.py`
compares each against the original on random inputs to 1e-12):

  flag_bits          <- ziplearner.flag_bits         the KT rate price of one exception flag (DESIGN §8 change 1)
  pay                <- ziplearner.pay               charge a structure for one prediction (the retired "flat" branch dropped)
  precision_bits     <- ziplearner.precision_bits    E15's 1/sqrt(n) law for a continuous parameter (§8 change 2)
  entropy            <- arcgames.entropy             the binary entropy in bits
  table_price        <- arcgames.LocalRule._price    the two-part code of a merged table; V is now an argument
  kt_bits            <- textlm.ContextLM._bits       the per-context Krichevsky-Trofimov code of a training text
  backoff_chain      <- textlm.ContextLM._chain_of   the recency-ordered backoff chain of a mask (E33)
  context_keys       <- textlm.ContextLM._keys       a masked context as one sorted integer key (the helper prequential_bits needs)
  prequential_bits   <- textlm.ContextLM._prequential  the blended-backoff code the predictor actually pays (E33's sleep price)

Conventions carried over unchanged: the alphabet is 0..V-1 with BORDER = V, so the integer key base is V + 1
(`ContextLM.base`); a context array `ctx` is (n, R) with column k-1 = k characters back; a mask lists kept positions in
characters back (1 = the previous character).
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ── the rate price (ziplearner.py) ───────────────────────────────────────────────────────────────────────────────────
def flag_bits(is_exception, n_right, n_wrong):
    """The RATE price of saying whether this observation is an exception (DESIGN §8, change 1): an adaptive code for a
    two-symbol source, P(exception) = (n_wrong + 1/2) / (n + 1) -- the Krichevsky-Trofimov estimator. Summed over n
    observations it converges to n * H(epsilon) + O(log n): exceptions are paid for by their RATE, not remembered."""
    n = n_right + n_wrong
    p_exc = (n_wrong + 0.5) / (n + 1.0)
    return -math.log2(p_exc if is_exception else 1.0 - p_exc)


def pay(s, right):
    """Charge structure `s` (any object with V, cost, n_right, n_wrong) for one predicted digit: the exception flag at
    its rate, plus log2(V - 1) for the digit when the prediction was wrong. (The retired "flat" branch of E4 is gone.)"""
    if right:
        s.cost += flag_bits(False, s.n_right, s.n_wrong)
        s.n_right += 1
    else:
        s.cost += flag_bits(True, s.n_right, s.n_wrong) + math.log2(s.V - 1)
        s.n_wrong += 1


def precision_bits(n, span, sigma):
    """§8 change 2: a continuous parameter estimated from n observations with noise sigma is written at the precision
    the data justify. E15 measured the step that minimises the expected total code length: delta = sigma * sqrt(12 / n)
    (the 1/sqrt(n) law, with the constant of a uniform quantisation error), so it costs log2(span / delta) =
    1/2 log2 n + log2(span / sigma) - 1/2 log2 12 bits."""
    return 0.5 * math.log2(max(1, n)) + math.log2(span / sigma) - 0.5 * math.log2(12)


# ── the table price (arcgames.py) ────────────────────────────────────────────────────────────────────────────────────
def entropy(p):
    """Binary entropy in bits; 0 at the ends."""
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def table_price(merged, V):
    """Two-part code of the evidence under a table: one parameter per entry (log2 V), exceptions by their rate
    (n * H(wrong / n)) plus the wrong digits (log2(V - 1) each). `merged` is {key: {y: count}}."""
    bits = 0.0
    for counts in merged.values():
        n = sum(counts.values())
        wrong = n - max(counts.values())
        bits += math.log2(V) + n * entropy(wrong / n) + wrong * math.log2(V - 1)
    return bits


# ── the text prices (textlm.py) ──────────────────────────────────────────────────────────────────────────────────────
def context_keys(ctx, mask, base):
    """A masked context as one integer key: the kept positions, in the mask's order, as digits in base `base`."""
    key = np.zeros(len(ctx), dtype=np.int64)
    for k in mask:
        key = key * base + ctx[:, k - 1]
    return key


def kt_bits(ctx_keys, nxt, V, base=None):
    """The training text's length under the adaptive KT code, one memoryless source per context: for a context with
    counts n_1..n_V (N in all), [lgamma(N + V/2) - lgamma(V/2) - sum_i (lgamma(n_i + 1/2) - lgamma(1/2))] / ln 2."""
    base = V + 1 if base is None else base
    pair = ctx_keys * base + nxt
    _u, pc = np.unique(pair, return_counts=True)
    _c, tc = np.unique(ctx_keys, return_counts=True)
    pc_t, tc_t = torch.tensor(pc, dtype=torch.float64), torch.tensor(tc, dtype=torch.float64)
    bits = (torch.lgamma(tc_t + V / 2).sum() - len(tc) * math.lgamma(V / 2)
            - (torch.lgamma(pc_t + 0.5).sum() - len(pc) * math.lgamma(0.5))) / math.log(2)
    return float(bits)


def backoff_chain(mask):
    """The mask, then the mask without its oldest kept position, and so on down to the empty context (the unigram):
    E28's nearest-stored-key default in its structured, recency-ordered form (E33)."""
    chain, m = [], list(mask)
    while True:
        chain.append(tuple(m))
        if not m:
            break
        m = [x for x in m if x != max(m)]                             # drop the oldest kept position
    return chain


def prequential_bits(ctx, nxt, mask, V, limit, base=None):
    """The code length of the (first `limit` characters of the) training text under the blended backoff predictor
    with this mask, counts growing as the text is read -- the price a mask is judged by (E24's rule with the code
    the predictor actually pays, not the per-context KT sum, which ignores backoff and punishes sparse orders).
    Scaled to the whole text by len(nxt) / n."""
    base = V + 1 if base is None else base
    n = min(limit, len(nxt))
    chain = backoff_chain(mask)
    keys = [context_keys(ctx[:n], list(m), base).tolist() for m in chain]
    nxt_l = nxt[:n].tolist()
    totals = [dict() for _ in chain]
    pairs = [dict() for _ in chain]
    bits = 0.0
    for i in range(n):
        c = nxt_l[i]
        p, mass = 0.0, 1.0
        for li in range(len(chain)):
            k = keys[li][i]
            N = totals[li].get(k, 0)
            cnt = pairs[li].get(k * base + c, 0)
            p += mass * cnt / (N + V / 2)
            mass *= (V / 2) / (N + V / 2)
        p += mass / V
        bits -= math.log2(p)
        for li in range(len(chain)):
            k = keys[li][i]
            totals[li][k] = totals[li].get(k, 0) + 1
            pk = k * base + c
            pairs[li][pk] = pairs[li].get(pk, 0) + 1
    return bits * len(nxt) / n                                        # scaled to the whole text


__all__ = ["flag_bits", "pay", "precision_bits", "entropy", "table_price", "context_keys", "kt_bits",
           "backoff_chain", "prequential_bits"]
