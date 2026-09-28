"""The blended-backoff chain (E33's text predictor), vectorised.

`e34.Chain` is the reference implementation and stays the definition of the model: nine Python dicts keyed by
tuples of characters, one inner dict of counts per context, one `update` call per character. It is correct and it
does not scale -- E38 measured 2.2 GB of resident memory at 2.73M characters and 4.2 GB at 10M, and every key is a
freshly allocated tuple, nine per character.

Nothing about the MODEL needs that. The online (prequential) code at position i depends only on two counts:

    n_ctx[i]   = #{ j < i : context_r(j) == context_r(i) }
    n_ctx_c[i] = #{ j < i : context_r(j) == context_r(i) and seq[j] == seq[i] }

Both are "how many earlier positions share my key" -- a WITHIN-GROUP OCCURRENCE RANK, which a stable argsort plus
an accumulate computes for the whole corpus at once, in C, with no hash table and no Python loop. The contexts
themselves are rolling base-(V+1) integers, so no tuples are built either. The blend across orders is then plain
numpy arithmetic over the per-position count arrays.

This is an APPARATUS change, not a model change: `verify_against_reference` asserts the per-character bits match
`e34.Chain` to 1e-9, and every use of this module should keep doing so on a slice.

    bits = online_bits(seq, V, R=8)                  # prequential: counts accumulate as the stream is coded
    bits = frozen_bits(train, test, V, R=8)          # counts fixed at the end of training
"""
from __future__ import annotations

import numpy as np

BORDER = None                                                          # = V; the symbol Stream.back() returns off the start


def _codes(seq: np.ndarray, V: int, r: int) -> np.ndarray:
    """Rolling base-(V+1) code of the r characters BEFORE each position; out-of-range positions read BORDER = V.

    Matches `e34.Stream.back(k)`, which returns V when the history is shorter than k. Order 0 is the single empty
    context, so every position shares one group -- exactly what `tables[0]` holds in the reference."""
    n = len(seq)
    if r == 0:
        return np.zeros(n, dtype=np.int64)
    B = V + 1
    padded = np.concatenate([np.full(r, V, dtype=np.int64), seq.astype(np.int64)])
    out = np.zeros(n, dtype=np.int64)
    for j in range(1, r + 1):                                          # back(j) = padded[i + r - j]
        out += padded[r - j: r - j + n] * (B ** (j - 1))
    return out


def _rank_within_group(keys: np.ndarray) -> np.ndarray:
    """For each position, how many EARLIER positions carry the same key. One stable sort, no hashing."""
    n = len(keys)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(keys, kind="stable")                            # groups contiguous, positions ascending in each
    sk = keys[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(sk[1:], sk[:-1], out=new[1:])
    idx = np.arange(n, dtype=np.int64)
    start = np.maximum.accumulate(np.where(new, idx, np.int64(0)))
    rank = idx - start
    out = np.empty(n, dtype=np.int64)
    out[order] = rank
    return out


def online_counts(seq: np.ndarray, V: int, r: int):
    """(n_ctx, n_ctx_c) at every position for one order, counting only strictly earlier positions."""
    ctx = _codes(seq, V, r)
    n_ctx = _rank_within_group(ctx)
    n_ctx_c = _rank_within_group(ctx * (V + 1) + seq.astype(np.int64))
    return n_ctx, n_ctx_c


def _blend(per_order, V: int) -> np.ndarray:
    """E33/e34's blend, vectorised over positions:
        p = 0; mass = 1
        for r = R..0:  if N_r: p += mass*c_r/(N_r+V/2);  mass *= (V/2)/(N_r+V/2)
        p += mass/V
    """
    n = len(per_order[0][0])
    p = np.zeros(n)
    mass = np.ones(n)
    half = V / 2.0
    for n_ctx, n_ctx_c in per_order:                                   # already deepest-first
        has = n_ctx > 0
        denom = n_ctx + half
        p += np.where(has, mass * n_ctx_c / denom, 0.0)
        mass = np.where(has, mass * half / denom, mass)
    return p + mass / V


def online_bits(seq, V: int, R: int = 8, chunk_orders: bool = True) -> np.ndarray:
    """Per-character code length in bits under the prequential blended-backoff chain. Equals `e34.Chain` run
    online over the same sequence."""
    seq = np.asarray(seq, dtype=np.int64)
    per_order = [online_counts(seq, V, r) for r in range(R, -1, -1)]
    p = _blend(per_order, V)
    return -np.log2(np.maximum(p, 1e-12))


def frozen_counts(train: np.ndarray, test: np.ndarray, V: int, r: int):
    """(n_ctx, n_ctx_c) for the TEST positions with the tables fixed at the end of training."""
    both = np.concatenate([train, test])
    ctx_all = _codes(both, V, r)
    nt = len(train)
    ctx_tr, ctx_te = ctx_all[:nt], ctx_all[nt:]
    key_tr = ctx_tr * (V + 1) + train.astype(np.int64)
    key_te = ctx_te * (V + 1) + test.astype(np.int64)

    def lookup(tr_keys, te_keys):
        u, c = np.unique(tr_keys, return_counts=True)
        pos = np.searchsorted(u, te_keys)
        pos_ok = np.clip(pos, 0, len(u) - 1) if len(u) else np.zeros(len(te_keys), dtype=np.int64)
        hit = (len(u) > 0) & (u[pos_ok] == te_keys) if len(u) else np.zeros(len(te_keys), dtype=bool)
        out = np.zeros(len(te_keys), dtype=np.int64)
        if len(u):
            out[hit] = c[pos_ok[hit]]
        return out

    return lookup(ctx_tr, ctx_te), lookup(key_tr, key_te)


def frozen_bits(train, test, V: int, R: int = 8) -> np.ndarray:
    """Per-character bits on `test` with the chain trained on `train` and NOT updated while coding."""
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    per_order = [frozen_counts(train, test, V, r) for r in range(R, -1, -1)]
    p = _blend(per_order, V)
    return -np.log2(np.maximum(p, 1e-12))


def verify_against_reference(seq, alphabet, R: int = 8, tol: float = 1e-9):
    """Assert this module reproduces `e34.Chain` per character. An apparatus change is worth nothing unverified."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from e34 import Chain, Stream

    V = len(alphabet)
    seq = np.asarray(seq, dtype=np.int64)
    chain, stream = Chain(V, R=R), Stream(V, alphabet)
    ref = np.empty(len(seq))
    for i, c in enumerate(seq.tolist()):
        keys = chain.predict(stream)
        ref[i] = chain.prob(keys, c)
        chain.update(keys, c)
        stream.push(c)
    ref = -np.log2(np.maximum(ref, 1e-12))
    got = online_bits(seq, V, R)
    err = float(np.abs(ref - got).max())
    assert err < tol, f"fast chain differs from e34.Chain by {err} bits at worst"
    return {"n": int(len(seq)), "max_abs_bits_error": err,
            "ref_bpc": float(ref.mean()), "fast_bpc": float(got.mean())}


__all__ = ["online_bits", "frozen_bits", "online_counts", "frozen_counts", "verify_against_reference"]


# ---------------------------------------------------------------------------------------------------------------
# §9 rule 3 applied to the context tree: capacity forces consolidation, and what to drop is chosen by PRICE.
#
# The chain above keeps every context it ever sees, for ever. That is not a continual learner, and it contradicts
# the section of the design that produced E6 (100% retention, 3 blocks, 36 bits against 452): "two blocks that
# differ by a recurring Q become one block plus Q; if merging is not enough, drop the block with the least
# (evidence x bits it saves)". A context at order r IS such a block, and its parent at order r-1 is the thing to
# merge it into. Most order-8 contexts are seen once and predict exactly what their parent predicts, so they cost
# storage and save nothing.
#
# NOTE this is a different question from E33's, which found the sleep MASK a no-op on text. That asked which
# context POSITIONS to keep (an offset mask, shared by every context). This asks which context INSTANCES to keep.
def keep_mask(train, V: int, r: int, lam: float = 1.0, prec_bits: float = 4.0):
    """Which order-r contexts pay for themselves on the training prefix.

    A context is kept when the bits its own distribution SAVES over its parent's, on its own observations, exceed
    the bits it COSTS to store (its support times `prec_bits`, scaled by `lam` -- the rate/distortion exchange of
    §8). Returns the sorted array of kept context codes."""
    train = np.asarray(train, dtype=np.int64)
    if r == 0:
        return None                                                    # order 0 is one context and is always kept
    ctx = _codes(train, V, r)
    par = _codes(train, V, r - 1)
    half = V / 2.0
    # final counts per (context) and per (context, symbol), for the child and for its parent
    cu, cinv, ccnt = np.unique(ctx, return_inverse=True, return_counts=True)
    pu, pinv, pcnt = np.unique(par, return_inverse=True, return_counts=True)
    ck, ckinv, ckcnt = np.unique(ctx * (V + 1) + train, return_inverse=True, return_counts=True)
    pk, pkinv, pkcnt = np.unique(par * (V + 1) + train, return_inverse=True, return_counts=True)
    # per-observation coding advantage of the child over the parent, under the same KT-style estimate
    p_child = ckcnt[ckinv] / (ccnt[cinv] + half)
    p_par = pkcnt[pkinv] / (pcnt[pinv] + half)
    adv = np.log2(np.maximum(p_child, 1e-12)) - np.log2(np.maximum(p_par, 1e-12))
    saved = np.zeros(len(cu))
    np.add.at(saved, cinv, adv)
    support = np.zeros(len(cu), dtype=np.int64)                        # distinct symbols the child has seen
    np.add.at(support, np.unique(np.stack([cinv, train]), axis=1)[0], 1)
    cost = lam * prec_bits * support
    return cu[saved > cost]


def online_bits_pruned(seq, V: int, R: int, keeps) -> np.ndarray:
    """`online_bits`, but a context outside its order's keep-set is treated as unseen, so its mass falls through
    to the shorter context -- exactly what the blend already does for a context it has never met."""
    seq = np.asarray(seq, dtype=np.int64)
    per_order = []
    for r in range(R, -1, -1):
        n_ctx, n_ctx_c = online_counts(seq, V, r)
        k = keeps.get(r)
        if k is not None:
            ctx = _codes(seq, V, r)
            pos = np.searchsorted(k, ctx)
            ok = (pos < len(k)) & (k[np.clip(pos, 0, max(0, len(k) - 1))] == ctx) if len(k) else np.zeros(len(ctx), bool)
            n_ctx = np.where(ok, n_ctx, 0)
            n_ctx_c = np.where(ok, n_ctx_c, 0)
        per_order.append((n_ctx, n_ctx_c))
    return -np.log2(np.maximum(_blend(per_order, V), 1e-12))


__all__ += ["keep_mask", "online_bits_pruned"]


def keep_mask_loo(train, V: int, R: int, lam: float = 1.0, prec_bits: float = 4.0):
    """Keep-sets chosen by what a context is ACTUALLY worth to the blend, not by a standalone comparison.

    `keep_mask` above asked whether a context predicts better than its parent on its own. That is the wrong
    question and it prunes almost everything: a context seen once has a KT estimate of 1/(1 + V/2), which looks
    terrible beside a well-populated parent -- but the model never USES it as a replacement, it interpolates it,
    contributing c/(N + V/2) of the mass and passing the rest down. So the worth of a context is the difference in
    coded bits, over its own observations, between the blend WITH its order and the blend WITHOUT it -- a
    leave-one-out on the real objective. Everything else is the same two-part decision: keep iff saved > cost."""
    train = np.asarray(train, dtype=np.int64)
    per_order = [online_counts(train, V, r) for r in range(R, -1, -1)]
    full = -np.log2(np.maximum(_blend(per_order, V), 1e-12))
    keeps = {}
    for i, r in enumerate(range(R, -1, -1)):
        if r == 0:
            continue
        without = [(np.zeros_like(a), np.zeros_like(b)) if j == i else (a, b)
                   for j, (a, b) in enumerate(per_order)]
        delta = -np.log2(np.maximum(_blend(without, V), 1e-12)) - full      # bits this order saves, per position
        ctx = _codes(train, V, r)
        u, inv = np.unique(ctx, return_inverse=True)
        saved = np.zeros(len(u))
        np.add.at(saved, inv, delta)
        support = np.zeros(len(u), dtype=np.int64)
        np.add.at(support, np.unique(np.stack([inv, train]), axis=1)[0], 1)
        keeps[r] = u[saved > lam * prec_bits * support]
    return keeps


__all__ += ["keep_mask_loo"]


def context_worth(train, V: int, R: int):
    """Per-context bits-saved, the quantity §9 rule 3 ranks by. Returns {order: (codes, saved_bits)}."""
    train = np.asarray(train, dtype=np.int64)
    per_order = [online_counts(train, V, r) for r in range(R, -1, -1)]
    full = -np.log2(np.maximum(_blend(per_order, V), 1e-12))
    worth = {}
    for i, r in enumerate(range(R, -1, -1)):
        if r == 0:
            continue
        without = [(np.zeros_like(a), np.zeros_like(b)) if j == i else (a, b)
                   for j, (a, b) in enumerate(per_order)]
        delta = -np.log2(np.maximum(_blend(without, V), 1e-12)) - full
        ctx = _codes(train, V, r)
        u, inv = np.unique(ctx, return_inverse=True)
        saved = np.zeros(len(u))
        np.add.at(saved, inv, delta)
        worth[r] = (u, saved)
    return worth


def keep_top(worth, budget: int):
    """§9 rule 3 as written: keep the highest-worth contexts across ALL orders until the budget is spent, and
    drop the rest -- a RANKING under a capacity budget, not a per-item threshold."""
    rows = [(r, i, s) for r, (u, sv) in worth.items() for i, s in enumerate(sv)]
    rows.sort(key=lambda t: -t[2])
    keeps = {r: [] for r in worth}
    for r, i, s in rows[:budget]:
        keeps[r].append(worth[r][0][i])
    return {r: np.sort(np.array(v, dtype=np.int64)) for r, v in keeps.items()}


__all__ += ["context_worth", "keep_top"]
