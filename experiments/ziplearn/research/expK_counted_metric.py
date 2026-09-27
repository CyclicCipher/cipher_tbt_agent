"""expK (S3, 2026-09-27) -- the counted semantic metric, tested where expL says the bits actually are.

expL: 13.4% of positions are the FIRST CHARACTER OF A WORD and they carry 32.1% of the code length (4.52 bits each
against 1.24 for word-interior characters). That is the one place a character model has nothing to lean on -- the
context ends at the space. expL also shows a counted word BIGRAM is WORSE than the character chain there (5.04 vs
4.52): with 15k+ word types the previous word alone is too sparse to count. So the question S23 asks is exactly the
question the data asks: can a metric that lets SIMILAR previous words share their successor statistics beat both?

Three predictors of the first character of a word, all gradient-free, all from counts:
  bigram     p(c | previous word), KT-smoothed, backing off to the marginal              (expL's 5.04)
  lowrank    the SAME count matrix, PPMI'd and truncated by SVD to rank r, renormalised  -- S23.2's asymmetric
             construction in its purest form: Sigma_qk = a counted cross-covariance, A^T B = its truncated SVD
  knn        p(c | w) pooled over the k nearest previous words under a PPMI+SVD word-word metric  -- the symmetric
             metric used as a smoother
Reference: the character chain's own bits at the same positions.

Also reports the morphology diagnostic S23.5 asks for: do the metric's nearest neighbours share stems?

Prints JSON to runs/research/expK.json. No neural network is built, run or trained.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from e34 import Chain, Stream                                          # noqa: E402
from textlm import load_corpus, training_slice                         # noqa: E402

STEM = 4
WIN = 4                                                                # +-WIN words for the word-word co-occurrence
MAXV = 12000                                                           # word types kept; the rest are UNK


def words_of(seq, alpha):
    out, cur = [], []
    for c in seq:
        if c in alpha:
            cur.append(c)
        elif cur:
            out.append(tuple(cur)); cur = []
    if cur:
        out.append(tuple(cur))
    return out


def first_chars(seq, alpha):
    """[(previous word, first character of the next word)] over a sequence."""
    ws, cur, pairs, prev = [], [], [], None
    lst = seq
    i, n = 0, len(lst)
    while i < n:
        if lst[i] in alpha:
            j = i
            while j < n and lst[j] in alpha:
                j += 1
            if prev is not None:
                pairs.append((prev, lst[i]))
            prev = tuple(lst[i:j])
            i = j
        else:
            i += 1
    return pairs


def ppmi(M, shift=1.0):
    """Positive PMI of a count matrix, as a dense array."""
    M = np.asarray(M, dtype=np.float64)
    tot = M.sum()
    if tot == 0:
        return M
    r = M.sum(1, keepdims=True); c = M.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.log((M * tot) / (r * c) / shift)
    P[~np.isfinite(P)] = 0.0
    return np.maximum(P, 0.0)


def main():
    t0 = time.time()
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 900_000
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    alpha = {i for i, ch in enumerate(alphabet) if ch.isalpha()}
    seqs = training_slice(train, n_train)
    out = {"n_train_chars": sum(len(s) for s in seqs), "V": V, "rank_grid": [8, 16, 32, 64]}

    # ---------- the word vocabulary ----------
    wc = Counter()
    for s in seqs:
        wc.update(words_of(s.tolist(), alpha))
    vocab = [w for w, _ in wc.most_common(MAXV)]
    wid = {w: i for i, w in enumerate(vocab)}
    UNK = len(vocab)
    out["word_types_total"], out["word_types_kept"] = len(wc), len(vocab)

    # ---------- Sigma_qk : counts of (previous word -> first character of the next word) ----------
    B = np.zeros((len(vocab) + 1, V), dtype=np.float64)
    for s in seqs:
        for pw, c in first_chars(s.tolist(), alpha):
            B[wid.get(pw, UNK), c] += 1.0
    marg = B.sum(0) + 0.5
    marg = marg / marg.sum()
    out["train_boundaries"] = int(B.sum())

    # ---------- the word-word PPMI metric (for the kNN smoother and the morphology diagnostic) ----------
    rows, cols, vals = [], [], []
    for s in seqs:
        ws = [wid.get(w, UNK) for w in words_of(s.tolist(), alpha)]
        for i, a in enumerate(ws):
            for j in range(max(0, i - WIN), min(len(ws), i + WIN + 1)):
                if j != i:
                    rows.append(a); cols.append(ws[j]); vals.append(1.0)
    C = sp.coo_matrix((vals, (rows, cols)), shape=(len(vocab) + 1, len(vocab) + 1)).tocsr()
    tot = C.sum(); rsum = np.asarray(C.sum(1)).ravel() + 1e-9; csum = np.asarray(C.sum(0)).ravel() + 1e-9
    Cc = C.tocoo()
    pm = np.log((Cc.data * tot) / (rsum[Cc.row] * csum[Cc.col]))
    keep = pm > 0
    P = sp.coo_matrix((pm[keep], (Cc.row[keep], Cc.col[keep])), shape=C.shape).tocsr()
    k_emb = 64
    U, S, _ = svds(P.astype(np.float32), k=k_emb)
    E = U[:, ::-1] * S[::-1]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    out["embedding_rank"] = k_emb

    # ---------- diagnostic: do the nearest neighbours share stems? ----------
    probe = [wid[w] for w, _ in wc.most_common(600) if w in wid and len(w) >= STEM][:500]
    Ep = E[probe]
    sims = Ep @ E[:len(vocab)].T
    for n, i in enumerate(probe):
        sims[n, i] = -9.0
    nn = np.argsort(-sims, axis=1)[:, :10]
    share = np.mean([[vocab[probe[n]][:STEM] == vocab[j][:STEM] for j in nn[n]] for n in range(len(probe))])
    rng = np.random.default_rng(0)
    base = np.mean([[vocab[probe[n]][:STEM] == vocab[j][:STEM]
                     for j in rng.integers(0, len(vocab), 10)] for n in range(len(probe))])
    out["morphology"] = {"probe_words": len(probe), "top10_share_stem": float(share),
                         "random_baseline": float(base),
                         "examples": [{"word": "".join(alphabet[c] for c in vocab[probe[n]]),
                                       "nn": ["".join(alphabet[c] for c in vocab[j]) for j in nn[n][:5]]}
                                      for n in range(0, 40, 8)]}

    # ---------- low-rank Sigma_qk at several ranks ----------
    Bp = ppmi(B)
    lowrank = {}
    for r in out["rank_grid"]:
        Ur, Sr, Vr = svds(Bp.astype(np.float32), k=min(r, min(Bp.shape) - 1))
        R = np.maximum((Ur * Sr) @ Vr, 0.0)
        R = R + 1e-3
        lowrank[r] = R / R.sum(1, keepdims=True)

    # ---------- kNN-smoothed successor table ----------
    KNN = 24
    Brow = B + 0.0
    nrm = Brow.sum(1, keepdims=True)
    simsA = E[:len(vocab)] @ E[:len(vocab)].T
    np.fill_diagonal(simsA, -9.0)
    top = np.argpartition(-simsA, KNN, axis=1)[:, :KNN]
    knn_tab = np.zeros_like(Brow)
    for i in range(len(vocab)):
        w = np.maximum(simsA[i, top[i]], 0.0)
        if w.sum() <= 0:
            knn_tab[i] = Brow[i]; continue
        knn_tab[i] = Brow[i] + (w[:, None] * Brow[top[i]]).sum(0) / (w.sum() + 1e-9) * 3.0
    knn_tab[UNK] = Brow[UNK]
    knn_p = (knn_tab + 0.5) / (knn_tab.sum(1, keepdims=True) + V * 0.5)

    # ---------- evaluate on the held-out book, at word-initial positions only ----------
    tst = test.tolist()
    pairs = first_chars(tst, alpha)
    idx = np.array([wid.get(pw, UNK) for pw, _ in pairs])
    cs = np.array([c for _, c in pairs])
    Np = B.sum(1)

    def bits_of(p):
        return float(np.mean(-np.log2(np.maximum(p[idx, cs], 1e-12))))

    pb = (B + 0.5) / (Np[:, None] + V * 0.5)
    seen = Np[idx] > 0
    pb_mix = np.where(seen[:, None], pb[idx], marg[None, :])
    res = {
        "n_boundaries": len(pairs),
        "marginal": float(np.mean(-np.log2(np.maximum(marg[cs], 1e-12)))),
        "bigram_kt": float(np.mean(-np.log2(np.maximum(pb_mix[np.arange(len(cs)), cs], 1e-12)))),
        "knn_smoothed": bits_of(knn_p),
        "lowrank": {str(r): bits_of(lowrank[r]) for r in out["rank_grid"]},
    }

    # the character chain's own bits at the same positions (the reference expL measured)
    chain, stream = Chain(V, R=8), Stream(V, alphabet)
    for s in seqs:
        for c in s.tolist():
            chain.update(chain.predict(stream), c); stream.push(c)
    chain_bits, at = [], set()
    i, n = 0, len(tst)
    prev_end = None
    while i < n:
        if tst[i] in alpha:
            if prev_end is not None:
                at.add(i)
            j = i
            while j < n and tst[j] in alpha:
                j += 1
            prev_end = j; i = j
        else:
            i += 1
    chain_p = np.zeros(n)
    for i, c in enumerate(tst):
        keys = chain.predict(stream)
        chain_p[i] = chain.prob(keys, c)
        chain.update(keys, c); stream.push(c)
        if i % 200000 == 0:
            print(f"  chain {i}/{n}", flush=True)
    order = sorted(at)
    cb = -np.log2(np.maximum(chain_p[order], 1e-12))
    res["chain_at_boundaries"] = float(np.mean(cb))

    # the combination that matters: the chain is one expert, the metric another. A LINEAR pool (which E34 showed
    # can only choose) and a GEOMETRIC pool with equal exponents (which E34 showed is where the gain lives).
    best_r = min(out["rank_grid"], key=lambda r: res["lowrank"][str(r)])
    pl = lowrank[best_r][idx, cs]
    pc = chain_p[order]
    m = min(len(pl), len(pc)); pl, pc = pl[:m], pc[:m]
    res["best_rank"] = best_r
    res["pool_linear_50_50"] = float(np.mean(-np.log2(np.maximum(0.5 * pl + 0.5 * pc, 1e-12))))
    g = np.sqrt(np.maximum(pl, 1e-12) * np.maximum(pc, 1e-12))
    res["pool_geometric_unnormalised"] = float(np.mean(-np.log2(np.maximum(g, 1e-12))))
    res["note_geometric"] = "unnormalised: a lower bound on the pool's cost, not a valid code (S24.2)"
    res["corpus_bpc_if_boundaries_improved"] = {
        k: float((np.sum(cb) - (res["chain_at_boundaries"] - v) * len(cb)) / len(tst))
        for k, v in [("lowrank_best", res["lowrank"][str(best_r)]), ("linear_pool", res["pool_linear_50_50"])]
    }
    out["word_initial_prediction"] = res
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expK.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
