"""Language probe — the column's SR-frame IS a word embedding (architecture doc; the geometry question).

Last turn's claim: the SR-eigenvector frame (the column's L6 location code) and a word embedding are the SAME
object — both are low-rank spectral embeddings of a co-occurrence/transition statistic. word2vec factorizes
shifted PMI (Levy & Goldberg 2014); `_sr_frame` is the eigenbasis of the normalized graph (Stachenfeld 2017).
So feed a column the next-token statistics of a corpus and its frame should come out as a semantic geometry,
the way human MTL and LLMs do (Goldstein 2022; Jamali 2024; Platonic convergence, Huh 2024).

We test that on the only text we have — classical Latin (Caesar, Tacitus, Apuleius, Ovid, Cato) and Middle
High German (Nibelungenlied) — with a NEXT-TOKEN objective:

  geometry (undirected) : symmetric windowed co-occurrence -> PPMI -> `_sr_frame` (the column's LITERAL code)
                          -> word vectors. Validated by nearest-neighbour coherence + inflection analogies.
  prediction (directed) : forward co-occurrence -> PPMI -> truncated SVD = the DIRECTED SR-frame (eigh is its
                          symmetric special case) -> P(next=j | cur=i) ~ P(j)*exp(w_i . c_j). Held-out
                          perplexity vs a unigram and an add-alpha bigram, with a rare-context breakdown.

The claim to land: the low-rank geometry GENERALISES the next-token statistics (beats the raw bigram on unseen
contexts) — the same reason E3 soft-retrieval beat the transformer on unseen pairs — and its neighbours are
morphologically/semantically coherent. No imported embeddings; the geometry is the column's own mechanism.

Run:  python -m precursor.language          (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch                                                          # noqa: E402

from tbt.column import _sr_frame                                      # the column's LITERAL frame  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "corpora")
TOKEN_RE = re.compile(r"[a-zà-ÿæœþð]+")
UNK = "<unk>"

# small Latin inflection analogies (EVAL probes — measuring the displacement structure, not teaching a rule):
# 2nd-declension nominative -> genitive singular, the most regular paradigm in the corpus.
ANALOGIES = [
    ("bellum", "belli", "regnum", "regni"), ("bellum", "belli", "verbum", "verbi"),
    ("bellum", "belli", "oppidum", "oppidi"), ("bellum", "belli", "consilium", "consilii"),
    ("bellum", "belli", "imperium", "imperii"), ("bellum", "belli", "auxilium", "auxilii"),
    ("dominus", "domini", "animus", "animi"), ("dominus", "domini", "populus", "populi"),
    ("dominus", "domini", "murus", "muri"), ("dominus", "domini", "servus", "servi"),
    ("dominus", "domini", "equus", "equi"), ("animus", "animi", "populus", "populi"),
    ("regnum", "regni", "imperium", "imperii"), ("populus", "populi", "servus", "servi"),
]
PROBES = ["rex", "bellum", "caesar", "miles", "urbs", "deus", "manus", "dies", "pater", "magnus"]


# ── corpus ────────────────────────────────────────────────────────────────────────────────────────────
def load_docs(subdir, fold_j=True):
    """Each .txt = one document (token list). Strip BOM + [section] markers, lowercase, j->i (Latin)."""
    root = os.path.join(CORPUS, subdir)
    docs = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".txt"):
            continue
        text = open(os.path.join(root, fn), encoding="utf-8").read().lstrip("﻿")
        text = re.sub(r"\[[^\]]*\]", " ", text).lower()
        if fold_j:
            text = text.replace("j", "i")
        toks = TOKEN_RE.findall(text)
        if toks:
            docs.append(toks)
    return docs


def chunk_docs(docs, size=1500):
    """Cut into fixed-length pseudo-documents so holdout works uniformly (the Nibelungenlied is one file)."""
    return [d[s:s + size] for d in docs for s in range(0, len(d), size) if d[s:s + size]]


def split_docs(docs, test_frac=0.15, seed=0):
    order = list(range(len(docs)))
    np.random.default_rng(seed).shuffle(order)
    ntok = sum(len(docs[i]) for i in order)
    test, acc = set(), 0
    for i in order:                                                   # whole documents held out (no leakage)
        if acc < test_frac * ntok:
            test.add(i); acc += len(docs[i])
    train = [docs[i] for i in range(len(docs)) if i not in test]
    held = [docs[i] for i in range(len(docs)) if i in test]
    return train, held


def build_vocab(train_docs, V):
    cnt = Counter(t for d in train_docs for t in d)
    keep = [w for w, _ in cnt.most_common(V - 1)]
    itos = [UNK] + keep
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(docs, stoi):
    unk = stoi[UNK]
    return [[stoi.get(t, unk) for t in d] for d in docs]


# ── counts ────────────────────────────────────────────────────────────────────────────────────────────
def forward_cooc(enc, V, window):
    """C[i,j] = sum over positions of (i at t) co-occurring with (j at t+1..t+window), 1/distance weighted."""
    rows, cols, vals = [], [], []
    for d in enc:
        for t, i in enumerate(d):
            for k in range(1, window + 1):
                if t + k < len(d):
                    rows.append(i); cols.append(d[t + k]); vals.append(1.0 / k)
    return coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()


def symmetric_cooc(enc, V, window):
    C = forward_cooc(enc, V, window)
    return C + C.T


def strict_bigram(enc, V):
    rows, cols = [], []
    uni = np.zeros(V)
    for d in enc:
        for t, i in enumerate(d):
            uni[i] += 1
            if t + 1 < len(d):
                rows.append(i); cols.append(d[t + 1])
    C = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr()
    return C, uni


def ppmi(C, shift=1.0):
    """Positive PMI: max(0, log[ P(i,j) / (P(i)P(j)) ] - log shift). Sparse, on the nonzero entries only."""
    C = C.tocoo()
    total = C.data.sum()
    row_sum = np.asarray(C.sum(1)).ravel()
    col_sum = np.asarray(C.sum(0)).ravel()
    pmi = np.log((C.data * total) / (row_sum[C.row] * col_sum[C.col]) + 1e-12) - math.log(shift)
    pmi = np.maximum(pmi, 0.0)
    keep = pmi > 0
    return coo_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape).tocsr()


# ── the directed SR-frame (SVD of the directed transition PPMI) ─────────────────────────────────────────
def directed_embed(P, d):
    u, s, vt = svds(P.astype(np.float64), k=d)
    order = np.argsort(s)[::-1]
    s, u, vt = s[order], u[:, order], vt[order, :]
    root = np.sqrt(s)[None, :]
    return u * root, vt.T * root                                      # w_i (current), c_j (next)


# ── next-token perplexity ───────────────────────────────────────────────────────────────────────────────
def pairs_of(enc):
    return [(d[t], d[t + 1]) for d in enc for t in range(len(d) - 1)]


def sr_lm_nll(w, c, uni_logp, pairs, beta):
    """mean -log P(j|i), P(j|i) = softmax_j( log P(j) + beta * w_i . c_j ) over the full vocabulary."""
    ctx = sorted({i for i, _ in pairs})
    row = {i: r for r, i in enumerate(ctx)}
    W = torch.tensor(w[ctx], dtype=torch.float32)
    C = torch.tensor(c, dtype=torch.float32)
    up = torch.tensor(uni_logp, dtype=torch.float32)
    logits = beta * (W @ C.t()) + up[None, :]                         # (U, V)
    logZ = torch.logsumexp(logits, dim=1)
    nll = 0.0
    for i, j in pairs:
        r = row[i]
        nll += (logZ[r] - logits[r, j]).item()
    return nll / len(pairs)


def interp_bigram_nll(Cbg, rs, uni_p, pairs, lam):
    """Jelinek-Mercer: P(j|i) = lam * count(i,j)/count(i) + (1-lam) * P(j). The FAIR n-gram baseline —
    it backs off to the unigram on unseen contexts (not the add-alpha strawman that over-penalizes)."""
    out = 0.0
    for i, j in pairs:
        p = lam * (Cbg[i, j] / rs[i]) + (1 - lam) * uni_p[j] if rs[i] > 0 else uni_p[j]
        out += -math.log(p + 1e-12)
    return out / len(pairs)


def unigram_nll(uni_logp, pairs):
    return -sum(uni_logp[j] for _, j in pairs) / len(pairs)


def nearest(w, itos, stoi, word, k=8):
    if word not in stoi:
        return []
    wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
    sims = wn @ wn[stoi[word]]
    idx = np.argsort(-sims)[1:k + 1]
    return [itos[i] for i in idx]


def analogy_acc(w, stoi, itos, k=5):
    wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
    hit = seen = 0
    for a, astar, b, bstar in ANALOGIES:
        if not all(x in stoi for x in (a, astar, b, bstar)):
            continue
        seen += 1
        q = wn[stoi[astar]] - wn[stoi[a]] + wn[stoi[b]]
        q /= np.linalg.norm(q) + 1e-9
        ranked = np.argsort(-(wn @ q))
        ban = {stoi[a], stoi[astar], stoi[b]}
        top = [i for i in ranked if i not in ban][:k]
        hit += int(stoi[bstar] in top)
    return hit, seen


# ── run ───────────────────────────────────────────────────────────────────────────────────────────────
def run(subdir, V=4000, d=200, window=5, test_frac=0.15, seed=0, fold_j=True, sym_V=1500):
    docs = chunk_docs(load_docs(subdir, fold_j=fold_j))
    train, held = split_docs(docs, test_frac, seed)
    stoi, itos = build_vocab(train, V)
    V = len(itos)
    enc_tr, enc_te = encode(train, stoi), encode(held, stoi)
    n_tr = sum(len(d) for d in enc_tr); n_te = sum(len(d) for d in enc_te)

    probes = [p for p in PROBES if p in stoi]                         # hand-picked Latin, kept where present
    probes += [wd for wd in itos[1:] if len(wd) >= 4 and wd not in probes][:8 - len(probes)]   # else frequent

    # directed SR-frame for prediction + geometry
    P = ppmi(forward_cooc(enc_tr, V, window))
    w, c = directed_embed(P, d)

    # next-token: tune beta on a dev slice carved from train, report test
    dev = pairs_of(enc_tr[-max(1, len(enc_tr) // 6):])
    test = pairs_of(enc_te)
    Cbg, uni = strict_bigram(enc_tr, V)
    rs = np.asarray(Cbg.sum(1)).ravel()
    uni_logp = np.log((uni + 1.0) / (uni.sum() + V))
    uni_p = np.exp(uni_logp)
    beta = min([0.25, 0.5, 1.0, 2.0, 4.0], key=lambda b: sr_lm_nll(w, c, uni_logp, dev, b))
    lam = min([0.0, 0.1, 0.3, 0.5, 0.7, 0.9], key=lambda l: interp_bigram_nll(Cbg, rs, uni_p, dev, l))

    ppl = lambda nll: math.exp(nll)
    res = {
        "unigram": ppl(unigram_nll(uni_logp, test)),
        "bigram(interp)": ppl(interp_bigram_nll(Cbg, rs, uni_p, test, lam)),
        "SR-frame": ppl(sr_lm_nll(w, c, uni_logp, test, beta)),
    }
    # rare-context split: is the win concentrated where counting has thin evidence?
    freq_rank = {i: r for r, i in enumerate(np.argsort(-uni))}
    rare = [(i, j) for i, j in test if freq_rank[i] > 500]
    rare_res = (ppl(interp_bigram_nll(Cbg, rs, uni_p, rare, lam)),
                ppl(sr_lm_nll(w, c, uni_logp, rare, beta))) if rare else None

    # geometry: the column's LITERAL undirected frame on the same corpus (smaller V for the dense eigh)
    sym_stoi = {wd: i for i, wd in enumerate(itos[:sym_V])}
    Wsym = symmetric_cooc([[t for t in d if t < sym_V] for d in enc_tr], sym_V, window)
    Wsym = ppmi(Wsym).toarray()
    Z = _sr_frame(torch.tensor(Wsym, dtype=torch.float64)).numpy()[:, -d:]   # top-d non-trivial modes
    overlap = _frame_overlap(w[:sym_V], Z, sym_stoi, probes)
    sym_itos = itos[:sym_V]
    und_neighbours = {p: nearest(Z, sym_itos, sym_stoi, p) for p in probes if p in sym_stoi}

    ha, sa = analogy_acc(w, stoi, itos)
    return dict(subdir=subdir, V=V, d=d, beta=beta, lam=lam, n_tr=n_tr, n_te=n_te, ndocs=len(docs),
                res=res, rare=rare_res, n_test=len(test), n_rare=len(rare), probes=probes,
                neighbours={p: nearest(w, itos, stoi, p) for p in probes},
                und_neighbours=und_neighbours, analogy=(ha, sa), overlap=overlap)


def _frame_overlap(w_dir, Z, sym_stoi, probe_words, k=10):
    """Do the directed model and the column's literal undirected `_sr_frame` find the SAME neighbours?"""
    a = w_dir / (np.linalg.norm(w_dir, axis=1, keepdims=True) + 1e-9)
    b = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    probes = [sym_stoi[p] for p in probe_words if p in sym_stoi]
    if not probes:
        return float("nan")
    tot = 0.0
    for i in probes:
        na = set(np.argsort(-(a @ a[i]))[1:k + 1])
        nb = set(np.argsort(-(b @ b[i]))[1:k + 1])
        tot += len(na & nb) / k
    return tot / len(probes)


if __name__ == "__main__":
    print("language probe — the column's SR-frame as a word embedding, next-token objective\n")
    for sub in ("latin books", "mittelhochdeutsch"):
        r = run(sub)
        print(f"=== {sub}  ({r['ndocs']} docs, {r['n_tr']:,} train / {r['n_te']:,} test tokens, "
              f"V={r['V']}, d={r['d']}, beta={r['beta']}) ===")
        print(f"  next-token perplexity (held-out, {r['n_test']:,} pairs):")
        for name, p in r["res"].items():
            print(f"      {name:>10}:  {p:8.1f}")
        if r["rare"]:
            print(f"  rare-context (ctx rank>500, {r['n_rare']:,} pairs):  bigram(interp) {r['rare'][0]:.1f}"
                  f"   SR-frame {r['rare'][1]:.1f}")
        ha, sa = r["analogy"]
        ana = f"inflection analogies (top-5): {ha}/{sa}    " if sa else ""
        print(f"  {ana}undirected `_sr_frame` agrees with directed on {r['overlap']*100:.0f}% of neighbours")
        print("  nearest neighbours — directed frame (prediction / next-token geometry):")
        for p in r["probes"][:6]:
            if r["neighbours"][p]:
                print(f"      {p:>10} -> {', '.join(r['neighbours'][p])}")
        print("  nearest neighbours — the column's LITERAL `_sr_frame` (undirected / similarity geometry):")
        for p in r["probes"][:6]:
            if r["und_neighbours"].get(p):
                print(f"      {p:>10} -> {', '.join(r['und_neighbours'][p])}")
        print()
