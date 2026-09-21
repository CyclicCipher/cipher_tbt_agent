"""Experiment F -- a gradient-free FEATURE inside the text denoising block: coarse keys learned from the windows.

E35's text block (experiments/ziplearn/e35.py, `SeqRule`) fills a masked character from the +-3 window around it,
keyed by the EXACT window; a window never seen backs off by shrinking the window symmetrically (r = 3 -> 2 -> 1 -> 0).
At mask rate 0.5 it reconstructs 0.320 of the masked characters one shot (0.5 -> 0) and 0.309 with the 2-block chain
(0.5 -> 0.25 -> 0). This script keeps the block (a table learned by counting, the majority target per key, backoff
through a fixed list of keys, finest first) and swaps the KEY:

  (i)   exact      the E35 key: the exact window, symmetric shrink backoff  [r3, r2, r1, r0]
  (ii)  km<k>      a k-means clustering of the windows (each window = the concatenation of one-hot characters over the
                   7 positions, Euclidean = Hamming; k in {256, 1024, 4096}; fitted on a sample of the training
                   windows at the block's own mask level), the table keyed by the cluster id.  Variants:
                     km<k>            [km, global]
                     exact3>km<k>     [r3, km, global]                 -- exact when the full window was seen, else cluster
                     exact3(n3)>km<k> [r3 with count >= 3, km, global] -- exact only when seen >= 3 times
                     km-hier          [r3, km4096, km1024, km256, global]
                     km-mi<k>         k-means with the positions weighted by MI(character at p; target), from counts
  (iii) pca<b>     the one-hot window projected on its top b principal directions (covariance from counts), each
                   quantised to its sign; key = the b-bit code, backoff by dropping the lowest-variance bits
                   [pca_b, pca_{b-2}, ..., pca_2, global]; also exact3>pca<b>.
  (iv)  nb         the naive-Bayes product of the 7 per-position tables P(y) prod_p P(x_p | y) (counts + 1/2): the
                   cheapest "shared structure" predictor, no window key at all (an extra reference); exact3>nb and
                   exact32>nb use it as the backoff behind the exact r3 (r3, r2) tables.

Every variant is fitted and predicted ONLY at masked positions (the centre is MASK); E35's SeqRule is also run as-is
for the reproduction row.  Measured on the held-out book (De Bello Civili), the first 60k characters, mask rate 0.5:
one shot (block 0.5 -> 0) and the chain (0.5 -> 0.25 then 0.25 -> 0); for the one shot also which backoff level
resolved each prediction and that level's accuracy.  Training: 300k characters (E35's slice), chunks of 64.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/research/expF_cluster_features_text.py
        -> experiments/ziplearn/research/expF_cluster_features_text.json      (CPU, ~4 min)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ZL = HERE.parent
sys.path.insert(0, str(ZL))
sys.path.insert(0, str(ZL.parent.parent / "src"))
from e35 import SeqRule, mask_chain                                  # noqa: E402
from textlm import load_corpus, training_slice                      # noqa: E402

R = 3
P = 2 * R + 1


# ── windows ──────────────────────────────────────────────────────────────────────────────────────────────────────
def join_chunks(chunks, BORDER):
    """One array: BORDER x3, chunk, BORDER x3, chunk, ... -- a +-3 window then sees exactly E35's per-chunk padding."""
    parts = [np.full(R, BORDER, np.int64)]
    for c in chunks:
        parts.append(c.astype(np.int64))
        parts.append(np.full(R, BORDER, np.int64))
    return np.concatenate(parts)


def windows(joined):
    """(n, 7) the +-3 window around every position of the joined array (its own BORDER padding covers the ends)."""
    n = len(joined)
    return np.stack([joined[R + off: n - R + off] for off in range(-R, R + 1)], 1), np.arange(R, n - R)


def exact_key(W, rr, base):
    key = np.zeros(len(W), dtype=np.int64)
    for p in range(R - rr, R + rr + 1):
        key = key * base + W[:, p]
    return key


# ── the table: majority target per key, backoff through a list of keys ──────────────────────────────────────────
def majority_table(keys, y, ybase):
    pair = keys * ybase + y
    up, pc = np.unique(pair, return_counts=True)
    ks, ys = up // ybase, up % ybase
    order = np.lexsort((-pc, ks))
    ks, ys, pc = ks[order], ys[order], pc[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return ks[first], ys[first], np.add.reduceat(pc, first)


def lookup(table, keys):
    mk, mv, tot = table
    pos = np.minimum(np.searchsorted(mk, keys), len(mk) - 1)
    hit = mk[pos] == keys
    return np.where(hit, mv[pos], -1), np.where(hit, tot[pos], 0)


class Backoff:
    """levels: list of (name, keyfn, n_min); the prediction is the first level whose key was seen >= n_min times.
    `cache` (shared across predictors over the same input level) holds each key kind's values per tagged array."""

    def __init__(self, levels, ybase, cache):
        self.levels, self.ybase, self.cache = levels, ybase, cache

    def _keys(self, name, fn, W, tag):
        c = self.cache.get((name, tag))
        if c is None:
            c = self.cache[(name, tag)] = fn(W)
        return c

    def fit(self, W, y, tag):
        self.tables = [majority_table(self._keys(n, fn, W, tag), y, self.ybase) for n, fn, _m in self.levels]
        return self

    def predict(self, W, tag):
        out = np.full(len(W), -1, np.int64)
        which = np.full(len(W), -1, np.int64)
        pending = np.ones(len(W), dtype=bool)
        for li, ((n, fn, n_min), tab) in enumerate(zip(self.levels, self.tables)):
            if not pending.any():
                break
            idx = np.flatnonzero(pending)
            v, tot = lookup(tab, self._keys(n, fn, W, tag)[idx])
            ok = (v >= 0) & (tot >= n_min)
            out[idx[ok]] = v[ok]
            which[idx[ok]] = li
            pending[idx[ok]] = False
        return out, which


# ── (ii) k-means over one-hot windows, by gathers (a one-hot dot product is a lookup) ────────────────────────────
def kmeans_assign(W, C, w, B=4096):
    k, _P, _base = C.shape
    ps = [p for p in range(P) if w[p] > 0]
    G = [np.ascontiguousarray((w[p] * C[:, p, :]).T.astype(np.float32)) for p in ps]         # (base, k)
    bias = (-0.5 * sum(w[p] * (C[:, p, :] ** 2).sum(1) for p in ps)).astype(np.float32)       # (k,)
    out = np.empty(len(W), np.int64)
    for s in range(0, len(W), B):
        Wb = W[s:s + B]
        sc = np.tile(bias, (len(Wb), 1))
        for g, p in zip(G, ps):
            sc += g[Wb[:, p]]
        out[s:s + B] = sc.argmax(1)
    return out


def kmeans_fit(W, k, base, rng, iters=8, sample=30000, w=None):
    w = np.ones(P) if w is None else np.asarray(w, dtype=np.float64)
    S = W[rng.choice(len(W), min(sample, len(W)), replace=False)]
    init = S[rng.choice(len(S), k, replace=False)]
    C = np.zeros((k, P, base), np.float32)
    for p in range(P):
        C[np.arange(k), p, init[:, p]] = 1.0
    prev = None
    for it in range(iters):
        a = kmeans_assign(S, C, w)
        nc = np.bincount(a, minlength=k)
        counts = np.zeros((k, P, base), np.float32)
        for p in range(P):
            counts[:, p, :] = np.bincount(a * base + S[:, p], minlength=k * base).reshape(k, base)
        empty = np.flatnonzero(nc == 0)
        if len(empty):
            re = S[rng.choice(len(S), len(empty), replace=False)]
            for p in range(P):
                counts[empty, p, re[:, p]] = 1.0
            nc[empty] = 1
        C = counts / nc[:, None, None]
        changed = int(len(a)) if prev is None else int((a != prev).sum())
        prev = a
        if changed < 0.002 * len(S):
            break
    return C, dict(iters=it + 1, changed_last=changed, sample=int(len(S)), used=int((nc > 0).sum()))


def mi_weights(W, y, base, ybase):
    """MI(character at position p; target) in bits, from the training counts -- the count-based relevance of a position."""
    out = np.zeros(P)
    py = np.bincount(y, minlength=ybase) / len(y)
    for p in range(P):
        j = np.bincount(W[:, p] * ybase + y, minlength=base * ybase).reshape(base, ybase) / len(y)
        px = j.sum(1)
        nz = j > 0
        out[p] = float((j[nz] * np.log2(j[nz] / (px[:, None] * py[None, :])[nz])).sum())
    return out


# ── (iii) PCA of the one-hot window from counts, sign-quantised ────────────────────────────────────────────────
def pca_fit(W, base, n_comp=16):
    D = P * base
    N = len(W)
    XtX = np.zeros((D, D))
    for p in range(P):
        for q in range(P):
            XtX[p * base:(p + 1) * base, q * base:(q + 1) * base] = np.bincount(W[:, p] * base + W[:, q], minlength=base * base).reshape(base, base)
    mu = np.diag(XtX) / N
    cov = XtX / N - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(cov)
    U = evecs[:, ::-1][:, :n_comp]                                    # (D, n_comp) top first
    return dict(Up=U.reshape(P, base, n_comp), offset=mu @ U, evals=evals[::-1][:n_comp], n=n_comp)


def pca_code(W, model):
    Up, off, nb = model["Up"], model["offset"], model["n"]
    proj = -off[None, :].repeat(len(W), 0)
    for p in range(P):
        proj += Up[p][W[:, p]]
    bits = proj > 0
    code = np.zeros(len(W), np.int64)
    for b in range(nb):
        code |= bits[:, b].astype(np.int64) << (nb - 1 - b)        # the top component is the most significant bit
    return code


# ── (iv) naive Bayes over the positions ─────────────────────────────────────────────────────────────────────────
class NaiveBayes:
    def __init__(self, base, ybase):
        self.base, self.ybase = base, ybase

    def fit(self, W, y):
        base, ybase = self.base, self.ybase
        self.logprior = np.log(np.bincount(y, minlength=ybase) + 0.5)
        self.logp = []
        for p in range(P):
            j = np.bincount(y * base + W[:, p], minlength=ybase * base).reshape(ybase, base) + 0.5
            self.logp.append(np.log(j / j.sum(1, keepdims=True)).T.astype(np.float32))         # (base, ybase)
        return self

    def predict(self, W, tag=None):
        sc = np.tile(self.logprior.astype(np.float32), (len(W), 1))
        for p in range(P):
            if p != R:
                sc += self.logp[p][W[:, p]]
        return sc.argmax(1), np.zeros(len(W), np.int64)


class Fallback:
    """A Backoff over some exact levels, and a keyless predictor (naive Bayes) for whatever they leave unresolved."""

    def __init__(self, first, rest):
        self.first, self.rest = first, rest
        self.levels = first.levels + [("nb", None, 1)]

    def predict(self, W, tag):
        out, which = self.first.predict(W, tag)
        pend = which < 0
        if pend.any():
            out[pend] = self.rest.predict(W[pend])[0]
            which[pend] = len(self.first.levels)
        return out, which


# ── a block = the keys fitted on one input level + all the predictors over them ──────────────────────────────────
class Block:
    def __init__(self, W, y, tag, base, ybase, rng, ks, kmeans_iters, kmeans_sample, pca_bits, log):
        self.base, self.ybase = base, ybase
        t0 = time.time()
        self.km = {}
        for k in ks:
            C, info = kmeans_fit(W, k, base, rng, iters=kmeans_iters, sample=kmeans_sample)
            self.km[k] = (C, np.ones(P))
            log(f"      k-means k={k}: {info['iters']} iters, {info['used']} clusters used, last pass moved {info['changed_last']}  [{time.time() - t0:.0f}s]")
        self.w_mi = mi_weights(W, y, base, ybase)
        log("      MI(position; target) bits: " + " ".join(f"{v:.3f}" for v in self.w_mi))
        for k in ks:
            C, info = kmeans_fit(W, k, base, rng, iters=kmeans_iters, sample=kmeans_sample, w=self.w_mi)
            self.km[("mi", k)] = (C, self.w_mi)
            log(f"      k-means (MI-weighted) k={k}: {info['iters']} iters, {info['used']} used  [{time.time() - t0:.0f}s]")
        self.pca = {b: pca_fit(W, base, b) for b in pca_bits}
        for b in pca_bits:
            ev = self.pca[b]["evals"]
            log(f"      PCA {b} directions: variance {ev[0]:.3f} {ev[1]:.3f} {ev[2]:.3f} ... {ev[-1]:.3f} (of {P - 1:.0f} total one-hot variance at most)  [{time.time() - t0:.0f}s]")
        # cached key functions
        ex = lambda rr: (f"r{rr}", (lambda Wq, rr=rr: exact_key(Wq, rr, base)))
        km = lambda key: (f"km{key}", (lambda Wq, key=key: kmeans_assign(Wq, *self.km[key])))
        pc = lambda b, nb: (f"pca{b}_{nb}", (lambda Wq, b=b, nb=nb: pca_code(Wq, self.pca[b]) >> (b - nb)))
        glob = ("global", lambda Wq: np.zeros(len(Wq), np.int64))
        lv = lambda pairs, n_min=1: [(n, f, n_min) for n, f in pairs]
        specs = {
            "exact": lv([ex(3), ex(2), ex(1), ex(0)]),
        }
        for k in ks:
            specs[f"km{k}"] = lv([km(k), glob])
            specs[f"exact3>km{k}"] = lv([ex(3), km(k), glob])
            specs[f"exact3(n3)>km{k}"] = [(*ex(3), 3)] + lv([km(k), glob])
            specs[f"km-mi{k}"] = lv([km(("mi", k)), glob])
            specs[f"exact3>km-mi{k}"] = lv([ex(3), km(("mi", k)), glob])
        if len(ks) > 1:
            specs["km-hier"] = lv([ex(3)] + [km(k) for k in sorted(ks, reverse=True)] + [glob])
            specs["km-mi-hier"] = lv([ex(3)] + [km(("mi", k)) for k in sorted(ks, reverse=True)] + [glob])
        for b in pca_bits:
            specs[f"pca{b}"] = lv([pc(b, nb) for nb in range(b, 0, -2)] + [glob])
            specs[f"exact3>pca{b}"] = lv([ex(3)] + [pc(b, nb) for nb in range(b, 0, -2)] + [glob])
        self.cache = {}
        self.pred = {name: Backoff(spec, ybase, self.cache).fit(W, y, tag) for name, spec in specs.items()}
        self.pred["nb"] = NaiveBayes(base, ybase).fit(W, y)
        self.pred["exact3>nb"] = Fallback(Backoff(lv([ex(3)]), ybase, self.cache).fit(W, y, tag), self.pred["nb"])
        self.pred["exact32>nb"] = Fallback(Backoff(lv([ex(3), ex(2)]), ybase, self.cache).fit(W, y, tag), self.pred["nb"])
        log(f"      {len(self.pred)} predictors fitted  [{time.time() - t0:.0f}s]")

    def names(self):
        return list(self.pred)


def masked_pairs(before, after, MASK):
    """Training rows of a block: the windows at the masked positions of the joined `before`, the target from `after`."""
    W, pos = windows(before)
    m = before[pos] == MASK
    return W[m], after[pos][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_chars", type=int, default=300000)
    ap.add_argument("--test_chars", type=int, default=60000)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--ks", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--pca_bits", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--kmeans_iters", type=int, default=8)
    ap.add_argument("--kmeans_sample", type=int, default=30000)
    ap.add_argument("--out", default=str(HERE / "expF_cluster_features_text.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    log = lambda s: print(s, flush=True)

    alphabet, train, test = load_corpus()
    V = len(alphabet)
    MASK, BORDER, base, ybase = V, V + 1, V + 2, V + 1                 # keys over symbols+MASK+BORDER; targets over symbols+MASK
    text = np.concatenate(training_slice(train, args.train_chars)).astype(np.int16)
    test = test[:args.test_chars].astype(np.int16)
    K = 4
    schedule = [k / K for k in range(K + 1)]
    chunks = [text[i:i + args.chunk] for i in range(0, len(text) - args.chunk, args.chunk)]
    tchunks = [test[i:i + args.chunk] for i in range(0, len(test) - args.chunk, args.chunk)]
    log(f"Experiment F: {len(text)} training characters ({len(chunks)} chunks of {args.chunk}), {len(test)} held-out, alphabet {V}, "
        f"window +-{R}, mask schedule {schedule}")

    # ── E35's own SeqRule, as-is (the reproduction row) ───────────────────────────────────────────────────────
    chains = [mask_chain(c, schedule, rng, MASK) for c in chunks]
    e35 = {}
    for name, (src, dst) in (("block1", (1, 0)), ("block2", (2, 1)), ("oneshot2", (2, 0))):
        rule = SeqRule(V)
        for ch in chains:
            rule.observe(ch[src], ch[dst])
        e35[name] = rule
    log(f"   E35 SeqRule fitted (full windows: block1 {len(e35['block1'].tables[3])}, block2 {len(e35['block2'].tables[3])})  [{time.time() - t_start:.0f}s]")

    # ── the held-out masks (one draw per chunk, rate 0.5), as E35 ─────────────────────────────────────────────
    t_eval = 0.5
    masks = [rng.random(len(c)) < t_eval for c in tchunks]
    noisy_chunks = []
    for c, m in zip(tchunks, masks):
        nz = c.copy(); nz[m] = MASK
        noisy_chunks.append(nz)
    n_masked = int(sum(m.sum() for m in masks))

    # E35 rows, per chunk
    acc = dict(one=0, chain=0)
    for c, m, nz in zip(tchunks, masks, noisy_chunks):
        one = e35["oneshot2"].predict(nz)
        cur = e35["block1"].predict(e35["block2"].predict(nz))
        acc["one"] += int((one[m] == c[m]).sum()); acc["chain"] += int((cur[m] == c[m]).sum())
    results = {"E35-SeqRule": dict(one_shot=acc["one"] / n_masked, chain=acc["chain"] / n_masked)}
    log(f"   E35 SeqRule at mask {t_eval}: one shot {results['E35-SeqRule']['one_shot']:.3f}, chain {results['E35-SeqRule']['chain']:.3f} "
        f"over {n_masked} masked characters (E35 logged 0.320 / 0.309)")

    # ── the joined training levels and the blocks ─────────────────────────────────────────────────────────────
    L = {k: join_chunks([ch[k] for ch in chains], BORDER) for k in (0, 1, 2)}
    W2, y2_one = masked_pairs(L[2], L[0], MASK)                      # one shot: 0.5 -> 0
    _W2, y2_chain = masked_pairs(L[2], L[1], MASK)                   # block 2: 0.5 -> 0.25 (target may be MASK)
    W1, y1 = masked_pairs(L[1], L[0], MASK)                          # block 1: 0.25 -> 0
    log(f"   masked training rows: level 2 {len(W2)}, level 1 {len(W1)}; block-2 targets that are MASK: {(y2_chain == MASK).mean():.3f}")
    log("   level-2 keys + predictors (one shot and block 2 share the input level):")
    B2 = Block(W2, y2_one, "W2", base, ybase, rng, args.ks, args.kmeans_iters, args.kmeans_sample, args.pca_bits, log)
    B2c = {name: Backoff(B2.pred[name].levels, ybase, B2.cache).fit(W2, y2_chain, "W2") for name in B2.names() if "nb" not in name}
    B2c["nb"] = NaiveBayes(base, ybase).fit(W2, y2_chain)
    B2c["exact3>nb"] = Fallback(B2c["exact"].__class__([B2c["exact"].levels[0]], ybase, B2.cache).fit(W2, y2_chain, "W2"), B2c["nb"])
    B2c["exact32>nb"] = Fallback(B2c["exact"].__class__(B2c["exact"].levels[:2], ybase, B2.cache).fit(W2, y2_chain, "W2"), B2c["nb"])
    log("   level-1 keys + predictors:")
    B1 = Block(W1, y1, "W1", base, ybase, rng, args.ks, args.kmeans_iters, args.kmeans_sample, args.pca_bits, log)

    # ── evaluation ────────────────────────────────────────────────────────────────────────────────────────────
    clean = join_chunks(tchunks, BORDER)
    noisy = join_chunks(noisy_chunks, BORDER)
    Wt, post = windows(noisy)
    mt = noisy[post] == MASK
    Wt, post = Wt[mt], post[mt]
    truth = clean[post]
    assert len(post) == n_masked
    log(f"   evaluating {len(B2.names())} predictors on {n_masked} masked held-out characters ...")
    for name in B2.names():
        t0 = time.time()
        one, which = B2.pred[name].predict(Wt, "Wt")
        row = dict(one_shot=float((one == truth).mean()))
        if name != "nb":
            lv = B2.pred[name].levels
            row["levels"] = [dict(level=lv[li][0], share=float((which == li).mean()), acc=float((one[which == li] == truth[which == li]).mean()) if (which == li).any() else None)
                             for li in range(len(lv))]
        # the chain: block 2 (0.5 -> 0.25) then block 1 (0.25 -> 0) on what is still MASK
        mid = noisy.copy()
        p2, _w = B2c[name].predict(Wt, "Wt")
        mid[post] = p2
        Wm, pm = windows(mid)
        still = mid[pm] == MASK
        Wm, pm = Wm[still], pm[still]
        p1, _w = B1.pred[name].predict(Wm, "Wm:" + name)
        final = mid.copy()
        final[pm] = p1
        row["chain"] = float((final[post] == truth).mean())
        row["chain_block2_filled"] = float((p2 != MASK).mean())
        row["chain_block2_filled_acc"] = float((p2[p2 != MASK] == truth[p2 != MASK]).mean()) if (p2 != MASK).any() else None
        row["seconds"] = round(time.time() - t0, 1)
        results[name] = row
        lvs = "" if name == "nb" else "  levels: " + " | ".join(f"{d['level']} {d['share']:.2f}@{d['acc']:.3f}" for d in row["levels"] if d["share"] > 0)
        log(f"   {name:<20} one shot {row['one_shot']:.3f}  chain {row['chain']:.3f} (block 2 filled {row['chain_block2_filled']:.2f} at {(row['chain_block2_filled_acc'] or float('nan')):.3f}){lvs}")

    out = dict(train_chars=int(len(text)), test_chars=int(len(test)), n_masked=n_masked, t_eval=t_eval, chunk=args.chunk, ks=args.ks,
               pca_bits=args.pca_bits, kmeans_iters=args.kmeans_iters, kmeans_sample=args.kmeans_sample, seed=args.seed,
               mi_bits_level2=[float(v) for v in B2.w_mi], mi_bits_level1=[float(v) for v in B1.w_mi],
               pca_evals_level2={str(b): [float(v) for v in B2.pca[b]["evals"]] for b in args.pca_bits},
               e35_logged=dict(one_shot=0.320, chain=0.309), results=results, seconds=round(time.time() - t_start, 1))
    json.dump(out, open(args.out, "w"), indent=1)
    best = max((n for n in results if n != "E35-SeqRule"), key=lambda n: results[n]["one_shot"])
    log(f"\nExperiment F: best one shot {best} {results[best]['one_shot']:.3f} vs exact {results['exact']['one_shot']:.3f} "
        f"(E35 SeqRule {results['E35-SeqRule']['one_shot']:.3f}); chain {results[best]['chain']:.3f} vs {results['exact']['chain']:.3f}  [{time.time() - t_start:.0f}s]")


if __name__ == "__main__":
    main()
