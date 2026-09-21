"""E33 (a) — the ZipLearner language model on the Latin corpus, and its classical references (DESIGN §19).

The discrete machinery of E24/E28 on text, character level. A context is the previous R characters (R = 8; BORDER before
the start of a text). The model is a table from the MASKED context (the positions the sleep pass keeps) to the counts
of the next character; its price is the length of the training text under the adaptive Krichevsky–Trofimov code, one
memoryless source per context -- the two-part code of E24 with the majority-or-exception rule replaced by the code a
language model actually pays: for a context with counts n_1..n_V (N in all),
    bits(context) = [lgamma(N + V/2) - lgamma(V/2) - sum_i (lgamma(n_i + 1/2) - lgamma(1/2))] / ln 2.
The sleep pass drops context positions while the training text gets shorter under that code (greedy, E24's rule with the
price alone). Prediction on unseen text is the KT probability from the masked context's counts, and for a context never
seen, the same mask minus its oldest kept position, and so on down to the unigram -- the structured form of E28's
"nearest stored key" default (recency-ordered backoff). Conceptually this is E28's block with the corpus as its memory
tokens; computationally it is sorted integer keys (a masked context and its next character fit in 63 bits).

References: the same table with the mask fixed to the last R positions for R in {2, 4, 6, 8} (a plain n-gram with KT
and backoff -- the control for the sleep pass), and a general compressor's conditional bits per character
(xz: size(train + test) - size(train), over the test length).

    python experiments/ziplearn/textlm.py --sizes 1000 10000 100000 1000000 all      (CPU) -> runs/e33/e33.json
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent.parent / "corpora" / "latin books"
HELDOUT = "COMMENTARIORUM LIBRI III DE BELLO CIVILI.txt"
MIN_COUNT = 200                                   # characters rarer than this in the corpus map to one OTHER symbol


def load_corpus(heldout=HELDOUT):
    """(alphabet, training texts as int16 arrays (one per book), the held-out book as an int16 array)."""
    books = {b.name: b.read_text(encoding="utf-8", errors="replace") for b in sorted(CORPUS.glob("*.txt"))}
    counts = Counter()
    for t in books.values():
        counts.update(t)
    alphabet = sorted(k for k, v in counts.items() if v >= MIN_COUNT)
    index = {ch: i for i, ch in enumerate(alphabet)}
    other = len(alphabet)                                              # the OTHER symbol
    def encode(t):
        return np.fromiter((index.get(ch, other) for ch in t), dtype=np.int16, count=len(t))
    train = [encode(t) for n, t in books.items() if n != heldout]
    test = encode(books[heldout])
    return alphabet + ["<other>"], train, test


def training_slice(train, n):
    """n characters spread over the training books: n / (number of books) contiguous characters from the start of each."""
    if n is None:
        return train
    per = max(1, n // len(train))
    out = [t[:per] for t in train]
    return out


class ContextLM:
    """A masked-context character model over sorted integer keys (module docstring)."""

    def __init__(self, V, R=8):
        self.V, self.R = V, R
        self.base = V + 1                                              # symbols 0..V-1, BORDER = V
        self.mask = list(range(1, R + 1))                              # kept positions, in characters back (1 = the previous)
        self.tables = {}                                               # tuple(mask) -> (ctx keys sorted, ctx totals, pair keys sorted, pair counts)
        self.n_train = 0

    # -- keys --------------------------------------------------------------------------------------------------
    def _contexts(self, seqs):
        """For every position of every text: the R previous characters (BORDER-padded), and the next character."""
        ctx, nxt = [], []
        for s in seqs:
            pad = np.concatenate([np.full(self.R, self.V, dtype=np.int64), s.astype(np.int64)])
            n = len(s)
            cols = [pad[self.R - k: self.R - k + n] for k in range(1, self.R + 1)]   # k back
            ctx.append(np.stack(cols, 1))                                          # (n, R): column k-1 = k characters back
            nxt.append(s.astype(np.int64))
        return np.concatenate(ctx), np.concatenate(nxt)

    def _keys(self, ctx, mask):
        key = np.zeros(len(ctx), dtype=np.int64)
        for k in mask:
            key = key * self.base + ctx[:, k - 1]
        return key

    # -- the price: the training text's length under the adaptive KT code, per context ---------------------------
    def _bits(self, ctx_keys, nxt):
        pair = ctx_keys * self.base + nxt
        _u, pc = np.unique(pair, return_counts=True)
        _c, tc = np.unique(ctx_keys, return_counts=True)
        V = self.V
        pc_t, tc_t = torch.tensor(pc, dtype=torch.float64), torch.tensor(tc, dtype=torch.float64)
        bits = (torch.lgamma(tc_t + V / 2).sum() - len(tc) * math.lgamma(V / 2)
                - (torch.lgamma(pc_t + 0.5).sum() - len(pc) * math.lgamma(0.5))) / math.log(2)
        return float(bits)

    @staticmethod
    def _chain_of(mask):
        chain, m = [], list(mask)
        while True:
            chain.append(tuple(m))
            if not m:
                break
            m = [x for x in m if x != max(m)]                          # drop the oldest kept position
        return chain

    def _prequential(self, ctx, nxt, mask, limit):
        """The code length of the (first `limit` characters of the) training text under the blended backoff predictor
        with this mask, counts growing as the text is read -- the price a mask is judged by (E24's rule with the code
        the predictor actually pays, not the per-context KT sum, which ignores backoff and punishes sparse orders)."""
        n = min(limit, len(nxt))
        chain = self._chain_of(mask)
        V, base = self.V, self.base
        keys = [self._keys(ctx[:n], list(m)).tolist() for m in chain]
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
        return bits * len(nxt) / n                                     # scaled to the whole text

    def fit(self, seqs, sleep=True, log=None, limit=200000):
        """Count, then (if sleep) drop kept positions while the training text's prequential code gets shorter (greedy,
        judged on its first `limit` characters); then build the backoff tables over the whole text."""
        ctx, nxt = self._contexts(seqs)
        self.n_train = len(nxt)
        mask = list(self.mask)
        cur = self._prequential(ctx, nxt, mask, limit) if sleep else float("nan")
        before = cur
        if sleep:
            while len(mask) > 0:
                best, best_bits = None, cur
                for k in mask:
                    cand = [m for m in mask if m != k]
                    b = self._prequential(ctx, nxt, cand, limit)
                    if b < best_bits:
                        best, best_bits = k, b
                if best is None:
                    break
                mask.remove(best)
                cur = best_bits
                if log is not None:
                    log.append((list(mask), cur))
        self.mask = mask
        self.bits_before, self.bits_after = before, cur
        chain = self._chain_of(mask)                                   # the mask, then without its oldest position, ..., unigram
        self.chain = chain
        self.tables = {}
        for m in chain:
            k = self._keys(ctx, list(m))
            pair = k * self.base + nxt
            ck, ct = np.unique(k, return_counts=True)
            pk, pc = np.unique(pair, return_counts=True)
            self.tables[m] = (ck, ct, pk, pc)
        return self

    # -- prediction ----------------------------------------------------------------------------------------------
    def _level_counts(self, ctx, nxt):
        """Per backoff level: (count of (context, next), total of the context) for every position; 0, 0 where unseen."""
        out = []
        for m in self.chain:
            ck, ct, pk, pc = self.tables[m]
            k = self._keys(ctx, list(m))
            pos = np.minimum(np.searchsorted(ck, k), len(ck) - 1)
            total = np.where(ck[pos] == k, ct[pos], 0)
            pair = k * self.base + nxt
            pp = np.minimum(np.searchsorted(pk, pair), len(pk) - 1)
            cnt = np.where(pk[pp] == pair, pc[pp], 0)
            out.append((cnt.astype(np.float64), total.astype(np.float64)))
        return out

    def bits_per_char(self, seq):
        """Frozen: mean -log2 P(next | context) over a text. P blends the backoff levels, deepest first: a level with
        counts n_c of N gives n_c / (N + V/2) and passes the KT mass reserved for unseen symbols, (V/2) / (N + V/2), to the
        next level; an unseen context (N = 0) passes everything; the empty context (unigram) is always seen."""
        ctx, nxt = self._contexts([seq])
        V = self.V
        p = np.zeros(len(nxt))
        mass = np.ones(len(nxt))
        for cnt, total in self._level_counts(ctx, nxt):
            p += mass * cnt / (total + V / 2)
            mass = mass * (V / 2) / (total + V / 2)
        p += mass / V                                                  # what escapes the unigram: uniform
        return float(-np.log2(p).mean())

    def bits_per_char_online(self, seq):
        """Prequential: the same predictor, but the counts grow with the text as it is predicted -- the model keeps
        learning through the test, which is what data efficiency means for a sequential learner (and what xz does)."""
        ctx, nxt = self._contexts([seq])
        V = self.V
        levels = []
        for m in self.chain:
            ck, ct, pk, pc = self.tables[m]
            levels.append((list(m), dict(zip(ck.tolist(), ct.tolist())), dict(zip(pk.tolist(), pc.tolist()))))
        keys_by_level = [self._keys(ctx, m).tolist() for m, _t, _p in levels]
        nxt_l = nxt.tolist()
        base = self.base
        total_bits = 0.0
        for i in range(len(nxt_l)):
            c = nxt_l[i]
            p, mass = 0.0, 1.0
            for li, (_m, totals, pairs) in enumerate(levels):
                k = keys_by_level[li][i]
                N = totals.get(k, 0)
                n = pairs.get(k * base + c, 0)
                p += mass * n / (N + V / 2)
                mass *= (V / 2) / (N + V / 2)
            p += mass / V
            total_bits -= math.log2(p)
            for li, (_m, totals, pairs) in enumerate(levels):
                k = keys_by_level[li][i]
                totals[k] = totals.get(k, 0) + 1
                pk = k * base + c
                pairs[pk] = pairs.get(pk, 0) + 1
        return total_bits / len(nxt_l)


def xz_bits_per_char(train_seqs, test_seq, alphabet):
    """A general compressor's conditional cost of the test text given the training text: 8 (|xz(train+test)| - |xz(train)|) / |test|."""
    def raw(seqs):
        return b"".join(bytes(np.asarray(s, dtype=np.uint8)) for s in seqs)
    tr = raw(train_seqs)
    a = len(lzma.compress(tr, preset=9 | lzma.PRESET_EXTREME))
    b = len(lzma.compress(tr + raw([test_seq]), preset=9 | lzma.PRESET_EXTREME))
    return 8.0 * (b - a) / len(test_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["1000", "10000", "100000", "1000000", "all"])
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "runs" / "e33"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    n_train_all = sum(len(t) for t in train)
    print(f"E33 (a): Latin corpus, {len(train)} training books ({n_train_all} characters), held out {HELDOUT} ({len(test)} characters), "
          f"alphabet {V} (rare characters as <other>)", flush=True)
    report = {"alphabet": V, "heldout": HELDOUT, "n_test": int(len(test)), "n_train_all": int(n_train_all), "sizes": {}}
    for s in args.sizes:
        n = None if s == "all" else int(s)
        seqs = training_slice(train, n)
        n_chars = sum(len(t) for t in seqs)
        row = {"n_train": int(n_chars)}
        t0 = time.time()
        log = []
        zip_lm = ContextLM(V, args.R).fit(seqs, sleep=True, log=log)
        row["ziplm"] = dict(mask=zip_lm.mask, bits_before=zip_lm.bits_before, bits_after=zip_lm.bits_after,
                            bpc=zip_lm.bits_per_char(test), bpc_online=zip_lm.bits_per_char_online(test), seconds=time.time() - t0)
        for R in (2, 4, 6, 8):
            m = ContextLM(V, R).fit(seqs, sleep=False)
            row[f"ngram{R}"] = dict(bpc=m.bits_per_char(test), bpc_online=m.bits_per_char_online(test))
        row["xz"] = dict(bpc_online=xz_bits_per_char(seqs, test, alphabet))
        report["sizes"][s] = row
        print(f"   {n_chars:>9} chars | ZipLM mask {zip_lm.mask} (training text {zip_lm.bits_before / 8 / 1024:.0f} -> {zip_lm.bits_after / 8 / 1024:.0f} KiB) | "
              f"frozen bpc: ZipLM {row['ziplm']['bpc']:.3f}, n-gram 2/4/6/8 " + "/".join(f"{row[f'ngram{R}']['bpc']:.3f}" for R in (2, 4, 6, 8))
              + f" | online bpc: ZipLM {row['ziplm']['bpc_online']:.3f}, n-gram 2/4/6/8 " + "/".join(f"{row[f'ngram{R}']['bpc_online']:.3f}" for R in (2, 4, 6, 8))
              + f", xz {row['xz']['bpc_online']:.3f} | {time.time() - t0:.0f}s", flush=True)
    json.dump(report, open(out / "e33.json", "w"), indent=1)


if __name__ == "__main__":
    main()
