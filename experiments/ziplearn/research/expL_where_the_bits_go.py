"""expL (D1', 2026-09-27) -- WHERE OUR BITS GO on Latin, with no neural network anywhere.

DESIGN.md S24.4 hypothesis: our blended backoff shares statistical strength only down the SUFFIX chain (drop
characters from the left), so on a heavily inflected language it should pay twice for the same stem -- `amabat`,
`amabant`, `amabamus` are four unrelated contexts to it. If that is where the surplus lives, the missing primitive
is a factored (stem, suffix) context, which is writable. If the surplus is uniform, the gap is a smoothing constant
and S24.4 is dead.

This is an ATTRIBUTION of the model's own code length, not a predictor: the grouping uses the whole word, which the
predictor never sees. Online (adaptive) coding, exactly as E33/E34's 1.820 number.

Prints JSON to runs/research/expL.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from e34 import Chain, Stream                                          # noqa: E402
from textlm import load_corpus, training_slice                         # noqa: E402

STEM = 4                                                               # characters of a word treated as its stem proxy


def word_spans(seq, alpha):
    """[(start, end)] of maximal runs of alphabetic characters, and a per-position (word_id, offset)."""
    spans, i, n = [], 0, len(seq)
    while i < n:
        if seq[i] in alpha:
            j = i
            while j < n and seq[j] in alpha:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    pos = np.full(n, -1, dtype=np.int64)
    off = np.zeros(n, dtype=np.int32)
    for w, (a, b) in enumerate(spans):
        pos[a:b] = w
        off[a:b] = np.arange(b - a)
    return spans, pos, off


def main():
    t0 = time.time()
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 800_000
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    alpha_ids = {i for i, ch in enumerate(alphabet) if ch.isalpha()}
    seqs = training_slice(train, n_train)

    # ---- pass 1: the model, trained online on the training slice then continued online on the held-out book ----
    chain, stream = Chain(V, R=8), Stream(V, alphabet)
    for s in seqs:
        for c in s.tolist():
            chain.update(chain.predict(stream), c)
            stream.push(c)
    n_seen = sum(len(s) for s in seqs)

    # ---- the training vocabulary and its stems ----
    tr_words, tr_stems = Counter(), Counter()
    for s in seqs:
        spans, _, _ = word_spans(s.tolist(), alpha_ids)
        lst = s.tolist()
        for a, b in spans:
            w = tuple(lst[a:b])
            tr_words[w] += 1
            if len(w) >= STEM:
                tr_stems[w[:STEM]] += 1
    # a stem is "shared" when training shows it with at least two different whole words
    stem_forms = defaultdict(set)
    for w in tr_words:
        if len(w) >= STEM:
            stem_forms[w[:STEM]].add(w)

    # ---- pass 2: code the held-out book, attributing every character's bits ----
    tst = test.tolist()
    spans, wpos, woff = word_spans(tst, alpha_ids)
    wkey = [tuple(tst[a:b]) for a, b in spans]

    bits = np.zeros(len(tst))
    for i, c in enumerate(tst):
        keys = chain.predict(stream)
        p = chain.prob(keys, c)
        bits[i] = -np.log2(max(p, 1e-12))
        chain.update(keys, c)
        stream.push(c)

    total = float(bits.sum())
    out = {
        "n_train_chars": n_seen, "n_test_chars": len(tst), "V": V,
        "bits_per_char_online": total / len(tst),
        "train_word_types": len(tr_words), "train_stems": len(stem_forms),
    }

    # (a) concentration -- D1's pass criterion is about whether the code length is concentrated at all
    srt = np.sort(bits)[::-1]
    out["concentration"] = {f"top_{int(q*100)}pct_share": float(srt[:max(1, int(q * len(srt)))].sum() / total)
                            for q in (0.01, 0.05, 0.10, 0.25, 0.50)}

    # (b) by position within a word (the S24.4 shape: stem region vs suffix region)
    by_off = defaultdict(list)
    for i in range(len(tst)):
        if wpos[i] >= 0:
            by_off[min(int(woff[i]), 8)].append(bits[i])
        else:
            by_off[-1].append(bits[i])                                 # non-letters
    out["by_word_offset"] = {str(k): {"n": len(v), "mean_bits": float(np.mean(v))}
                             for k, v in sorted(by_off.items())}

    # (c) the morphology test: characters in the SUFFIX region (offset >= STEM) of a word whose WHOLE form was
    #     never seen in training, split by whether its STEM was seen with other forms.
    groups = {"unseen_word_shared_stem": [], "unseen_word_new_stem": [], "seen_word": [], "short_word": []}
    for i in range(len(tst)):
        w = wpos[i]
        if w < 0:
            continue
        key = wkey[w]
        if len(key) < STEM:
            groups["short_word"].append(bits[i]); continue
        if tr_words.get(key, 0) > 0:
            groups["seen_word"].append(bits[i]); continue
        forms = stem_forms.get(key[:STEM], ())
        groups["unseen_word_shared_stem" if len(forms) >= 2 else "unseen_word_new_stem"].append(bits[i])
    out["by_word_novelty"] = {k: {"n": len(v), "mean_bits": float(np.mean(v)) if v else None,
                                  "share_of_total": float(np.sum(v) / total) if v else 0.0}
                              for k, v in groups.items()}

    # (d) the same split restricted to the SUFFIX characters, which is where a factored context would pay off
    suf = {"unseen_word_shared_stem": [], "unseen_word_new_stem": [], "seen_word": []}
    for i in range(len(tst)):
        w = wpos[i]
        if w < 0 or woff[i] < STEM:
            continue
        key = wkey[w]
        if len(key) < STEM:
            continue
        if tr_words.get(key, 0) > 0:
            suf["seen_word"].append(bits[i])
        else:
            forms = stem_forms.get(key[:STEM], ())
            suf["unseen_word_shared_stem" if len(forms) >= 2 else "unseen_word_new_stem"].append(bits[i])
    out["suffix_chars_only"] = {k: {"n": len(v), "mean_bits": float(np.mean(v)) if v else None}
                                for k, v in suf.items()}

    # (e) by training frequency of the word (head vs tail -- where a counted model should beat an averaging one)
    bands = {"unseen": [], "1-2": [], "3-10": [], "11-100": [], "100+": [], "non_letter": []}
    for i in range(len(tst)):
        w = wpos[i]
        if w < 0:
            bands["non_letter"].append(bits[i]); continue
        f = tr_words.get(wkey[w], 0)
        k = "unseen" if f == 0 else "1-2" if f <= 2 else "3-10" if f <= 10 else "11-100" if f <= 100 else "100+"
        bands[k].append(bits[i])
    out["by_word_frequency"] = {k: {"n": len(v), "mean_bits": float(np.mean(v)) if v else None,
                                    "share_of_total": float(np.sum(v) / total) if v else 0.0}
                                for k, v in bands.items()}

    # (f) THE WORD BOUNDARY. (b) says offset 0 is where the bits are: character context ends at the space, so the
    #     first character of a word is the one place a character model has nothing to lean on. How much of that is
    #     recoverable by a WORD-level context -- i.e. by the semantic landscape of S23 -- rather than by more
    #     characters? Compare our per-character cost at offset 0 against the empirical conditional entropy of the
    #     first character given the previous word, and given the word itself, both counted on training.
    tr_bigram = Counter()                                              # (prev word, first char of next word)
    tr_prev = Counter()
    tr_first = Counter()
    for s in seqs:
        lst = s.tolist()
        sp, _, _ = word_spans(lst, alpha_ids)
        for a, b in zip(sp, sp[1:]):
            pw, nx = tuple(lst[a[0]:a[1]]), lst[b[0]]
            tr_bigram[(pw, nx)] += 1
            tr_prev[pw] += 1
            tr_first[nx] += 1
    n_first = sum(tr_first.values())

    init_seen, init_unseen, cond_bits, marg_bits, ours = [], [], [], [], []
    for w, (a, b) in enumerate(spans):
        if w == 0:
            continue
        pw = wkey[w - 1]
        c0 = tst[a]
        ours.append(bits[a])
        (init_seen if tr_bigram.get((pw, c0), 0) > 0 else init_unseen).append(bits[a])
        # counted predictor of the first character: KT-smoothed p(c | previous word), backing off to p(c)
        Np = tr_prev.get(pw, 0)
        pm = (tr_first.get(c0, 0) + 0.5) / (n_first + V / 2)
        pc = ((tr_bigram.get((pw, c0), 0) + 0.5) / (Np + V / 2)) if Np else pm
        cond_bits.append(-np.log2(max(pc, 1e-12)))
        marg_bits.append(-np.log2(max(pm, 1e-12)))
    out["word_initial"] = {
        "n": len(ours),
        "share_of_all_positions": len(ours) / len(tst),
        "share_of_total_bits": float(np.sum(ours) / total),
        "ours_mean_bits": float(np.mean(ours)),
        "prev_word_bigram_seen": {"n": len(init_seen), "mean_bits": float(np.mean(init_seen)) if init_seen else None},
        "prev_word_bigram_unseen": {"n": len(init_unseen), "mean_bits": float(np.mean(init_unseen)) if init_unseen else None},
        "counted_given_prev_word_bits": float(np.mean(cond_bits)),
        "counted_marginal_bits": float(np.mean(marg_bits)),
        "headroom_vs_ours": float(np.mean(ours) - np.mean(cond_bits)),
        "headroom_bits_per_char_of_corpus": float((np.sum(ours) - np.sum(cond_bits)) / len(tst)),
    }

    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expL.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
