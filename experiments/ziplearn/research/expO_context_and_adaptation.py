"""expO (2026-09-27) -- two questions the user asked, turned into measurements. No neural network anywhere.

Q1 "static vs contextual is the biggest hurdle". expL located the cost: 13.4% of positions are word-initial and
carry 32% of the code length (4.52 bits each). Two competing explanations, and they imply different work:
  (i) the character model simply runs out of WINDOW at a boundary -- then more characters fix it, no semantics;
  (ii) the information needed at a boundary is not in the recent characters at all -- then S23's landscape is the
       only route, and expK's failure is about DATA, not principle.
Measured: the chain's bits overall and at boundaries at R = 8, 12, 16, 24 (does more context close the gap? where
does it saturate?), and the counted word-metric's gap to the chain at four training sizes (is it data-starved?).

Q2 "what would it take for the legibility tax to be NEGATIVE". One axis is measurable with no network: ADAPTATION.
A counted model updates per symbol at zero cost; a deployed transformer is frozen. Measured: frozen vs online bits
per character at each training size, which is the size of that advantage and how it scales.

Prints JSON to runs/research/expO.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from e34 import Chain, Stream                                          # noqa: E402
from textlm import load_corpus, training_slice                         # noqa: E402

sys.path.insert(0, str(HERE))
from expK_counted_metric import first_chars, words_of                  # noqa: E402

from collections import Counter                                        # noqa: E402


def boundary_positions(tst, alpha):
    at, prev_end, i, n = [], None, 0, len(tst)
    while i < n:
        if tst[i] in alpha:
            if prev_end is not None:
                at.append(i)
            j = i
            while j < n and tst[j] in alpha:
                j += 1
            prev_end = j; i = j
        else:
            i += 1
    return np.array(at)


def code(chain, stream, seq, learn):
    p = np.empty(len(seq))
    for i, c in enumerate(seq):
        keys = chain.predict(stream)
        p[i] = chain.prob(keys, c)
        if learn:
            chain.update(keys, c)
        stream.push(c)
    return -np.log2(np.maximum(p, 1e-12))


def train_chain(R, seqs, V, alphabet):
    chain, stream = Chain(V, R=R), Stream(V, alphabet)
    for s in seqs:
        for c in s.tolist():
            chain.update(chain.predict(stream), c)
            stream.push(c)
    return chain, stream


def main():
    t0 = time.time()
    alphabet, train, test = load_corpus()
    V = len(alphabet)
    alpha = {i for i, ch in enumerate(alphabet) if ch.isalpha()}
    tst = test.tolist()
    at = boundary_positions(tst, alpha)
    out = {"V": V, "n_test": len(tst), "n_boundaries": int(len(at)),
           "boundary_share_of_positions": float(len(at) / len(tst))}

    # ---------- Q1(i): does more character context close the boundary gap? ----------
    depth = {}
    for R in (8, 12, 16, 24):
        seqs = training_slice(train, 900_000)
        chain, stream = train_chain(R, seqs, V, alphabet)
        b = code(chain, stream, tst, learn=True)
        depth[R] = {"bpc": float(b.mean()), "boundary_bits": float(b[at].mean()),
                    "interior_bits": float(np.delete(b, at).mean()),
                    "boundary_share_of_bits": float(b[at].sum() / b.sum())}
        print(f"  R={R}: bpc {depth[R]['bpc']:.4f} boundary {depth[R]['boundary_bits']:.3f}", flush=True)
    out["context_depth"] = depth

    # ---------- Q1(ii) + Q2: data size curve, frozen vs online, and the metric's gap ----------
    sizes = [100_000, 300_000, 900_000, None]
    curve = []
    for n in sizes:
        seqs = training_slice(train, n)
        n_chars = sum(len(s) for s in seqs)
        chain, stream = train_chain(8, seqs, V, alphabet)
        import copy
        froz_chain, froz_stream = copy.deepcopy(chain), copy.deepcopy(stream)
        b_on = code(chain, stream, tst, learn=True)
        b_fr = code(froz_chain, froz_stream, tst, learn=False)

        # the counted word metric at this size: bigram vs kNN-smoothed, both at boundaries
        wc = Counter()
        for s in seqs:
            wc.update(words_of(s.tolist(), alpha))
        vocab = [w for w, _ in wc.most_common(12000)]
        wid = {w: i for i, w in enumerate(vocab)}
        UNK = len(vocab)
        B = np.zeros((len(vocab) + 1, V))
        for s in seqs:
            for pw, c in first_chars(s.tolist(), alpha):
                B[wid.get(pw, UNK), c] += 1.0
        marg = (B.sum(0) + 0.5); marg /= marg.sum()
        pairs = first_chars(tst, alpha)
        idx = np.array([wid.get(pw, UNK) for pw, _ in pairs])
        cs = np.array([c for _, c in pairs])
        Np = B.sum(1)
        pb = (B + 0.5) / (Np[:, None] + V * 0.5)
        seen = Np[idx] > 0
        pv = np.where(seen, pb[idx, cs], marg[cs])
        curve.append({
            "train_chars": n_chars, "word_types": len(wc),
            "tokens_per_type": float(sum(wc.values()) / max(1, len(wc))),
            "bpc_online": float(b_on.mean()), "bpc_frozen": float(b_fr.mean()),
            "adaptation_gain_bits": float(b_fr.mean() - b_on.mean()),
            "boundary_bits_chain": float(b_on[at].mean()),
            "boundary_bits_word_bigram": float(np.mean(-np.log2(np.maximum(pv, 1e-12)))),
            "bigram_minus_chain": float(np.mean(-np.log2(np.maximum(pv, 1e-12))) - b_on[at].mean()),
            "boundary_coverage": float(seen.mean()),
        })
        print(f"  n={n_chars}: on {curve[-1]['bpc_online']:.4f} fr {curve[-1]['bpc_frozen']:.4f} "
              f"adapt {curve[-1]['adaptation_gain_bits']:.4f} bigram-chain {curve[-1]['bigram_minus_chain']:+.3f}",
              flush=True)
    out["data_curve"] = curve
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expO.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
