"""expP (2026-09-27) -- Q2: what would it take for the legibility tax to be NEGATIVE?

expO measured the one axis on which a counted model beats a frozen one on ITS OWN distribution: adaptation. The gain
halves with every ~3x of data (0.339 -> 0.189 -> 0.104 -> 0.050 bits/char at 100k -> 2.7M), so on a homogeneous
corpus it is a small-data effect that is going away.

The hypothesis this tests: the adaptation advantage is not about data SIZE, it is about STATIONARITY. A deployed
gradient-trained model is frozen; a counted model rewrites itself per symbol at zero cost. So the tax should go
sharply negative exactly where the test distribution differs from training -- which is the ordinary condition of a
deployed model, and never the condition of a benchmark.

Measured: a chain trained on the full Latin corpus, then made to code four streams, frozen against online --
(a) the held-out Latin book (same register), (b) Middle High German (Nibelungenlied), (c) Old High German
(Hildebrandslied + Muspilli), (d) the Latin book REVERSED (a synthetic shift with the same alphabet and unigram
statistics). The alphabet is built jointly over all three corpora so no arm is charged for unseen symbols.

Prints JSON to runs/research/expP.json. No neural network is built, run or trained.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from e34 import Chain, Stream                                          # noqa: E402

CORPORA = HERE.parent.parent.parent / "corpora"
HELDOUT = "COMMENTARIORUM LIBRI III DE BELLO CIVILI.txt"
MIN_COUNT = 200


def joint_corpus():
    """(alphabet, latin training texts, {stream name: int array}) with ONE alphabet over all three corpora."""
    texts = {}
    for d in ("latin books", "mittelhochdeutsch", "althochdeutsch"):
        for f in sorted((CORPORA / d).glob("*.txt")):
            texts[(d, f.name)] = f.read_text(encoding="utf-8", errors="replace")
    counts = Counter()
    for t in texts.values():
        counts.update(t)
    alphabet = sorted(k for k, v in counts.items() if v >= MIN_COUNT)
    index = {ch: i for i, ch in enumerate(alphabet)}
    other = len(alphabet)

    def enc(t):
        return np.fromiter((index.get(ch, other) for ch in t), dtype=np.int16, count=len(t))

    train = [enc(t) for (d, n), t in texts.items() if d == "latin books" and n != HELDOUT]
    streams = {
        "latin_heldout": enc(texts[("latin books", HELDOUT)]),
        "mittelhochdeutsch": np.concatenate([enc(t) for (d, _), t in texts.items() if d == "mittelhochdeutsch"]),
        "althochdeutsch": np.concatenate([enc(t) for (d, _), t in texts.items() if d == "althochdeutsch"]),
    }
    streams["latin_reversed"] = streams["latin_heldout"][::-1].copy()
    return alphabet + ["<other>"], train, streams


def code(chain, stream, seq, learn):
    p = np.empty(len(seq))
    for i, c in enumerate(seq):
        keys = chain.predict(stream)
        p[i] = chain.prob(keys, c)
        if learn:
            chain.update(keys, c)
        stream.push(c)
    return -np.log2(np.maximum(p, 1e-12))


def main():
    t0 = time.time()
    alphabet, train, streams = joint_corpus()
    V = len(alphabet)
    out = {"V": V, "train_chars": int(sum(len(s) for s in train)), "streams": {}}

    base_chain, base_stream = Chain(V, R=8), Stream(V, alphabet)
    for s in train:
        for c in s.tolist():
            base_chain.update(base_chain.predict(base_stream), c)
            base_stream.push(c)
    print(f"trained on {out['train_chars']} Latin characters, V={V}", flush=True)

    for name, seq in streams.items():
        lst = seq.tolist()
        on = code(copy.deepcopy(base_chain), copy.deepcopy(base_stream), lst, learn=True)
        fr = code(copy.deepcopy(base_chain), copy.deepcopy(base_stream), lst, learn=False)
        # the adaptation profile: the gain over the first / last tenth of the stream
        tenth = max(1, len(lst) // 10)
        rec = {
            "n_chars": len(lst),
            "bpc_frozen": float(fr.mean()), "bpc_online": float(on.mean()),
            "adaptation_gain_bits": float(fr.mean() - on.mean()),
            "relative_gain": float((fr.mean() - on.mean()) / fr.mean()),
            "gain_first_tenth": float(fr[:tenth].mean() - on[:tenth].mean()),
            "gain_last_tenth": float(fr[-tenth:].mean() - on[-tenth:].mean()),
        }
        out["streams"][name] = rec
        print(f"  {name:20s} n={len(lst):7d} frozen {rec['bpc_frozen']:.3f} online {rec['bpc_online']:.3f} "
              f"GAIN {rec['adaptation_gain_bits']:+.3f} ({100*rec['relative_gain']:.1f}%) "
              f"first10% {rec['gain_first_tenth']:+.3f} last10% {rec['gain_last_tenth']:+.3f}", flush=True)

    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expP.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
