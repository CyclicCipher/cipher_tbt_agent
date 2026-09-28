"""expT -- why does bits/char sit near 1.8 and barely move with data? (the user's question, 2026-09-27)

E38: 1.8613 -> 1.8085 bits/char for 7x more data, about 0.06 bits per decade. Two candidate explanations, and
they imply opposite next moves:
  (i)  SPARSITY-BOUND: the deep contexts exist but have too little evidence, so the model rarely gets to use
       them. Then more data is the cure and the decline should accelerate as coverage fills in.
  (ii) CONTEXT-BOUND: the model already matches its deepest order almost everywhere, and eight characters of
       history simply do not contain the missing information. Then more data is nearly worthless and the cure is
       a different kind of context.
expO already showed R = 8 -> 24 changes nothing (1.8901 -> 1.8903), which points at (ii), but that is indirect.
This measures it: at each training size, the distribution of the DEEPEST order that matched, and the bits paid at
each depth, split by word-initial against word-interior positions (E36's split, where 13.4% of positions carry
32% of the code length).

No network anywhere. Writes runs/research/expT.json.
"""
from __future__ import annotations

import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ziplib.chain import _blend, online_counts                        # noqa: E402

CORPUS = HERE.parent.parent.parent / "corpora" / "latin_classical"
HELDOUT = "caesar.txt"
R = 8


def load(n_train=None):
    texts = {f.name: unicodedata.normalize("NFC", f.read_text(encoding="utf-8", errors="replace"))
             for f in sorted(CORPUS.glob("*.txt"))}
    cnt = Counter()
    for t in texts.values():
        cnt.update(t)
    alpha = sorted(k for k, v in cnt.items() if v >= 200)
    idx = {c: i for i, c in enumerate(alpha)}
    other = len(alpha)
    enc = lambda t: np.fromiter((idx.get(c, other) for c in t), dtype=np.int64, count=len(t))  # noqa: E731
    test = enc(texts.pop(HELDOUT))
    parts = [enc(t) for _, t in sorted(texts.items())]
    total = sum(len(p) for p in parts)
    if n_train:
        parts = [p[:max(1, int(len(p) * n_train / total))] for p in parts]
    return alpha + ["<other>"], np.concatenate(parts), test


def word_initial_mask(seq, alpha_ids):
    is_a = np.isin(seq, list(alpha_ids))
    prev = np.empty_like(is_a); prev[0] = False; prev[1:] = is_a[:-1]
    return is_a & ~prev


def main():
    t0 = time.time()
    out = {"R": R, "heldout": HELDOUT, "sizes": []}
    for n_train in (2_730_000, 10_000_000, 34_000_000):
        alphabet, train, test = load(n_train)
        V = len(alphabet)
        alpha_ids = {i for i, ch in enumerate(alphabet) if ch.isalpha()}
        both = np.concatenate([train, test])
        nt = len(train)
        per_order = [online_counts(both, V, r) for r in range(R, -1, -1)]
        bits = -np.log2(np.maximum(_blend(per_order, V), 1e-12))[nt:]

        # deepest order whose context had been seen before this position
        deepest = np.zeros(len(both), dtype=np.int8)
        for i, r in enumerate(range(R, -1, -1)):
            hit = per_order[i][0] > 0
            deepest = np.where((deepest == 0) & hit, np.int8(r), deepest)
        deepest = deepest[nt:]
        wi = word_initial_mask(test, alpha_ids)

        rec = {"train_chars": int(nt), "bpc": float(bits.mean()),
               "bpc_word_initial": float(bits[wi].mean()), "bpc_interior": float(bits[~wi].mean()),
               "word_initial_share_of_positions": float(wi.mean()),
               "word_initial_share_of_bits": float(bits[wi].sum() / bits.sum()),
               "by_depth": {}, "by_depth_word_initial": {}}
        for d in range(R, -1, -1):
            m = deepest == d
            if m.sum():
                rec["by_depth"][str(d)] = {"share_of_positions": float(m.mean()),
                                           "mean_bits": float(bits[m].mean()),
                                           "share_of_bits": float(bits[m].sum() / bits.sum())}
            mm = m & wi
            if mm.sum():
                rec["by_depth_word_initial"][str(d)] = {"share": float(mm.sum() / max(1, wi.sum())),
                                                        "mean_bits": float(bits[mm].mean())}
        rec["frac_reaching_R"] = float((deepest == R).mean())
        rec["mean_depth"] = float(deepest.mean())
        out["sizes"].append(rec)
        print(f"n={nt:>10,}  bpc {rec['bpc']:.4f}  initial {rec['bpc_word_initial']:.3f}  "
              f"interior {rec['bpc_interior']:.3f}  mean depth {rec['mean_depth']:.2f}  "
              f"at depth {R}: {100*rec['frac_reaching_R']:.1f}%", flush=True)
        for d in range(R, -1, -1):
            v = rec["by_depth"].get(str(d))
            if v:
                print(f"    depth {d}: {100*v['share_of_positions']:5.1f}% of positions, "
                      f"{v['mean_bits']:5.2f} bits, {100*v['share_of_bits']:5.1f}% of the code", flush=True)
        del per_order, both
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expT.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
