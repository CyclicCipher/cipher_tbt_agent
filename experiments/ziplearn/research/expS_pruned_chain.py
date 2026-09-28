"""expS (E39) -- §9 rule 3 on the text store: does PRICE choose which contexts to keep, and what does it cost?

The user's point (2026-09-27): the memory problem should be solved by the continual-learning machinery we already
have, not by a faster hash table. §9 rule 3 says capacity forces consolidation and what to drop is chosen by
(evidence × bits saved); E6 proved it on rules (100% retention, 3 blocks, 36 bits against 452). The text chain
never consolidates anything -- it keeps every context it has ever seen, for ever, which is why E38 hit 6.4 GB at
20M characters and stopped.

A context at order r is exactly a §9 block, and its parent at order r−1 is the thing to merge it into. So:
    keep a context iff the bits its own distribution saves over its parent's, on its own observations,
    exceed the bits it costs to store.
The keep-set is decided on the TRAINING prefix only and applied while coding held-out text, so nothing is chosen
with knowledge of the test.

Pass (pre-registered): ≥ 5× fewer retained contexts at no more than +0.02 bits/char against the unpruned chain.
Refute: any setting that saves memory costs more than +0.05 bits/char.

NOT the same question as E33's, which found the sleep MASK a no-op on text: that asked which context POSITIONS to
keep (one offset mask shared by all contexts); this asks which context INSTANCES to keep.
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
from ziplib.chain import (_codes, context_worth, keep_top, online_bits,   # noqa: E402
                          online_bits_pruned)

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
    return len(alpha) + 1, np.concatenate(parts), test


def main():
    t0 = time.time()
    n_train = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000
    V, train, test = load(n_train)
    both = np.concatenate([train, test])
    nt = len(train)
    out = {"V": V, "train_chars": int(nt), "test_chars": int(len(test)), "R": R, "heldout": HELDOUT, "rows": []}
    print(json.dumps({k: out[k] for k in ("V", "train_chars", "test_chars")}, indent=1), flush=True)

    full = online_bits(both, V, R)[nt:]
    n_full = sum(len(np.unique(_codes(train, V, r))) for r in range(1, R + 1))
    out["unpruned"] = {"bpc": float(full.mean()), "contexts": int(n_full)}
    print(f"unpruned: bpc {full.mean():.4f} over {n_full:,} contexts", flush=True)

    worth = context_worth(train, V, R)
    n_all = sum(len(u) for u, _ in worth.values())
    out["contexts_available"] = int(n_all)
    for frac in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        budget = max(1, int(n_all * frac))
        keeps = keep_top(worth, budget)
        kept = sum(len(k) for k in keeps.values())
        b = online_bits_pruned(both, V, R, keeps)[nt:]
        rec = {"budget_fraction": frac, "contexts": int(kept), "reduction": n_full / max(1, kept),
               "bpc": float(b.mean()), "delta_bpc": float(b.mean() - full.mean())}
        out["rows"].append(rec)
        print(f"  keep {frac:<5} = {kept:>9,} contexts ({rec['reduction']:5.1f}x fewer)  bpc {rec['bpc']:.4f}  "
              f"delta {rec['delta_bpc']:+.4f}", flush=True)

    ok = [r for r in out["rows"] if r["reduction"] >= 5 and r["delta_bpc"] <= 0.02]
    bad = [r for r in out["rows"] if r["delta_bpc"] > 0.05 and r["reduction"] > 1.05]
    out["verdict"] = {"pass": bool(ok), "best": (min(ok, key=lambda r: -r["reduction"]) if ok else None),
                      "any_setting_costs_over_0.05": bool(bad)}
    out["seconds"] = time.time() - t0
    d = HERE.parent / "runs" / "research"; d.mkdir(parents=True, exist_ok=True)
    (d / "expS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nverdict:", json.dumps(out["verdict"]), f"{out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
