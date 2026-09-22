"""Check that every function in ziplib.price equals the code it MOVED from, on random inputs, to 1e-12.

    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe experiments/ziplearn/ziplib/_check_price.py      (CPU, seconds)

Originals: ziplearner.flag_bits / pay / precision_bits; arcgames.entropy / LocalRule._price (V = arcgames.V);
textlm.ContextLM._bits / _chain_of / _keys / _prequential. Run from anywhere; the old files are imported, not copied.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # experiments/ziplearn: the originals
import ziplearner as ZL                                               # noqa: E402
import arcgames as AG                                                 # noqa: E402
import textlm as TL                                                   # noqa: E402
from ziplib import price as P                                         # noqa: E402

TOL = 1e-12


class _Struct:
    def __init__(self, V, price="rate"):
        self.V, self.price, self.cost, self.n_right, self.n_wrong = V, price, 0.0, 0, 0


def check(name, a, b, tol=TOL):
    err = abs(a - b) if isinstance(a, float) or isinstance(a, int) else float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    ok = err <= tol and (not isinstance(a, float) or not math.isnan(a))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: |new - old| = {err:.3e}")
    return ok


def main():
    rng = np.random.default_rng(0)
    ok = True
    # flag_bits
    worst = 0.0
    for _ in range(2000):
        nr, nw = int(rng.integers(0, 500)), int(rng.integers(0, 500))
        for exc in (True, False):
            worst = max(worst, abs(P.flag_bits(exc, nr, nw) - ZL.flag_bits(exc, nr, nw)))
    ok &= check("flag_bits (2000 x 2 random counts)", worst, 0.0)
    # pay -- the rate branch, a random sequence of right/wrong on two structures with the same V
    worst = 0.0
    for V in (2, 10, 16, 37):
        s_new, s_old = _Struct(V), _Struct(V, "rate")
        for _ in range(500):
            right = bool(rng.integers(0, 2))
            P.pay(s_new, right)
            ZL.pay(s_old, right)
            worst = max(worst, abs(s_new.cost - s_old.cost), abs(s_new.n_right - s_old.n_right), abs(s_new.n_wrong - s_old.n_wrong))
    ok &= check("pay (4 V x 500 steps, rate)", worst, 0.0)
    # precision_bits
    worst = 0.0
    for _ in range(2000):
        n = int(rng.integers(0, 10000))
        span, sigma = float(rng.uniform(0.01, 1000.0)), float(rng.uniform(1e-4, 10.0))
        worst = max(worst, abs(P.precision_bits(n, span, sigma) - ZL.precision_bits(n, span, sigma)))
    ok &= check("precision_bits (2000 random (n, span, sigma))", worst, 0.0)
    # entropy
    worst = 0.0
    for p in np.concatenate([[0.0, 1.0], rng.uniform(0.0, 1.0, 5000)]):
        worst = max(worst, abs(P.entropy(float(p)) - AG.entropy(float(p))))
    ok &= check("entropy (5002 p)", worst, 0.0)
    # table_price vs LocalRule._price (which reads the module constant V = 16)
    worst = 0.0
    for _ in range(200):
        merged = {}
        for k in range(int(rng.integers(1, 60))):
            counts = {int(c): int(rng.integers(1, 50)) for c in rng.choice(AG.V, size=int(rng.integers(1, 5)), replace=False)}
            merged[bytes([k])] = counts
        worst = max(worst, abs(P.table_price(merged, AG.V) - AG.LocalRule._price(merged)))
    ok &= check("table_price (200 random merged tables, V = 16)", worst, 0.0)
    # the text prices: random texts over V symbols, R = 8 contexts
    for V, R, n in ((11, 8, 3000), (42, 8, 5000), (5, 4, 800)):
        lm = TL.ContextLM(V, R)
        seqs = [rng.integers(0, V, size=int(rng.integers(50, n // 2 + 51))).astype(np.int16) for _ in range(3)]
        ctx, nxt = lm._contexts(seqs)
        masks = [list(range(1, R + 1)), [1, 2, 4], [1], [], [2, min(3, R), R]]
        w_keys = w_kt = w_chain = w_preq = 0.0
        for mask in masks:
            k_new, k_old = P.context_keys(ctx, mask, V + 1), lm._keys(ctx, mask)
            w_keys = max(w_keys, float(np.max(np.abs(k_new - k_old))) if len(k_new) else 0.0)
            w_kt = max(w_kt, abs(P.kt_bits(k_new, nxt, V) - lm._bits(k_old, nxt)))
            w_chain = max(w_chain, 0.0 if P.backoff_chain(mask) == TL.ContextLM._chain_of(mask) else 1.0)
            for limit in (100, 1000, 10 ** 9):
                w_preq = max(w_preq, abs(P.prequential_bits(ctx, nxt, mask, V, limit) - lm._prequential(ctx, nxt, mask, limit)))
        ok &= check(f"context_keys (V={V}, R={R}, {len(masks)} masks)", w_keys, 0.0)
        ok &= check(f"kt_bits (V={V}, R={R}, {len(masks)} masks)", w_kt, 0.0)
        ok &= check(f"backoff_chain (V={V}, R={R}, {len(masks)} masks)", w_chain, 0.0)
        ok &= check(f"prequential_bits (V={V}, R={R}, {len(masks)} masks x 3 limits)", w_preq, 0.0)
    print("PASS: ziplib.price equals the moved code to 1e-12" if ok else "FAIL: see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
