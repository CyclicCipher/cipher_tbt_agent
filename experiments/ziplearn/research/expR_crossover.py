"""expR (E38, DESIGN §24.7) -- the crossover, on a corpus big enough to reach it. No network anywhere.

E36 measured the counted word bigram's deficit against the character chain AT WORD BOUNDARIES shrinking with data
(+0.644 / +0.574 / +0.425 / +0.259 at 100k / 289k / 769k / 2.73M characters) and the refit put the crossover at
17-26M. `corpora/latin_classical` is 35.0M characters, so the prediction is testable rather than extrapolated.

Pass: the deficit reaches zero at or below 26M characters. Refute: still positive at the largest size that fits.

APPARATUS LIMIT, watched and reported rather than hidden: `e34.Chain` keeps nine Python dicts keyed by character
tuples, and the order-8 table holds close to one context per training character. The run measures its own resident
memory after each size and stops before it would ask this machine for more than MEM_CAP_GB; a size that does not
fit is recorded as an apparatus limit, never as a result.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from e34 import Chain, Stream                                          # noqa: E402
from expK_counted_metric import first_chars, words_of                  # noqa: E402

CORPUS = HERE.parent.parent.parent / "corpora" / "latin_classical"
HELDOUT = "caesar.txt"
MIN_COUNT = 200
MEM_CAP_GB = float(os.environ.get("MEM_CAP_GB", "6.0"))


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        try:                                                           # windows fallback, no psutil
            import ctypes
            import ctypes.wintypes as wt

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                                     ctypes.byref(c), c.cb)
            return c.WorkingSetSize / 1e9
        except Exception:
            return float("nan")


def load():
    """One alphabet over the whole classical corpus; the held-out author kept apart."""
    texts = {}
    for f in sorted(CORPUS.glob("*.txt")):
        texts[f.name] = unicodedata.normalize("NFC", f.read_text(encoding="utf-8", errors="replace"))
    counts = Counter()
    for t in texts.values():
        counts.update(t)
    alphabet = sorted(k for k, v in counts.items() if v >= MIN_COUNT)
    idx = {c: i for i, c in enumerate(alphabet)}
    other = len(alphabet)

    def enc(t):
        return np.fromiter((idx.get(c, other) for c in t), dtype=np.int16, count=len(t))

    test = enc(texts.pop(HELDOUT))
    train = [enc(t) for n, t in sorted(texts.items())]
    return alphabet + ["<other>"], train, test


def slice_to(train, n):
    """n characters taken proportionally from every training author, so no size is one author's style."""
    total = sum(len(t) for t in train)
    out = []
    for t in train:
        k = max(1, int(len(t) * n / total))
        out.append(t[:k])
    return out


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


def main():
    t0 = time.time()
    alphabet, train, test = load()
    V = len(alphabet)
    alpha = {i for i, ch in enumerate(alphabet) if ch.isalpha()}
    tst = test.tolist()
    at = boundary_positions(tst, alpha)
    total_train = sum(len(t) for t in train)
    out = {"corpus": "latin_classical", "heldout": HELDOUT, "V": V,
           "train_chars_available": total_train, "test_chars": len(tst),
           "n_boundaries": int(len(at)), "mem_cap_gb": MEM_CAP_GB, "sizes": []}
    print(json.dumps({k: out[k] for k in list(out)[:7]}, indent=1), flush=True)

    if "--sizes" in sys.argv:
        sizes = [int(float(x)) for x in sys.argv[sys.argv.index("--sizes") + 1].split(",")]
    else:
        sizes = [int(x) for x in (2.73e6, 5e6, 1e7, 2e7, 3.4e7)]
    sizes = [s for s in sizes if s <= total_train]
    out["out_name"] = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "expR"
    for n in sizes:
        seqs = slice_to(train, n)
        n_chars = sum(len(s) for s in seqs)
        chain, stream = Chain(V, R=8), Stream(V, alphabet)
        for s in seqs:
            for c in s.tolist():
                chain.update(chain.predict(stream), c)
                stream.push(c)
        mem = rss_gb()
        b = np.empty(len(tst))
        for i, c in enumerate(tst):
            keys = chain.predict(stream)
            b[i] = chain.prob(keys, c)
            chain.update(keys, c)
            stream.push(c)
        b = -np.log2(np.maximum(b, 1e-12))

        wc = Counter()
        for s in seqs:
            wc.update(words_of(s.tolist(), alpha))
        vocab = [w for w, _ in wc.most_common(400000)]
        wid = {w: i for i, w in enumerate(vocab)}
        UNK = len(vocab)
        B = np.zeros((len(vocab) + 1, V))
        for s in seqs:
            for pw, c in first_chars(s.tolist(), alpha):
                B[wid.get(pw, UNK), c] += 1.0
        marg = B.sum(0) + 0.5; marg /= marg.sum()
        pairs = first_chars(tst, alpha)
        ii = np.array([wid.get(pw, UNK) for pw, _ in pairs])
        cs = np.array([c for _, c in pairs])
        Np = B.sum(1)
        pb = (B + 0.5) / (Np[:, None] + V * 0.5)
        seen = Np[ii] > 0
        pv = np.where(seen, pb[ii, cs], marg[cs])
        bigram = float(np.mean(-np.log2(np.maximum(pv, 1e-12))))
        chain_at = float(b[at].mean())

        rec = {"train_chars": n_chars, "rss_gb_after_training": round(mem, 2),
               "bpc_online": float(b.mean()), "boundary_bits_chain": chain_at,
               "boundary_bits_word_bigram": bigram, "bigram_minus_chain": bigram - chain_at,
               "word_types": len(wc), "tokens_per_type": float(sum(wc.values()) / max(1, len(wc))),
               "boundary_coverage": float(seen.mean()),
               "boundary_share_of_bits": float(b[at].sum() / b.sum())}
        out["sizes"].append(rec)
        print(f"  n={n_chars:>9,} rss {mem:4.1f}GB  bpc {rec['bpc_online']:.4f}  "
              f"chain@bnd {chain_at:.3f}  bigram {bigram:.3f}  DEFICIT {rec['bigram_minus_chain']:+.3f}  "
              f"types {len(wc):>7,} tok/type {rec['tokens_per_type']:.1f}", flush=True)
        del chain, stream, B, seqs
        gc.collect()
        if mem > MEM_CAP_GB:
            out["apparatus_limit"] = (f"stopped after {n_chars} characters: resident memory {mem:.1f} GB exceeded "
                                      f"MEM_CAP_GB={MEM_CAP_GB}; larger sizes are an apparatus limit, not a result")
            print("  " + out["apparatus_limit"], flush=True)
            break

    d = out["sizes"]
    if len(d) >= 2:
        x = np.log10([r["train_chars"] for r in d]); y = np.array([r["bigram_minus_chain"] for r in d])
        s, c = np.polyfit(x, y, 1)
        out["fit"] = {"slope_per_decade": float(s), "crossover_chars": float(10 ** (-c / s)) if s else None,
                      "crossed": bool(y[-1] <= 0)}
    out["seconds"] = time.time() - t0
    p = HERE.parent / "runs" / "research"; p.mkdir(parents=True, exist_ok=True)
    (p / f"{out['out_name']}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n" + json.dumps({k: out[k] for k in ("fit", "apparatus_limit", "seconds") if k in out}, indent=1))


if __name__ == "__main__":
    main()
