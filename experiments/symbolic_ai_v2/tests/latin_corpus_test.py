"""latin_corpus_test.py — Bigram vs multi-level perplexity on 12 Latin books.

Loads up to N_BOOKS from the GT4HistOCR EarlyModernLatin corpus (welded line
files identical to the format used by char_latin.py), trains a MorphismGraph,
and compares character-level perplexity for two prediction modes:

  bigram      — predict(prev_atom, next_etype)        [standard edge counts]
  multilevel  — predict(composition | prev_atom, ...) [richer trigram+ context]

SKIP automatically if the corpus is absent; run as a standalone script or via
pytest -v -s.

Run:  python experiments/symbolic_ai_v2/tests/latin_corpus_test.py
 or:  python -m pytest experiments/symbolic_ai_v2/tests/latin_corpus_test.py -v -s
"""

import sys
import math
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.symbolic_ai_v2.core.topology import sequence_1d
from experiments.symbolic_ai_v2.core.morphism import MorphismGraph
from experiments.symbolic_ai_v2.core.predict  import (
    perplexity,
    perplexity_multilevel,
)

# ── Config ─────────────────────────────────────────────────────────────────────

N_BOOKS    = 12       # cap at 12 for faster runs (vs. full 89-book corpus)
TRAIN_FRAC = 0.9      # fraction of books used for training

CORPUS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "GT4HistOCR" / "corpus" / "EarlyModernLatin"
)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_latin_books(n: int = N_BOOKS) -> list[str]:
    """Return up to n book sequences as single strings.

    Each book directory contains multiple .gt.txt line files that are welded
    into one long character sequence — identical to char_latin.py's
    load_sequences() logic for fair comparison with PCH v1 results.
    """
    if not CORPUS_PATH.exists():
        return []
    book_dirs = sorted(d for d in CORPUS_PATH.iterdir() if d.is_dir())[:n]
    books: list[str] = []
    for bdir in book_dirs:
        lines: list[str] = []
        for gt_file in sorted(bdir.glob("*.gt.txt")):
            text = gt_file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                lines.append(text)
        if lines:
            books.append("".join(lines))
    return books

# ── Core benchmark ─────────────────────────────────────────────────────────────

_RESULT_CACHE: dict | None = None   # computed once, shared across all tests


def _benchmark() -> dict:
    """Train MorphismGraph on N_BOOKS×TRAIN_FRAC books; test on the rest."""
    global _RESULT_CACHE
    if _RESULT_CACHE is not None:
        return _RESULT_CACHE

    books = load_latin_books(N_BOOKS)
    if not books:
        _RESULT_CACHE = {"skipped": True}
        return _RESULT_CACHE

    split      = max(1, int(len(books) * TRAIN_FRAC))
    train_seqs = books[:split]
    test_seqs  = books[split:] if split < len(books) else [books[-1]]

    topo = sequence_1d()
    mg   = MorphismGraph()

    t0 = time.time()
    for seq in train_seqs:
        mg.observe_sequence(seq, topo)
    train_secs = time.time() - t0

    baseline     = math.log2(max(mg.n_atoms(), 2))
    ppl_bi       = perplexity(mg, test_seqs, topo)
    ppl_multi    = perplexity_multilevel(mg, test_seqs, topo)
    total_secs   = time.time() - t0

    _RESULT_CACHE = {
        "skipped":         False,
        "n_books":         len(books),
        "n_train":         len(train_seqs),
        "n_test":          len(test_seqs),
        "train_chars":     sum(len(b) for b in train_seqs),
        "test_chars":      sum(len(b) for b in test_seqs),
        "n_atoms":         mg.n_atoms(),
        "n_compositions":  mg.n_compositions(),
        "n_edges":         mg.n_edges(),
        "baseline_bits":   baseline,
        "bigram_ppl":      ppl_bi,
        "multilevel_ppl":  ppl_multi,
        "bigram_gain":     baseline - ppl_bi,
        "multilevel_gain": baseline - ppl_multi,
        "ml_improvement":  ppl_bi - ppl_multi,
        "train_secs":      train_secs,
        "total_secs":      total_secs,
    }
    return _RESULT_CACHE

# ── Tests ──────────────────────────────────────────────────────────────────────

def test_corpus_available():
    """Corpus must be present (SKIP gracefully if not)."""
    r = _benchmark()
    if r.get("skipped"):
        print(f"  SKIPPED — corpus not found at:\n  {CORPUS_PATH}")
        return
    print(f"  Found {r['n_books']} books  |  "
          f"train={r['n_train']} ({r['train_chars']:,} chars)  |  "
          f"test={r['n_test']} ({r['test_chars']:,} chars)")
    assert r["n_books"] > 0


def test_bigram_beats_baseline():
    """Bigram MorphismGraph must beat the uniform-over-alphabet baseline."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print(f"\n  Baseline (log2 alphabet = {r['n_atoms']} chars): "
          f"{r['baseline_bits']:.3f} bits/char")
    print(f"  Bigram perplexity:       {r['bigram_ppl']:.3f} bits/char  "
          f"(gain {r['bigram_gain']:+.3f})")
    assert r["bigram_ppl"] < r["baseline_bits"], (
        f"Bigram {r['bigram_ppl']:.3f} must be below baseline {r['baseline_bits']:.3f}"
    )


def test_multilevel_beats_bigram():
    """Multi-level context must give strictly lower perplexity than bigram."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print(f"  Multilevel perplexity:   {r['multilevel_ppl']:.3f} bits/char  "
          f"(gain {r['multilevel_gain']:+.3f})")
    print(f"  Improvement vs bigram:   {r['ml_improvement']:+.3f} bits/char  "
          f"({r['n_compositions']} compositions as context,  "
          f"{r['n_edges']} total edges)")
    print(f"  Training time: {r['train_secs']:.1f}s  |  "
          f"Total (incl. eval): {r['total_secs']:.1f}s")
    assert r["multilevel_ppl"] < r["bigram_ppl"], (
        f"Multilevel {r['multilevel_ppl']:.3f} must be below bigram {r['bigram_ppl']:.3f}\n"
        f"  ({r['n_compositions']} compositions available)"
    )


def test_summary():
    """Print a concise comparison table; always passes (informational)."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print()
    print(f"  {'Model':<20}  {'bits/char':>10}  {'vs baseline':>12}  {'vs bigram':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (uniform)':<20}  {r['baseline_bits']:>10.3f}  {'—':>12}  {'—':>10}")
    print(f"  {'Bigram':<20}  {r['bigram_ppl']:>10.3f}  "
          f"{r['bigram_gain']:>+11.3f}  {'—':>10}")
    print(f"  {'Multilevel':<20}  {r['multilevel_ppl']:>10.3f}  "
          f"{r['multilevel_gain']:>+11.3f}  "
          f"{r['ml_improvement']:>+9.3f}")
    print()
    print(f"  Target to beat small transformer (~6-layer 256-dim): < 2.0 bits/char")


# ── Test runner ────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_corpus_available,
        test_bigram_beats_baseline,
        test_multilevel_beats_bigram,
        test_summary,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"Running {t.__name__}...")
            t()
            print(f"  PASSED\n")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}\n")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print()
    print(f"{passed}/{len(tests)} tests passed.")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
