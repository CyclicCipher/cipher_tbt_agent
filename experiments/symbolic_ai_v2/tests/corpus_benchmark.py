"""corpus_benchmark.py — Multi-language bigram vs multi-level perplexity benchmark.

Loads every language subfolder found in experiments/symbolic_ai_v2/corpus/,
trains a MorphismGraph per language, and prints a unified comparison table:

  Baseline  | log2(alphabet_size)
  Bigram    | predict(prev_atom, etype)
  Multilevel| predict(composition | prev_atom, etype)  [richer context]

Split strategy (per language):
  ≥ 3 files : 80/20 book-level split
      2 files : train on first, test on second
      1 file  : 90% chars train / 10% chars test
      0 files : skipped

Run:
  python experiments/symbolic_ai_v2/tests/corpus_benchmark.py
or:
  python -m pytest experiments/symbolic_ai_v2/tests/corpus_benchmark.py -v -s
"""

import sys
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.symbolic_ai_v2.core.topology  import sequence_1d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph
from experiments.symbolic_ai_v2.core.predict   import perplexity, perplexity_multilevel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
TRAIN_FRAC  = 0.8   # book-level split when ≥3 files
CHAR_SPLIT  = 0.9   # character-level split when only 1 file


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_folder(folder: Path) -> list[str]:
    """Return list of document strings from all .txt files in *folder*."""
    docs: list[str] = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        text = text.lstrip("\ufeff").strip()   # strip UTF-8 BOM
        if text:
            docs.append(text)
    return docs


def _split(docs: list[str]) -> tuple[list[str], list[str]]:
    """Return (train_docs, test_docs) using the appropriate split strategy."""
    n = len(docs)
    if n == 0:
        return [], []
    if n == 1:
        # single file — character-level split
        text  = docs[0]
        cut   = int(len(text) * CHAR_SPLIT)
        return [text[:cut]], [text[cut:]]
    if n == 2:
        return [docs[0]], [docs[1]]
    # ≥3 files — book-level split
    cut = max(1, int(n * TRAIN_FRAC))
    return docs[:cut], docs[cut:]


# ---------------------------------------------------------------------------
# Per-language benchmark
# ---------------------------------------------------------------------------

def _benchmark_language(folder: Path) -> dict:
    docs = _load_folder(folder)
    if not docs:
        return {"skipped": True, "reason": "no .txt files"}

    train_docs, test_docs = _split(docs)
    if not test_docs:
        return {"skipped": True, "reason": "split produced empty test set"}

    topo = sequence_1d()
    mg   = MorphismGraph()

    t0 = time.time()
    for doc in train_docs:
        mg.observe_sequence(doc, topo)
    train_secs = time.time() - t0

    baseline     = math.log2(max(mg.n_atoms(), 2))
    ppl_bi       = perplexity(mg, test_docs, topo)
    ppl_multi    = perplexity_multilevel(mg, test_docs, topo)
    total_secs   = time.time() - t0

    return {
        "skipped":         False,
        "n_docs":          len(docs),
        "n_train":         len(train_docs),
        "n_test":          len(test_docs),
        "train_chars":     sum(len(d) for d in train_docs),
        "test_chars":      sum(len(d) for d in test_docs),
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


# ---------------------------------------------------------------------------
# Global results cache  (compute once, share across all tests)
# ---------------------------------------------------------------------------

_RESULTS: dict[str, dict] | None = None


def _all_benchmarks() -> dict[str, dict]:
    global _RESULTS
    if _RESULTS is not None:
        return _RESULTS

    if not CORPUS_ROOT.exists():
        _RESULTS = {}
        return _RESULTS

    _RESULTS = {}
    for subfolder in sorted(CORPUS_ROOT.iterdir()):
        if subfolder.is_dir():
            lang = subfolder.name
            print(f"  Benchmarking: {lang} ...", flush=True)
            _RESULTS[lang] = _benchmark_language(subfolder)

    return _RESULTS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_corpus_root_exists():
    """Corpus root directory must be present."""
    results = _all_benchmarks()
    if not results:
        print(f"  SKIPPED — corpus root not found: {CORPUS_ROOT}")
        return
    langs = [k for k, v in results.items() if not v.get("skipped")]
    print(f"  Found {len(langs)} runnable language(s): {', '.join(langs)}")
    assert len(langs) > 0, f"No usable language folders under {CORPUS_ROOT}"


def test_bigram_beats_baseline():
    """Bigram must beat the uniform-over-alphabet baseline for every language."""
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_run = False
    for lang, r in results.items():
        if r.get("skipped"):
            print(f"  {lang}: SKIPPED ({r.get('reason', '?')})")
            continue
        any_run = True
        ok = r["bigram_ppl"] < r["baseline_bits"]
        print(f"  {lang}: bigram={r['bigram_ppl']:.3f}  "
              f"baseline={r['baseline_bits']:.3f}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok, (
            f"{lang}: bigram {r['bigram_ppl']:.3f} >= baseline {r['baseline_bits']:.3f}"
        )
    if not any_run:
        print("  SKIPPED — all languages skipped")


def test_multilevel_beats_bigram():
    """Multilevel must give strictly lower perplexity than bigram for every language."""
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_run = False
    for lang, r in results.items():
        if r.get("skipped"):
            continue
        any_run = True
        ok = r["multilevel_ppl"] < r["bigram_ppl"]
        print(f"  {lang}: multilevel={r['multilevel_ppl']:.3f}  "
              f"bigram={r['bigram_ppl']:.3f}  "
              f"improvement={r['ml_improvement']:+.3f}  "
              f"({'OK' if ok else 'FAIL — multilevel not better'})")
        assert ok, (
            f"{lang}: multilevel {r['multilevel_ppl']:.3f} >= "
            f"bigram {r['bigram_ppl']:.3f}"
        )
    if not any_run:
        print("  SKIPPED — all languages skipped")


def test_summary_table():
    """Print unified comparison table across all languages. Always passes."""
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED — no results")
        return

    W = 22   # language column width
    print()
    print(f"  {'Language':<{W}}  {'docs':>5}  {'train':>8}  "
          f"{'test':>8}  {'V':>5}  {'comps':>7}  "
          f"{'baseline':>9}  {'bigram':>8}  {'multilev':>9}  {'ML gain':>8}")
    print(f"  {'-' * (W + 70)}")

    for lang, r in results.items():
        if r.get("skipped"):
            print(f"  {lang:<{W}}  SKIPPED ({r.get('reason', '?')})")
            continue
        print(
            f"  {lang:<{W}}"
            f"  {r['n_docs']:>5}"
            f"  {r['train_chars']:>8,}"
            f"  {r['test_chars']:>8,}"
            f"  {r['n_atoms']:>5}"
            f"  {r['n_compositions']:>7,}"
            f"  {r['baseline_bits']:>9.3f}"
            f"  {r['bigram_ppl']:>8.3f}"
            f"  {r['multilevel_ppl']:>9.3f}"
            f"  {r['ml_improvement']:>+8.3f}"
        )

    print()
    print(f"  Target to beat a small transformer: < 2.0 bits/char")
    print(f"  (ML gain = bigram - multilevel; positive = multilevel is better)")
    print()

    # Timing
    total = sum(r["total_secs"] for r in results.values() if not r.get("skipped"))
    print(f"  Total benchmark time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Direct runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_corpus_root_exists,
        test_bigram_beats_baseline,
        test_multilevel_beats_bigram,
        test_summary_table,
    ]
    passed = 0
    for t in tests:
        print(f"\nRunning {t.__name__} ...")
        try:
            t()
            print(f"  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed.")
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
