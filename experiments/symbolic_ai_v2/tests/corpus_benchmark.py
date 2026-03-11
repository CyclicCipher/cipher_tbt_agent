"""corpus_benchmark.py — Multi-language benchmark using the full model stack.

Creates one MorphismGraph(topo) per language, which auto-wires LiveCTKG.
All prediction functions (perplexity, perplexity_multilevel) then
automatically use the full back-off chain:

  1. Hopf-smoothed composition context   (Graph-SEQUITUR depth-k grammar)
  2. FCA type-group back-off             (LiveCTKG adjunction, if multi-edge)
  3. Raw atom bigram
  4. FCA type-group back-off on atom
  5. Corpus-wide marginal

Additional metrics reported per language:
  - Grammar quality   : max composition level, boundary rate, avg segment length
  - CTKG live stats   : types, concepts, successful merges, disambiguations
  - MDL cost          : free_energy() — bits needed to encode the model

AIT is disabled (no_ait marker) because its O(|edges|) hook is too slow for
million-character corpora.  Use ActiveInferenceTracker manually on a subset
when an online learning curve is needed.

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
from collections import Counter
from pathlib import Path

try:
    import pytest
    # Disable AIT autouse fixture — O(|edges|) hook is too slow for large corpora.
    pytestmark = pytest.mark.no_ait
except ImportError:
    pass   # running directly (python corpus_benchmark.py) — pytest not needed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.symbolic_ai_v2.core.topology  import sequence_1d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph
from experiments.symbolic_ai_v2.core.predict   import perplexity, perplexity_multilevel
from experiments.symbolic_ai_v2.reasoning.active_inference import free_energy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
TRAIN_FRAC  = 0.8
CHAR_SPLIT  = 0.9
_MIN_TRAIN_FOR_MULTILEVEL = 50_000   # chars; FCA/composition context only reliable above this


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_folder(folder: Path) -> list[str]:
    docs: list[str] = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        text = text.lstrip("\ufeff").strip()
        if text:
            docs.append(text)
    return docs


def _split(docs: list[str]) -> tuple[list[str], list[str]]:
    n = len(docs)
    if n == 0:
        return [], []
    if n == 1:
        text = docs[0]
        cut  = int(len(text) * CHAR_SPLIT)
        return [text[:cut]], [text[cut:]]
    if n == 2:
        return [docs[0]], [docs[1]]
    cut = max(1, int(n * TRAIN_FRAC))
    return docs[:cut], docs[cut:]


# ---------------------------------------------------------------------------
# Grammar helpers
# ---------------------------------------------------------------------------

def _depth_distribution(
    mg: MorphismGraph, test_docs: list[str], topo
) -> tuple[Counter, int]:
    """Distribution of ctx_id.level during multilevel test-set evaluation."""
    hist  = Counter()
    total = 0
    for seq in test_docs:
        ctx_id  = None
        atom_id = None
        for value, etype in topo.stream_tokens(seq):
            sid = mg.atoms.get(value)
            if sid is None:
                ctx_id = atom_id = None
                continue
            if ctx_id is not None and etype is not None:
                hist[mg.symbols[ctx_id].level] += 1
                total += 1
                comp   = mg.rules_inv.get((ctx_id, etype, sid))
                ctx_id = comp if comp is not None else sid
            else:
                ctx_id = sid
            atom_id = sid
    return hist, total


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
    mg   = MorphismGraph(topo)   # LiveCTKG auto-wired; full back-off chain active

    t0 = time.time()
    for doc in train_docs:
        mg.observe_sequence(doc, topo)
        mg.prune()          # drop stale singletons at document boundary
    train_secs = time.time() - t0

    train_chars = sum(len(d) for d in train_docs)

    baseline     = math.log2(max(mg.n_atoms(), 2))
    ppl_bi       = perplexity(mg, test_docs, topo)
    ppl_multi    = perplexity_multilevel(mg, test_docs, topo)
    total_secs   = time.time() - t0

    # Grammar quality
    max_level = max((mg.symbols[s].level for s in mg.rules), default=0)
    n_bounds  = mg._n_boundaries
    boundary_rate   = n_bounds / max(train_chars, 1)
    avg_segment_len = train_chars / max(n_bounds, 1)

    # MDL cost
    fe = free_energy(mg)

    # LiveCTKG stats (always populated because MorphismGraph(topo) wires it)
    ctkg          = mg._ctkg
    n_ctkg_types  = len(ctkg.global_kg.types)
    n_ctkg_conc   = len(ctkg.global_kg.concepts)
    n_ctkg_merges = ctkg._n_merges
    n_ctkg_viols  = ctkg._n_violations

    # Context depth during multilevel evaluation
    depth_hist, depth_total = _depth_distribution(mg, test_docs, topo)
    pct_comp = (
        sum(v for k, v in depth_hist.items() if k > 0) / max(depth_total, 1)
    )
    mean_depth = (
        sum(k * v for k, v in depth_hist.items()) / max(depth_total, 1)
    )

    return {
        "skipped":          False,
        "n_docs":           len(docs),
        "n_train":          len(train_docs),
        "n_test":           len(test_docs),
        "train_chars":      train_chars,
        "test_chars":       sum(len(d) for d in test_docs),
        "n_atoms":          mg.n_atoms(),
        "n_compositions":   mg.n_compositions(),
        "n_edges":          mg.n_edges(),
        "baseline_bits":    baseline,
        "bigram_ppl":       ppl_bi,
        "multilevel_ppl":   ppl_multi,
        "bigram_gain":      baseline - ppl_bi,
        "multilevel_gain":  baseline - ppl_multi,
        "ml_improvement":   ppl_bi - ppl_multi,
        "train_secs":       train_secs,
        "total_secs":       total_secs,
        # Grammar
        "max_comp_level":   max_level,
        "n_boundaries":     n_bounds,
        "boundary_rate":    boundary_rate,
        "avg_segment_len":  avg_segment_len,
        # MDL
        "free_energy":      fe,
        # LiveCTKG
        "n_ctkg_types":     n_ctkg_types,
        "n_ctkg_concepts":  n_ctkg_conc,
        "n_ctkg_merges":    n_ctkg_merges,
        "n_ctkg_viols":     n_ctkg_viols,
        # Context depth
        "depth_hist":       dict(depth_hist),
        "pct_comp_context": pct_comp,
        "mean_ctx_depth":   mean_depth,
    }


# ---------------------------------------------------------------------------
# Global results cache
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
    """Corpus root directory must be present and have at least one runnable language."""
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
              f"baseline={r['baseline_bits']:.3f}  {'OK' if ok else 'FAIL'}")
        assert ok, (
            f"{lang}: bigram {r['bigram_ppl']:.3f} >= baseline {r['baseline_bits']:.3f}"
        )
    if not any_run:
        print("  SKIPPED — all languages skipped")


def test_multilevel_beats_bigram():
    """Multilevel context must give strictly lower perplexity than bigram.

    Only asserted for languages with >= _MIN_TRAIN_FOR_MULTILEVEL training chars;
    smaller corpora do not have enough data for the composition context to be
    reliable and are reported but not asserted.
    """
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_checked = False
    for lang, r in results.items():
        if r.get("skipped"):
            print(f"  {lang}: SKIPPED ({r.get('reason', '?')})")
            continue
        ok    = r["multilevel_ppl"] < r["bigram_ppl"]
        large = r["train_chars"] >= _MIN_TRAIN_FOR_MULTILEVEL
        tag   = "OK" if ok else ("FAIL" if large else "small corpus — not asserted")
        print(f"  {lang}: multilevel={r['multilevel_ppl']:.3f}  "
              f"bigram={r['bigram_ppl']:.3f}  "
              f"improvement={r['ml_improvement']:+.3f}  "
              f"train={r['train_chars']:,}  ({tag})")
        if large:
            any_checked = True
            assert ok, (
                f"{lang}: multilevel {r['multilevel_ppl']:.3f} >= "
                f"bigram {r['bigram_ppl']:.3f} "
                f"(train_chars={r['train_chars']:,})"
            )
    if not any_checked:
        print(f"  NOTE — no language has >= {_MIN_TRAIN_FOR_MULTILEVEL:,} training chars")


def test_grammar_quality():
    """Grammar must show meaningful compositional structure for large corpora.

    Asserts for languages with >= _MIN_TRAIN_FOR_MULTILEVEL training chars:
      - At least one composition with level >= 2   (trigram+ context exists)
      - Boundary rate in (0, 0.5)                  (neither trivial nor pathological)
      - Average segment length > 2                 (segments span multiple tokens)
    """
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_checked = False
    for lang, r in results.items():
        if r.get("skipped") or r["train_chars"] < _MIN_TRAIN_FOR_MULTILEVEL:
            continue
        any_checked = True
        print(f"  {lang}:  max_level={r['max_comp_level']}  "
              f"boundaries={r['n_boundaries']:,}  "
              f"boundary_rate={r['boundary_rate']:.4f}  "
              f"avg_seg={r['avg_segment_len']:.1f} chars")
        assert r["max_comp_level"] >= 2, (
            f"{lang}: max composition level {r['max_comp_level']} < 2 "
            f"(no trigram+ context learned)"
        )
        assert 0 < r["boundary_rate"] < 0.5, (
            f"{lang}: boundary_rate={r['boundary_rate']:.4f} out of (0, 0.5)"
        )
        assert r["avg_segment_len"] > 2, (
            f"{lang}: avg_segment_len={r['avg_segment_len']:.1f} <= 2"
        )
    if not any_checked:
        print(f"  NOTE — no language has >= {_MIN_TRAIN_FOR_MULTILEVEL:,} training chars")


def test_ctkg_live():
    """LiveCTKG must accumulate concepts and complete merges for large corpora.

    MorphismGraph(topo) wires LiveCTKG automatically; this test verifies the
    pipeline actually ran and produced a non-empty global CTKG.
    """
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_checked = False
    for lang, r in results.items():
        if r.get("skipped") or r["train_chars"] < _MIN_TRAIN_FOR_MULTILEVEL:
            continue
        any_checked = True
        print(f"  {lang}:  ctkg_types={r['n_ctkg_types']}  "
              f"concepts={r['n_ctkg_concepts']}  "
              f"merges={r['n_ctkg_merges']}  "
              f"disambig={r['n_ctkg_viols']}")
        assert r["n_ctkg_merges"] > 0, (
            f"{lang}: LiveCTKG made 0 successful merges — pipeline did not run"
        )
        assert r["n_ctkg_concepts"] > 0, (
            f"{lang}: LiveCTKG has 0 concepts — FCA produced nothing"
        )
    if not any_checked:
        print(f"  NOTE — no language has >= {_MIN_TRAIN_FOR_MULTILEVEL:,} training chars")


def test_free_energy_finite():
    """MDL free energy must be a finite positive number for every trained model."""
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_run = False
    for lang, r in results.items():
        if r.get("skipped"):
            continue
        any_run = True
        fe = r["free_energy"]
        ok = math.isfinite(fe) and fe > 0
        print(f"  {lang}: free_energy={fe:,.1f} bits  {'OK' if ok else 'FAIL'}")
        assert ok, f"{lang}: free_energy={fe} is not finite and positive"
    if not any_run:
        print("  SKIPPED — all languages skipped")


def test_context_depth():
    """Majority of test predictions must use composition context (level > 0).

    Verifies that the model has learned enough compositional structure for
    multi-level prediction to dominate over raw bigram context.
    Only checked for large corpora.
    """
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED")
        return
    any_checked = False
    for lang, r in results.items():
        if r.get("skipped") or r["train_chars"] < _MIN_TRAIN_FOR_MULTILEVEL:
            continue
        any_checked = True
        pct  = r["pct_comp_context"]
        mean = r["mean_ctx_depth"]
        hist = r["depth_hist"]
        print(f"  {lang}: {pct*100:.1f}% composition context  "
              f"mean_depth={mean:.2f}  "
              f"max_depth={max(hist, default=0)}")
        assert pct > 0.5, (
            f"{lang}: only {pct*100:.1f}% of predictions use composition context "
            f"(expected > 50%)"
        )
    if not any_checked:
        print(f"  NOTE — no language has >= {_MIN_TRAIN_FOR_MULTILEVEL:,} training chars")


def test_summary_table():
    """Print comprehensive table across all languages. Always passes."""
    results = _all_benchmarks()
    if not results:
        print("  SKIPPED — no results")
        return

    W = 20
    print()
    print(f"  {'Language':<{W}}  {'train':>8}  {'baseline':>8}  "
          f"{'bigram':>7}  {'multilev':>8}  {'ML gain':>7}  "
          f"{'max_lvl':>7}  {'seg_len':>7}  {'ctkg_c':>6}  {'F(bits)':>10}")
    print(f"  {'-' * (W + 82)}")
    for lang, r in results.items():
        if r.get("skipped"):
            print(f"  {lang:<{W}}  SKIPPED ({r.get('reason', '?')})")
            continue
        print(
            f"  {lang:<{W}}"
            f"  {r['train_chars']:>8,}"
            f"  {r['baseline_bits']:>8.3f}"
            f"  {r['bigram_ppl']:>7.3f}"
            f"  {r['multilevel_ppl']:>8.3f}"
            f"  {r['ml_improvement']:>+7.3f}"
            f"  {r['max_comp_level']:>7}"
            f"  {r['avg_segment_len']:>7.1f}"
            f"  {r['n_ctkg_concepts']:>6,}"
            f"  {r['free_energy']:>10,.0f}"
        )
    print()
    print(f"  Context depth (% comp / mean level / max level):")
    for lang, r in results.items():
        if r.get("skipped"):
            continue
        hist = r["depth_hist"]
        print(f"    {lang:<{W}}  {r['pct_comp_context']*100:5.1f}% comp  "
              f"mean={r['mean_ctx_depth']:.2f}  "
              f"max={max(hist, default=0)}")
    print()
    print(f"  Target to beat a small transformer: < 2.0 bits/char")
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
        test_grammar_quality,
        test_ctkg_live,
        test_free_energy_finite,
        test_context_depth,
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
