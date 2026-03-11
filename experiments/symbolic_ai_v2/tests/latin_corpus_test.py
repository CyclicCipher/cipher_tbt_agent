"""latin_corpus_test.py — Full-stack benchmark on the GT4HistOCR EarlyModernLatin corpus.

Uses MorphismGraph(topo) so LiveCTKG is automatically wired and the full
prediction back-off chain is active:
  1. Hopf-smoothed composition context
  2. FCA type-group back-off  (from LiveCTKG; trivial for sequence_1d)
  3. Raw atom bigram
  4. FCA type-group back-off on atom
  5. Corpus-wide marginal

AIT is disabled (no_ait marker) — its O(|edges|) hook is too slow for
million-character corpora.

Reports:
  - Perplexity (baseline / bigram / multilevel)
  - Grammar quality: max composition level, boundary rate, avg segment length
  - LiveCTKG stats: types, concepts, merges, disambiguations
  - MDL cost: free_energy() in bits
  - Context depth distribution during test evaluation

SKIP automatically if the corpus is absent.

Run:  python experiments/symbolic_ai_v2/tests/latin_corpus_test.py
 or:  python -m pytest experiments/symbolic_ai_v2/tests/latin_corpus_test.py -v -s
"""

import sys
import math
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import pytest
    pytestmark = pytest.mark.no_ait
except ImportError:
    pass

from experiments.symbolic_ai_v2.core.topology import sequence_1d
from experiments.symbolic_ai_v2.core.morphism import MorphismGraph
from experiments.symbolic_ai_v2.core.predict  import perplexity, perplexity_multilevel
from experiments.symbolic_ai_v2.reasoning.active_inference import free_energy

# ── Config ─────────────────────────────────────────────────────────────────────

N_BOOKS    = 12
TRAIN_FRAC = 0.9

CORPUS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "GT4HistOCR" / "corpus" / "EarlyModernLatin"
)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_latin_books(n: int = N_BOOKS) -> list[str]:
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


# ── Depth distribution helper ──────────────────────────────────────────────────

def _depth_distribution(mg: MorphismGraph, test_seqs: list[str], topo) -> tuple[Counter, int]:
    hist  = Counter()
    total = 0
    for seq in test_seqs:
        ctx_id = None
        for value, etype in topo.stream_tokens(seq):
            sid = mg.atoms.get(value)
            if sid is None:
                ctx_id = None
                continue
            if ctx_id is not None and etype is not None:
                hist[mg.symbols[ctx_id].level] += 1
                total += 1
                comp   = mg.rules_inv.get((ctx_id, etype, sid))
                ctx_id = comp if comp is not None else sid
            else:
                ctx_id = sid
    return hist, total


# ── Core benchmark ─────────────────────────────────────────────────────────────

_RESULT_CACHE: dict | None = None


def _benchmark() -> dict:
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
    mg   = MorphismGraph(topo)   # LiveCTKG auto-wired

    t0 = time.time()
    for seq in train_seqs:
        mg.observe_sequence(seq, topo)
        mg.prune()          # drop stale singletons at document boundary
    train_secs = time.time() - t0

    train_chars = sum(len(s) for s in train_seqs)
    baseline    = math.log2(max(mg.n_atoms(), 2))
    ppl_bi      = perplexity(mg, test_seqs, topo)
    ppl_multi   = perplexity_multilevel(mg, test_seqs, topo)
    total_secs  = time.time() - t0

    # Grammar quality
    max_level       = max((mg.symbols[s].level for s in mg.rules), default=0)
    n_bounds        = mg._n_boundaries
    boundary_rate   = n_bounds / max(train_chars, 1)
    avg_segment_len = train_chars / max(n_bounds, 1)

    # MDL cost
    fe = free_energy(mg)

    # LiveCTKG
    ctkg = mg._ctkg
    n_ctkg_types  = len(ctkg.global_kg.types)
    n_ctkg_conc   = len(ctkg.global_kg.concepts)
    n_ctkg_merges = ctkg._n_merges
    n_ctkg_viols  = ctkg._n_violations

    # Context depth during evaluation
    depth_hist, depth_total = _depth_distribution(mg, test_seqs, topo)
    pct_comp   = sum(v for k, v in depth_hist.items() if k > 0) / max(depth_total, 1)
    mean_depth = sum(k * v for k, v in depth_hist.items()) / max(depth_total, 1)

    _RESULT_CACHE = {
        "skipped":          False,
        "n_books":          len(books),
        "n_train":          len(train_seqs),
        "n_test":           len(test_seqs),
        "train_chars":      train_chars,
        "test_chars":       sum(len(s) for s in test_seqs),
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
        "max_comp_level":   max_level,
        "n_boundaries":     n_bounds,
        "boundary_rate":    boundary_rate,
        "avg_segment_len":  avg_segment_len,
        "free_energy":      fe,
        "n_ctkg_types":     n_ctkg_types,
        "n_ctkg_concepts":  n_ctkg_conc,
        "n_ctkg_merges":    n_ctkg_merges,
        "n_ctkg_viols":     n_ctkg_viols,
        "depth_hist":       dict(depth_hist),
        "pct_comp_context": pct_comp,
        "mean_ctx_depth":   mean_depth,
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
    print(f"  Baseline (log2 V={r['n_atoms']}): {r['baseline_bits']:.3f} bits/char")
    print(f"  Bigram:                          {r['bigram_ppl']:.3f} bits/char  "
          f"(gain {r['bigram_gain']:+.3f})")
    assert r["bigram_ppl"] < r["baseline_bits"]


def test_multilevel_beats_bigram():
    """Multi-level composition context must give strictly lower perplexity."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print(f"  Multilevel: {r['multilevel_ppl']:.3f} bits/char  "
          f"(gain {r['multilevel_gain']:+.3f}  |  "
          f"improvement vs bigram {r['ml_improvement']:+.3f})")
    print(f"  Grammar: {r['n_compositions']:,} compositions  "
          f"max_level={r['max_comp_level']}  "
          f"{r['n_edges']:,} edges")
    assert r["multilevel_ppl"] < r["bigram_ppl"]


def test_grammar_quality():
    """Grammar must have multi-level compositions and sensible boundary rate."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print(f"  max_comp_level={r['max_comp_level']}  "
          f"boundaries={r['n_boundaries']:,}  "
          f"boundary_rate={r['boundary_rate']:.4f}  "
          f"avg_segment={r['avg_segment_len']:.1f} chars")
    assert r["max_comp_level"] >= 2, "No trigram+ compositions learned"
    assert 0 < r["boundary_rate"] < 0.5
    assert r["avg_segment_len"] > 2


def test_ctkg_live():
    """LiveCTKG must have accumulated concepts and completed merges."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print(f"  LiveCTKG: types={r['n_ctkg_types']}  "
          f"concepts={r['n_ctkg_concepts']}  "
          f"merges={r['n_ctkg_merges']}  "
          f"disambig={r['n_ctkg_viols']}")
    assert r["n_ctkg_merges"] > 0, "LiveCTKG made 0 successful merges"
    assert r["n_ctkg_concepts"] > 0, "LiveCTKG has 0 concepts"


def test_free_energy_finite():
    """MDL free energy must be finite and positive."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    fe = r["free_energy"]
    print(f"  free_energy = {fe:,.0f} bits")
    assert math.isfinite(fe) and fe > 0, f"free_energy={fe}"


def test_context_depth():
    """Majority of test predictions must use composition context (level > 0)."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    pct  = r["pct_comp_context"]
    hist = r["depth_hist"]
    print(f"  {pct*100:.1f}% composition context  "
          f"mean_depth={r['mean_ctx_depth']:.2f}  "
          f"max_depth={max(hist, default=0)}")
    print("  Distribution:", {k: v for k, v in sorted(hist.items()) if v > 0})
    assert pct > 0.5, f"Only {pct*100:.1f}% of predictions use composition context"


def test_summary():
    """Print comprehensive comparison table; always passes."""
    r = _benchmark()
    if r.get("skipped"):
        print("  SKIPPED — corpus not found")
        return
    print()
    print(f"  {'Model':<22}  {'bits/char':>10}  {'vs baseline':>12}  {'vs bigram':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Baseline (uniform)':<22}  {r['baseline_bits']:>10.3f}  {'—':>12}  {'—':>10}")
    print(f"  {'Bigram':<22}  {r['bigram_ppl']:>10.3f}  "
          f"{r['bigram_gain']:>+11.3f}  {'—':>10}")
    print(f"  {'Multilevel':<22}  {r['multilevel_ppl']:>10.3f}  "
          f"{r['multilevel_gain']:>+11.3f}  "
          f"{r['ml_improvement']:>+9.3f}")
    print()
    print(f"  Grammar:  max_level={r['max_comp_level']}  "
          f"avg_seg={r['avg_segment_len']:.1f} chars  "
          f"boundaries={r['n_boundaries']:,}  "
          f"comps={r['n_compositions']:,}")
    print(f"  LiveCTKG: types={r['n_ctkg_types']}  "
          f"concepts={r['n_ctkg_concepts']}  "
          f"merges={r['n_ctkg_merges']}  "
          f"disambig={r['n_ctkg_viols']}")
    print(f"  MDL:      free_energy={r['free_energy']:,.0f} bits")
    print(f"  Context:  {r['pct_comp_context']*100:.1f}% composition  "
          f"mean_depth={r['mean_ctx_depth']:.2f}  "
          f"max_depth={max(r['depth_hist'], default=0)}")
    print()
    print(f"  Target: < 2.0 bits/char (small transformer baseline)")
    print(f"  Train {r['train_secs']:.1f}s  |  Total {r['total_secs']:.1f}s")


# ── Direct runner ───────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_corpus_available,
        test_bigram_beats_baseline,
        test_multilevel_beats_bigram,
        test_grammar_quality,
        test_ctkg_live,
        test_free_energy_finite,
        test_context_depth,
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
