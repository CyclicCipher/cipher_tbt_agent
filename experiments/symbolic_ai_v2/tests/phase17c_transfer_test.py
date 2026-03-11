"""Phase 17c — Cross-domain transfer benchmark.

One algorithm, two domains, shared topology.

Both language and mathematics are processed with the same sequence_1d topology
and the same MorphismGraph.  Two models are trained in opposite orders:

  Model A:  language corpus FIRST, then math sequences
  Model B:  math sequences FIRST, then language corpus

And two single-domain baselines:

  Model M:  math only
  Model L:  language only

Metrics collected for each model:

  lang_ppl   — perplexity on held-out language test set  (bits/char)
  math_ppl   — perplexity on held-out math test set      (bits/token)
  math_acc   — top-1 accuracy on math test set           (%)
  n_rules    — number of algebraic rules discovered

Hypotheses:
  H1 (domain-agnostic):      order does not matter — |ppl_A - ppl_B| < 20% of ppl_M
  H2 (no interference):      both A and B achieve reasonable ppl on BOTH domains
  H3 (math structure helps): math_acc(A) ≈ math_acc(M)  (language doesn't hurt math)

Language source:
  Multilingual corpus files under corpus/ (if available) or a synthetic
  Zipfian-like word sequence drawn from a 50-word vocabulary.  Both exercise
  the same statistical learning algorithm; only richness differs.

Math source:
  LEVELS generator — all 11 levels, 3× repetition.

Run:
  pytest experiments/symbolic_ai_v2/tests/phase17c_transfer_test.py -v -s
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from experiments.symbolic_ai_v2.core.topology  import sequence_1d
from experiments.symbolic_ai_v2.core.morphism  import MorphismGraph
from experiments.symbolic_ai_v2.core.predict   import (
    perplexity as _perplexity,
    perplexity_multilevel as _ppl_ml,
)
from experiments.symbolic_ai_v2.corpus.math_generator import LEVELS
from experiments.symbolic_ai_v2.reasoning.rule_store       import build_rule_store
from experiments.symbolic_ai_v2.reasoning.variable_binding import build_variable_binding

# Reuse the math accuracy helper from corpus_benchmark
from experiments.symbolic_ai_v2.tests.corpus_benchmark import _math_accuracy

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"
_LANG_LIMIT  = 30_000   # chars per language file; keeps the test fast
_MATH_REPS   = 3        # repetitions per math sequence (matches math_benchmark.py)


# ── Language corpus loading ────────────────────────────────────────────────────

def _load_language(limit_chars: int = _LANG_LIMIT) -> tuple[list[str], list[str]]:
    """Return (train_docs, test_docs) as lists of strings.

    Uses the multilingual corpus if available; otherwise generates a synthetic
    Zipfian word sequence so the test always runs without file dependencies.
    The synthetic corpus uses a 50-word vocabulary with rank-frequency p(w) ∝ 1/rank.
    """
    docs: list[str] = []
    if CORPUS_ROOT.exists():
        for subfolder in sorted(CORPUS_ROOT.iterdir()):
            if not subfolder.is_dir():
                continue
            for path in sorted(subfolder.glob("*.txt"))[:2]:    # max 2 files/lang
                text = path.read_text(encoding="utf-8", errors="replace")
                text = text.lstrip("\ufeff").strip()[:limit_chars]
                if text:
                    docs.append(text)

    if not docs:
        # Synthetic Zipfian fallback — always available
        vocab = [
            "the", "of", "and", "to", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "with", "as", "at", "this", "be",
            "have", "from", "or", "an", "by", "not", "but", "had", "his", "they",
            "its", "one", "all", "were", "we", "when", "your", "can", "said", "there",
            "use", "each", "which", "she", "do", "how", "their", "if", "will", "up",
        ]
        weights = [1.0 / (i + 1) for i in range(len(vocab))]
        rng = random.Random(42)
        for _ in range(4):
            words = rng.choices(vocab, weights=weights, k=5000)
            docs.append(" ".join(words))

    # Split: 80/20 by document count, or first/last if only one doc
    n = len(docs)
    if n == 1:
        cut = int(len(docs[0]) * 0.9)
        return [docs[0][:cut]], [docs[0][cut:]]
    cut = max(1, int(n * 0.8))
    return docs[:cut], docs[cut:]


# ── Math sequences ─────────────────────────────────────────────────────────────

def _load_math() -> tuple[list[list[str]], list[list[str]]]:
    """Return (train_seqs, test_seqs) for all 11 math levels."""
    train: list[list[str]] = []
    test:  list[list[str]] = []
    for _name, gen_fn in LEVELS:
        tr, te = gen_fn()
        train.extend(tr * _MATH_REPS)
        test.extend(te)
    return train, test


# ── Model builder ──────────────────────────────────────────────────────────────

def _build_model(
    first_domain:  tuple,   # (seqs_or_docs, is_language)
    second_domain: tuple,   # (seqs_or_docs, is_language) or None
    topo,
) -> MorphismGraph:
    """Train a MorphismGraph on first_domain then second_domain."""
    mg = MorphismGraph(topo)

    for seqs, is_language in [first_domain, second_domain]:
        if seqs is None:
            continue
        for seq in seqs:
            mg.observe_sequence(seq, topo)
        mg.prune()

    build_rule_store(mg, topo)
    build_variable_binding(mg, topo)
    return mg


# ── Module-scope fixture: train all four models once ──────────────────────────

@pytest.fixture(scope="module")
def transfer_results():
    """Train models A, B, M, L and return their evaluation metrics.

    Returns a dict with keys: 'A', 'B', 'M', 'L', each containing:
      lang_ppl, math_ppl, math_acc, n_rules, n_compositions
    """
    topo = sequence_1d()

    lang_train, lang_test = _load_language()
    math_train, math_test = _load_math()

    has_language = bool(lang_train and lang_test)

    def _eval(mg):
        math_ppl  = _ppl_ml(mg, math_test, topo)
        math_acc  = _math_accuracy(mg, math_test, topo)
        n_rules   = len(getattr(mg, '_algebraic_rules', {}))
        n_comps   = mg.n_compositions()
        lang_ppl  = _ppl_ml(mg, lang_test, topo) if has_language else float('nan')
        return dict(
            lang_ppl  = lang_ppl,
            math_ppl  = math_ppl,
            math_acc  = math_acc,
            n_rules   = n_rules,
            n_comps   = n_comps,
        )

    # Model M: math only (baseline)
    mg_M = _build_model((math_train, False), (None, False), topo)
    res_M = _eval(mg_M)

    # Model L: language only (baseline); skip when no corpus
    if has_language:
        mg_L = _build_model((lang_train, True), (None, True), topo)
        res_L = _eval(mg_L)
    else:
        res_L = None

    # Model A: language → math
    if has_language:
        mg_A = _build_model((lang_train, True), (math_train, False), topo)
    else:
        mg_A = _build_model((math_train, False), (None, False), topo)
    res_A = _eval(mg_A)

    # Model B: math → language
    if has_language:
        mg_B = _build_model((math_train, False), (lang_train, True), topo)
    else:
        mg_B = _build_model((math_train, False), (None, False), topo)
    res_B = _eval(mg_B)

    print("\n")
    print("=" * 72)
    print("PHASE 17c - Cross-Domain Transfer Benchmark")
    print("=" * 72)
    print(f"  Language corpus: {'available (' + str(sum(len(d) for d in lang_train)) + ' chars)' if has_language else 'USING SYNTHETIC FALLBACK'}")
    print(f"  Math sequences:  {len(math_train)} train, {len(math_test)} test")
    print()
    print(f"  {'Model':<10}  {'lang_ppl':>9}  {'math_ppl':>9}  {'math_acc':>9}  {'n_rules':>8}  {'n_comps':>8}")
    print(f"  {'-' * 60}")
    for label, r in [("M (math)", res_M), ("L (lang)", res_L),
                     ("A (L->M)", res_A), ("B (M->L)", res_B)]:
        if r is None:
            print(f"  {label:<10}  {'SKIPPED (no corpus)':>50}")
            continue
        lang_s = f"{r['lang_ppl']:9.3f}" if math.isfinite(r['lang_ppl']) else "      n/a"
        print(f"  {label:<10}  {lang_s}  {r['math_ppl']:9.3f}  {r['math_acc']*100:8.1f}%"
              f"  {r['n_rules']:>8}  {r['n_comps']:>8}")
    print()

    # Transfer analysis
    print("  Transfer analysis:")
    baseline_math_ppl = res_M['math_ppl']
    baseline_math_acc = res_M['math_acc']
    for label, r in [("A (L->M)", res_A), ("B (M->L)", res_B)]:
        if r is None:
            continue
        ppl_change = (r['math_ppl'] - baseline_math_ppl) / max(baseline_math_ppl, 1e-6)
        acc_change = r['math_acc'] - baseline_math_acc
        direction  = "positive" if ppl_change < -0.05 else (
                     "interference" if ppl_change > 0.20 else "neutral")
        print(f"  {label}: math_ppl {ppl_change:+.1%} vs M-only  "
              f"math_acc {acc_change:+.1%}  => {direction} transfer")

    if has_language and res_L is not None:
        baseline_lang_ppl = res_L['lang_ppl']
        for label, r in [("A (L->M)", res_A), ("B (M->L)", res_B)]:
            if r is None:
                continue
            lang_change = (r['lang_ppl'] - baseline_lang_ppl) / max(baseline_lang_ppl, 1e-6)
            direction   = "positive" if lang_change < -0.05 else (
                          "interference" if lang_change > 0.20 else "neutral")
            print(f"  {label}: lang_ppl {lang_change:+.1%} vs L-only  => {direction} transfer")

    print("=" * 72)

    return {
        "M": res_M, "L": res_L, "A": res_A, "B": res_B,
        "has_language": has_language,
        "baseline_math_ppl": baseline_math_ppl,
        "baseline_math_acc": baseline_math_acc,
        "topo": topo,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestMathBaseline:
    """Model M (math only) — sanity checks for the baseline."""

    def test_math_rules_discovered(self, transfer_results):
        """Math-only model must discover algebraic rules."""
        r = transfer_results["M"]
        assert r["n_rules"] >= 4, (
            f"Math-only model discovered only {r['n_rules']} rules "
            f"(expected >= 4: succ, add, sub, mul)"
        )

    def test_math_ppl_reasonable(self, transfer_results):
        """Math-only model must beat a uniform baseline."""
        r = transfer_results["M"]
        # Uniform baseline ≈ log2(n_atoms); math model should be well below.
        assert r["math_ppl"] < 8.0, (
            f"Math-only ppl {r['math_ppl']:.3f} too high (expected < 8.0)"
        )

    def test_math_accuracy_reasonable(self, transfer_results):
        """Math-only model must achieve > 60% test accuracy."""
        r = transfer_results["M"]
        assert r["math_acc"] >= 0.60, (
            f"Math-only accuracy {r['math_acc']*100:.1f}% (expected >= 60%)"
        )


class TestNoInterference:
    """Neither training order should cause catastrophic interference."""

    def test_model_A_math_not_catastrophic(self, transfer_results):
        """Model A (language→math) must reach reasonable math ppl."""
        r = transfer_results["A"]
        baseline = transfer_results["baseline_math_ppl"]
        # Allow up to 50% degradation vs math-only baseline
        threshold = baseline * 1.50
        assert r["math_ppl"] < threshold, (
            f"Language-first model A math_ppl={r['math_ppl']:.3f} "
            f"> 1.5 × baseline ({baseline:.3f}): catastrophic interference"
        )

    def test_model_B_math_not_catastrophic(self, transfer_results):
        """Model B (math→language) must reach reasonable math ppl."""
        r = transfer_results["B"]
        baseline = transfer_results["baseline_math_ppl"]
        threshold = baseline * 1.50
        assert r["math_ppl"] < threshold, (
            f"Language-first model B math_ppl={r['math_ppl']:.3f} "
            f"> 1.5 × baseline ({baseline:.3f}): catastrophic interference"
        )

    def test_model_A_math_accuracy_maintained(self, transfer_results):
        """Adding language training must not destroy math accuracy."""
        r       = transfer_results["A"]
        baseline = transfer_results["baseline_math_acc"]
        # Allow 25% absolute drop
        assert r["math_acc"] >= baseline - 0.25, (
            f"Model A math_acc={r['math_acc']*100:.1f}% "
            f"dropped >25 pts vs baseline ({baseline*100:.1f}%)"
        )

    def test_model_B_math_accuracy_maintained(self, transfer_results):
        """Switching to language after math must not destroy accuracy."""
        r       = transfer_results["B"]
        baseline = transfer_results["baseline_math_acc"]
        assert r["math_acc"] >= baseline - 0.25, (
            f"Model B math_acc={r['math_acc']*100:.1f}% "
            f"dropped >25 pts vs baseline ({baseline*100:.1f}%)"
        )


class TestOrderIndependence:
    """Domain-agnostic hypothesis: training order should not matter much."""

    def test_math_ppl_similar_both_orders(self, transfer_results):
        """Math ppl of A and B should agree within 30%."""
        a = transfer_results["A"]["math_ppl"]
        b = transfer_results["B"]["math_ppl"]
        lo, hi = min(a, b), max(a, b)
        ratio = hi / max(lo, 1e-6)
        assert ratio < 1.30, (
            f"Math ppl differs too much by order: A={a:.3f}, B={b:.3f} "
            f"(ratio {ratio:.2f} > 1.30)"
        )

    def test_math_rules_survive_language(self, transfer_results):
        """Both A and B must have discovered algebraic rules."""
        for label in ("A", "B"):
            r = transfer_results[label]
            assert r["n_rules"] >= 4, (
                f"Model {label}: only {r['n_rules']} algebraic rules after mixed training"
            )

    def test_language_ppl_similar_both_orders(self, transfer_results):
        """If language corpus present: A and B should reach similar lang_ppl."""
        if not transfer_results["has_language"]:
            pytest.skip("No language corpus available")
        a = transfer_results["A"]["lang_ppl"]
        b = transfer_results["B"]["lang_ppl"]
        if not (math.isfinite(a) and math.isfinite(b)):
            pytest.skip("lang_ppl not finite (corpus too small)")
        lo, hi = min(a, b), max(a, b)
        ratio = hi / max(lo, 1e-6)
        assert ratio < 1.30, (
            f"Lang ppl differs too much by order: A={a:.3f}, B={b:.3f} "
            f"(ratio {ratio:.2f} > 1.30)"
        )
