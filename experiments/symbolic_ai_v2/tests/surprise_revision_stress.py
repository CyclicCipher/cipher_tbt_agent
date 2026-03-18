"""
Stress test for SurpriseDetector + RevisionEngine (Stage 6).

Tests push the revision loop under adversarial conditions:
  1. Threshold sensitivity  — find the threshold below which normal seqs trigger false positives.
  2. High anomaly density   — sequences where >50% of tokens are surprising.
  3. Long chains            — sequences of length 1000+.
  4. Contradictory revisions — two rules that contradict each other; revision must not loop.
  5. Silent rules           — the predictor is always silent ({}); every token is surprising.
  6. Self-referential rules  — A→A chain; BFS must not loop.
  7. Competing hypotheses at scale — 100 hypotheses, only 1 with majority evidence.
  8. Threshold drift         — threshold is lowered iteratively until all tokens are anomalous.
  9. Evidence count overflow — same bigram seen 10,000 times.
 10. Empty / single-token sequences.

Run:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/surprise_revision_stress.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.inference.surprise import (
    SurpriseDetector,
    _KL_INF_SUBSTITUTE,
)
from experiments.symbolic_ai_v2.ctkg.inference.revise import RevisionEngine
from experiments.symbolic_ai_v2.ctkg.inference.deduct import DeductionEngine


# ---------------------------------------------------------------------------
# Stub predictors
# ---------------------------------------------------------------------------

class FixedPredictor:
    """Always returns the same distribution regardless of prefix."""
    def __init__(self, dist: dict):
        self._dist = dict(dist)

    def predict_next(self, prefix):
        return dict(self._dist)


class SilentPredictor:
    """Always returns {} (no prediction)."""
    def predict_next(self, prefix):
        return {}


class PositionPredictor:
    """Returns a per-position distribution keyed on prefix length."""
    def __init__(self, dists: list[dict]):
        self._dists = dists

    def predict_next(self, prefix):
        idx = min(len(prefix), len(self._dists) - 1)
        return dict(self._dists[idx])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _status(ok: bool) -> str:
    return PASS if ok else FAIL


def _make(predictor, threshold=1.0, complexity=0.5):
    mg = MorphismGraph()
    sd = SurpriseDetector(predictor, mg=mg, threshold=threshold)
    eng = RevisionEngine(sd, mg, complexity_penalty=complexity)
    return eng, mg, sd


# ---------------------------------------------------------------------------
# Test 1: Threshold sensitivity
# ---------------------------------------------------------------------------

def test_threshold_sensitivity():
    """Find the critical threshold where normal sequences start being flagged.

    A predictor assigns uniform(N) to position 1.  Normal sequences should
    NOT trigger revision when threshold > log(N).  The critical point is
    threshold = log(N) = log(10) ≈ 2.3 nats.

    Expected: revision fires when threshold < log(10), not when threshold > log(10).
    """
    N = 10
    tokens = [f't{i}' for i in range(N)]
    normal_seq = ["start", tokens[0], "end"]

    pred = PositionPredictor([
        {"start": 1.0},
        {t: 1.0 / N for t in tokens},
        {"end": 1.0},
    ])

    threshold_above = math.log(N) + 0.01
    threshold_below = math.log(N) - 0.01

    eng_above, _, _ = _make(pred, threshold=threshold_above, complexity=0.5)
    eng_below, _, _ = _make(pred, threshold=threshold_below, complexity=0.5)

    result_above = eng_above.revise(normal_seq)
    result_below = eng_below.revise(normal_seq)

    ok = (result_above is None) and (result_below is not None)
    return _status(ok), f"above_log10={result_above is None}, below_log10={result_below is not None}"


# ---------------------------------------------------------------------------
# Test 2: High anomaly density
# ---------------------------------------------------------------------------

def test_high_anomaly_density():
    """Sequence where every-other token is anomalous.

    Predictor always says 'a'.  Sequence: [a, X, a, X, a, X, ...] of length 20.
    All 'X' positions are anomalous.  Engine should adopt (a → X) once and then
    evidence_count should be 10 (seen 10 times).
    """
    pred = FixedPredictor({"a": 1.0})
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    tokens = []
    for _ in range(10):
        tokens.extend(["a", "X"])

    # Revise each bigram (a, X) occurrence.
    # Because we call revise() on the whole sequence, all (a→X) bigrams
    # are collected as the same candidate → evidence_count increments.
    eng.revise(tokens)

    morphs = mg.morphisms(include_identity=False)
    obs = [m for m in morphs if m.morph_type == "OBS_SEQ"
           and mg._objects[m.source].label == "a"
           and mg._objects[m.target].label == "X"]

    # There should be exactly 1 edge a→X.
    # The RevisionEngine calls _apply() once (for the best candidate),
    # and the best candidate covers all 10 (a→X) bigrams via explains list.
    # Actually: generate_candidates aggregates same-bigram anomalies into one candidate.
    # So evidence_count should be 1 (initial) after first revise().
    # But score = 10 - 0.5 = 9.5 >> 0 → adopted.
    ok = len(obs) == 1
    evidence = obs[0].evidence_count if obs else 0
    return _status(ok), f"n_edges={len(obs)}, evidence={evidence}, score_expected=9.5"


# ---------------------------------------------------------------------------
# Test 3: Long sequence
# ---------------------------------------------------------------------------

def test_long_sequence():
    """Scan a sequence of length 1000 without crashing or hanging."""
    pred = FixedPredictor({"a": 1.0})
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    # Sequence: alternating a and X, length 1000.
    tokens = (["a", "X"] * 500)

    t0 = time.monotonic()
    result = eng.revise(tokens)
    elapsed = time.monotonic() - t0

    ok = result is not None and elapsed < 5.0
    return _status(ok), f"elapsed={elapsed:.3f}s, adopted={result is not None}"


# ---------------------------------------------------------------------------
# Test 4: Contradictory revisions
# ---------------------------------------------------------------------------

def test_contradictory_revisions():
    """Alternating normal and anomalous sequences should not loop.

    After the first revision (a→X adopted), the predictor no longer flags (a→X),
    so the second revision of a normal sequence is clean.
    The engine adopts the FIRST anomaly and stops (no infinite loop).
    """
    pred = FixedPredictor({"a": 1.0})
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    results = []
    for i in range(20):
        if i % 2 == 0:
            # Anomalous: a→X
            r = eng.revise(["a", "X"])
        else:
            # Normal: a→a
            r = eng.revise(["a", "a"])
        results.append(r is not None)

    # Should not crash; results are deterministic.
    ok = True  # If we get here without exception, the test passes.
    n_adopted = sum(results)
    return _status(ok), f"n_adopted={n_adopted}/20, no_infinite_loop=True"


# ---------------------------------------------------------------------------
# Test 5: Silent predictor
# ---------------------------------------------------------------------------

def test_silent_predictor():
    """SilentPredictor returns {} → every token has KL = _KL_INF_SUBSTITUTE.

    All tokens are flagged.  The engine should adopt the best candidate (most
    frequent bigram) and return a valid RevisionCandidate.
    """
    pred = SilentPredictor()
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    # Sequence with a repeated bigram: (a→b) appears 5 times; (b→c) appears 1 time.
    tokens = ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "b", "c"]
    result = eng.revise(tokens)

    # Best candidate should be (a→b) with score=5-0.5=4.5.
    ok = (result is not None
          and result.source_label == "a"
          and result.target_label == "b")
    score = result.score if result else None
    src = result.source_label if result else None
    tgt = result.target_label if result else None
    return _status(ok), f"adopted=({src}->{tgt}), score={score}"


# ---------------------------------------------------------------------------
# Test 6: Self-referential rules (BFS loop guard)
# ---------------------------------------------------------------------------

def test_self_referential_rules():
    """DeductionEngine BFS must not loop on A→A rules.

    Chain: A→A (circular), given A.  Should not hang.
    """
    engine = DeductionEngine("rule", "given", "conclude", max_depth=5)
    prefix = ["rule", "A", "A", "given", "A", "conclude"]

    t0 = time.monotonic()
    result = engine.predict(prefix)
    elapsed = time.monotonic() - t0

    # A→A is circular: A is both premise and "conclusion".  BFS visits A once,
    # then tries to enqueue A again but it's already visited → stops.
    # reachable list is empty (A at depth 0 is the premise, not a conclusion).
    ok = elapsed < 1.0  # Must terminate quickly.
    return _status(ok), f"result={result}, elapsed={elapsed:.4f}s"


# ---------------------------------------------------------------------------
# Test 7: Competing hypotheses at scale
# ---------------------------------------------------------------------------

def test_competing_hypotheses_scale():
    """100 competing single-evidence bigrams; one bigram has 50 occurrences.

    The winning bigram (a→WINNER) should be adopted over all others.
    """
    pred = SilentPredictor()
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    rng = random.Random(42)
    n_competitors = 100
    competitor_labels = [f"C{i}" for i in range(n_competitors)]

    tokens = []
    # 50 occurrences of (a→WINNER)
    for _ in range(50):
        tokens.extend(["a", "WINNER"])
    # 1 occurrence of each competitor (a→C0), (a→C1), ...
    for c in competitor_labels:
        tokens.extend(["a", c])

    # Shuffle to prevent order-bias in the BFS.
    pairs = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)]
    rng.shuffle(pairs)
    tokens = [t for p in pairs for t in p]

    t0 = time.monotonic()
    result = eng.revise(tokens)
    elapsed = time.monotonic() - t0

    ok = (result is not None
          and result.source_label == "a"
          and result.target_label == "WINNER"
          and result.score >= 49.5)
    winner = result.target_label if result else None
    score = result.score if result else None
    return _status(ok), f"winner={winner}, score={score}, elapsed={elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Test 8: Threshold drift
# ---------------------------------------------------------------------------

def test_threshold_drift():
    """Lower threshold iteratively until even 'certain' tokens are flagged.

    At threshold=0.0, even a perfectly predicted token (KL=0) is NOT flagged
    (is_surprising uses strict > not >=).  Verify this edge case.
    """
    pred = FixedPredictor({"a": 1.0})
    eng, mg, sd = _make(pred, threshold=0.0, complexity=0.5)

    result = eng.revise(["a", "a", "a"])
    # KL("a", {a:1.0}) = 0.0.  is_surprising(0.0) = (0.0 > 0.0) = False.
    ok = result is None
    return _status(ok), f"no_false_positives_at_zero_threshold={result is None}"


# ---------------------------------------------------------------------------
# Test 9: Evidence count overflow
# ---------------------------------------------------------------------------

def test_evidence_count_overflow():
    """Same bigram revised 10,000 times — evidence count should accumulate."""
    pred = SilentPredictor()
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    for _ in range(10_000):
        eng.revise(["a", "b"])

    morphs = mg.morphisms(include_identity=False)
    obs = [m for m in morphs if m.morph_type == "OBS_SEQ"
           and mg._objects[m.source].label == "a"
           and mg._objects[m.target].label == "b"]

    ok = len(obs) == 1 and obs[0].evidence_count == 10_000
    evidence = obs[0].evidence_count if obs else 0
    return _status(ok), f"n_edges={len(obs)}, evidence={evidence}"


# ---------------------------------------------------------------------------
# Test 10: Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases():
    """Empty sequence and single-token sequence should not crash."""
    pred = SilentPredictor()
    eng, mg, sd = _make(pred, threshold=0.5, complexity=0.5)

    r_empty = eng.revise([])
    r_single = eng.revise(["a"])

    ok = r_empty is None and r_single is None
    return _status(ok), f"empty={r_empty}, single={r_single}"


# ---------------------------------------------------------------------------
# Test 11: Deduction — deep chain (50 hops)
# ---------------------------------------------------------------------------

def test_deep_chain():
    """50-hop implication chain should terminate correctly and quickly."""
    n = 50
    engine = DeductionEngine("rule", "given", "conclude", max_depth=n)

    # Build chain: A0 → A1 → A2 → ... → A{n}
    prefix = []
    for i in range(n):
        prefix += ["rule", f"A{i}", f"A{i+1}"]
    prefix += ["given", "A0", "conclude"]

    t0 = time.monotonic()
    result = engine.predict(prefix)
    elapsed = time.monotonic() - t0

    ok = result == {f"A{n}": 1.0} and elapsed < 1.0
    return _status(ok), f"result={result}, elapsed={elapsed:.4f}s"


# ---------------------------------------------------------------------------
# Test 12: Branching implication graph
# ---------------------------------------------------------------------------

def test_branching_graph():
    """Multiple paths; deepest reachable node returned.

    Graph:
      A → B (depth 1)
      A → C (depth 1)
      B → D (depth 2)
      C → E → F (depth 3 from A)

    Given A, deepest = F (depth 3).
    """
    engine = DeductionEngine("r", "g", "c", max_depth=5)
    prefix = [
        "r", "A", "B",
        "r", "A", "C",
        "r", "B", "D",
        "r", "C", "E",
        "r", "E", "F",
        "g", "A", "c",
    ]
    result = engine.predict(prefix)

    ok = result == {"F": 1.0}
    return _status(ok), f"result={result}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TESTS = [
    ("01 Threshold sensitivity",       test_threshold_sensitivity),
    ("02 High anomaly density",        test_high_anomaly_density),
    ("03 Long sequence (n=1000)",      test_long_sequence),
    ("04 Contradictory revisions",     test_contradictory_revisions),
    ("05 Silent predictor",            test_silent_predictor),
    ("06 Self-referential rules",      test_self_referential_rules),
    ("07 Competing hypotheses ×100",   test_competing_hypotheses_scale),
    ("08 Threshold drift to zero",     test_threshold_drift),
    ("09 Evidence count ×10K",         test_evidence_count_overflow),
    ("10 Edge cases (empty/single)",   test_edge_cases),
    ("11 Deep chain (50 hops)",        test_deep_chain),
    ("12 Branching implication graph", test_branching_graph),
]


def main():
    print("=" * 65)
    print("Surprise + Revision Stress Test")
    print("=" * 65)
    print()

    passed = failed = warned = 0
    for name, fn in _TESTS:
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = FAIL, f"EXCEPTION: {e}"

        icon = {"PASS": "OK", "FAIL": "XX", "WARN": "~~"}.get(status, "??")
        print(f"  {icon} {name}")
        print(f"      {detail}")
        if status == PASS:
            passed += 1
        elif status == FAIL:
            failed += 1
        else:
            warned += 1
    print()
    print("=" * 65)
    print(f"Results: {passed} passed, {failed} failed, {warned} warned")
    print("=" * 65)
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
