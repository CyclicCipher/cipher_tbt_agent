"""Sense disambiguation and context-dependent prediction benchmark.

Tests two capabilities of the CTKG Predictor:

  1. SENSE SPLITTING (FCA polysemy detection)
     Token 'X' is polysemous: it predicts 'OUT_A' after 'CTX_A' and 'OUT_B'
     after 'CTX_B'.  The FCA concept lattice should discover two distinct
     concepts — one concentrating on OUT_A, one on OUT_B — from positions
     where X appears in the left context.

  2. CONTEXT-DEPENDENT PREDICTION (disambiguation)
     Given prefix ['CTX_A', 'X']:  predict OUT_A   (context within r → 100%)
     Given prefix ['CTX_B', 'X']:  predict OUT_B   (context within r → 100%)
     Given prefix ['X'] alone:     predict uniform  (no disambiguation signal)
     The contrast demonstrates that the predictor actively uses context.

  3. CONTEXT-BLIND ABLATION
     Verify that removing the disambiguating context token (predicting from
     prefix=['X'] only) destroys performance — confirming the accuracy gain
     is due to context use, not a trivial bias.

Training corpus (r=2, sequences of length 3):
    ['CTX_A', 'X', 'OUT_A']  × N_COPIES
    ['CTX_B', 'X', 'OUT_B']  × N_COPIES

At r=2, predicting the token after 'X' (position 2 in the sequence):
    _left_only_hash(['CTX_A', 'X'], r=2) = 'r2|-2,CTX_A|-1,X|+1,<pad>|+2,<pad>'
    This exactly matches the training context hash → Level 2 fires → 100% accuracy.

Pass criteria:
  - FCA polysemy: ≥ 2 concepts exist with intent distributions that differ
    by JSD > 0.3 (one concentrating on OUT_A, one on OUT_B).
  - Context-in-window accuracy: 100% (both senses correct).
  - Context-blind accuracy: < 70% (no context = no disambiguation signal).

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/sense_context_benchmark.py
"""

from __future__ import annotations

import sys
import os
import math

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import discover_concepts
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import discover_morphisms
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import discover_processes
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

R = 2          # context radius
K = 5          # k-nearest for Kan extension
N_COPIES = 10  # repetitions per sense (ensures FCA min_support satisfied)

# Token names
CTX_A = "CTX_A"
CTX_B = "CTX_B"
X     = "X"
OUT_A = "OUT_A"
OUT_B = "OUT_B"


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence helper
# ---------------------------------------------------------------------------

def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence in [0, 1] (log2 scale).

    Returns 0 for identical distributions and 1 for fully disjoint ones.
    """
    all_keys = set(p) | set(q)
    p_tot = sum(p.values()) or 1.0
    q_tot = sum(q.values()) or 1.0
    pn = {k: p.get(k, 0.0) / p_tot for k in all_keys}
    qn = {k: q.get(k, 0.0) / q_tot for k in all_keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a[k] * math.log2(a[k] / ((a[k] + b[k]) / 2))
                   for k in a if a[k] > 0 and (a[k] + b[k]) > 0)

    m = {k: (pn[k] + qn[k]) / 2 for k in all_keys}
    jsd = 0.5 * _kl(pn, m) + 0.5 * _kl(qn, m)
    return min(1.0, max(0.0, jsd))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> bool:
    # ---- Build corpus ----
    seqs: list[list[str]] = []
    seqs += [[CTX_A, X, OUT_A]] * N_COPIES
    seqs += [[CTX_B, X, OUT_B]] * N_COPIES
    total = len(seqs)

    print(f"Sense/context benchmark — {total} training sequences (r={R}, k={K})")
    print(f"  Corpus: {N_COPIES}x [CTX_A, X, OUT_A]  +  {N_COPIES}x [CTX_B, X, OUT_B]")
    print()

    # ---- HankelCount ----
    hc = HankelCount(r_max=R)
    hc.update_batch(seqs)

    # ---- FCA ----
    lattices = discover_concepts(
        hankel=hc,
        r_levels=[R],
        lambda_productivity=0.1,
        merge_threshold=0.15,
        min_support=1.0,
    )
    lattice = lattices[0]
    concepts = lattice.concepts
    print(f"FCA concepts discovered: {len(concepts)}")

    # ---- Morphism / process discovery ----
    mg = discover_morphisms(seqs, hc, lattice, r=R)
    process_rules = discover_processes(seqs, op_atoms=[])

    # ---- Build Predictor ----
    pred = Predictor(
        hankel=hc,
        lattice=lattice,
        morphism_graph=mg,
        process_rules=process_rules,
        k_neighbours=K,
        r=R,
    )

    # ==================================================================
    # Test 1: FCA polysemy
    # ==================================================================
    # We expect two concepts with high intent weight on OUT_A and OUT_B
    # respectively.  These arise because at r=2 the two sense contexts
    # ("r2|-2,CTX_A|-1,X|+1,<pad>|+2,<pad>" vs "r2|-2,CTX_B|-1,X|+1,...")
    # have disjoint intent distributions and are not merged.

    # Find top concepts by intent weight for OUT_A and OUT_B
    best_a: dict[str, float] = {}   # concept intent with most OUT_A weight
    best_b: dict[str, float] = {}   # concept intent with most OUT_B weight
    best_a_w = 0.0
    best_b_w = 0.0

    for c in concepts:
        wa = c.intent_weights.get(OUT_A, 0.0)
        wb = c.intent_weights.get(OUT_B, 0.0)
        if wa > best_a_w:
            best_a_w = wa
            best_a = dict(c.intent_weights)
        if wb > best_b_w:
            best_b_w = wb
            best_b = dict(c.intent_weights)

    polysemy_jsd = _jsd(best_a, best_b) if best_a and best_b else 0.0
    polysemy_pass = polysemy_jsd > 0.3

    print("Test 1 — FCA polysemy (sense splitting):")
    print(f"  Best concept for OUT_A: weight={best_a_w:.3f}, "
          f"top-intent={sorted(best_a, key=lambda k: -best_a[k])[:3]}")
    print(f"  Best concept for OUT_B: weight={best_b_w:.3f}, "
          f"top-intent={sorted(best_b, key=lambda k: -best_b[k])[:3]}")
    print(f"  JSD between the two sense-concepts: {polysemy_jsd:.3f}  "
          f"(threshold > 0.3)")
    print(f"  Status: {'PASS' if polysemy_pass else 'FAIL'}")
    print()

    # ==================================================================
    # Test 2: Context-in-window prediction (context present, within r)
    # ==================================================================
    # With r=2, the disambiguating context token CTX_A or CTX_B is at
    # offset -2 from the prediction position → within the window.

    cases_in_window = [
        ([CTX_A, X], OUT_A, "sense A"),
        ([CTX_B, X], OUT_B, "sense B"),
    ]

    n_in_window_correct = 0
    print("Test 2 — Context-in-window prediction:")
    for prefix, truth, label in cases_in_window:
        dist = pred.predict_next(prefix)
        best_tok = max(dist, key=lambda t: dist[t]) if dist else None
        ok = best_tok == truth
        n_in_window_correct += ok
        top3 = sorted(dist, key=lambda t: -dist[t])[:3] if dist else []
        print(f"  [{label}] prefix={prefix} -> truth={truth!r}, "
              f"got={best_tok!r}, top3={top3}  {'OK' if ok else 'FAIL'}")

    in_window_acc = n_in_window_correct / len(cases_in_window)
    in_window_pass = in_window_acc >= 0.999
    print(f"  Accuracy: {in_window_acc:.1%}  (required: 100%)")
    print(f"  Status: {'PASS' if in_window_pass else 'FAIL'}")
    print()

    # ==================================================================
    # Test 3: Context-blind ablation (no disambiguating context)
    # ==================================================================
    # With only prefix=[X], the left context is <pad>,<pad> at r=2.
    # Level 2 won't match any training context; Kan extension has zero
    # overlap with all training hashes (none have <pad> at -2 AND -1).
    # Expected: falls through to morphism / marginal → 50% or less.

    dist_blind_a = pred.predict_next([X])
    best_blind = max(dist_blind_a, key=lambda t: dist_blind_a[t]) if dist_blind_a else None
    # Test: the blind prediction should NOT be consistently OUT_A or OUT_B
    # (it might pick one arbitrarily, but we verify it doesn't reach 100%)
    blind_correct_for_a = best_blind == OUT_A
    blind_correct_for_b = best_blind == OUT_B
    top3_blind = sorted(dist_blind_a, key=lambda t: -dist_blind_a[t])[:3] if dist_blind_a else []

    # Run across both senses to get an accuracy rate
    n_blind_correct = 0
    for prefix_tok, truth in [([X], OUT_A), ([X], OUT_B)]:
        d = pred.predict_next(prefix_tok)
        got = max(d, key=lambda t: d[t]) if d else None
        n_blind_correct += (got == truth)
    blind_acc = n_blind_correct / 2.0

    context_lift = in_window_acc / max(blind_acc, 0.01)
    context_dependence_pass = blind_acc < 0.70 and in_window_pass

    print("Test 3 — Context-blind ablation:")
    print(f"  predict_next([X]) -> best={best_blind!r}, top3={top3_blind}")
    print(f"  Blind accuracy (vs OUT_A and OUT_B): {blind_acc:.1%}")
    print(f"  Context lift: {context_lift:.1f}x (in-window / blind)")
    print(f"  Status: {'PASS' if context_dependence_pass else 'FAIL'}  "
          f"(requires blind < 70% AND in-window = 100%)")
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    all_pass = polysemy_pass and in_window_pass and context_dependence_pass

    print(f"FCA polysemy:        {'PASS' if polysemy_pass         else 'FAIL'}")
    print(f"In-window accuracy:  {'PASS' if in_window_pass        else 'FAIL'}")
    print(f"Context dependence:  {'PASS' if context_dependence_pass else 'FAIL'}")
    print()
    if all_pass:
        print("RESULT: ALL TESTS PASS")
    else:
        print("RESULT: SOME TESTS FAILED")
    return all_pass


if __name__ == "__main__":
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
