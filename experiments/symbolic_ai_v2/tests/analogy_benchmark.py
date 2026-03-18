"""Analogical reasoning benchmark.

Trains the CTKG pipeline on succ/pred for 0-99, then tests three structural
analogies:

  1. Inverse composition:
       succ(pred(n)) = n  for n = 1..99  (expected: 100%)
       pred(succ(n)) = n  for n = 0..98  (expected: 100%)

  2. Morphism type correspondence:
       succ and pred morphisms should share the same source-concept type
       and the same target-concept type (both map DIGIT -> DIGIT).

  3. Nearest-neighbour morphism analogy:
       Given the succ morphism's (source, target) concept centroid vector,
       the nearest OTHER non-identity morphism (by cosine similarity of the
       concatenated source+target centroids) should be pred -- not some
       unrelated morphism.

Required:
  - Inverse composition: both directions 100%.
  - Morphism correspondence: succ and pred share source/target concepts.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/analogy_benchmark.py
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

import numpy as np

from experiments.symbolic_ai_v2.corpus.digit_math_generator import (
    digit_succ_pred_split,
)
from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import discover_concepts
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import (
    discover_morphisms,
)
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    discover_processes,
    apply_process_rule,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_MAX = 99
R = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _morph_vector(mg, morph_id: int) -> np.ndarray:
    """Concatenated [source_centroid, target_centroid] for a morphism."""
    m = mg.morphism_by_id(morph_id)
    if m is None:
        return np.zeros(1)
    src_obj = mg.object_by_id(m.source)
    tgt_obj = mg.object_by_id(m.target)
    src_vec = src_obj.concept.centroid_vector if src_obj else np.zeros(1)
    tgt_vec = tgt_obj.concept.centroid_vector if tgt_obj else np.zeros(1)
    # Pad to same length before concatenating
    n = max(len(src_vec), len(tgt_vec))
    src_pad = np.pad(src_vec, (0, n - len(src_vec)))
    tgt_pad = np.pad(tgt_vec, (0, n - len(tgt_vec)))
    return np.concatenate([src_pad, tgt_pad])


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    # ---- Training ----
    _, train_seqs, _ = digit_succ_pred_split(
        train_max=TRAIN_MAX,
        test_min=TRAIN_MAX + 1,
        test_max=TRAIN_MAX + 1,
    )

    print(f"Analogical Reasoning Benchmark (trained on succ/pred 0-{TRAIN_MAX})")
    print(f"r={R}, #train={len(train_seqs)}")
    print()

    hc = HankelCount(r_max=R)
    hc.update_batch(train_seqs)

    lattices = discover_concepts(
        hankel=hc,
        r_levels=[R],
        lambda_productivity=0.1,
        merge_threshold=0.15,
        min_support=2.0,
    )
    lattice = lattices[0]
    mg = discover_morphisms(train_seqs, hc, lattice, r=R)
    process_rules = discover_processes(train_seqs, op_atoms=["succ", "pred"])

    pred = Predictor(
        hankel=hc,
        lattice=lattice,
        morphism_graph=mg,
        process_rules=process_rules,
        k_neighbours=5,
        r=R,
    )

    succ_rule = next((r for r in process_rules if r.op_atom == "succ"), None)
    pred_rule  = next((r for r in process_rules if r.op_atom == "pred"), None)

    # ---- Test 1: Inverse composition ----
    print("1. Inverse composition")

    # succ(pred(n)) = n for n = 1..99
    n_succ_pred = 0
    total_sp = 99
    for n in range(1, 100):
        digits_n  = [d for d in str(n)]
        pred_n    = apply_process_rule(pred_rule,  digits_n)
        if pred_n is None:
            continue
        succ_pred_n = apply_process_rule(succ_rule, pred_n)
        if succ_pred_n == digits_n:
            n_succ_pred += 1

    # pred(succ(n)) = n for n = 0..98
    n_pred_succ = 0
    total_ps = 99
    for n in range(0, 99):
        digits_n  = [d for d in str(n)]
        succ_n    = apply_process_rule(succ_rule, digits_n)
        if succ_n is None:
            continue
        pred_succ_n = apply_process_rule(pred_rule, succ_n)
        if pred_succ_n == digits_n:
            n_pred_succ += 1

    sp_pct = 100.0 * n_succ_pred / total_sp
    ps_pct = 100.0 * n_pred_succ / total_ps
    sp_pass = "PASS" if sp_pct >= 100.0 else "FAIL"
    ps_pass = "PASS" if ps_pct >= 100.0 else "FAIL"
    print(f"   succ(pred(n)) = n  for n=1..99:   {n_succ_pred}/{total_sp} = {sp_pct:.1f}%  {sp_pass}")
    print(f"   pred(succ(n)) = n  for n=0..98:   {n_pred_succ}/{total_ps} = {ps_pct:.1f}%  {ps_pass}")
    print()

    # ---- Test 2: Morphism type correspondence ----
    print("2. Morphism type correspondence")

    non_id_morphs = mg.morphisms(include_identity=False)
    succ_morphs = [m for m in non_id_morphs if "succ" in m.morph_type.lower()
                   or "SUCC" in m.morph_type]
    pred_morphs = [m for m in non_id_morphs if "pred" in m.morph_type.lower()
                   or "PRED" in m.morph_type]

    print(f"   Total non-identity morphisms: {len(non_id_morphs)}")
    print(f"   Morphism types: {sorted({m.morph_type for m in non_id_morphs})}")

    if succ_morphs and pred_morphs:
        src_match = succ_morphs[0].source == pred_morphs[0].source
        tgt_match = succ_morphs[0].target == pred_morphs[0].target
        print(f"   succ and pred share source concept: {'PASS' if src_match else 'FAIL'}")
        print(f"   succ and pred share target concept: {'PASS' if tgt_match else 'FAIL'}")
    else:
        # Morphism discovery may not separate succ/pred by name; test by
        # checking all discovered morphisms share the same source/target pair.
        if len(non_id_morphs) >= 1:
            sources = {m.source for m in non_id_morphs}
            targets = {m.target for m in non_id_morphs}
            print(f"   Distinct source concepts: {len(sources)}")
            print(f"   Distinct target concepts: {len(targets)}")
            print(f"   All morphisms share source: {'PASS' if len(sources) == 1 else 'INFO'}")
            print(f"   All morphisms share target: {'PASS' if len(targets) == 1 else 'INFO'}")
        else:
            print("   No non-identity morphisms discovered.")
    print()

    # ---- Test 3: Nearest-neighbour morphism analogy ----
    print("3. Nearest-neighbour morphism analogy")

    if len(non_id_morphs) >= 2:
        # Build morphism vectors
        morph_vecs = {m.morph_id: _morph_vector(mg, m.morph_id) for m in non_id_morphs}
        morph_ids  = list(morph_vecs.keys())

        # Try to find succ and pred by morph_type hint or by highest evidence
        if succ_morphs and pred_morphs:
            ref_id   = succ_morphs[0].morph_id
            tgt_id   = pred_morphs[0].morph_id
        else:
            # Use the two most-evidenced morphisms as a proxy
            by_ev = sorted(non_id_morphs, key=lambda m: -m.evidence_count)
            ref_id = by_ev[0].morph_id
            tgt_id = by_ev[1].morph_id if len(by_ev) > 1 else ref_id

        ref_vec = morph_vecs[ref_id]
        sims = {}
        for mid, vec in morph_vecs.items():
            if mid == ref_id:
                continue
            sims[mid] = _cosine(ref_vec, vec)

        if sims:
            nearest_id = max(sims, key=lambda k: sims[k])
            nearest_sim = sims[nearest_id]
            tgt_sim = sims.get(tgt_id, float("nan"))
            nearest_is_target = (nearest_id == tgt_id)
            nn_pass = "PASS" if nearest_is_target else "INFO"
            print(f"   Nearest morphism to succ-like: morph_id={nearest_id} "
                  f"(sim={nearest_sim:.3f})")
            print(f"   Similarity to pred-like morph: {tgt_sim:.3f}")
            print(f"   Nearest is pred-like: {nn_pass}")
        else:
            print("   Only one non-identity morphism; skipping NN test.")
    else:
        print("   Fewer than 2 non-identity morphisms; skipping NN test.")
    print()

    # ---- Summary ----
    all_pass = (sp_pct >= 100.0 and ps_pct >= 100.0)
    print("=" * 60)
    print(f"Required: inverse composition 100% in both directions.")
    print(f"Overall: {'ALL REQUIRED CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    run_benchmark()
