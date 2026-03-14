"""Math benchmark — CTKG Predictor on math levels 1-8 + Phase F/G traces.

Design:
  - Train on train_seqs ONLY (no test leakage).
  - Evaluate on BOTH train_seqs (recall) and test_seqs (OOD generalisation).
  - The two scores reveal whether the predictor memorised or generalised.

Levels included:
    successor, addition, subtraction, multiplication, powers,
    linear_eval, derivatives, integrals          (plain eq-format)
    power_trace, linear_trace, algebra_trace      (Phase D traces)
    conservation_scenario                         (Phase F -- cs4/cs3/cs2/cs1)
    bernoulli_trace                               (Phase F -- bern_p1/bern_p2)
    derivative_trace, integral_trace              (Phase G)

Levels excluded and why:
    counting      -- no 'eq' boundary; ordinal structure tested in ordinal_benchmark.py
    conservation  -- ambiguous: multiple valid RHS decompositions share the same
                     prefix up to 'eq add', causing irreducible ties in H
    bernoulli     -- compact form has no 'eq'; expanded conservation form inherits
                     the same ambiguity as Level 9 (multiple valid (P2,V2) pairs)

Pass criterion (per level):
    train strict == 100%  (basic recall sanity)
    test  strict >= 99.9% (true generalisation)
    Both must hold for PASS.  Levels that recall but fail OOD are marked MEMORISED.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/math_benchmark.py
"""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.corpus.math_generator import (
    successor_seqs,
    addition_seqs,
    subtraction_seqs,
    multiplication_seqs,
    power_seqs,
    linear_eval_seqs,
    derivative_seqs,
    integral_seqs,
    power_trace_seqs,
    linear_eval_trace_seqs,
    algebra_trace_seqs,
    conservation_scenario_seqs,
    bernoulli_trace_seqs,
    derivative_trace_seqs,
    integral_trace_seqs,
)
from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.em_loop import em_loop
from experiments.symbolic_ai_v2.ctkg.learning.mdl_prune import mdl_prune
from experiments.symbolic_ai_v2.ctkg.learning.lens_update import compute_gradients, apply_gradients
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    discover_processes,
    discover_compose_chains,
    build_free_category,
    enrich_morphism_graph,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor

EOS = "<eos>"
R = 12   # radius -- must exceed longest input prefix (derivatives need >=9)
K = 1    # k=1: single best-matching context, prevents contamination


# ---------------------------------------------------------------------------
# Level registry  (label, gen_fn, is_trace)
# ---------------------------------------------------------------------------

LEVELS = [
    # Plain eq-format levels
    ("successor",             successor_seqs,             False),
    ("addition",              addition_seqs,              False),
    ("subtraction",           subtraction_seqs,           False),
    ("multiplication",        multiplication_seqs,        False),
    ("powers",                power_seqs,                 False),
    ("linear_eval",           linear_eval_seqs,           False),
    ("derivatives",           derivative_seqs,            False),
    ("integrals",             integral_seqs,              False),
    # Phase D: step/ans traces
    ("power_trace",           power_trace_seqs,           True),
    ("linear_trace",          linear_eval_trace_seqs,     True),
    ("algebra_trace",         algebra_trace_seqs,         True),
    # Phase F: conservation scenario + Bernoulli traces
    ("conservation_scenario", conservation_scenario_seqs, True),
    ("bernoulli_trace",       bernoulli_trace_seqs,       True),
    # Phase G: expanded calculus traces
    ("derivative_trace",      derivative_trace_seqs,      True),
    ("integral_trace",        integral_trace_seqs,        True),
]


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def _strip_eos(seq: list[str]) -> list[str]:
    if seq and seq[-1] == EOS:
        return seq[:-1]
    return seq


def _ground_truth(seq: list[str], is_trace: bool) -> list[str]:
    seq = _strip_eos(seq)
    if is_trace:
        for delim in ("step", "ans"):
            try:
                return seq[seq.index(delim):]
            except ValueError:
                pass
        return []
    else:
        try:
            eq_idx = seq.index("eq")
            return seq[eq_idx + 1:]
        except ValueError:
            return []


def _input_prefix(seq: list[str], is_trace: bool) -> list[str]:
    seq = _strip_eos(seq)
    if is_trace:
        for delim in ("step", "ans"):
            try:
                return seq[:seq.index(delim)]
            except ValueError:
                pass
        return seq
    else:
        try:
            eq_idx = seq.index("eq")
            return seq[:eq_idx + 1]
        except ValueError:
            return seq[:1]


def _generate_n(predictor: Predictor, prefix: list[str], n: int) -> list[str]:
    generated: list[str] = []
    current = list(prefix)
    for _ in range(n):
        dist = predictor.predict_next(current)
        if not dist:
            break
        next_tok = max(dist, key=lambda x: dist[x])
        generated.append(next_tok)
        current.append(next_tok)
    return generated


# ---------------------------------------------------------------------------
# Per-level evaluator
# ---------------------------------------------------------------------------

def _eval_level(
    predictor: Predictor,
    seqs: list[list[str]],
    is_trace: bool,
    early_exit: bool = False,
) -> tuple[float, float, list]:
    """Return (strict_pct, tok_pct, fail_examples).

    If early_exit=True, stop and return 0% as soon as the first failure is
    found.  Used for train-set evaluation where the pass threshold is 100%:
    the first failure proves the level cannot pass, so all remaining sequences
    are skipped.  This gives up to 100x speedup for failing trace levels.
    """
    n_strict = 0
    n_tok_correct = 0
    n_tok_total = 0
    fail_examples = []

    for seq in seqs:
        truth  = _ground_truth(seq, is_trace)
        prefix = _input_prefix(seq, is_trace)
        gen    = _generate_n(predictor, prefix, len(truth))

        ok = (gen == truth)
        if ok:
            n_strict += 1
        else:
            if len(fail_examples) < 2:
                fail_examples.append((seq, truth, gen))
            if early_exit:
                return 0.0, 0.0, fail_examples

        n = len(truth)
        if n > 0:
            gen_cmp = gen[:n]
            n_tok_correct += sum(g == t for g, t in zip(gen_cmp, truth))
            n_tok_total   += n

    n = len(seqs)
    strict_pct = 100.0 * n_strict / n if n else float("nan")
    tok_pct    = 100.0 * n_tok_correct / n_tok_total if n_tok_total else float("nan")
    return strict_pct, tok_pct, fail_examples


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> bool:
    # ---- Collect per-level train/test splits ----
    per_level: dict[str, tuple[list, list, bool]] = {}
    train_all: list[list[str]] = []

    for name, gen_fn, is_trace in LEVELS:
        train, test = gen_fn()
        train = [_strip_eos(s) for s in train]
        test  = [_strip_eos(s) for s in test]
        per_level[name] = (train, test, is_trace)
        train_all.extend(train)

    n_train_total = len(train_all)
    n_test_total  = sum(len(t) for _, (_, t, _) in per_level.items())
    print(f"Math benchmark -- {n_train_total} train seqs, {n_test_total} test seqs "
          f"(no <eos>), r={R}, k={K}")
    print()

    # ---- Build HankelCount on TRAINING sequences only (for Predictor) ----
    hc = HankelCount(r_max=R)
    hc.update_batch(train_all)

    # ---- Phase 2+4 EM loop (FCA ↔ SEQUITUR) ----
    print("Running EM loop (Phase 2+4)...")
    lattices, mg, dl_history = em_loop(
        corpus=train_all,
        r_levels=[R],
        n_em_max=10,
        lambda_productivity=0.1,
        merge_threshold=0.15,
        min_support=2.0,
        verbose=False,
    )
    lattice = lattices[0]
    print(f"  EM converged in {len(dl_history)} iterations, "
          f"final DL={dl_history[-1]:.1f}, "
          f"n_concepts={len(lattice.concepts)}, "
          f"n_morphisms={len(list(mg.morphisms()))}")
    print()

    # ---- Phase 7: MDL pruning of morphism graph ----
    vocab_size = len(list(hc.vocabulary()))
    mg = mdl_prune(mg, vocab_size=vocab_size, lambda_prune=0.05, min_support=2)
    print(f"After MDL pruning: {len(list(mg.morphisms()))} morphisms retained")
    print()

    # ---- Free category construction (CT_REFERENCE §17) ----
    # Discovers all arithmetic morphisms, NNO structure, adjunctions,
    # natural transformations, and equations from training sequences.
    fc = build_free_category(train_all)
    print(f"Free category: {len(fc.edges)} edges, "
          f"{len(fc.nno_candidates)} NNO operators, "
          f"{len(fc.adjunctions)} adjunctions, "
          f"{len(fc.nat_transforms)} nat-transforms, "
          f"{len(fc.equations)} equations")
    if fc.nno_candidates:
        print(f"  NNO operators : {[n.op for n in fc.nno_candidates]}")
    if fc.adjunctions:
        print(f"  Adjunctions   : {[(a.left_op, a.right_op) for a in fc.adjunctions]}")
    if fc.nat_transforms:
        print(f"  Nat-transforms: {[(n.op, n.kind) for n in fc.nat_transforms]}")

    # ---- MorphismGraph enrichment (routes FC(G) into Level 2+3) ----
    nno, adj, nat = enrich_morphism_graph(fc, mg, lattice, hc)
    print(f"MorphismGraph after enrichment: {len(list(mg.morphisms()))} morphisms")
    print()

    # ---- Process rules / chain rules — stubs returning [] under Option B ----
    process_rules = discover_processes(train_all)
    chain_rules = discover_compose_chains(train_all)
    # (both return [] — Predictor falls through to Level 2+3 for all arithmetic)

    # ---- Phase 8: Lens calibration on training data ----
    # Deduplicate over unique (last_tok, observed_tok) bigrams.
    # compute_gradients only uses prefix[-1], so every occurrence of the same
    # bigram produces identical LensGradient objects — we compute once and
    # scale by frequency (weight field).  Reduces ~28K calls → ~1K unique bigrams.
    print("Running lens calibration (Phase 8)...")
    from collections import Counter
    bigram_counter: Counter = Counter()
    for seq in train_all:
        for i in range(1, len(seq)):
            bigram_counter[(seq[i - 1], seq[i])] += 1

    all_grads: list = []
    for (last_tok, obs_tok), count in bigram_counter.items():
        grads = compute_gradients(mg, [last_tok], obs_tok, r=R)
        for g in grads:
            g.weight = float(count)
        all_grads.extend(grads)

    if all_grads:
        mg = apply_gradients(mg, all_grads, learning_rate=0.05)
    print(f"  Lens calibration: {len(bigram_counter)} unique bigrams, "
          f"{len(all_grads)} gradient signals, 1 batch apply")
    print()

    # ---- Build Predictor ----
    pred = Predictor(
        hankel=hc,
        lattice=lattice,
        morphism_graph=mg,
        process_rules=process_rules,
        chain_rules=chain_rules,
        k_neighbours=K,
        r=R,
        fc=fc,
    )

    # ---- Evaluate per level ----
    header = (
        f"{'Level':<24}  {'#tr':>4}  {'#te':>4}  "
        f"{'train%':>7}  {'test%':>7}  status"
    )
    sep = "-" * (len(header) + 2)
    print(header)
    print(sep)

    all_pass  = True
    memorised = []
    fail_detail: dict[str, list] = {}

    for name, (train, test, is_trace) in per_level.items():
        tr_pct, _, _          = _eval_level(pred, train, is_trace, early_exit=True)
        te_pct, _, te_fails   = _eval_level(pred, test,  is_trace)

        train_ok = tr_pct >= 99.9
        test_ok  = te_pct >= 99.9

        if train_ok and test_ok:
            status = "PASS"
        elif train_ok and not test_ok:
            status = "MEMORISED"
            memorised.append(name)
            all_pass = False
        else:
            status = "FAIL"
            all_pass = False

        print(
            f"{name:<24}  {len(train):>4}  {len(test):>4}  "
            f"{tr_pct:>6.1f}%  {te_pct:>6.1f}%  {status}"
        )

        if not test_ok and te_fails:
            fail_detail[name] = (te_fails, is_trace)

    # ---- Failure details ----
    if fail_detail:
        print()
        print("Test-set failures (up to 2 per level):")
        for name, (examples, is_trace) in fail_detail.items():
            print(f"  [{name}]")
            for seq, truth, gen in examples:
                pfx_str = " ".join(_input_prefix(seq, is_trace))
                print(f"    prefix: {pfx_str[:70]}")
                print(f"    truth:  {truth}")
                print(f"    got:    {gen}")

    print()
    if all_pass:
        print("RESULT: ALL LEVELS PASS (100% train + 100% test generalisation)")
    else:
        if memorised:
            print(f"MEMORISED (recall OK, OOD fails): {', '.join(memorised)}")
        print("RESULT: SOME LEVELS FAILED OR DID NOT GENERALISE")
    return all_pass


if __name__ == "__main__":
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
