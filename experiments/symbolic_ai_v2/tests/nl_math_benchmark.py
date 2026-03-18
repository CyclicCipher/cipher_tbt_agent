"""Natural-language math benchmark — surface-form independence test.

Every digit token '0'..'9' is replaced with its English name before training
AND before evaluation.  The system must pass the benchmark without any special
cases for number words.

Bijection used:
    DIGIT_WORDS = {'0':'zero','1':'one','2':'two','3':'three','4':'four',
                   '5':'five','6':'six','7':'seven','8':'eight','9':'nine'}

What this tests:
    The corpus uses individual digit tokens (so 12 = ['one','two'], 35 =
    ['three','five'] in this mode).  The system must discover:
    - that 'zero','one','two',... form a successor chain
    - that arithmetic over these word-tokens works identically to digit-tokens
    - no code path may check `tok in '0123456789'` or compare number words by
      their English alphabetical order

Pass condition: |nl_accuracy - standard_accuracy| <= 2%  for EVERY level.

Note on "twelve" and "thirty-five":
    Because the corpus tokenizes numbers digit-by-digit, 12 becomes the pair
    ['one','two'] and 35 becomes ['three','five'] — not the compound English
    words 'twelve' and 'thirty-five'.  Compound number words require Phase V
    normalization (unify_surface_forms recognising that ['one','two'] and
    ['twelve'] are the same object in ℕ).  This benchmark tests the simpler
    digit-name bijection, which is achievable now without special cases.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/nl_math_benchmark.py
"""

from __future__ import annotations

import sys
import os
from typing import Optional

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
from experiments.symbolic_ai_v2.ctkg.learning.mdl_prune import (
    mdl_prune,
    semantic_deduplicate,
    compute_storage_policy,
)
from experiments.symbolic_ai_v2.ctkg.learning.lens_update import compute_gradients, apply_gradients
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    discover_processes,
    discover_compose_chains,
    build_free_category,
    enrich_morphism_graph,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor

EOS = "<eos>"
R = 12
K = 1

# ---------------------------------------------------------------------------
# Bijection: digit character → English digit name
# ---------------------------------------------------------------------------

DIGIT_WORDS: dict[str, str] = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
}
DIGIT_WORDS_INV: dict[str, str] = {v: k for k, v in DIGIT_WORDS.items()}


def _nl_token(tok: str) -> str:
    """Replace a single digit token with its English name; others pass through."""
    return DIGIT_WORDS.get(tok, tok)


def _nl_seq(seq: list[str]) -> list[str]:
    """Replace every digit token in a sequence with its English name."""
    return [_nl_token(t) for t in seq]


# ---------------------------------------------------------------------------
# Level registry (identical to math_benchmark.py)
# ---------------------------------------------------------------------------

LEVELS = [
    ("successor",             successor_seqs,             False),
    ("addition",              addition_seqs,              False),
    ("subtraction",           subtraction_seqs,           False),
    ("multiplication",        multiplication_seqs,        False),
    ("powers",                power_seqs,                 False),
    ("linear_eval",           linear_eval_seqs,           False),
    ("derivatives",           derivative_seqs,            False),
    ("integrals",             integral_seqs,              False),
    ("power_trace",           power_trace_seqs,           True),
    ("linear_trace",          linear_eval_trace_seqs,     True),
    ("algebra_trace",         algebra_trace_seqs,         True),
    ("conservation_scenario", conservation_scenario_seqs, True),
    ("bernoulli_trace",       bernoulli_trace_seqs,       True),
    ("derivative_trace",      derivative_trace_seqs,      True),
    ("integral_trace",        integral_trace_seqs,        True),
]

# ---------------------------------------------------------------------------
# Helpers (copied from math_benchmark.py)
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


def _eval_level(
    predictor: Predictor,
    seqs: list[list[str]],
    is_trace: bool,
    early_exit: bool = False,
) -> tuple[float, float, list]:
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
# Build predictor
# ---------------------------------------------------------------------------

def _build_predictor(train_all: list[list[str]], raw_corpus: Optional[list[list[str]]] = None) -> Predictor:
    hc = HankelCount(r_max=R)
    hc.update_batch(train_all)

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
    print(f"  EM: {len(dl_history)} iters, DL={dl_history[-1]:.1f}, "
          f"concepts={len(lattice.concepts)}, "
          f"morphisms={len(list(mg.morphisms()))}")

    vocab_size = len(list(hc.vocabulary()))
    mg = mdl_prune(mg, vocab_size=vocab_size, lambda_prune=0.05, min_support=2)
    mg_before_dedup = len(list(mg.morphisms()))
    mg = semantic_deduplicate(mg, rules=[])
    print(f"  Semantic dedup: {mg_before_dedup} -> {len(list(mg.morphisms()))} morphisms")
    storage_policy = compute_storage_policy(mg, rules=[], k_steps=5)
    n_store = sum(1 for v in storage_policy.values() if v)
    print(f"  Storage policy: {n_store}/{len(storage_policy)} types must store")

    fc = build_free_category(train_all)
    print(f"  Free category: {len(fc.edges)} edges, "
          f"{len(fc.nno_candidates)} NNO, "
          f"{len(fc.adjunctions)} adj, "
          f"{len(fc.nat_transforms)} nat-trans")
    enrich_morphism_graph(fc, mg, lattice, hc)
    print(f"  Morphisms after enrichment: {len(list(mg.morphisms()))}")

    process_rules = discover_processes(train_all)
    chain_rules = discover_compose_chains(train_all)

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

    return Predictor(
        hankel=hc,
        lattice=lattice,
        morphism_graph=mg,
        process_rules=process_rules,
        chain_rules=chain_rules,
        k_neighbours=K,
        r=R,
        fc=fc,
        raw_corpus=raw_corpus if raw_corpus is not None else train_all,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> bool:
    # ---- Collect raw splits ----
    per_level_raw: dict[str, tuple[list, list, bool]] = {}
    for name, gen_fn, is_trace in LEVELS:
        train, test = gen_fn()
        train = [_strip_eos(s) for s in train]
        test  = [_strip_eos(s) for s in test]
        per_level_raw[name] = (train, test, is_trace)

    train_all_std = []
    for name, (train, _, _) in per_level_raw.items():
        train_all_std.extend(train)

    # ---- Apply NL bijection ----
    per_level_nl: dict[str, tuple[list, list, bool]] = {}
    for name, (train, test, is_trace) in per_level_raw.items():
        per_level_nl[name] = (
            [_nl_seq(s) for s in train],
            [_nl_seq(s) for s in test],
            is_trace,
        )
    train_all_nl = []
    for name, (train, _, _) in per_level_nl.items():
        train_all_nl.extend(train)

    n_train = len(train_all_std)
    n_test  = sum(len(t) for _, (_, t, _) in per_level_raw.items())
    print(f"Natural-language math benchmark — {n_train} train, {n_test} test, r={R}, k={K}")
    print(f"Bijection: digit token -> English name (e.g. '3' -> 'three', '1','2' -> 'one','two')")
    print()

    # ---- Standard predictor ----
    print("Building STANDARD predictor...")
    pred_std = _build_predictor(train_all_std)
    print()

    # ---- NL predictor ----
    print("Building NATURAL-LANGUAGE predictor...")
    pred_nl = _build_predictor(train_all_nl)
    print()

    # ---- Evaluate both ----
    TOLERANCE = 2.0  # ±2% is acceptable drift

    header = (
        f"{'Level':<24}  {'#tr':>4}  {'#te':>4}  "
        f"{'tr_std%':>7}  {'tr_nl%':>7}  "
        f"{'te_std%':>7}  {'te_nl%':>7}  {'delta':>7}  result"
    )
    sep = "-" * (len(header) + 2)
    print(header)
    print(sep)

    all_compliant = True
    deltas: dict[str, float] = {}

    for name, (train_std, test_std, is_trace) in per_level_raw.items():
        train_nl, test_nl, _ = per_level_nl[name]

        tr_std, _, _ = _eval_level(pred_std, train_std, is_trace)
        tr_nl,  _, _ = _eval_level(pred_nl,  train_nl,  is_trace)
        te_std, _, _ = _eval_level(pred_std, test_std,  is_trace)
        te_nl,  _, _ = _eval_level(pred_nl,  test_nl,   is_trace)

        delta = te_nl - te_std
        deltas[name] = delta
        compliant = abs(delta) <= TOLERANCE
        if not compliant:
            all_compliant = False

        result = "OK" if compliant else f"DRIFT({delta:+.1f}%)"
        print(
            f"{name:<24}  {len(train_std):>4}  {len(test_std):>4}  "
            f"{tr_std:>6.1f}%  {tr_nl:>6.1f}%  "
            f"{te_std:>6.1f}%  {te_nl:>6.1f}%  {delta:>+6.1f}%  {result}"
        )

    print()
    if all_compliant:
        print("RESULT: NL-COMPLIANT — all levels within +-2% under English digit names")
    else:
        failing = [n for n, d in deltas.items() if abs(d) > TOLERANCE]
        print(f"RESULT: NOT COMPLIANT — surface-form dependency in: {', '.join(failing)}")
        print("        These levels rely on token numeric identity, not structural discovery.")
        print()
        print("Note: compound English numbers (twelve, thirty-five) require Phase V")
        print("      normalization (unify_surface_forms) — not yet implemented.")

    # ---- Knowledge graph comparison ----
    _compare_kgs(pred_std, pred_nl)

    return all_compliant


def _compare_kgs(pred_std: object, pred_nl: object) -> None:
    """Print a side-by-side comparison of the two knowledge graphs."""
    from collections import Counter

    print()
    print("=== Knowledge Graph Comparison: STANDARD vs NL ===")

    for label, pred in [("STANDARD", pred_std), ("NL", pred_nl)]:
        mg = pred._morphism_graph
        lattice = pred._lattice
        objs = list(mg.objects())
        morphs = list(mg.morphisms(include_identity=False))
        type_counts: Counter = Counter(m.morph_type for m in morphs)
        print(f"\n[{label}] objects={len(objs)}  morphisms={len(morphs)}"
              f"  types={len(type_counts)}  concepts={len(lattice.concepts)}")
        top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:10]
        for mt, cnt in top_types:
            print(f"  {mt:<40} count={cnt}")

    # Structural comparison: do both KGs have the same morphism type set?
    mg_std = pred_std._morphism_graph
    mg_nl  = pred_nl._morphism_graph
    types_std = {m.morph_type for m in mg_std.morphisms(include_identity=False)}
    types_nl  = {m.morph_type for m in mg_nl.morphisms(include_identity=False)}
    shared  = types_std & types_nl
    std_only = types_std - types_nl
    nl_only  = types_nl  - types_std
    print(f"\nShared morph types : {len(shared)}")
    print(f"STANDARD-only      : {len(std_only)}")
    print(f"NL-only            : {len(nl_only)}")
    if std_only:
        for t in sorted(std_only)[:5]:
            print(f"  STD: {t}")
    if nl_only:
        for t in sorted(nl_only)[:5]:
            print(f"  NL:  {t}")
    print()


if __name__ == "__main__":
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
