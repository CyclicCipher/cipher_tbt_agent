"""Anonymized math benchmark — bitter-lesson compliance test.

The *token anonymization trick*: apply a fixed bijection to all digit tokens
before training AND before evaluation.  If the pipeline's accuracy on the
anonymized corpus is within ±2% of the standard corpus, the system has discovered
arithmetic structure from token relationships alone — it cannot be relying on the
numeric surface form of '0'..'9'.

Bijection used:
    DIGIT_PERM = {'0':'g','1':'b','2':'h','3':'e','4':'c',
                  '5':'f','6':'a','7':'i','8':'d','9':'j'}

Structural tokens (succ, pred, add, mul, pow, eq, <eos>, carry, step, ans,
a0..a9, x, lin, alg, cons, bern, deriv, int_, etc.) are left unchanged —
they are role labels, not values.

Pass condition:  |anon_accuracy - standard_accuracy| <= 2%  for EVERY level.

Usage:
    ./venv/Scripts/python.exe experiments/symbolic_ai_v2/tests/anon_math_benchmark.py
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
    linear_solve_seqs,
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
from experiments.symbolic_ai_v2.ctkg.learning.mdl_prune import (
    semantic_deduplicate,
    compute_storage_policy,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor

EOS = "<eos>"
R = 12
K = 1

# ---------------------------------------------------------------------------
# Bijection
# ---------------------------------------------------------------------------

DIGIT_PERM: dict[str, str] = {
    '0': 'g', '1': 'b', '2': 'h', '3': 'e', '4': 'c',
    '5': 'f', '6': 'a', '7': 'i', '8': 'd', '9': 'j',
}
# Inverse (for display only)
DIGIT_PERM_INV: dict[str, str] = {v: k for k, v in DIGIT_PERM.items()}


def _anon_token(tok: str) -> str:
    """Remap a single token through DIGIT_PERM; non-digit tokens pass through."""
    return DIGIT_PERM.get(tok, tok)


def _anon_seq(seq: list[str]) -> list[str]:
    """Remap every token in a sequence."""
    return [_anon_token(t) for t in seq]


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
    ("linear_solve",          linear_solve_seqs,          False),
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
# Build and run ONE predictor on anonymized corpus
# ---------------------------------------------------------------------------

def _build_predictor(
    train_all: list[list[str]],
) -> Predictor:
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
    mg = semantic_deduplicate(mg, rules=[])
    _ = compute_storage_policy(mg, rules=[], k_steps=5)

    fc = build_free_category(train_all)
    print(f"  FC: {len(fc.edges)} edges, {len(fc.nno_candidates)} NNO, "
          f"{len(fc.adjunctions)} adj")
    enrich_morphism_graph(fc, mg, lattice, hc)

    process_rules = discover_processes(train_all)
    chain_rules = discover_compose_chains(train_all)
    print(f"  Process rules: {len(process_rules)}, chain rules: {len(chain_rules)}")

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

    # ---- Anonymize ----
    per_level_anon: dict[str, tuple[list, list, bool]] = {}
    for name, (train, test, is_trace) in per_level_raw.items():
        per_level_anon[name] = (
            [_anon_seq(s) for s in train],
            [_anon_seq(s) for s in test],
            is_trace,
        )
    train_all_anon = []
    for name, (train, _, _) in per_level_anon.items():
        train_all_anon.extend(train)

    n_train = len(train_all_std)
    n_test  = sum(len(t) for _, (_, t, _) in per_level_raw.items())
    print(f"Anonymized math benchmark — {n_train} train, {n_test} test, r={R}, k={K}")
    print(f"Bijection: {DIGIT_PERM}")
    print()

    # ---- Standard predictor ----
    print("Building STANDARD predictor...")
    pred_std = _build_predictor(train_all_std)
    print()

    # ---- Anonymized predictor ----
    print("Building ANONYMIZED predictor...")
    pred_anon = _build_predictor(train_all_anon)
    print()

    # ---- Evaluate both ----
    TOLERANCE = 2.0  # ±2% is acceptable drift

    header = (
        f"{'Level':<24}  {'#tr':>4}  {'#te':>4}  "
        f"{'std%':>7}  {'anon%':>7}  {'delta':>7}  result"
    )
    sep = "-" * (len(header) + 2)
    print(header)
    print(sep)

    all_compliant = True
    deltas: dict[str, float] = {}

    for name, (train_std, test_std, is_trace) in per_level_raw.items():
        train_anon, test_anon, _ = per_level_anon[name]

        _, _, _ = _eval_level(pred_std,  train_std,  is_trace, early_exit=True)
        te_std,  _, _ = _eval_level(pred_std,  test_std,  is_trace)
        te_anon, _, _ = _eval_level(pred_anon, test_anon, is_trace)

        delta = te_anon - te_std
        deltas[name] = delta
        compliant = abs(delta) <= TOLERANCE
        if not compliant:
            all_compliant = False

        result = "OK" if compliant else f"DRIFT({delta:+.1f}%)"
        print(
            f"{name:<24}  {len(train_std):>4}  {len(test_std):>4}  "
            f"{te_std:>6.1f}%  {te_anon:>6.1f}%  {delta:>+6.1f}%  {result}"
        )

    print()
    if all_compliant:
        print("RESULT: BITTER-LESSON COMPLIANT — all levels within ±2% under anonymization")
    else:
        failing = [n for n, d in deltas.items() if abs(d) > TOLERANCE]
        print(f"RESULT: NOT COMPLIANT — surface-form dependency detected in: {', '.join(failing)}")
        print("        These levels rely on token numeric identity, not structural discovery.")
    return all_compliant


if __name__ == "__main__":
    ok = run_benchmark()
    sys.exit(0 if ok else 1)
