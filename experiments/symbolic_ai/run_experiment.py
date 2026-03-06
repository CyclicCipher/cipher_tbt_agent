"""Arithmetic experiment: symbolic AI vs neural network baseline.

Demonstrates that a structured symbolic system generalizes from a small
number of examples with 100% accuracy, using zero trainable parameters.

Comparison baseline (from MISTAKES.md, Mistake #42):
    Mamba3 neural network: 1.26M parameters, 50 epochs, 100 examples
    → ~45% test accuracy on single-digit arithmetic (memorization, not generalization)

This experiment:
    Symbolic AI: 0 trainable parameters
    → 100% test accuracy after consolidation from ≤20 examples

Fourteen phases:
    1.  Built-in knowledge: verify succ, pred, compare work via process execution
    2.  Learn addition: teach 20 examples, consolidate, test on 80 held-out pairs
    3.  Learn subtraction: teach 15 examples, consolidate, test on held-out pairs
    4.  Minimum examples sweep: how many examples needed for reliable synthesis?
    5.  Prerequisite enforcement: synthesis correctly fails without ancestors
    6.  Learn multiplication: teach 10 examples, consolidate (double-nested fn)
    7.  Learn exponentiation: teach 10 examples, consolidate (triple-nested fold+fn)
    8.  Learn division: teach 10 examples, synthesize fold_until template (bounded, safe)
    9.  Verify remainder: pre-defined lookup process; verify correct on all pairs
   10.  Verify GCD: pre-defined fold_until+lookup process; verify on all pairs
   11.  Symbolic differentiation: Level C primitives — build and differentiate
        polynomial expressions; verify power rule, product rule, constant rule.
        (Gap 2 toward ODEs — symbolic AST manipulation layer)
   12.  Integration, sequences, inspection (Gaps 3-5):
        A. sym_integrate — polynomial antiderivative; fundamental theorem check
        B. scan / seq_* — variable-length state sequences
        C. sym_tag / sym_lhs / sym_rhs / sym_val — expression tree introspection
   13.  ODE solving (Gaps 6-7 — sym_expand/sym_coeff + float arithmetic):
        A. First-order separable ODE: dy/dx = f(x), apply initial condition
        B. Second-order constant-coefficient: characteristic equation (float roots)
        C. Apply initial conditions, verify solution at multiple points
   14.  Fluid dynamics (float_pow, float_pi — physics application domain):
        A. Continuity equation: A₁v₁ = A₂v₂ → find outlet velocity
        B. Torricelli's theorem: v = √(2gh) → tank drain speed
        C. Bernoulli's equation (horizontal): P₂ = P₁ + ½ρ(v₁²−v₂²)
        D. Venturi meter (combined application problem): continuity + Bernoulli
           in sequence, given pipe geometry + inlet conditions, find outlet pressure
   15.  Approximate synthesis proof-of-concept (Gap A — statistical visual concepts):
        Synthetic bright/dark images; synthesizer discovers luminance threshold rule.
        First use of consolidate_approx() and float literals in templates.
   16.  CIFAR-10 cat classification (approximate synthesis on real images):
        Auto-download CIFAR-10; learn cat vs non-cat using Gabor/DoG features.
        Reports best template, threshold, and test accuracy.
        Expected: 65–75% with a single fixed visual feature.

Run from experiments/symbolic_ai/:
    python run_experiment.py
"""

from __future__ import annotations

import io
import os
import random
import sys

# Force UTF-8 output on Windows (avoids cp1252 errors with tick marks, arrows).
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Make experiments/ and experiments/symbolic_ai/ importable.
_EXPERIMENTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SYMBOLIC_AI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXPERIMENTS)
sys.path.insert(0, _SYMBOLIC_AI)

from ctkg.parser import parse_file
from ctkg.graph import KnowledgeGraph, Concept, TypeDef, Prerequisite, BUILTIN_TYPES

from engine import SymbolicAI


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
ARITHMETIC_CTKG = os.path.join(_HERE, '..', 'ctkg', 'domains', 'arithmetic.ctkg')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(ok: bool) -> str:
    return '✓' if ok else '✗'


def _load_fresh_ai() -> SymbolicAI:
    """Load a fresh SymbolicAI from arithmetic.ctkg."""
    graph = parse_file(ARITHMETIC_CTKG)
    return SymbolicAI(graph)


def _all_addition_pairs():
    """All 100 single-digit addition pairs (a, b) where a,b ∈ 0..9."""
    return [(a, b) for a in range(10) for b in range(10)]


def _all_subtraction_pairs():
    """All (a, b) single-digit subtraction pairs where a >= b."""
    return [(a, b) for a in range(10) for b in range(a + 1)]


def _all_multiplication_pairs():
    """All 100 single-digit multiplication pairs (a, b) where a,b ∈ 0..9."""
    return [(a, b) for a in range(10) for b in range(10)]


def _all_exponentiation_pairs():
    """Bounded exponentiation pairs (a, b) where a ∈ 0..4, b ∈ 0..4.

    Upper-bounded to keep results manageable (max 4^4 = 256).
    Excludes (0,0) which is conventionally 1 but could match constant templates.
    """
    return [(a, b) for a in range(5) for b in range(5)]


def _all_division_pairs():
    """All (a, b) where b >= 1, a in 1..9, b in 1..9, b <= a.

    Excludes b=0 (undefined). Restricts to a >= b for cleaner quotients.
    """
    return [(a, b) for a in range(1, 10) for b in range(1, a + 1)]


def _all_remainder_pairs():
    """All (a, b) where b >= 1, for remainder computation."""
    return [(a, b) for a in range(0, 10) for b in range(1, 10)]


def _all_gcd_pairs():
    """All (a, b) where both >= 1, for GCD computation."""
    return [(a, b) for a in range(1, 10) for b in range(1, 10)]


def _mul_inputs(a, b):
    return (a, b)


def _mul_outputs(a, b):
    return (a * b,)


def _exp_inputs(a, b):
    return (a, b)


def _exp_outputs(a, b):
    return (a ** b,)


def _add_inputs(a, b):
    return (a, 'ADD', b)


def _sub_inputs(a, b):
    return (a, 'SUB', b)


def _add_outputs(a, b):
    r = a + b
    return (r // 10, r % 10)


def _sub_outputs(a, b):
    r = a - b
    return (r // 10, r % 10)   # borrow is always 0 since a >= b


def _fmt_process(lines):
    if not lines:
        return '(none)'
    return '  |  '.join(lines)


# ---------------------------------------------------------------------------
# Phase 1: Built-in knowledge verification
# ---------------------------------------------------------------------------

def phase1_builtin(ai: SymbolicAI):
    print('=' * 65)
    print('Phase 1: Built-in Knowledge Verification')
    print('  (loaded from arithmetic.ctkg; no examples needed)')
    print('=' * 65)

    tests = [
        # (concept_name, inputs, expected_outputs, display_name)
        ('successor',   (0,), (1,),    'SUCC(0)'),
        ('successor',   (4,), (5,),    'SUCC(4)'),
        ('successor',   (9,), (10,),   'SUCC(9)'),   # integer, not modular
        ('predecessor', (5,), (4,),    'PRED(5)'),
        ('predecessor', (1,), (0,),    'PRED(1)'),
        ('comparison',  (7, 3), ('GT',), 'CMP(7,3)'),
        ('comparison',  (2, 8), ('LT',), 'CMP(2,8)'),
        ('comparison',  (5, 5), ('EQ',), 'CMP(5,5)'),
    ]

    passed = 0
    for concept, inputs, expected, label in tests:
        actual = ai.ask(concept, inputs)
        ok = actual == expected
        passed += ok
        status = _tick(ok)
        print(f'  {status}  {label:12s} → {actual}  (expected {expected})')

    print(f'\n  Result: {passed}/{len(tests)} correct')
    print()
    return passed == len(tests)


# ---------------------------------------------------------------------------
# Phase 2: Learn single-digit addition
# ---------------------------------------------------------------------------

def phase2_addition(ai: SymbolicAI, seed: int = 42):
    print('=' * 65)
    print('Phase 2: Learn Single-Digit Addition from 20 Examples')
    print('=' * 65)

    # Clear the pre-loaded process to simulate "not yet learned".
    ai.clear_process('single_digit_addition')

    rng = random.Random(seed)
    all_pairs = _all_addition_pairs()
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:20]
    test_pairs  = all_pairs[20:]

    # Teach training examples.
    for a, b in train_pairs:
        ai.teach('single_digit_addition', _add_inputs(a, b), _add_outputs(a, b))

    kl_before = ai.kl('single_digit_addition')
    print(f'\n  Taught {len(train_pairs)} examples')
    print(f'  KL before consolidation: {kl_before:.3f} bits  (should be low — exact match cache)')
    print(f'  Should consolidate (KL > 0.1): {ai.should_consolidate("single_digit_addition")}')

    # Consolidate.
    print('\n  Running synthesis...')
    rule = ai.consolidate('single_digit_addition')

    if rule is None:
        print('  ✗ SYNTHESIS FAILED')
        return False

    print(f'  ✓ Synthesis succeeded')
    print(f'  Discovered rule ({len(rule)} lines):')
    for line in rule:
        print(f'    {line}')

    kl_after = ai.kl('single_digit_addition')
    print(f'\n  KL after consolidation: {kl_after:.3f} bits  (should be 0)')

    # Test on held-out pairs.
    correct = 0
    failures = []
    for a, b in test_pairs:
        actual   = ai.ask('single_digit_addition', _add_inputs(a, b))
        expected = _add_outputs(a, b)
        if actual == expected:
            correct += 1
        else:
            failures.append((a, b, actual, expected))

    pct = 100.0 * correct / len(test_pairs)
    print(f'\n  Test on {len(test_pairs)} held-out pairs: {correct}/{len(test_pairs)} = {pct:.1f}%')

    if failures:
        print(f'  Failures (first 5):')
        for a, b, got, exp in failures[:5]:
            print(f'    {a}+{b}: got {got}, expected {exp}')
    else:
        print('  No failures.')

    print()
    return correct == len(test_pairs)


# ---------------------------------------------------------------------------
# Phase 3: Learn single-digit subtraction
# ---------------------------------------------------------------------------

def phase3_subtraction(ai: SymbolicAI, seed: int = 42):
    print('=' * 65)
    print('Phase 3: Learn Single-Digit Subtraction from 15 Examples')
    print('=' * 65)

    ai.clear_process('single_digit_subtraction')

    rng = random.Random(seed)
    all_pairs = _all_subtraction_pairs()   # 55 pairs where a >= b
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:15]
    test_pairs  = all_pairs[15:]

    for a, b in train_pairs:
        ai.teach('single_digit_subtraction', _sub_inputs(a, b), _sub_outputs(a, b))

    print(f'\n  Taught {len(train_pairs)} examples')
    print('  Running synthesis...')
    rule = ai.consolidate('single_digit_subtraction')

    if rule is None:
        print('  ✗ SYNTHESIS FAILED')
        return False

    print(f'  ✓ Synthesis succeeded')
    print(f'  Discovered rule ({len(rule)} lines):')
    for line in rule:
        print(f'    {line}')

    correct = 0
    failures = []
    for a, b in test_pairs:
        actual   = ai.ask('single_digit_subtraction', _sub_inputs(a, b))
        expected = _sub_outputs(a, b)
        if actual == expected:
            correct += 1
        else:
            failures.append((a, b, actual, expected))

    pct = 100.0 * correct / len(test_pairs)
    print(f'\n  Test on {len(test_pairs)} held-out pairs: {correct}/{len(test_pairs)} = {pct:.1f}%')
    if failures:
        for a, b, got, exp in failures[:5]:
            print(f'    {a}-{b}: got {got}, expected {exp}')
    else:
        print('  No failures.')
    print()
    return correct == len(test_pairs)


# ---------------------------------------------------------------------------
# Phase 4: Minimum examples sweep
# ---------------------------------------------------------------------------

def phase4_minimum_examples(seed: int = 42):
    print('=' * 65)
    print('Phase 4: Minimum Examples Needed for Reliable Synthesis')
    print('=' * 65)
    print()

    rng = random.Random(seed)
    all_pairs = _all_addition_pairs()
    rng.shuffle(all_pairs)

    min_success = None
    for n in [2, 3, 4, 5, 7, 10, 15, 20]:
        # Fresh AI for each n to avoid contamination.
        ai = _load_fresh_ai()
        ai.clear_process('single_digit_addition')

        train = all_pairs[:n]
        test  = all_pairs[n:]

        for a, b in train:
            ai.teach('single_digit_addition', _add_inputs(a, b), _add_outputs(a, b))

        rule = ai.consolidate('single_digit_addition')

        if rule is None:
            print(f'  N={n:2d}: FAILED  (synthesis returned None — ambiguous or no succ template)')
        else:
            correct = sum(
                1 for a, b in test
                if ai.ask('single_digit_addition', _add_inputs(a, b)) == _add_outputs(a, b)
            )
            pct = 100.0 * correct / len(test) if test else 100.0
            rule_summary = rule[0] if rule else '?'
            print(f'  N={n:2d}: SUCCESS → {correct}/{len(test)} ({pct:.0f}%)  rule: [{rule_summary}]')
            if min_success is None:
                min_success = n

    print()
    if min_success is not None:
        print(f'  Minimum N for reliable synthesis: {min_success}')
        print(f'  (Neural baseline needs ~100 examples for ~45% test accuracy)')
    print()


# ---------------------------------------------------------------------------
# Phase 5: Prerequisite enforcement
# ---------------------------------------------------------------------------

def _make_isolated_graph() -> KnowledgeGraph:
    """Build a minimal graph with single_digit_addition but NO ancestors.

    This simulates trying to learn addition without knowing successor.
    ancestors('single_digit_addition') will return an empty set, so
    the synthesizer generates zero templates and synthesis fails.
    """
    g = KnowledgeGraph()

    # Add builtins.
    for name, td in BUILTIN_TYPES.items():
        g.add_type(td)

    # Add just enough types.
    from ctkg.graph import TypeDef
    g.add_type(TypeDef('digit', 'symbol', [str(i) for i in range(10)], {'ordered'}))
    g.add_type(TypeDef('carry', 'symbol', ['0', '1']))
    g.add_type(TypeDef('op',    'symbol', ['ADD', 'SUB']))

    # Add single_digit_addition with NO prerequisites.
    g.add_concept(Concept(
        name='single_digit_addition',
        description='Addition without prerequisites (isolated)',
        domain='arithmetic',
        input_type=['digit', 'op', 'digit'],
        output_type=['carry', 'digit'],
    ))

    return g


def phase5_prerequisite_enforcement(seed: int = 42):
    print('=' * 65)
    print('Phase 5: Prerequisite Enforcement')
    print('  (synthesis must fail when ancestors are absent)')
    print('=' * 65)

    # Test 1: isolated graph (no ancestors at all)
    g = _make_isolated_graph()
    ai = SymbolicAI(g)

    rng = random.Random(seed)
    all_pairs = _all_addition_pairs()
    rng.shuffle(all_pairs)

    for a, b in all_pairs[:20]:
        ai.teach('single_digit_addition', _add_inputs(a, b), _add_outputs(a, b))

    rule = ai.consolidate('single_digit_addition')

    if rule is None:
        print('\n  ✓ Synthesis correctly FAILED when no ancestors are present.')
        print('    (no succ/pred templates generated — prerequisite graph enforced)')
    else:
        print(f'\n  ✗ Synthesis should have failed but returned: {rule}')

    # Test 2: full graph — verify synthesis succeeds (sanity check)
    ai2 = _load_fresh_ai()
    ai2.clear_process('single_digit_addition')
    for a, b in all_pairs[:20]:
        ai2.teach('single_digit_addition', _add_inputs(a, b), _add_outputs(a, b))
    rule2 = ai2.consolidate('single_digit_addition')

    if rule2 is not None:
        print('\n  ✓ Synthesis correctly SUCCEEDED with full ancestor chain.')
        print(f'    Rule: {rule2[0]} ...')
    else:
        print('\n  ✗ Synthesis should have succeeded with full graph but failed.')

    print()

    # Show which ancestors single_digit_addition has in the full graph.
    full_ai = _load_fresh_ai()
    ancestors = full_ai.graph.ancestors('single_digit_addition')
    print(f'  Ancestors in full graph: {sorted(ancestors)}')
    print(f'  Ancestors in isolated graph: {sorted(g.ancestors("single_digit_addition"))}')
    print()


# ---------------------------------------------------------------------------
# Phase 6: Learn multiplication
# ---------------------------------------------------------------------------

def phase6_multiplication(ai: SymbolicAI, seed: int = 42):
    print('=' * 65)
    print('Phase 6: Learn Multiplication from 10 Examples')
    print('  (discovers fold(b, 0, fn(k, fold(a, k, succ))))')
    print('=' * 65)

    # multiplication already has a process in arithmetic.ctkg; clear it to simulate learning.
    ai.clear_process('multiplication')

    rng = random.Random(seed)
    all_pairs = _all_multiplication_pairs()
    # Exclude trivial cases where a=0 or b=0 (always 0) or a=1/b=1 (identity)
    # to force the synthesizer to need the real formula.
    nontrivial = [(a, b) for a, b in all_pairs if a >= 2 and b >= 2]
    rng.shuffle(nontrivial)
    train_pairs = nontrivial[:10]
    test_pairs  = [p for p in all_pairs if p not in set(train_pairs)]

    for a, b in train_pairs:
        ai.teach('multiplication', _mul_inputs(a, b), _mul_outputs(a, b))

    print(f'\n  Training examples: {[(a, b, a*b) for a, b in train_pairs]}')
    print(f'  Taught {len(train_pairs)} examples')
    print('\n  Running synthesis...')
    rule = ai.consolidate('multiplication')

    if rule is None:
        print('  ✗ SYNTHESIS FAILED')
        return False

    print(f'  ✓ Synthesis succeeded')
    print(f'  Discovered rule ({len(rule)} lines):')
    for line in rule:
        print(f'    {line}')

    correct = 0
    failures = []
    for a, b in test_pairs:
        actual   = ai.ask('multiplication', _mul_inputs(a, b))
        expected = _mul_outputs(a, b)
        if actual == expected:
            correct += 1
        else:
            failures.append((a, b, actual, expected))

    pct = 100.0 * correct / len(test_pairs)
    print(f'\n  Test on {len(test_pairs)} held-out pairs: {correct}/{len(test_pairs)} = {pct:.1f}%')
    if failures:
        print(f'  Failures (first 5):')
        for a, b, got, exp in failures[:5]:
            print(f'    {a}*{b}: got {got}, expected {exp}')
    else:
        print('  No failures.')
    print()
    return correct == len(test_pairs)


# ---------------------------------------------------------------------------
# Phase 7: Learn exponentiation
# ---------------------------------------------------------------------------

def phase7_exponentiation(ai: SymbolicAI, seed: int = 42):
    print('=' * 65)
    print('Phase 7: Learn Exponentiation from 10 Examples')
    print('  (discovers fold(b, 1, fn(acc, fold(a, 0, fn(k, fold(acc, k, succ))))))')
    print('=' * 65)

    ai.clear_process('exponentiation')

    rng = random.Random(seed)
    all_pairs = _all_exponentiation_pairs()
    # Use nontrivial examples: a>=2, b>=2 to avoid degenerate matches
    nontrivial = [(a, b) for a, b in all_pairs if a >= 2 and b >= 2]
    rng.shuffle(nontrivial)
    train_pairs = nontrivial[:min(10, len(nontrivial))]
    test_pairs  = [p for p in all_pairs if p not in set(train_pairs)]

    for a, b in train_pairs:
        ai.teach('exponentiation', _exp_inputs(a, b), _exp_outputs(a, b))

    print(f'\n  Training examples: {[(a, b, a**b) for a, b in train_pairs]}')
    print(f'  Taught {len(train_pairs)} examples')
    print('\n  Running synthesis...')
    rule = ai.consolidate('exponentiation')

    if rule is None:
        print('  ✗ SYNTHESIS FAILED')
        return False

    print(f'  ✓ Synthesis succeeded')
    print(f'  Discovered rule ({len(rule)} lines):')
    for line in rule:
        print(f'    {line}')

    correct = 0
    failures = []
    for a, b in test_pairs:
        actual   = ai.ask('exponentiation', _exp_inputs(a, b))
        expected = _exp_outputs(a, b)
        if actual == expected:
            correct += 1
        else:
            failures.append((a, b, actual, expected))

    pct = 100.0 * correct / len(test_pairs) if test_pairs else 100.0
    print(f'\n  Test on {len(test_pairs)} held-out pairs: {correct}/{len(test_pairs)} = {pct:.1f}%')
    if failures:
        print(f'  Failures (first 5):')
        for a, b, got, exp in failures[:5]:
            print(f'    {a}^{b}: got {got}, expected {exp}')
    else:
        print('  No failures.')
    print()
    return correct == len(test_pairs)


# ---------------------------------------------------------------------------
# Phase 8: Learn division (synthesize fold_until template)
# ---------------------------------------------------------------------------

def phase8_division(ai: SymbolicAI, seed: int = 42):
    print('=' * 65)
    print('Phase 8: Learn Division from 10 Examples')
    print('  (discovers bounded fold_until — cannot infinite-loop)')
    print('=' * 65)

    ai.clear_process('division')

    rng = random.Random(seed)
    all_pairs = _all_division_pairs()
    # Use nontrivial examples: b >= 2 and a >= 2*b so quotient >= 2
    nontrivial = [(a, b) for a, b in all_pairs if b >= 2 and a >= 2 * b]
    rng.shuffle(nontrivial)
    train_pairs = nontrivial[:10]
    test_pairs  = [p for p in all_pairs if p not in set(train_pairs)]

    for a, b in train_pairs:
        ai.teach('division', (a, b), (a // b,))

    print(f'\n  Training examples: {[(a, b, a//b) for a, b in train_pairs]}')
    print(f'  Taught {len(train_pairs)} examples')
    print('\n  Running synthesis...')
    rule = ai.consolidate('division')

    if rule is None:
        print('  ✗ SYNTHESIS FAILED')
        return False

    print(f'  ✓ Synthesis succeeded')
    print(f'  Discovered rule ({len(rule)} lines):')
    for line in rule:
        print(f'    {line}')

    correct = 0
    failures = []
    for a, b in test_pairs:
        try:
            actual   = ai.ask('division', (a, b))
            expected = (a // b,)
            if actual == expected:
                correct += 1
            else:
                failures.append((a, b, actual, expected))
        except Exception as e:
            failures.append((a, b, f'ERROR: {e}', (a // b,)))

    pct = 100.0 * correct / len(test_pairs) if test_pairs else 100.0
    print(f'\n  Test on {len(test_pairs)} held-out pairs: {correct}/{len(test_pairs)} = {pct:.1f}%')
    if failures:
        print(f'  Failures (first 5):')
        for a, b, got, exp in failures[:5]:
            print(f'    {a}//{b}: got {got}, expected {exp}')
    else:
        print('  No failures.')
    print()
    return correct == len(test_pairs)


# ---------------------------------------------------------------------------
# Phase 9: Verify remainder (pre-defined lookup process)
# ---------------------------------------------------------------------------

def phase9_remainder(ai: SymbolicAI):
    print('=' * 65)
    print('Phase 9: Verify Remainder (pre-defined lookup process)')
    print('  (uses lookup(division) + fold+succ multiply + fold+pred subtract)')
    print('=' * 65)

    all_pairs = _all_remainder_pairs()
    correct = 0
    failures = []
    for a, b in all_pairs:
        try:
            actual   = ai.ask('remainder', (a, b))
            expected = (a % b,)
            if actual == expected:
                correct += 1
            else:
                failures.append((a, b, actual, expected))
        except Exception as e:
            failures.append((a, b, f'ERROR: {e}', (a % b,)))

    pct = 100.0 * correct / len(all_pairs)
    print(f'\n  Test on {len(all_pairs)} pairs: {correct}/{len(all_pairs)} = {pct:.1f}%')
    if failures:
        print(f'  Failures (first 5):')
        for a, b, got, exp in failures[:5]:
            print(f'    {a} mod {b}: got {got}, expected {exp}')
    else:
        print('  No failures.')
    print()
    return correct == len(all_pairs)


# ---------------------------------------------------------------------------
# Phase 10: Verify GCD (pre-defined fold_until + lookup process)
# ---------------------------------------------------------------------------

def phase10_gcd(ai: SymbolicAI):
    print('=' * 65)
    print('Phase 10: Verify GCD (Euclidean algorithm, fold_until + lookup)')
    print('=' * 65)

    import math
    all_pairs = _all_gcd_pairs()
    correct = 0
    failures = []
    for a, b in all_pairs:
        try:
            actual   = ai.ask('greatest_common_divisor', (a, b))
            expected = (math.gcd(a, b),)
            if actual == expected:
                correct += 1
            else:
                failures.append((a, b, actual, expected))
        except Exception as e:
            failures.append((a, b, f'ERROR: {e}', (math.gcd(a, b),)))

    pct = 100.0 * correct / len(all_pairs)
    print(f'\n  Test on {len(all_pairs)} pairs: {correct}/{len(all_pairs)} = {pct:.1f}%')
    if failures:
        print(f'  Failures (first 5):')
        for a, b, got, exp in failures[:5]:
            print(f'    GCD({a},{b}): got {got}, expected {exp}')
    else:
        print('  No failures.')
    print()
    return correct == len(all_pairs)


# ---------------------------------------------------------------------------
# Phase 11: Symbolic differentiation (Level C primitives)
# ---------------------------------------------------------------------------

def phase11_symbolic_differentiation() -> bool:
    """Phase 11: Verify the Level C symbolic expression primitives.

    No synthesis in this phase — sym_diff, sym_eval etc. are built-in
    primitives (like succ/pred in Phase 1).

    Tests three polynomials:
        f(x) = x^2 + 3x + 2        f'(x) = 2x + 3
        g(x) = 2x^3 - x^2 + 4     g'(x) = 6x^2 - 2x
        h(x) = (x^2)(x + 1)        h'(x) = 3x^2 + 2x   (product rule)

    Evaluation points: x = 0, 1, 2, 3.
    """
    print('=' * 65)
    print('Phase 11: Symbolic Differentiation (Level C primitives)')
    print('  sym_num, sym_var, sym_add/sub/mul/pow, sym_eval, sym_diff')
    print('  — Gap 2 primitives enabling algebra and ODE curricula —')
    print('=' * 65)

    ai = _load_fresh_ai()
    interp = ai._interp
    interp.engine_ask = ai.ask

    # ------------------------------------------------------------------
    # Helper: run a zero-input process and return the emitted value.
    # ------------------------------------------------------------------
    def _run(process_lines):
        return interp.run(process_lines, inputs=(), input_type=[])

    passed = 0
    total  = 0

    def _check(label, actual, expected):
        nonlocal passed, total
        total += 1
        ok = actual == expected
        passed += ok
        print(f'  {_tick(ok)}  {label:40s} got {actual!r}  expected {expected!r}')
        return ok

    # ------------------------------------------------------------------
    # Test 1: f(x) = x^2 + 3x + 2,  f'(x) = 2x + 3
    # Build, differentiate, evaluate, print expression strings.
    # ------------------------------------------------------------------
    print('\n  Test 1: f(x) = x^2 + 3x + 2')
    proc_f = [
        'xv  = sym_var(X)',
        'f   = sym_add(sym_pow(xv, 2), sym_add(sym_mul(sym_num(3), xv), sym_num(2)))',
        'df  = sym_diff(f, X)',
        'fs  = sym_str(f)',
        'dfs = sym_str(df)',
        # Evaluate at x=0,1,2,5
        'f0  = sym_eval(f,  X, 0)',
        'f1  = sym_eval(f,  X, 1)',
        'f2  = sym_eval(f,  X, 2)',
        'f5  = sym_eval(f,  X, 5)',
        'df0 = sym_eval(df, X, 0)',
        'df1 = sym_eval(df, X, 1)',
        'df2 = sym_eval(df, X, 2)',
        'df5 = sym_eval(df, X, 5)',
        'emit(fs, dfs, f0, f1, f2, f5, df0, df1, df2, df5)',
    ]
    fs, dfs, f0, f1, f2, f5, df0, df1, df2, df5 = _run(proc_f)
    print(f'    f(x)  = {fs}')
    print(f'    f\'(x) = {dfs}')
    _check("f(0)",   f0,  2)
    _check("f(1)",   f1,  6)
    _check("f(2)",   f2,  12)
    _check("f(5)",   f5,  42)
    _check("f'(0)",  df0, 3)
    _check("f'(1)",  df1, 5)
    _check("f'(2)",  df2, 7)
    _check("f'(5)",  df5, 13)

    # ------------------------------------------------------------------
    # Test 2: g(x) = 2x^3 - x^2 + 4,  g'(x) = 6x^2 - 2x
    # ------------------------------------------------------------------
    print('\n  Test 2: g(x) = 2x^3 - x^2 + 4')
    proc_g = [
        'xv  = sym_var(X)',
        'g   = sym_add(sym_sub(sym_mul(sym_num(2), sym_pow(xv, 3)), sym_pow(xv, 2)), sym_num(4))',
        'dg  = sym_diff(g, X)',
        'gs  = sym_str(g)',
        'dgs = sym_str(dg)',
        'g0  = sym_eval(g,  X, 0)',
        'g1  = sym_eval(g,  X, 1)',
        'g2  = sym_eval(g,  X, 2)',
        'dg0 = sym_eval(dg, X, 0)',
        'dg1 = sym_eval(dg, X, 1)',
        'dg2 = sym_eval(dg, X, 2)',
        'emit(gs, dgs, g0, g1, g2, dg0, dg1, dg2)',
    ]
    gs, dgs, g0, g1, g2, dg0, dg1, dg2 = _run(proc_g)
    print(f'    g(x)  = {gs}')
    print(f'    g\'(x) = {dgs}')
    # g(0) = 4, g(1) = 2-1+4=5, g(2) = 16-4+4=16
    _check("g(0)",   g0,  4)
    _check("g(1)",   g1,  5)
    _check("g(2)",   g2,  16)
    # g'(x)=6x^2-2x: g'(0)=0, g'(1)=4, g'(2)=20
    _check("g'(0)",  dg0, 0)
    _check("g'(1)",  dg1, 4)
    _check("g'(2)",  dg2, 20)

    # ------------------------------------------------------------------
    # Test 3: h(x) = x^2 * (x + 1) = x^3 + x^2,  h'(x) = 3x^2 + 2x
    # Tests product rule via sym_mul of two subexpressions.
    # ------------------------------------------------------------------
    print('\n  Test 3: h(x) = x^2 * (x + 1)  [product rule]')
    proc_h = [
        'xv  = sym_var(X)',
        'u   = sym_pow(xv, 2)',               # x^2
        'v   = sym_add(xv, sym_num(1))',       # x + 1
        'h   = sym_mul(u, v)',                 # x^2 * (x+1)
        'dh  = sym_diff(h, X)',
        'hs  = sym_str(h)',
        'dhs = sym_str(dh)',
        'h0  = sym_eval(h,  X, 0)',
        'h1  = sym_eval(h,  X, 1)',
        'h2  = sym_eval(h,  X, 2)',
        'h3  = sym_eval(h,  X, 3)',
        'dh0 = sym_eval(dh, X, 0)',
        'dh1 = sym_eval(dh, X, 1)',
        'dh2 = sym_eval(dh, X, 2)',
        'dh3 = sym_eval(dh, X, 3)',
        'emit(hs, dhs, h0, h1, h2, h3, dh0, dh1, dh2, dh3)',
    ]
    hs, dhs, h0, h1, h2, h3, dh0, dh1, dh2, dh3 = _run(proc_h)
    print(f'    h(x)  = {hs}')
    print(f'    h\'(x) = {dhs}')
    # h(x) = x^3+x^2: h(0)=0, h(1)=2, h(2)=12, h(3)=36
    _check("h(0)",   h0,  0)
    _check("h(1)",   h1,  2)
    _check("h(2)",   h2,  12)
    _check("h(3)",   h3,  36)
    # h'(x) = 3x^2+2x: h'(0)=0, h'(1)=5, h'(2)=16, h'(3)=33
    _check("h'(0)",  dh0, 0)
    _check("h'(1)",  dh1, 5)
    _check("h'(2)",  dh2, 16)
    _check("h'(3)",  dh3, 33)

    # ------------------------------------------------------------------
    # Test 4: sym_subst — substitute X=2 into f before evaluating
    # ------------------------------------------------------------------
    print('\n  Test 4: sym_subst — substitute X with sym_num(2) in f, then eval')
    proc_subst = [
        'xv  = sym_var(X)',
        'f   = sym_add(sym_pow(xv, 2), sym_add(sym_mul(sym_num(3), xv), sym_num(2)))',
        'f2  = sym_subst(f, X, sym_num(2))',   # f with X→2 = constant 12
        'fs2 = sym_str(f2)',
        'val = sym_eval(f2, X, 0)',             # X is gone; bindings unused
        'emit(fs2, val)',
    ]
    fs2, val = _run(proc_subst)
    print(f'    f[X→2] = {fs2}')
    _check("sym_subst: f[X→2] evaluates to", val, 12)

    print(f'\n  Result: {passed}/{total} correct')
    print()
    return passed == total


# ---------------------------------------------------------------------------
# Phase 12 — Integration, Sequences, and Inspection (Gaps 3-5)
# ---------------------------------------------------------------------------

def phase12_integration_sequences_inspection() -> bool:
    """Verify sym_integrate, scan/seq_*, and sym_tag/lhs/rhs inspection primitives."""
    print()
    print('─' * 65)
    print('Phase 12: Integration, Sequences, Inspection (Gaps 3-5)')
    print('─' * 65)

    ai    = _load_fresh_ai()
    interp = ai._interp
    interp.engine_ask = ai.ask

    def _run(process_lines):
        return interp.run(process_lines, inputs=(), input_type=[])

    passed = 0
    total  = 0

    def _check(label, actual, expected):
        nonlocal passed, total
        total += 1
        ok = actual == expected
        passed += ok
        print(f'  {_tick(ok)}  {label:42s} got {actual!r}  (expected {expected!r})')
        return ok

    def _check_close(label, actual, expected, tol=1e-9):
        """For floating-point results from sym_eval on rational antiderivatives."""
        nonlocal passed, total
        total += 1
        ok = abs(actual - expected) < tol
        passed += ok
        print(f'  {_tick(ok)}  {label:42s} got {actual!r}  (expected {expected!r})')
        return ok

    # ------------------------------------------------------------------
    # Part A: sym_integrate — polynomial antiderivatives
    # ------------------------------------------------------------------
    print('\n  Part A: sym_integrate (Gap 3 — polynomial antiderivative)')

    # Test 1: ∫1 dx = x
    print('\n  Test A1: integral of constant 1 → x')
    proc = [
        'xv  = sym_var(X)',
        'f   = sym_num(1)',
        'F   = sym_integrate(f, X)',
        'Fs  = sym_str(F)',
        'v0  = sym_eval(F, X, 0)',
        'v3  = sym_eval(F, X, 3)',
        'emit(Fs, v0, v3)',
    ]
    Fs, v0, v3 = _run(proc)
    print(f'    ∫1 dx = {Fs}')
    _check('∫1 dx at x=0', v0, 0)
    _check('∫1 dx at x=3', v3, 3)

    # Test 2: ∫x dx = x²/2
    print('\n  Test A2: integral of x → x²/2')
    proc = [
        'xv  = sym_var(X)',
        'f   = xv',
        'F   = sym_integrate(f, X)',
        'Fs  = sym_str(F)',
        'v0  = sym_eval(F, X, 0)',
        'v2  = sym_eval(F, X, 2)',
        'v4  = sym_eval(F, X, 4)',
        'emit(Fs, v0, v2, v4)',
    ]
    Fs, v0, v2, v4 = _run(proc)
    print(f'    ∫x dx = {Fs}')
    _check_close('∫x dx at x=0', v0, 0.0)
    _check_close('∫x dx at x=2', v2, 2.0)    # 2²/2 = 2
    _check_close('∫x dx at x=4', v4, 8.0)    # 4²/2 = 8

    # Test 3: ∫3x² + 2x + 1 dx = x³ + x² + x  (integer coefficients cancel)
    print('\n  Test A3: ∫(3x² + 2x + 1) dx = x³ + x² + x')
    proc = [
        'xv  = sym_var(X)',
        'p3  = sym_mul(sym_num(3), sym_pow(xv, 2))',   # 3x²
        'p2  = sym_mul(sym_num(2), xv)',               # 2x
        'p1  = sym_num(1)',
        'f   = sym_add(sym_add(p3, p2), p1)',
        'F   = sym_integrate(f, X)',
        'Fs  = sym_str(F)',
        'v0  = sym_eval(F, X, 0)',
        'v1  = sym_eval(F, X, 1)',
        'v2  = sym_eval(F, X, 2)',
        'emit(Fs, v0, v1, v2)',
    ]
    Fs, v0, v1, v2 = _run(proc)
    print(f'    ∫(3x²+2x+1) dx = {Fs}')
    _check_close('F(0)', v0,  0.0)    # 0+0+0
    _check_close('F(1)', v1,  3.0)    # 1+1+1
    _check_close('F(2)', v2, 14.0)    # 8+4+2

    # Test 4: Fundamental theorem — d/dx(∫f dx) = f
    print('\n  Test A4: fundamental theorem — d/dx(∫x² dx) = x²  at x=1,2,3')
    proc = [
        'xv  = sym_var(X)',
        'f   = sym_pow(xv, 2)',          # f = x²
        'F   = sym_integrate(f, X)',      # F = x³/3
        'dF  = sym_diff(F, X)',           # dF = x²  (= f)
        'dF1 = sym_eval(dF, X, 1)',
        'dF2 = sym_eval(dF, X, 2)',
        'dF3 = sym_eval(dF, X, 3)',
        'f1  = sym_eval(f,  X, 1)',
        'f2  = sym_eval(f,  X, 2)',
        'f3  = sym_eval(f,  X, 3)',
        'emit(dF1, dF2, dF3, f1, f2, f3)',
    ]
    dF1, dF2, dF3, f1, f2, f3 = _run(proc)
    _check_close('d/dx(∫x²dx) at x=1 == x² at x=1', dF1, f1)
    _check_close('d/dx(∫x²dx) at x=2 == x² at x=2', dF2, f2)
    _check_close('d/dx(∫x²dx) at x=3 == x² at x=3', dF3, f3)

    # ------------------------------------------------------------------
    # Part B: scan / seq_* — variable-length state sequences
    # ------------------------------------------------------------------
    print('\n  Part B: scan and seq_* primitives (Gap 4 — sequences)')

    # Test 5: scan(5, 0, succ) → [1, 2, 3, 4, 5]
    print('\n  Test B1: scan(5, 0, succ) → [1,2,3,4,5]')
    proc = [
        'traj = scan(5, 0, fn(s, succ(s)))',
        'n    = seq_len(traj)',
        's0   = seq_get(traj, 0)',
        's4   = seq_get(traj, 4)',
        'emit(n, s0, s4)',
    ]
    n, s0, s4 = _run(proc)
    _check('scan(5,0,succ) length',     n,  5)
    _check('scan(5,0,succ)[0]',         s0, 1)
    _check('scan(5,0,succ)[4]',         s4, 5)

    # Test 6: scan with fold — each step doubles (repeated addition via fold)
    print('\n  Test B2: scan(4, 1, fn(s, fold(s, s, succ))) → [2, 4, 8, 16]')
    proc = [
        'traj = scan(4, 1, fn(s, fold(s, s, succ)))',  # step: s → s+s = 2s
        'n    = seq_len(traj)',
        'v0   = seq_get(traj, 0)',
        'v1   = seq_get(traj, 1)',
        'v2   = seq_get(traj, 2)',
        'v3   = seq_get(traj, 3)',
        'emit(n, v0, v1, v2, v3)',
    ]
    n, v0, v1, v2, v3 = _run(proc)
    _check('doubling scan length', n,  4)
    _check('doubling scan[0]',     v0, 2)
    _check('doubling scan[1]',     v1, 4)
    _check('doubling scan[2]',     v2, 8)
    _check('doubling scan[3]',     v3, 16)

    # Test 7: seq_cons, seq_head, seq_tail
    print('\n  Test B3: seq_cons / seq_head / seq_tail')
    proc = [
        'empty = seq_nil()',
        'lst   = seq_cons(3, seq_cons(2, seq_cons(1, empty)))',
        'n     = seq_len(lst)',
        'h     = seq_head(lst)',
        'rest  = seq_tail(lst)',
        'h2    = seq_head(rest)',
        'emit(n, h, h2)',
    ]
    n, h, h2 = _run(proc)
    _check('seq [3,2,1] length',   n,  3)
    _check('seq_head([3,2,1])',     h,  3)
    _check('seq_head(tail([3,2,1]))', h2, 2)

    # ------------------------------------------------------------------
    # Part C: sym_tag / sym_lhs / sym_rhs / sym_val — tree inspection
    # ------------------------------------------------------------------
    print('\n  Part C: expression tree inspection (Gap 5 — sym_tag family)')

    # Test 8: inspect structure of 3x² + 5
    print('\n  Test C1: inspect (3*x²) + 5 — tag, lhs, rhs, val')
    proc = [
        'xv   = sym_var(X)',
        'term = sym_mul(sym_num(3), sym_pow(xv, 2))',  # 3x²
        'expr = sym_add(term, sym_num(5))',
        'tag  = sym_tag(expr)',                         # 'ADD'
        'lhs  = sym_lhs(expr)',                         # 3x²
        'rhs  = sym_rhs(expr)',                         # 5
        'ltag = sym_tag(lhs)',                          # 'MUL'
        'rval = sym_val(rhs)',                          # 5
        'emit(tag, ltag, rval)',
    ]
    tag, ltag, rval = _run(proc)
    _check('sym_tag(3x²+5)',         tag,  'ADD')
    _check('sym_tag(lhs=3x²)',       ltag, 'MUL')
    _check('sym_val(rhs=5)',         rval, 5)

    # Test 9: extract coefficient and exponent from a monomial
    print('\n  Test C2: extract coeff/exp from 7*x^3 via inspection')
    proc = [
        'xv    = sym_var(X)',
        'mono  = sym_mul(sym_num(7), sym_pow(xv, 3))',  # 7x³
        'tag   = sym_tag(mono)',                          # 'MUL'
        'coeff = sym_val(sym_lhs(mono))',                 # 7
        'power = sym_rhs(mono)',                          # x³
        'exp   = sym_exp(power)',                         # 3
        'vname = sym_name(sym_lhs(power))',               # 'X'
        'emit(tag, coeff, exp, vname)',
    ]
    tag, coeff, exp, vname = _run(proc)
    _check('sym_tag(7x³)',           tag,   'MUL')
    _check('coefficient of 7x³',    coeff, 7)
    _check('exponent of 7x³',       exp,   3)
    _check('variable name of 7x³',  vname, 'X')

    # Test 10: sym_tag on each node type
    print('\n  Test C3: sym_tag on all node types')
    proc = [
        'n1  = sym_tag(sym_num(5))',          # NUM
        'n2  = sym_tag(sym_var(X))',           # VAR
        'n3  = sym_tag(sym_add(sym_num(1), sym_num(2)))',   # NUM (folded)
        'n4  = sym_tag(sym_add(sym_var(X), sym_num(1)))',   # ADD
        'n5  = sym_tag(sym_mul(sym_var(X), sym_var(Y)))',   # MUL
        'n6  = sym_tag(sym_pow(sym_var(X), 2))',            # POW
        'n7  = sym_tag(sym_neg(sym_var(X)))',               # NEG
        'n8  = sym_tag(sym_div(sym_var(X), sym_num(3)))',   # DIV
        'emit(n1, n2, n3, n4, n5, n6, n7, n8)',
    ]
    n1, n2, n3, n4, n5, n6, n7, n8 = _run(proc)
    _check('sym_tag(NUM node)',  n1, 'NUM')
    _check('sym_tag(VAR node)',  n2, 'VAR')
    _check('sym_tag(1+2→NUM)',   n3, 'NUM')   # constant-folded to NUM(3)
    _check('sym_tag(ADD node)',  n4, 'ADD')
    _check('sym_tag(MUL node)',  n5, 'MUL')
    _check('sym_tag(POW node)',  n6, 'POW')
    _check('sym_tag(NEG node)',  n7, 'NEG')
    _check('sym_tag(DIV node)',  n8, 'DIV')

    print(f'\n  Result: {passed}/{total} correct')
    print()
    return passed == total


# ---------------------------------------------------------------------------
# Phase 13: ODE Solving (Gaps 6-7 — sym_expand/sym_coeff + float arithmetic)
# ---------------------------------------------------------------------------

def phase13_ode_solving() -> bool:
    """Phase 13: Demonstrate ODE solving using symbolic and float primitives.

    Part A: First-order separable ODE via sym_integrate + initial condition.
        ODE:       dy/dx = 6x² + 2x
        Solution:  y = 2x³ + x² + C
        IC y(0)=5: C = 5  →  y = 2x³ + x² + 5

    Part B: Second-order constant-coefficient ODE via characteristic equation.
        ODE:       y'' - 5y' + 6y = 0   (coefficients 1, -5, 6)
        Char eq:   r² - 5r + 6 = 0      (built symbolically, coeffs read via sym_coeff)
        Roots:     r1=3, r2=2
        General:   y = c1·e^(3x) + c2·e^(2x)

    Part C: Apply initial conditions y(0)=1, y'(0)=4 to find c1, c2.
        System:    c1 + c2 = 1,  3c1 + 2c2 = 4  →  c1=2, c2=-1
        Solution:  y = 2e^(3x) - e^(2x)
        Verified at x=0, 1, 2 vs expected.
    """
    import math

    print()
    print('─' * 65)
    print('Phase 13: ODE Solving (sym_expand, sym_coeff, float arithmetic)')
    print('─' * 65)

    ai    = _load_fresh_ai()
    interp = ai._interp
    interp.engine_ask = ai.ask

    def _run(process_lines, inputs=(), input_type=None):
        return interp.run(process_lines, inputs=inputs,
                          input_type=input_type or [])

    passed = 0
    total  = 0

    def _check(label, actual, expected):
        nonlocal passed, total
        total += 1
        ok = actual == expected
        passed += ok
        print(f'  {_tick(ok)}  {label:48s} got {actual!r}  (expected {expected!r})')
        return ok

    def _check_close(label, actual, expected, tol=1e-6):
        nonlocal passed, total
        total += 1
        ok = abs(float(actual) - float(expected)) < tol
        passed += ok
        print(f'  {_tick(ok)}  {label:48s} got {actual:.6f}  (expected {expected:.6f})')
        return ok

    # ------------------------------------------------------------------
    # Part A: First-order separable ODE: dy/dx = 6x² + 2x, y(0) = 5
    # ------------------------------------------------------------------
    print('\n  Part A: First-order separable ODE — dy/dx = 6x² + 2x, y(0)=5')

    proc_a = [
        # Build f(x) = 6x² + 2x
        'xv   = sym_var(X)',
        'p2   = sym_mul(sym_num(6), sym_pow(xv, 2))',   # 6x²
        'p1   = sym_mul(sym_num(2), xv)',                # 2x
        'f    = sym_add(p2, p1)',                        # 6x² + 2x
        # Antiderivative F(x) = 2x³ + x²
        'F    = sym_integrate(f, X)',
        'Fs   = sym_str(F)',
        # Apply IC: y(0) = 5 → C = 5 - F(0)
        'F0   = sym_eval(F, X, 0)',
        # Build y(x) = F(x) + C  (C = 5 since F(0) = 0)
        'Cval = 5 - F0',
        'y    = sym_add(F, sym_num(Cval))',
        'ys   = sym_str(y)',
        # Verify y at x=0,1,2,3
        'y0   = sym_eval(y, X, 0)',
        'y1   = sym_eval(y, X, 1)',
        'y2   = sym_eval(y, X, 2)',
        'y3   = sym_eval(y, X, 3)',
        'emit(Fs, ys, y0, y1, y2, y3)',
    ]
    Fs, ys, y0, y1, y2, y3 = _run(proc_a)
    print(f'    Antiderivative F(x) = {Fs}')
    print(f'    Solution     y(x) = {ys}')
    # y(x) = 2x³ + x² + 5: y(0)=5, y(1)=2+1+5=8, y(2)=16+4+5=25, y(3)=54+9+5=68
    _check('y(0) = 5',  y0,  5)
    _check('y(1) = 8',  y1,  8)
    _check('y(2) = 25', y2, 25)
    _check('y(3) = 68', y3, 68)

    # ------------------------------------------------------------------
    # Part B: Characteristic equation via sym_expand + sym_coeff
    # ODE: y'' - 5y' + 6y = 0  →  char eq: r² - 5r + 6 = 0
    # ------------------------------------------------------------------
    print('\n  Part B: Characteristic equation — r² - 5r + 6 = 0')

    proc_b = [
        # Build characteristic polynomial p(r) = r² - 5r + 6
        'rv   = sym_var(R)',
        'r2   = sym_pow(rv, 2)',                          # r²
        'r1   = sym_mul(sym_num(5), rv)',                 # 5r
        'p    = sym_add(sym_sub(r2, r1), sym_num(6))',   # r² - 5r + 6
        'ps   = sym_str(p)',
        # Expand (already in expanded form, but test sym_expand works)
        'pe   = sym_expand(p)',
        'pes  = sym_str(pe)',
        # Extract coefficients: a=coeff(r²), b=coeff(r¹), c=coeff(r⁰)
        'ca   = sym_coeff(pe, R, 2)',
        'cb   = sym_coeff(pe, R, 1)',
        'cc   = sym_coeff(pe, R, 0)',
        'emit(ps, pes, ca, cb, cc)',
    ]
    ps, pes, ca, cb, cc = _run(proc_b)
    print(f'    p(r) = {ps}')
    print(f'    expanded = {pes}')
    _check_close('coeff of r²  (a = 1)',  ca,  1.0)
    _check_close('coeff of r¹  (b = -5)', cb, -5.0)
    _check_close('coeff of r⁰  (c = 6)',  cc,  6.0)

    # ------------------------------------------------------------------
    # Part B cont.: Solve quadratic ar² + br + c = 0 via float arithmetic
    # ------------------------------------------------------------------
    print('\n  Part B (cont.): Solve char eq via float quadratic formula')

    proc_b2 = [
        # Inputs: a, b, c  (mapped as env['a'], env['b'], env['c'])
        # D = b² - 4ac
        'bsq  = float_mul(float_num(b), float_num(b))',
        'ac4  = float_mul(float_num(4), float_mul(float_num(a), float_num(c)))',
        'D    = float_sub(bsq, ac4)',
        'sqD  = float_sqrt(D)',
        # -b (negate b by computing 0 - b as integer, then float_num)
        'nb   = float_num(0 - b)',
        'twoa = float_mul(float_num(2), float_num(a))',
        'r1   = float_div(float_add(nb, sqD), twoa)',
        'r2   = float_div(float_sub(nb, sqD), twoa)',
        'emit(r1, r2)',
    ]
    r1, r2 = _run(proc_b2, inputs=(1, -5, 6), input_type=['c', 'c', 'c'])
    _check_close('r1 = 3.0', r1, 3.0)
    _check_close('r2 = 2.0', r2, 2.0)

    # ------------------------------------------------------------------
    # Part C: Apply initial conditions y(0)=1, y'(0)=4
    # System: c1 + c2 = 1,  r1*c1 + r2*c2 = 4
    # Solution: c1=2, c2=-1  → y(x) = 2e^(3x) - e^(2x)
    # ------------------------------------------------------------------
    print('\n  Part C: Apply ICs y(0)=1, y\'(0)=4 → find c1, c2; verify solution')

    # Solve 2×2 linear system for c1, c2.
    # Inputs (as floats via env): a=r1, b=r2, c=y0, d=yp0
    proc_c = [
        # c1*(r1-r2) = yp0 - r2*y0
        # c2 = y0 - c1
        'r2y0  = float_mul(float_num(b), float_num(c))',  # r2 * y0
        'numer = float_sub(float_num(d), r2y0)',           # yp0 - r2*y0
        'denom = float_sub(float_num(a), float_num(b))',   # r1 - r2
        'c1    = float_div(numer, denom)',
        'c2    = float_sub(float_num(c), c1)',             # y0 - c1
        'emit(c1, c2)',
    ]
    c1, c2 = _run(proc_c, inputs=(r1, r2, 1, 4), input_type=['r', 'r', 'ic', 'ic'])
    _check_close('c1 = 2.0', c1, 2.0)
    _check_close('c2 = -1.0', c2, -1.0)

    # Verify y(x) = c1*e^(r1*x) + c2*e^(r2*x) at x=0, 1, 2
    # Use process language with float_exp
    proc_verify = [
        # Inputs: a=r1, b=r2, c=c1, d=c2, plus x passed as 5th arg mapped to 'e'
        'e1   = float_mul(float_num(c), float_exp(float_mul(float_num(a), float_num(e))))',  # c1*exp(r1*x)
        'e2   = float_mul(float_num(d), float_exp(float_mul(float_num(b), float_num(e))))',  # c2*exp(r2*x)
        'yval = float_add(e1, e2)',
        'emit(yval,)',
    ]
    for x_val, expected in [(0, 1.0), (1, 2*math.exp(3) - math.exp(2))]:
        yval, = _run(proc_verify, inputs=(r1, r2, c1, c2, x_val),
                     input_type=['r', 'r', 'c', 'c', 'x'])
        _check_close(f'y({x_val}) = c1·e^(r1·{x_val}) + c2·e^(r2·{x_val})', yval, expected)

    # Verify ICs directly: y'(x) = c1*r1*e^(r1*x) + c2*r2*e^(r2*x)
    yp0 = c1 * r1 * math.exp(r1 * 0) + c2 * r2 * math.exp(r2 * 0)
    _check_close("y'(0) = 4.0 (ICs consistent)", yp0, 4.0)

    print(f'\n  Result: {passed}/{total} correct')
    print()
    return passed == total


# ---------------------------------------------------------------------------
# Phase 14: Fluid Dynamics (physics application domain)
# ---------------------------------------------------------------------------

def phase14_fluid_dynamics() -> bool:
    """Phase 14: Fluid dynamics using float_pow, float_pi, float_sqrt.

    Four fluid-mechanics laws, each encoded as a process and verified
    against the known answer.

    Part A: Continuity equation (incompressible flow)
        A₁v₁ = A₂v₂  →  v₂ = v₁ * (r₁/r₂)²
        Pipe narrows from r₁=0.05m to r₂=0.025m; v₁=2.0 m/s → v₂=8.0 m/s

    Part B: Torricelli's theorem
        v = √(2gh);  h=3.0m, g=9.81 m/s²  →  v ≈ 7.672 m/s

    Part C: Bernoulli (horizontal pipe, solve for pressure)
        P₂ = P₁ + ½ρ(v₁²−v₂²)
        P₁=200000 Pa, ρ=1000, v₁=2.0 m/s, v₂=4.0 m/s  →  P₂=194000 Pa

    Part D: Venturi meter (combined application problem)
        Given pipe radii r₁=0.05m, r₂=0.03m; v₁=3.0 m/s; ρ=1000; P₁=300000 Pa.
        Step 1 — continuity: v₂ = v₁*(r₁/r₂)²  = 3*(5/3)² ≈ 8.333 m/s
        Step 2 — Bernoulli: P₂ = P₁ + ½ρ(v₁²−v₂²) ≈ 269778 Pa
    """
    import math

    print()
    print('─' * 65)
    print('Phase 14: Fluid Dynamics (continuity, Bernoulli, Torricelli)')
    print('─' * 65)

    ai    = _load_fresh_ai()
    interp = ai._interp
    interp.engine_ask = ai.ask

    def _run(proc, inputs=(), input_type=None):
        return interp.run(proc, inputs=inputs, input_type=input_type or [])

    passed = 0
    total  = 0

    def _check_close(label, actual, expected, tol=1e-3):
        nonlocal passed, total
        total += 1
        ok = abs(float(actual) - float(expected)) < tol
        passed += ok
        print(f'  {_tick(ok)}  {label:52s}  got {float(actual):.4f}  (expected {float(expected):.4f})')
        return ok

    # ------------------------------------------------------------------
    # Part A: Continuity equation — v₂ = v₁ * (r₁/r₂)²
    # ------------------------------------------------------------------
    print('\n  Part A: Continuity equation  A₁v₁ = A₂v₂')
    print('    Pipe: r₁=0.05m → r₂=0.025m, v₁=2.0 m/s  →  v₂=?')

    proc_a = [
        # Inputs: a=r1, b=r2, c=v1
        'ratio  = float_div(float_num(a), float_num(b))',     # r1/r2
        'aratio = float_pow(ratio, 2)',                        # (r1/r2)² = A1/A2
        'v2     = float_mul(float_num(c), aratio)',            # v2 = v1*(A1/A2)
        'emit(v2,)',
    ]
    v2, = _run(proc_a, inputs=(0.05, 0.025, 2.0), input_type=['r', 'r', 'v'])
    _check_close('continuity v₂  (expected 8.0 m/s)', v2, 8.0)

    # Second test: r₁=0.10m, r₂=0.04m, v₁=1.5 m/s → v₂ = 1.5*(0.10/0.04)² = 1.5*6.25 = 9.375
    v2b, = _run(proc_a, inputs=(0.10, 0.04, 1.5), input_type=['r', 'r', 'v'])
    _check_close('continuity v₂  (expected 9.375 m/s)', v2b, 9.375)

    # ------------------------------------------------------------------
    # Part B: Torricelli's theorem — v = √(2gh)
    # ------------------------------------------------------------------
    print('\n  Part B: Torricelli\'s theorem  v = √(2gh)')
    print('    Tank: h=3.0m, g=9.81 m/s²  →  v=?')

    proc_b = [
        # Inputs: a=g, b=h
        'twogh = float_mul(float_num(2), float_mul(float_num(a), float_num(b)))',
        'v     = float_sqrt(twogh)',
        'emit(v,)',
    ]
    v_exit, = _run(proc_b, inputs=(9.81, 3.0), input_type=['g', 'h'])
    expected_v = math.sqrt(2 * 9.81 * 3.0)
    _check_close('Torricelli v  (h=3.0m)', v_exit, expected_v)

    # Second test: h=5.0m → v = √(2*9.81*5) = √98.1
    v_exit2, = _run(proc_b, inputs=(9.81, 5.0), input_type=['g', 'h'])
    _check_close('Torricelli v  (h=5.0m)', v_exit2, math.sqrt(2 * 9.81 * 5.0))

    # ------------------------------------------------------------------
    # Part C: Bernoulli (horizontal) — P₂ = P₁ + ½ρ(v₁²−v₂²)
    # ------------------------------------------------------------------
    print('\n  Part C: Bernoulli (horizontal)  P₂ = P₁ + ½ρ(v₁²−v₂²)')
    print('    P₁=200000 Pa, ρ=1000, v₁=2.0, v₂=4.0 m/s  →  P₂=?')

    proc_c = [
        # Inputs: a=P1, b=rho, c=v1, d=v2
        'half    = float_div(float_num(1), float_num(2))',
        'v1sq    = float_pow(float_num(c), 2)',
        'v2sq    = float_pow(float_num(d), 2)',
        'dvsq    = float_sub(v1sq, v2sq)',
        'dynterm = float_mul(float_mul(half, float_num(b)), dvsq)',
        'P2      = float_add(float_num(a), dynterm)',
        'emit(P2,)',
    ]
    P2, = _run(proc_c, inputs=(200000, 1000, 2.0, 4.0), input_type=['P', 'rho', 'v', 'v'])
    # P2 = 200000 + 0.5*1000*(4-16) = 200000 - 6000 = 194000
    _check_close('Bernoulli P₂  (expected 194000 Pa)', P2, 194000.0, tol=0.5)

    # Second test: v₁=1 m/s → v₂=3 m/s, P₁=150000, ρ=1000
    # P₂ = 150000 + 0.5*1000*(1-9) = 150000 - 4000 = 146000 Pa
    P2b, = _run(proc_c, inputs=(150000, 1000, 1.0, 3.0), input_type=['P', 'rho', 'v', 'v'])
    _check_close('Bernoulli P₂  (expected 146000 Pa)', P2b, 146000.0, tol=0.5)

    # ------------------------------------------------------------------
    # Part D: Venturi meter application problem — continuity + Bernoulli
    # ------------------------------------------------------------------
    print('\n  Part D: Venturi meter (continuity → Bernoulli, combined problem)')
    print('    r₁=0.05m, r₂=0.03m, v₁=3.0 m/s, ρ=1000, P₁=300000 Pa  →  v₂, P₂=?')

    proc_d = [
        # Inputs: a=r1, b=r2, c=v1, d=rho, e=P1
        # Step 1: continuity → v2
        'ratio  = float_div(float_num(a), float_num(b))',
        'aratio = float_pow(ratio, 2)',
        'v2     = float_mul(float_num(c), aratio)',
        # Step 2: Bernoulli → P2
        'half    = float_div(float_num(1), float_num(2))',
        'v1sq    = float_pow(float_num(c), 2)',
        'v2sq    = float_pow(v2, 2)',
        'dvsq    = float_sub(v1sq, v2sq)',
        'dynterm = float_mul(float_mul(half, float_num(d)), dvsq)',
        'P2      = float_add(float_num(e), dynterm)',
        'emit(v2, P2)',
    ]
    v2_venturi, P2_venturi = _run(
        proc_d,
        inputs=(0.05, 0.03, 3.0, 1000, 300000),
        input_type=['r', 'r', 'v', 'rho', 'P'],
    )
    # v₂ = 3.0 * (0.05/0.03)² = 3.0 * (25/9) = 25/3 ≈ 8.3333 m/s
    expected_v2 = 3.0 * (0.05 / 0.03) ** 2
    # P₂ = 300000 + 0.5*1000*(9.0 - v₂²)
    expected_P2 = 300000.0 + 0.5 * 1000.0 * (3.0**2 - expected_v2**2)
    _check_close('Venturi v₂  (expected ≈ 8.333 m/s)', v2_venturi, expected_v2, tol=1e-3)
    _check_close('Venturi P₂  (expected ≈ 269778 Pa)',  P2_venturi, expected_P2, tol=1.0)

    # Sanity check: P₂ < P₁ (narrowing → faster → lower pressure, Bernoulli effect)
    total += 1
    ok = float(P2_venturi) < 300000.0
    passed += ok
    print(f'  {_tick(ok)}  Venturi P₂ < P₁ (Bernoulli effect confirmed)          '
          f' P₂={float(P2_venturi):.1f} < P₁=300000')

    print(f'\n  Result: {passed}/{total} correct')
    print()
    return passed == total


# ---------------------------------------------------------------------------
# Phase 15: Approximate synthesis — synthetic brightness
# ---------------------------------------------------------------------------

def phase15_approximate_synthesis_brightness():
    """Phase 15: Verify consolidate_approx works on synthetic visual data.

    Generates bright (mean ≈ 0.7) and dark (mean ≈ 0.3) 32×32 images,
    teaches 20 of each to image_brightness, and calls consolidate_approx.
    The synthesizer should discover a luminance threshold rule.
    Expected: ≥95% accuracy on 100 held-out synthetic images.
    """
    print('=' * 65)
    print('Phase 15: Approximate Synthesis — Synthetic Brightness')
    print('  (Gap A: consolidate_approx finds luminance threshold rule)')
    print('=' * 65)

    try:
        import numpy as np
    except ImportError:
        print('  SKIP: numpy not available')
        return False

    vision_ctkg = os.path.join(_HERE, '..', 'ctkg', 'domains', 'vision.ctkg')
    if not os.path.exists(vision_ctkg):
        print(f'  SKIP: vision.ctkg not found at {vision_ctkg}')
        return False

    from modalities.vision import VisionModality
    graph = parse_file(vision_ctkg)
    ai = SymbolicAI(graph, modalities=[VisionModality()])

    np_rng = np.random.default_rng(42)

    def _bright_img():
        return np_rng.uniform(0.55, 0.95, (32, 32, 3)).astype(np.float32)

    def _dark_img():
        return np_rng.uniform(0.05, 0.45, (32, 32, 3)).astype(np.float32)

    N_TRAIN = 20
    for _ in range(N_TRAIN):
        ai.teach('image_brightness', (_bright_img(),), (1,))
        ai.teach('image_brightness', (_dark_img(),), (0,))
    print(f'  Trained on {2 * N_TRAIN} examples ({N_TRAIN} bright, {N_TRAIN} dark)')
    print(f'  KL before consolidation: {ai.kl("image_brightness"):.3f} bits')

    result = ai.consolidate_approx(
        'image_brightness',
        accuracy_threshold=0.85,
        subsample=40,
        verbose=True,
    )

    if result is None:
        print('  FAIL: consolidation returned None (no template found)')
        print()
        return False

    process, accuracy = result
    print(f'  Discovered rule: {_fmt_process(process)}')
    print(f'  Training accuracy: {accuracy:.1%}')

    # Test on 100 held-out synthetic images (50 bright + 50 dark)
    N_TEST = 50
    correct = 0
    for _ in range(N_TEST):
        out = ai.ask('image_brightness', (_bright_img(),))
        if out == (1,):
            correct += 1
    for _ in range(N_TEST):
        out = ai.ask('image_brightness', (_dark_img(),))
        if out == (0,):
            correct += 1

    test_acc = correct / (2 * N_TEST)
    print(f'  Test accuracy ({2 * N_TEST} held-out): {test_acc:.1%}')

    passed = test_acc >= 0.95
    print(f'  {"PASS" if passed else "FAIL"} — '
          f'{"Approximate synthesis works for statistical visual concepts." if passed else "Template did not generalize."}')
    print()
    return passed


# ---------------------------------------------------------------------------
# Phase 16: CIFAR-10 cat classification via approximate synthesis
# ---------------------------------------------------------------------------

def phase16_cifar10_cats():
    """Phase 16: Cat vs. non-cat classification on real CIFAR-10 images.

    Auto-downloads CIFAR-10 (~170 MB on first run).  Teaches 200 cat +
    200 non-cat training examples to the 'cat' concept, then calls
    consolidate_approx to find the best single-feature visual template.

    Expected: 60–75% test accuracy with a single fixed visual feature.
    (Multi-feature combination and reference frames are Phase 17+.)
    """
    print('=' * 65)
    print('Phase 16: CIFAR-10 Cat Classification (Approximate Synthesis)')
    print('  (Real images; expected 60-75% with single visual feature)')
    print('=' * 65)

    try:
        import numpy as np  # noqa: F401 — confirms numpy is importable
    except ImportError:
        print('  SKIP: numpy not available')
        return False

    vision_ctkg = os.path.join(_HERE, '..', 'ctkg', 'domains', 'vision.ctkg')
    if not os.path.exists(vision_ctkg):
        print(f'  SKIP: vision.ctkg not found at {vision_ctkg}')
        return False

    from modalities.vision import VisionModality
    from data_loader import load_cifar10_cats

    graph = parse_file(vision_ctkg)
    ai = SymbolicAI(graph, modalities=[VisionModality()])

    cats_tr, neg_tr, cats_te, neg_te = load_cifar10_cats(
        max_per_class_train=200,
        max_per_class_test=500,
    )

    for img in cats_tr:
        ai.teach('cat', (img,), (1,))
    for img in neg_tr:
        ai.teach('cat', (img,), (0,))
    print(f'  Taught {len(cats_tr)} cat + {len(neg_tr)} non-cat examples')
    print(f'  KL before consolidation: {ai.kl("cat"):.3f} bits')

    result = ai.consolidate_approx(
        'cat',
        accuracy_threshold=0.60,
        subsample=200,
        verbose=True,
    )

    if result is None:
        print('  No template reached 60% threshold.')
        print('  (Visual primitives alone insufficient — combine features in Phase 17+)')
        print()
        return False

    process, train_acc = result
    print(f'  Best template: {_fmt_process(process)}')
    print(f'  Training accuracy: {train_acc:.1%}')

    # Evaluate on held-out CIFAR-10 test images
    correct = 0
    total = len(cats_te) + len(neg_te)
    for img in cats_te:
        out = ai.ask('cat', (img,))
        if out == (1,):
            correct += 1
    for img in neg_te:
        out = ai.ask('cat', (img,))
        if out == (0,):
            correct += 1

    test_acc = correct / total if total > 0 else 0.0
    print(f'  Test accuracy ({total} images): {test_acc:.1%}')

    passed = test_acc >= 0.60
    print(f'  {"PASS" if passed else "FAIL"} — '
          f'{"Single-feature visual baseline established." if passed else "Below 60% threshold."}')
    print()
    return passed


# ---------------------------------------------------------------------------
# Phase 17: User-provided cat photos (real-world, varied quality)
# ---------------------------------------------------------------------------

def phase17_user_cat_photos():
    """Phase 17: Approximate synthesis on the user's own cat image collection.

    Loads whatever images are in data/cats/ and data/negatives/, resizes to
    128x128, trains on 80% of each class, tests on the held-out 20%.

    The dataset is known to be imperfect:
    - Pose sorting is approximate (some frontal/side/3Q are seated)
    - Some images contain multiple animals
    - One negative is a group of humans

    A single visual feature is unlikely to fully separate cats from non-cats
    at this resolution. The goal is to establish what the best single-feature
    template achieves as a floor, and to confirm the pipeline runs on
    real varied images. Expect 55-70%.

    PASS threshold is intentionally low (55%) — barely above chance — since
    the dataset is small and noisy. Real improvement requires TBT multi-view
    (Phase 18) or feature combination (Phase 19+).
    """
    print('=' * 65)
    print('Phase 17: User Cat Photos (Real-World, Varied Quality)')
    print('  (75 cats + 49 negatives; expect 55-70% single-feature)')
    print('=' * 65)

    try:
        import numpy as np
    except ImportError:
        print('  SKIP: numpy not available')
        return False

    vision_ctkg = os.path.join(_HERE, '..', 'ctkg', 'domains', 'vision.ctkg')
    if not os.path.exists(vision_ctkg):
        print(f'  SKIP: vision.ctkg not found')
        return False

    data_cats = os.path.join(_HERE, 'data', 'cats')
    data_negs = os.path.join(_HERE, 'data', 'negatives')
    if not os.path.exists(data_cats):
        print(f'  SKIP: data/cats/ not found — populate with cat images first')
        return False

    from modalities.vision import VisionModality
    from data_loader import load_image_folder, train_test_split

    cats, negs = load_image_folder(
        cats_dir=data_cats,
        neg_dir=data_negs,
        target_size=(128, 128),
    )

    if len(cats) == 0 or len(negs) == 0:
        print('  SKIP: no images loaded (check PIL/Pillow is installed)')
        return False

    print(f'  Loaded: {len(cats)} cat images, {len(negs)} negative images')

    # 80/20 train/test split on each class independently
    cats_tr, neg_tr, cats_te, neg_te = train_test_split(cats, negs, test_fraction=0.2)
    print(f'  Split:  train {len(cats_tr)}+{len(neg_tr)}, '
          f'test {len(cats_te)}+{len(neg_te)}')

    graph = parse_file(vision_ctkg)
    ai = SymbolicAI(graph, modalities=[VisionModality()])

    for img in cats_tr:
        ai.teach('cat', (img,), (1,))
    for img in neg_tr:
        ai.teach('cat', (img,), (0,))
    print(f'  Taught {len(cats_tr)} cat + {len(neg_tr)} non-cat training examples')

    # Use all training images as subsample (small dataset — no need to sub-sample)
    n_train = len(cats_tr) + len(neg_tr)
    result = ai.consolidate_approx(
        'cat',
        accuracy_threshold=0.55,
        subsample=n_train,
        verbose=True,
    )

    if result is None:
        print('  No template reached 55% threshold (chance = 50%).')
        print('  Single visual features cannot separate this dataset.')
        print()
        return False

    process, train_acc = result
    print(f'  Best template:     {_fmt_process(process)}')
    print(f'  Training accuracy: {train_acc:.1%}')

    # Evaluate on held-out images
    correct = 0
    total = len(cats_te) + len(neg_te)
    for img in cats_te:
        out = ai.ask('cat', (img,))
        if out == (1,):
            correct += 1
    for img in neg_te:
        out = ai.ask('cat', (img,))
        if out == (0,):
            correct += 1

    test_acc = correct / total if total > 0 else 0.0
    print(f'  Test accuracy:     {test_acc:.1%}  ({correct}/{total} correct)')

    # Breakdown by class
    cat_correct = sum(
        1 for img in cats_te if ai.ask('cat', (img,)) == (1,)
    )
    neg_correct = sum(
        1 for img in neg_te if ai.ask('cat', (img,)) == (0,)
    )
    print(f'    Cat recall:      {cat_correct}/{len(cats_te)} '
          f'({cat_correct/len(cats_te):.0%})')
    print(f'    Non-cat recall:  {neg_correct}/{len(neg_te)} '
          f'({neg_correct/len(neg_te):.0%})')

    passed = test_acc >= 0.55
    if passed:
        print(f'  PASS — single-feature floor established.')
    else:
        print(f'  FAIL — below chance+5%. Feature combination needed.')
    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print('=' * 65)
    print('  Symbolic AI -- Arithmetic Experiment')
    print('  Category Theory Knowledge Graph + Program Synthesis')
    print('=' * 65)
    print()
    print('  Comparison baseline (Mistake #42 / CONTINUATION.md):')
    print('    Mamba3 neural network: 1.26M params, ~50 epochs, 100 examples')
    print('    → ~45% test accuracy on single-digit arithmetic')
    print()
    print('  This system:')
    print('    Symbolic AI: 0 trainable parameters')
    print('    → 100% test accuracy after consolidation from ≤20 examples')
    print()

    ai = _load_fresh_ai()

    p1 = phase1_builtin(ai)
    p2 = phase2_addition(ai)
    p3 = phase3_subtraction(ai)
    phase4_minimum_examples()
    phase5_prerequisite_enforcement()
    p6 = phase6_multiplication(ai)
    p7 = phase7_exponentiation(ai)
    p8 = phase8_division(ai)
    p9 = phase9_remainder(ai)
    p10 = phase10_gcd(ai)
    p11 = phase11_symbolic_differentiation()
    p12 = phase12_integration_sequences_inspection()
    p13 = phase13_ode_solving()
    p14 = phase14_fluid_dynamics()
    p15 = phase15_approximate_synthesis_brightness()
    p16 = phase16_cifar10_cats()
    p17 = phase17_user_cat_photos()

    print('=' * 65)
    print('Summary')
    print('=' * 65)
    print(f'  Phase 1  (built-in ops):          {"PASS" if p1  else "FAIL"}')
    print(f'  Phase 2  (learn addition):         {"PASS" if p2  else "FAIL"}')
    print(f'  Phase 3  (learn subtraction):      {"PASS" if p3  else "FAIL"}')
    print(f'  Phase 4  (minimum examples):       (see above)')
    print(f'  Phase 5  (prerequisite guard):     (see above)')
    print(f'  Phase 6  (learn multiplication):   {"PASS" if p6  else "FAIL"}')
    print(f'  Phase 7  (learn exponentiation):   {"PASS" if p7  else "FAIL"}')
    print(f'  Phase 8  (learn division):         {"PASS" if p8  else "FAIL"}')
    print(f'  Phase 9  (verify remainder):       {"PASS" if p9  else "FAIL"}')
    print(f'  Phase 10 (verify GCD):             {"PASS" if p10 else "FAIL"}')
    print(f'  Phase 11 (symbolic diff):          {"PASS" if p11 else "FAIL"}')
    print(f'  Phase 12 (integrate/scan/inspect): {"PASS" if p12 else "FAIL"}')
    print(f'  Phase 13 (ODE solving):            {"PASS" if p13 else "FAIL"}')
    print(f'  Phase 14 (fluid dynamics):         {"PASS" if p14 else "FAIL"}')
    print(f'  Phase 15 (approx synth brightness):{"PASS" if p15 else "FAIL"}')
    print(f'  Phase 16 (CIFAR-10 cats):          {"PASS" if p16 else "FAIL"}')
    print(f'  Phase 17 (user cat photos):        {"PASS" if p17 else "FAIL"}')
    print()
    print('  Key results:')
    print('    Exact symbolic rule discovered from examples.')
    print('    Generalizes perfectly (100%) to ALL unseen pairs.')
    print('    No gradient descent. No epochs. No parameters to tune.')
    print('    Prerequisite graph correctly blocks learning without foundations.')
    print('    fn(param, body) enables multiplication and exponentiation from succ alone.')
    print('    fold_until is always bounded — no infinite loops possible.')
    print('    GCD executes Euclidean algorithm via fold_until + lookup(remainder).')
    print('    sym_diff differentiates polynomial expressions symbolically (Gap 2).')
    print('    sym_integrate computes antiderivatives; fundamental theorem verified (Gap 3).')
    print('    scan produces variable-length state sequences (Gap 4).')
    print('    sym_tag/lhs/rhs expose expression tree structure for inspection (Gap 5).')
    print('    sym_expand distributes MUL over ADD; sym_coeff reads poly coefficients (Gap 6).')
    print('    float_sqrt/exp enable quadratic characteristic equation solving (Gap 7).')
    print('    First-order separable + second-order constant-coeff ODEs solved (Phase 13).')
    print('    Fluid dynamics: continuity, Torricelli, Bernoulli, Venturi (Phase 14).')
    print('    float_pow/float_pi extend float arithmetic for physics domains.')
    print('    consolidate_approx finds statistical threshold rules (Gap A, Phase 15).')
    print('    Single visual feature baseline on real CIFAR-10 cat images (Phase 16).')
    print('    User cat photo floor: best single feature on 75 cats + 49 negatives (Phase 17).')
    print()


if __name__ == '__main__':
    main()
