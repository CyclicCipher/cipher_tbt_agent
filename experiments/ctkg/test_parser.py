"""Tests for the CTKG DSL parser and type system.

Tests:
  1. Type definitions parse correctly (symbol, seq, tuple, tagged, annotations)
  2. Arithmetic domain loads and validates (types resolve, graph is acyclic)
  3. Concepts have process lines preserved
  4. Adjunctions parse correctly
  5. Type validation catches undefined types
  6. Error handling (malformed type defs, missing '=', bad names)
  7. Topological sort produces valid training order
  8. build_arithmetic_graph() works via .ctkg file
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.ctkg.parser import parse, parse_file, ParseError
from experiments.ctkg.graph import TypeDef, BUILTIN_TYPES, UndefinedType
from experiments.ctkg.domains.arithmetic import build_arithmetic_graph


def test_type_parsing():
    """Test that type definitions parse correctly."""
    text = """
type digit = symbol(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ordered
type carry = symbol(0, 1)
type op = symbol(ADD, SUB)
type count_seq = seq(digit)
type pair = tuple(digit, digit)
type result = tagged(ok: nat, err: bool)
type index = nat ordered metric
"""
    graph = parse(text)

    errors = []

    # digit
    d = graph.types.get('digit')
    if not d:
        errors.append("type 'digit' not found")
    else:
        if d.constructor != 'symbol':
            errors.append(f"digit.constructor: {d.constructor} != 'symbol'")
        if len(d.params) != 10:
            errors.append(f"digit.params: {len(d.params)} != 10")
        if 'ordered' not in d.annotations:
            errors.append(f"digit.annotations missing 'ordered': {d.annotations}")

    # carry
    c = graph.types.get('carry')
    if not c:
        errors.append("type 'carry' not found")
    elif len(c.params) != 2:
        errors.append(f"carry.params: {len(c.params)} != 2")

    # seq
    s = graph.types.get('count_seq')
    if not s:
        errors.append("type 'count_seq' not found")
    elif s.constructor != 'seq' or s.params != ['digit']:
        errors.append(f"count_seq: {s}")

    # tuple
    p = graph.types.get('pair')
    if not p:
        errors.append("type 'pair' not found")
    elif p.constructor != 'tuple' or p.params != ['digit', 'digit']:
        errors.append(f"pair: {p}")

    # tagged
    r = graph.types.get('result')
    if not r:
        errors.append("type 'result' not found")
    elif r.constructor != 'tagged' or len(r.params) != 2:
        errors.append(f"result: {r}")

    # nat with annotations
    idx = graph.types.get('index')
    if not idx:
        errors.append("type 'index' not found")
    elif idx.constructor != 'nat' or idx.annotations != {'ordered', 'metric'}:
        errors.append(f"index: {idx}")

    # Builtins should still be present
    for builtin in ['nat', 'bool', 'expr', 'proposition']:
        if builtin not in graph.types:
            errors.append(f"builtin type '{builtin}' missing")

    return errors


def test_arithmetic_domain():
    """Test that the full arithmetic.ctkg parses and validates."""
    graph = build_arithmetic_graph()

    errors = []

    # --- Types ---
    expected_types = [
        'digit', 'carry', 'op', 'cmp_result', 'object_token', 'stop',
        'object_seq', 'count_seq', 'digit_pair',
        'query_result', 'counting_result', 'arith_result', 'column_result',
    ]
    for tname in expected_types:
        if tname not in graph.types:
            errors.append(f"Missing type: {tname}")

    # digit should be symbol with 10 params and ordered
    d = graph.types.get('digit')
    if d:
        if d.constructor != 'symbol':
            errors.append(f"digit.constructor = {d.constructor}")
        if len(d.params) != 10:
            errors.append(f"digit has {len(d.params)} params, expected 10")
        if 'ordered' not in d.annotations:
            errors.append(f"digit missing 'ordered' annotation")

    # --- Concepts ---
    expected_concepts = [
        'query_counting', 'combined_counting',
        'successor', 'predecessor', 'comparison',
        'single_digit_addition', 'single_digit_subtraction',
        'two_digit_single_arithmetic', 'two_digit_arithmetic',
    ]
    for cname in expected_concepts:
        if cname not in graph.concepts:
            errors.append(f"Missing concept: {cname}")

    # Concept count
    if len(graph.concepts) != 9:
        errors.append(f"Expected 9 concepts, got {len(graph.concepts)}")

    # All concepts should be in arithmetic domain
    for c in graph.concepts.values():
        if c.domain != 'arithmetic':
            errors.append(f"{c.name}.domain = '{c.domain}', expected 'arithmetic'")

    # --- Process lines preserved ---
    cc = graph.concepts.get('combined_counting')
    if cc:
        if len(cc.process) != 6:
            errors.append(
                f"combined_counting.process has {len(cc.process)} lines, "
                f"expected 6: {cc.process}")

    sda = graph.concepts.get('single_digit_addition')
    if sda:
        if len(sda.process) != 4:
            errors.append(
                f"single_digit_addition.process has {len(sda.process)} lines, "
                f"expected 4: {sda.process}")
        if not sda.supports_reverse:
            errors.append("single_digit_addition should be reversible")
        if not sda.is_atomic:
            errors.append("single_digit_addition should be atomic")

    # --- Prerequisites ---
    prereq_count = len(graph.prerequisites)
    if prereq_count != 12:
        edges = [(p.source, p.target) for p in graph.prerequisites]
        errors.append(f"Expected 12 prerequisites, got {prereq_count}: {edges}")

    # --- Adjunction ---
    if 'add_sub' not in graph.adjunctions:
        errors.append("Missing adjunction: add_sub")
    else:
        adj = graph.adjunctions['add_sub']
        if adj.forward != 'single_digit_addition':
            errors.append(f"add_sub.forward = '{adj.forward}'")
        if adj.inverse != 'single_digit_subtraction':
            errors.append(f"add_sub.inverse = '{adj.inverse}'")

    # --- Topological sort ---
    try:
        order = graph.topological_sort()
        # query_counting must come before combined_counting
        qc_idx = order.index('query_counting')
        cc_idx = order.index('combined_counting')
        if qc_idx >= cc_idx:
            errors.append(
                f"query_counting ({qc_idx}) should come before "
                f"combined_counting ({cc_idx})")

        # two_digit_arithmetic must be last
        if order[-1] != 'two_digit_arithmetic':
            errors.append(f"Last in topological order: {order[-1]}")

    except ValueError as e:
        errors.append(f"Topological sort failed: {e}")

    # --- Validation (type checking) ---
    val_errors = graph.validate(check_types=True)
    if val_errors:
        errors.append(f"Validation errors: {val_errors}")

    return errors


def test_type_validation():
    """Test that undefined types are caught by validation."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept foo
  domain test
  description "uses an undefined type"
  input digit nonexistent_type
  output digit
"""
    graph = parse(text)
    val_errors = graph.validate(check_types=True)

    errors = []
    undefined = [e for e in val_errors if isinstance(e, UndefinedType)]
    if len(undefined) != 1:
        errors.append(
            f"Expected 1 UndefinedType error, got {len(undefined)}: {undefined}")
    elif 'nonexistent_type' not in undefined[0].message:
        errors.append(
            f"UndefinedType should mention 'nonexistent_type': {undefined[0]}")

    return errors


def test_parse_errors():
    """Test that malformed inputs produce useful ParseError."""
    errors = []

    # Missing '=' in type def
    try:
        parse("type digit symbol(0, 1)")
        errors.append("Should have raised ParseError for type without '='")
    except ParseError as e:
        if "requires '='" not in str(e):
            errors.append(f"Wrong error message: {e}")

    # Unmatched paren
    try:
        parse("type digit = symbol(0, 1")
        errors.append("Should have raised ParseError for unmatched '('")
    except ParseError as e:
        if "Unmatched" not in str(e):
            errors.append(f"Wrong error message: {e}")

    # concept without name
    try:
        parse("concept")
        errors.append("Should have raised ParseError for concept without name")
    except ParseError as e:
        if "requires a name" not in str(e):
            errors.append(f"Wrong error message: {e}")

    return errors


def test_curriculum_generation():
    """Test that curriculum generation works end-to-end from .ctkg."""
    graph = build_arithmetic_graph()
    stages = graph.generate_curriculum()

    errors = []
    if not stages:
        errors.append("Empty curriculum")
        return errors

    # First stage should be query_counting (no prereqs)
    if stages[0].concept.name != 'query_counting':
        errors.append(f"First stage: {stages[0].concept.name}")

    # Last stage should be two_digit_arithmetic
    if stages[-1].concept.name != 'two_digit_arithmetic':
        errors.append(f"Last stage: {stages[-1].concept.name}")

    # Should have 9 stages
    if len(stages) != 9:
        errors.append(f"Expected 9 stages, got {len(stages)}")

    # Each stage's replay should only contain ancestors
    for s in stages:
        order_up_to = [st.concept.name for st in stages[:s.number]]
        for r in s.replay_concepts:
            if r not in order_up_to:
                errors.append(
                    f"Stage {s.number} ({s.concept.name}) replays {r} "
                    f"which hasn't been taught yet")

    return errors


def test_summary():
    """Test that summary output works."""
    graph = build_arithmetic_graph()
    summary = graph.summary()

    errors = []
    if '9 concepts' not in summary:
        errors.append(f"Summary should mention 9 concepts: {summary}")
    if 'custom types' not in summary:
        errors.append(f"Summary should mention custom types: {summary}")

    return errors


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ('Type parsing', test_type_parsing),
        ('Arithmetic domain', test_arithmetic_domain),
        ('Type validation', test_type_validation),
        ('Parse errors', test_parse_errors),
        ('Curriculum generation', test_curriculum_generation),
        ('Summary', test_summary),
    ]

    total = 0
    passed = 0
    failed_tests = []

    for name, fn in tests:
        total += 1
        try:
            errors = fn()
            if errors:
                print(f"  FAIL: {name}")
                for e in errors:
                    print(f"    - {e}")
                failed_tests.append(name)
            else:
                print(f"  PASS: {name}")
                passed += 1
        except Exception as e:
            print(f"  ERROR: {name}")
            traceback.print_exc()
            failed_tests.append(name)

    print(f"\n{passed}/{total} tests passed")
    if failed_tests:
        print(f"Failed: {', '.join(failed_tests)}")

    return len(failed_tests) == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
