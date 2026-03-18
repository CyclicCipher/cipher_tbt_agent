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

from experiments.ctkg.parser import parse, parse_file, merge, ParseError
from experiments.ctkg.graph import (
    TypeDef, BUILTIN_TYPES, UndefinedType, SheafViolation,
    ChallengedConjecture, UngroundedAssumption,
    Interface, Challenge, Override, types_compatible, MasteryState,
    CausalEdge, CompositionEdge, InstanceEdge, TemporalEdge,
    MissingPrerequisite,
)
from experiments.ctkg.domains.arithmetic import build_arithmetic_graph
from experiments.ctkg.domains.logic import build_logic_graph


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

    # Concept count (arithmetic.ctkg has grown: counting, ordinality, comparison,
    # single-digit +/-, two-digit ±, multiplication, exponentiation, division,
    # remainder, GCD = 14 concepts as of Phase B)
    if len(graph.concepts) != 14:
        errors.append(f"Expected 14 concepts, got {len(graph.concepts)}")

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
    if prereq_count != 22:
        edges = [(p.source, p.target) for p in graph.prerequisites]
        errors.append(f"Expected 22 prerequisites, got {prereq_count}: {edges}")

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

    # Should have 14 stages (one per concept in arithmetic domain)
    if len(stages) != 14:
        errors.append(f"Expected 14 stages, got {len(stages)}")

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
    if '14 concepts' not in summary:
        errors.append(f"Summary should mention 14 concepts: {summary}")
    if 'custom types' not in summary:
        errors.append(f"Summary should mention custom types: {summary}")

    return errors


def test_logic_domain():
    """Test that logic.ctkg parses and validates."""
    graph = build_logic_graph()

    errors = []

    # Types
    expected_types = ['connective', 'quantifier', 'truth_value', 'prop_var',
                      'literal', 'clause', 'formula', 'proof_step']
    for tname in expected_types:
        if tname not in graph.types:
            errors.append(f"Missing type: {tname}")

    # Concepts
    expected_concepts = ['truth_eval', 'negation', 'compound_eval',
                         'tautology_check', 'modus_ponens']
    for cname in expected_concepts:
        if cname not in graph.concepts:
            errors.append(f"Missing concept: {cname}")

    if len(graph.concepts) != 5:
        errors.append(f"Expected 5 concepts, got {len(graph.concepts)}")

    # All concepts should be in logic domain
    for c in graph.concepts.values():
        if c.domain != 'logic':
            errors.append(f"{c.name}.domain = '{c.domain}', expected 'logic'")

    # Interface should be parsed
    if 'logic' not in graph.interfaces:
        errors.append("Missing interface: logic")
    else:
        iface = graph.interfaces['logic']
        if len(iface.types) != 6:
            errors.append(
                f"Interface exports {len(iface.types)} types, expected 6: "
                f"{iface.types}")
        if len(iface.concepts) != 5:
            errors.append(
                f"Interface exports {len(iface.concepts)} concepts, "
                f"expected 5: {iface.concepts}")

    # Validation
    val_errors = graph.validate(check_types=True)
    if val_errors:
        errors.append(f"Validation errors: {val_errors}")

    return errors


def test_sheaf_compatible_merge():
    """Test that compatible domains merge without sheaf violations."""
    arith = build_arithmetic_graph()
    logic = build_logic_graph()

    errors = []

    # These domains share only builtin types (nat, bool, etc.)
    # They should merge cleanly
    violations = arith.sheaf_check(logic)
    if violations:
        errors.append(f"Unexpected sheaf violations: {violations}")

    # Perform sheaf merge
    violations = arith.sheaf_merge(logic)
    if violations:
        errors.append(f"Sheaf merge failed: {violations}")

    # After merge, should have concepts from both domains
    if 'successor' not in arith.concepts:
        errors.append("Lost arithmetic concept after merge")
    if 'truth_eval' not in arith.concepts:
        errors.append("Logic concept not merged")

    # Should have types from both domains
    if 'digit' not in arith.types:
        errors.append("Lost arithmetic type after merge")
    if 'connective' not in arith.types:
        errors.append("Logic type not merged")

    # Should have interfaces from both domains
    if 'arithmetic' not in arith.interfaces:
        errors.append("Lost arithmetic interface after merge")
    if 'logic' not in arith.interfaces:
        errors.append("Logic interface not merged")

    # Merged graph should still validate
    val_errors = arith.validate(check_types=True)
    if val_errors:
        errors.append(f"Merged graph validation errors: {val_errors}")

    return errors


def test_sheaf_violation():
    """Test that incompatible type definitions produce SheafViolation."""
    errors = []

    # Create two graphs with conflicting type definitions
    graph_a = parse("""
type status = symbol(OK, ERR)

concept check_a
  domain domain_a
  description "uses status"
  input status
  output bool
""")

    graph_b = parse("""
type status = symbol(GOOD, BAD, UNKNOWN)

concept check_b
  domain domain_b
  description "uses status differently"
  input status
  output bool
""")

    # sheaf_check should detect the conflict
    violations = graph_a.sheaf_check(graph_b)
    if len(violations) != 1:
        errors.append(
            f"Expected 1 SheafViolation, got {len(violations)}: {violations}")
    elif not isinstance(violations[0], SheafViolation):
        errors.append(
            f"Expected SheafViolation, got {type(violations[0])}: "
            f"{violations[0]}")
    elif 'status' not in violations[0].message:
        errors.append(
            f"SheafViolation should mention 'status': {violations[0]}")

    # sheaf_merge should refuse
    violations = graph_a.sheaf_merge(graph_b)
    if not violations:
        errors.append("sheaf_merge should have refused incompatible types")

    # graph_a should be unchanged
    if 'check_b' in graph_a.concepts:
        errors.append("graph_a was modified despite sheaf violation")

    return errors


def test_type_compatibility():
    """Test the types_compatible function directly."""
    errors = []

    # Same definitions should be compatible
    a = TypeDef('digit', 'symbol', ['0', '1', '2'], {'ordered'})
    b = TypeDef('digit', 'symbol', ['0', '1', '2'], {'ordered'})
    if not types_compatible(a, b):
        errors.append("Identical type defs should be compatible")

    # Different constructor
    c = TypeDef('digit', 'nat', [], set())
    if types_compatible(a, c):
        errors.append("Different constructors should be incompatible")

    # Different params
    d = TypeDef('digit', 'symbol', ['0', '1'], {'ordered'})
    if types_compatible(a, d):
        errors.append("Different params should be incompatible")

    # Different annotations
    e = TypeDef('digit', 'symbol', ['0', '1', '2'], {'metric'})
    if types_compatible(a, e):
        errors.append("Different annotations should be incompatible")

    return errors


def test_interface_parsing():
    """Test that interface blocks parse correctly."""
    text = """
type digit = symbol(0, 1, 2)

concept foo
  domain test
  description "test"
  input digit
  output bool

interface test_domain
  exports types digit
  exports concepts foo
"""
    graph = parse(text)

    errors = []

    if 'test_domain' not in graph.interfaces:
        errors.append("Interface 'test_domain' not parsed")
        return errors

    iface = graph.interfaces['test_domain']
    if iface.name != 'test_domain':
        errors.append(f"Interface name: {iface.name}")
    if iface.types != ['digit']:
        errors.append(f"Interface types: {iface.types}")
    if iface.concepts != ['foo']:
        errors.append(f"Interface concepts: {iface.concepts}")

    return errors


def test_transfer_probability_parsing():
    """Test that transfer_probability parses from requires syntax."""
    text = """
type digit = symbol(0, 1, 2)

concept base
  domain test
  description "base skill"
  input digit
  output digit
  atomic

concept derived_hard
  domain test
  description "hard prerequisite (default 1.0)"
  input digit
  output digit
  requires base via "foundation"

concept derived_soft
  domain test
  description "soft prerequisite"
  input digit
  output digit
  requires base via "partial foundation" [0.75]
"""
    graph = parse(text)

    errors = []

    # Hard prerequisite: default transfer_probability = 1.0
    hard_prereqs = [p for p in graph.prerequisites
                    if p.target == 'derived_hard']
    if len(hard_prereqs) != 1:
        errors.append(f"Expected 1 hard prereq, got {len(hard_prereqs)}")
    elif hard_prereqs[0].transfer_probability != 1.0:
        errors.append(
            f"Hard prereq transfer_probability: "
            f"{hard_prereqs[0].transfer_probability} != 1.0")

    # Soft prerequisite: explicit transfer_probability = 0.75
    soft_prereqs = [p for p in graph.prerequisites
                    if p.target == 'derived_soft']
    if len(soft_prereqs) != 1:
        errors.append(f"Expected 1 soft prereq, got {len(soft_prereqs)}")
    elif soft_prereqs[0].transfer_probability != 0.75:
        errors.append(
            f"Soft prereq transfer_probability: "
            f"{soft_prereqs[0].transfer_probability} != 0.75")
    elif soft_prereqs[0].role != 'partial foundation':
        errors.append(
            f"Soft prereq role: '{soft_prereqs[0].role}' != 'partial foundation'")

    return errors


def test_d_separation():
    """Test d-separation (Bayes-ball algorithm) on the arithmetic graph."""
    graph = build_arithmetic_graph()

    errors = []

    # Chain: query_counting -> combined_counting -> successor
    # If we observe combined_counting, query_counting and successor
    # should be d-separated.
    if not graph.d_separated('query_counting', 'successor',
                              {'combined_counting'}):
        errors.append(
            "query_counting and successor should be d-separated "
            "given combined_counting")

    # Without observing combined_counting, they should NOT be d-separated
    # (information flows through the chain).
    if graph.d_separated('query_counting', 'successor', set()):
        errors.append(
            "query_counting and successor should NOT be d-separated "
            "given nothing (chain is open)")

    # Fork: successor and predecessor share parent combined_counting.
    # Observing combined_counting blocks the fork.
    if not graph.d_separated('successor', 'predecessor',
                              {'combined_counting'}):
        errors.append(
            "successor and predecessor should be d-separated "
            "given combined_counting (fork blocked)")

    # Without observing combined_counting, successor and predecessor
    # are NOT d-separated (fork is open).
    if graph.d_separated('successor', 'predecessor', set()):
        errors.append(
            "successor and predecessor should NOT be d-separated "
            "given nothing (fork is open)")

    # Collider: single_digit_addition has parents successor and comparison.
    # Without observing single_digit_addition, successor and comparison
    # ARE d-separated via the collider path (collider blocks by default).
    # But comparison also depends on successor, so they're NOT d-separated
    # via the chain path successor -> comparison.
    # Test a cleaner collider: two_digit_single_arithmetic has parents
    # single_digit_addition and single_digit_subtraction.
    # The only path between addition and subtraction goes through their
    # shared ancestors OR through two_digit_single_arithmetic (collider).
    # Observing all shared ancestors blocks the chain paths.
    # If we also observe two_digit_single_arithmetic, the collider opens.
    shared = {'successor', 'predecessor', 'comparison', 'combined_counting',
              'query_counting'}
    if not graph.d_separated('single_digit_addition',
                              'single_digit_subtraction', shared):
        errors.append(
            "addition and subtraction should be d-separated given "
            "shared ancestors (collider closed)")

    # Observing two_digit_single_arithmetic (the collider child) opens it
    shared_plus_collider = shared | {'two_digit_single_arithmetic'}
    if graph.d_separated('single_digit_addition',
                          'single_digit_subtraction',
                          shared_plus_collider):
        errors.append(
            "addition and subtraction should NOT be d-separated when "
            "collider child is observed")

    return errors


def test_entropy():
    """Test entropy computations on a graph with known problem counts."""
    import math

    text = """
type digit = symbol(0, 1, 2, 3)

concept base
  domain test
  description "base"
  input digit
  output digit
  atomic

concept mid
  domain test
  description "middle"
  input digit
  output digit
  requires base via "foundation" [0.8]

concept top
  domain test
  description "top"
  input digit
  output digit
  requires mid via "bridge"
"""
    graph = parse(text)
    # Set problem counts for entropy computation
    graph.concepts['base'].n_problems = 16
    graph.concepts['mid'].n_problems = 32
    graph.concepts['top'].n_problems = 64

    errors = []

    # H(base) = log2(16) = 4.0
    h_base = graph.concept_entropy('base')
    if abs(h_base - 4.0) > 1e-9:
        errors.append(f"H(base) = {h_base}, expected 4.0")

    # H(mid) = log2(32) = 5.0
    h_mid = graph.concept_entropy('mid')
    if abs(h_mid - 5.0) > 1e-9:
        errors.append(f"H(mid) = {h_mid}, expected 5.0")

    # H(top) = log2(64) = 6.0
    h_top = graph.concept_entropy('top')
    if abs(h_top - 6.0) > 1e-9:
        errors.append(f"H(top) = {h_top}, expected 6.0")

    # Conditional entropy: H(mid | {base}) = H(mid) - H(base) * 0.8
    # = 5.0 - 4.0 * 0.8 = 5.0 - 3.2 = 1.8
    h_mid_given_base = graph.conditional_entropy('mid', {'base'})
    if abs(h_mid_given_base - 1.8) > 1e-9:
        errors.append(
            f"H(mid|base) = {h_mid_given_base}, expected 1.8")

    # Mutual information: I(mid; base) = H(mid) - H(mid|base) = 3.2
    mi = graph.mutual_information('mid', {'base'})
    if abs(mi - 3.2) > 1e-9:
        errors.append(f"I(mid; base) = {mi}, expected 3.2")

    # H(mid | {}) = H(mid) (no prerequisites learned)
    h_mid_given_nothing = graph.conditional_entropy('mid', set())
    if abs(h_mid_given_nothing - 5.0) > 1e-9:
        errors.append(
            f"H(mid|nothing) = {h_mid_given_nothing}, expected 5.0")

    # Information flow
    flows = graph.information_flow()
    expected_flow = 4.0 * 0.8  # H(base) * transfer_prob
    if abs(flows.get('base->mid', 0) - expected_flow) > 1e-9:
        errors.append(
            f"Flow base->mid = {flows.get('base->mid')}, "
            f"expected {expected_flow}")

    return errors


def test_intervention():
    """Test do-calculus / diagram surgery."""
    graph = build_arithmetic_graph()

    errors = []

    # Intervene on combined_counting: removes its incoming edge
    # from query_counting
    mutilated = graph.intervene({'combined_counting'})

    # combined_counting should have no parents in mutilated graph
    cc_parents = mutilated._parents.get('combined_counting', set())
    if cc_parents:
        errors.append(
            f"Intervened combined_counting still has parents: {cc_parents}")

    # query_counting should have no children pointing to combined_counting
    qc_children = mutilated._children.get('query_counting', set())
    if 'combined_counting' in qc_children:
        errors.append(
            "query_counting still has combined_counting as child "
            "after intervention")

    # Original graph should be unmodified
    orig_cc_parents = graph._parents.get('combined_counting', set())
    if 'query_counting' not in orig_cc_parents:
        errors.append("Original graph was modified by intervention")

    # All concepts should still be present
    if set(mutilated.concepts.keys()) != set(graph.concepts.keys()):
        errors.append("Intervention changed the concept set")

    # Remaining edges should be intact
    orig_non_cc = [p for p in graph.prerequisites
                   if p.target != 'combined_counting']
    mut_edges = [(p.source, p.target) for p in mutilated.prerequisites]
    for p in orig_non_cc:
        if (p.source, p.target) not in mut_edges:
            errors.append(
                f"Edge {p.source}->{p.target} missing after intervention")

    # After intervention, combined_counting and query_counting should
    # be d-separated given anything (no connecting edges)
    if not mutilated.d_separated('query_counting', 'combined_counting',
                                  set()):
        errors.append(
            "After intervening on combined_counting, it should be "
            "d-separated from query_counting")

    return errors


def test_mastery_state():
    """Test MasteryState operations."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept A
  domain test
  description "root"
  input digit
  output digit
  atomic

concept B
  domain test
  description "mid"
  input digit
  output digit
  requires A via "foundation" [0.9]

concept C
  domain test
  description "top"
  input digit
  output digit
  requires A via "also needed"
  requires B via "bridge" [0.8]
"""
    graph = parse(text)

    errors = []

    state = graph.mastery_state()

    # Initially all mastery is 0
    if any(v != 0.0 for v in state.levels.values()):
        errors.append(f"Initial mastery not all zero: {state.levels}")

    # A has no prereqs, so readiness = 1.0
    if state.expected_readiness('A') != 1.0:
        errors.append(
            f"A readiness = {state.expected_readiness('A')}, expected 1.0")

    # B depends on A (mastery 0), so readiness = 0 * 0.9 = 0.0
    if state.expected_readiness('B') != 0.0:
        errors.append(
            f"B readiness = {state.expected_readiness('B')}, expected 0.0")

    # Frontier with threshold 0.8: only A (no prereqs)
    front = state.frontier(threshold=0.8)
    if front != {'A'}:
        errors.append(f"Initial frontier: {front}, expected {{'A'}}")

    # Learn A
    state.observe('A', 0.95)

    # Now B readiness = 0.95 * 0.9 = 0.855
    b_ready = state.expected_readiness('B')
    if abs(b_ready - 0.855) > 1e-9:
        errors.append(f"B readiness after A=0.95: {b_ready}, expected 0.855")

    # C readiness = min(0.95 * 1.0, 0.0 * 0.8) = min(0.95, 0.0) = 0.0
    # (B not yet learned)
    c_ready = state.expected_readiness('C')
    if abs(c_ready - 0.0) > 1e-9:
        errors.append(f"C readiness: {c_ready}, expected 0.0")

    # Frontier: A is mastered, B is ready (0.855 > 0.8), C is not (0.0)
    front = state.frontier(threshold=0.8)
    if front != {'B'}:
        errors.append(f"Frontier after learning A: {front}, expected {{'B'}}")

    # Learn B
    state.observe('B', 0.95)

    # C readiness = min(0.95 * 1.0, 0.95 * 0.8) = min(0.95, 0.76) = 0.76
    c_ready = state.expected_readiness('C')
    if abs(c_ready - 0.76) > 1e-9:
        errors.append(
            f"C readiness after B=0.95: {c_ready}, expected 0.76")

    # Frontier with threshold 0.7: C should be ready
    front = state.frontier(threshold=0.7)
    if front != {'C'}:
        errors.append(
            f"Frontier(0.7) after A,B: {front}, expected {{'C'}}")

    # Frontier with threshold 0.8: C not ready (0.76 < 0.8)
    front = state.frontier(threshold=0.8)
    if front != set():
        errors.append(
            f"Frontier(0.8) after A,B: {front}, expected empty")

    return errors


def test_epistemic_tiers():
    """Test that epistemic tiers parse correctly from DSL."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept axiom_concept
  domain test
  description "a mathematical necessity"
  tier axiom
  input digit
  output digit
  atomic

concept theorem_concept
  domain test
  description "derived from premises"
  tier theorem
  assumes some_assumption
  input digit
  output digit
  requires axiom_concept via "foundation"

concept conjecture_concept
  domain test
  description "widely believed but unproven"
  tier conjecture
  input digit
  output digit
  requires theorem_concept via "derived"

concept heuristic_concept
  domain test
  description "dogs have 4 legs"
  tier heuristic
  input digit
  output digit
  default legs = 4
  default color = brown
"""
    graph = parse(text)

    errors = []

    # Check tiers
    ax = graph.concepts.get('axiom_concept')
    if not ax or ax.tier != 'axiom':
        errors.append(f"axiom_concept tier: {ax.tier if ax else 'missing'}")

    th = graph.concepts.get('theorem_concept')
    if not th or th.tier != 'theorem':
        errors.append(f"theorem_concept tier: {th.tier if th else 'missing'}")
    if not th or th.assumes != ['some_assumption']:
        errors.append(f"theorem_concept assumes: {th.assumes if th else 'missing'}")

    conj = graph.concepts.get('conjecture_concept')
    if not conj or conj.tier != 'conjecture':
        errors.append(f"conjecture_concept tier: {conj.tier if conj else 'missing'}")

    heur = graph.concepts.get('heuristic_concept')
    if not heur or heur.tier != 'heuristic':
        errors.append(f"heuristic_concept tier: {heur.tier if heur else 'missing'}")
    if not heur or heur.defaults != {'legs': '4', 'color': 'brown'}:
        errors.append(f"heuristic_concept defaults: {heur.defaults if heur else 'missing'}")

    # Default tier should be 'theorem' (axiom_concept explicitly sets 'axiom')
    # theorem_concept explicitly sets 'theorem' but that's also the default

    return errors


def test_challenge_edges():
    """Test challenge edges parse and validate correctly."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept negative_energy
  domain physics
  tier conjecture
  description "warp drives require exotic matter"
  input digit
  output digit

concept lentz_soliton
  domain physics
  tier theorem
  description "positive-energy warp metric"
  input digit
  output digit
  challenges negative_energy via "positive-energy reformulation"

concept standard_model
  domain physics
  tier axiom
  description "well-tested axiom"
  input digit
  output digit
"""
    graph = parse(text)

    errors = []

    # Check that challenge was parsed
    if len(graph.challenges) != 1:
        errors.append(f"Expected 1 challenge, got {len(graph.challenges)}")
        return errors

    ch = graph.challenges[0]
    if ch.source != 'lentz_soliton':
        errors.append(f"Challenge source: {ch.source}")
    if ch.target != 'negative_energy':
        errors.append(f"Challenge target: {ch.target}")
    if ch.role != 'positive-energy reformulation':
        errors.append(f"Challenge role: {ch.role}")

    # challenged_concepts() should return the challenged concept
    challenged = graph.challenged_concepts()
    if 'negative_energy' not in challenged:
        errors.append(f"challenged_concepts missing negative_energy: {challenged}")
    elif len(challenged['negative_energy']) != 1:
        errors.append(f"Expected 1 challenger, got {len(challenged['negative_energy'])}")

    # Validation should produce a ChallengedConjecture warning
    val_errors = graph.validate(check_types=True)
    challenged_warnings = [e for e in val_errors if isinstance(e, ChallengedConjecture)]
    if len(challenged_warnings) != 1:
        errors.append(
            f"Expected 1 ChallengedConjecture warning, got "
            f"{len(challenged_warnings)}: {val_errors}")
    elif 'negative_energy' not in challenged_warnings[0].message:
        errors.append(
            f"Warning should mention negative_energy: {challenged_warnings[0]}")

    return errors


def test_overrides_fido():
    """Test the Fido problem: defaults with instance overrides."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept dog
  domain biology
  tier heuristic
  description "typical dog properties"
  input digit
  output digit
  default legs = 4
  default tail = 1

concept fido
  domain biology
  description "a specific dog"
  input digit
  output digit
  overrides dog with legs = 3 via "lost a leg in accident"

concept rex
  domain biology
  description "another dog"
  input digit
  output digit
"""
    graph = parse(text)

    errors = []

    # Check override was parsed
    if len(graph.overrides) != 1:
        errors.append(f"Expected 1 override, got {len(graph.overrides)}")
        return errors

    ov = graph.overrides[0]
    if ov.instance != 'fido':
        errors.append(f"Override instance: {ov.instance}")
    if ov.default_concept != 'dog':
        errors.append(f"Override default_concept: {ov.default_concept}")
    if ov.property != 'legs':
        errors.append(f"Override property: {ov.property}")
    if ov.value != '3':
        errors.append(f"Override value: {ov.value}")
    if ov.reason != 'lost a leg in accident':
        errors.append(f"Override reason: {ov.reason}")

    # resolve_default for fido should return 3 (override)
    legs_fido = graph.resolve_default('dog', 'legs', 'fido')
    if legs_fido != '3':
        errors.append(f"resolve_default(dog, legs, fido) = {legs_fido}, expected '3'")

    # resolve_default for rex should return 4 (default, no override)
    legs_rex = graph.resolve_default('dog', 'legs', 'rex')
    if legs_rex != '4':
        errors.append(f"resolve_default(dog, legs, rex) = {legs_rex}, expected '4'")

    # resolve_default for tail should return 1 (no override for any instance)
    tail_fido = graph.resolve_default('dog', 'tail', 'fido')
    if tail_fido != '1':
        errors.append(f"resolve_default(dog, tail, fido) = {tail_fido}, expected '1'")

    # resolve_default without instance should return default
    legs_default = graph.resolve_default('dog', 'legs')
    if legs_default != '4':
        errors.append(f"resolve_default(dog, legs) = {legs_default}, expected '4'")

    return errors


def test_assumption_conditioned_prereqs():
    """Test assumption-conditioned prerequisites parse correctly."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept original_metric
  domain physics
  tier axiom
  description "original Alcubierre metric"
  input digit
  output digit
  atomic

concept negative_energy
  domain physics
  tier conjecture
  description "warp drives need negative energy"
  input digit
  output digit

concept warp_drive
  domain physics
  description "FTL via spacetime warping"
  input digit
  output digit
  requires negative_energy via "metric solution" assuming original_metric [derived]
  requires original_metric via "field equations"
"""
    graph = parse(text)

    errors = []

    # Find the assumption-conditioned prerequisite
    cond_prereqs = [p for p in graph.prerequisites if p.assuming is not None]
    if len(cond_prereqs) != 1:
        errors.append(f"Expected 1 conditioned prereq, got {len(cond_prereqs)}")
        return errors

    p = cond_prereqs[0]
    if p.assuming != 'original_metric':
        errors.append(f"assuming: {p.assuming}")
    if p.assumption_status != 'derived':
        errors.append(f"assumption_status: {p.assumption_status}")
    if p.source != 'negative_energy':
        errors.append(f"source: {p.source}")
    if p.target != 'warp_drive':
        errors.append(f"target: {p.target}")

    # assumption_dependents should find the prereq
    deps = graph.assumption_dependents('original_metric')
    if 'negative_energy->warp_drive' not in deps['prerequisites']:
        errors.append(f"assumption_dependents prereqs: {deps['prerequisites']}")

    return errors


def test_what_if_not():
    """Test counterfactual exploration via what_if_not()."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept A
  domain test
  description "root"
  input digit
  output digit
  atomic

concept B
  domain test
  description "blocker"
  input digit
  output digit
  requires A via "foundation"

concept C
  domain test
  description "blocked by B"
  input digit
  output digit
  requires B via "needs B"

concept D
  domain test
  description "blocked by both A and B"
  input digit
  output digit
  requires A via "needs A"
  requires B via "needs B"

concept E
  domain test
  description "only needs A, not B"
  input digit
  output digit
  requires A via "needs A"
"""
    graph = parse(text)

    errors = []

    # what_if_not(B): C depends only on B, so C should be opened.
    # D depends on both A and B. With B removed, D still needs A,
    # and A exists in the reduced graph, so D becomes frontier-eligible.
    # E doesn't depend on B at all, so it's not "blocked_in_original".
    opened = graph.what_if_not('B')

    # C and D are descendants of B, so both were blocked.
    # In the reduced graph (without B), C has no parents (B was removed),
    # so all its prereqs are satisfied trivially → opened.
    # D has parent A (which exists) → opened.
    if 'C' not in opened:
        errors.append(f"C should be opened when B removed: {opened}")
    if 'D' not in opened:
        errors.append(f"D should be opened when B removed: {opened}")
    if 'E' in opened:
        errors.append(f"E should NOT be in opened (not a descendant of B): {opened}")
    if 'A' in opened:
        errors.append(f"A should NOT be in opened: {opened}")

    return errors


def test_ungrounded_assumption():
    """Test that ungrounded assumptions are caught by validation."""
    text = """
type digit = symbol(0, 1, 2, 3)

concept foo
  domain test
  description "has a dangling assumption"
  tier theorem
  assumes nonexistent_assumption
  input digit
  output digit
"""
    graph = parse(text)
    val_errors = graph.validate(check_types=True)

    errors = []
    ungrounded = [e for e in val_errors if isinstance(e, UngroundedAssumption)]
    if len(ungrounded) != 1:
        errors.append(
            f"Expected 1 UngroundedAssumption, got {len(ungrounded)}: {val_errors}")
    elif 'nonexistent_assumption' not in ungrounded[0].message:
        errors.append(f"Should mention the assumption: {ungrounded[0]}")

    return errors


def test_tier_parse_error():
    """Test that invalid tier values produce ParseError."""
    errors = []

    try:
        parse("""
type digit = symbol(0, 1)
concept foo
  domain test
  description "bad tier"
  tier invalid_tier
  input digit
  output digit
""")
        errors.append("Should have raised ParseError for invalid tier")
    except ParseError as e:
        if "Invalid tier" not in str(e):
            errors.append(f"Wrong error message: {e}")

    return errors


# ---------------------------------------------------------------------------
# Phase B: new edge type tests
# ---------------------------------------------------------------------------

def test_causal_edges():
    """Test that 'causes' edges parse correctly with all optional qualifiers."""
    text = """
concept state_B
  domain test
  description "result state"

concept side_effect
  domain test
  description "optional side effect"

concept guarded_result
  domain test
  description "conditional result"

concept trigger
  domain test
  description "trigger concept"
  causes state_B via "produces B" [0.80] delay=3
  causes side_effect via "causes side effect"
  causes guarded_result via "conditional" [0.60] guard="health < 10"
"""
    graph = parse(text)
    errors = []

    trigger_edges = [e for e in graph.causal_edges if e.source == 'trigger']
    if len(trigger_edges) != 3:
        errors.append(
            f"Expected 3 causal edges from trigger, got {len(trigger_edges)}")
        return errors

    by_target = {e.target: e for e in trigger_edges}

    # causes state_B via "produces B" [0.80] delay=3
    e1 = by_target.get('state_B')
    if not e1:
        errors.append("Missing causal edge trigger -> state_B")
    else:
        if abs(e1.probability - 0.80) > 1e-9:
            errors.append(
                f"state_B probability: {e1.probability}, expected 0.80")
        if e1.delay_steps != 3:
            errors.append(
                f"state_B delay_steps: {e1.delay_steps}, expected 3")
        if e1.role != 'produces B':
            errors.append(f"state_B role: {e1.role!r}, expected 'produces B'")
        if e1.guard != '':
            errors.append(f"state_B guard: {e1.guard!r}, expected ''")

    # causes side_effect via "causes side effect"  (defaults)
    e2 = by_target.get('side_effect')
    if not e2:
        errors.append("Missing causal edge trigger -> side_effect")
    else:
        if abs(e2.probability - 1.0) > 1e-9:
            errors.append(
                f"side_effect probability: {e2.probability}, expected 1.0")
        if e2.delay_steps != 0:
            errors.append(
                f"side_effect delay_steps: {e2.delay_steps}, expected 0")

    # causes guarded_result via "conditional" [0.60] guard="health < 10"
    e3 = by_target.get('guarded_result')
    if not e3:
        errors.append("Missing causal edge trigger -> guarded_result")
    else:
        if abs(e3.probability - 0.60) > 1e-9:
            errors.append(
                f"guarded_result probability: {e3.probability}, expected 0.60")
        if e3.guard != 'health < 10':
            errors.append(
                f"guarded_result guard: {e3.guard!r}, expected 'health < 10'")

    return errors


def test_composition_edges():
    """Test that 'composes_into' edges parse correctly."""
    text = """
concept crafting_table
  domain minecraft
  description "a crafting table"

concept wooden_pickaxe
  domain minecraft
  description "a wooden pickaxe"

concept log
  domain minecraft
  description "a log"
  composes_into crafting_table via "4 planks to table" [0.95]
  composes_into wooden_pickaxe via "planks for head" [0.90]
"""
    graph = parse(text)
    errors = []

    log_edges = [e for e in graph.composition_edges if e.source == 'log']
    if len(log_edges) != 2:
        errors.append(
            f"Expected 2 composition edges from log, got {len(log_edges)}")
        return errors

    by_target = {e.target: e for e in log_edges}

    e1 = by_target.get('crafting_table')
    if not e1:
        errors.append("Missing composition edge log -> crafting_table")
    else:
        if abs(e1.probability - 0.95) > 1e-9:
            errors.append(
                f"crafting_table probability: {e1.probability}, expected 0.95")
        if e1.role != '4 planks to table':
            errors.append(
                f"crafting_table role: {e1.role!r}, expected '4 planks to table'")

    e2 = by_target.get('wooden_pickaxe')
    if not e2:
        errors.append("Missing composition edge log -> wooden_pickaxe")
    else:
        if abs(e2.probability - 0.90) > 1e-9:
            errors.append(
                f"wooden_pickaxe probability: {e2.probability}, expected 0.90")

    return errors


def test_instance_of_and_analogous():
    """Test 'instance_of' edges and analogous_concepts() span traversal."""
    text = """
concept animal
  domain biology
  description "abstract animal"

concept vehicle
  domain biology
  description "abstract vehicle"

concept dog
  domain biology
  description "a dog"
  instance_of animal via "dogs are animals"

concept cat
  domain biology
  description "a cat"
  instance_of animal via "cats are animals"

concept car
  domain biology
  description "a car"
  instance_of vehicle via "cars are vehicles"
"""
    graph = parse(text)
    errors = []

    if len(graph.instance_edges) != 3:
        errors.append(
            f"Expected 3 instance edges, got {len(graph.instance_edges)}")
        return errors

    dog_edge = next(
        (e for e in graph.instance_edges if e.source == 'dog'), None)
    if not dog_edge:
        errors.append("Missing instance_of edge for dog")
    elif dog_edge.target != 'animal':
        errors.append(f"dog -> {dog_edge.target!r}, expected 'animal'")
    elif dog_edge.role != 'dogs are animals':
        errors.append(f"dog role: {dog_edge.role!r}, expected 'dogs are animals'")

    # analogous_concepts(dog): shares 'animal' supertype with cat, not car
    dog_analogs = graph.analogous_concepts('dog')
    if 'cat' not in dog_analogs:
        errors.append(
            f"cat should be analogous to dog via animal: {dog_analogs}")
    if 'car' in dog_analogs:
        errors.append(
            f"car should NOT be analogous to dog (different supertype): "
            f"{dog_analogs}")
    if 'dog' in dog_analogs:
        errors.append(f"dog should NOT be its own analog: {dog_analogs}")
    if 'animal' in dog_analogs:
        errors.append(
            f"supertype 'animal' should not appear in analogs: {dog_analogs}")

    # analogous_concepts(cat): should find dog
    cat_analogs = graph.analogous_concepts('cat')
    if 'dog' not in cat_analogs:
        errors.append(f"dog should be analogous to cat: {cat_analogs}")

    # analogous_concepts(car): no shared supertypes with dog/cat
    car_analogs = graph.analogous_concepts('car')
    if 'dog' in car_analogs or 'cat' in car_analogs:
        errors.append(
            f"car analogs should not include animals: {car_analogs}")

    # analogous_concepts(animal): no supertype → empty
    animal_analogs = graph.analogous_concepts('animal')
    if animal_analogs:
        errors.append(
            f"animal has no supertype, analogs should be empty: {animal_analogs}")

    return errors


def test_temporal_edges():
    """Test that 'precedes' edges parse correctly."""
    text = """
concept place_table
  domain minecraft
  description "place crafting table"

concept open_table
  domain minecraft
  description "open crafting table UI"

concept sequence_start
  domain minecraft
  description "trigger sequence"
  precedes place_table via "must place before opening"
  precedes open_table via "opens UI after placing"
"""
    graph = parse(text)
    errors = []

    temporal_edges = graph.temporal_edges
    if len(temporal_edges) != 2:
        errors.append(
            f"Expected 2 temporal edges, got {len(temporal_edges)}")
        return errors

    by_target = {e.target: e for e in temporal_edges}

    e1 = by_target.get('place_table')
    if not e1:
        errors.append("Missing temporal edge sequence_start -> place_table")
    else:
        if e1.source != 'sequence_start':
            errors.append(
                f"Expected source 'sequence_start', got {e1.source!r}")
        if e1.role != 'must place before opening':
            errors.append(f"role: {e1.role!r}")

    e2 = by_target.get('open_table')
    if not e2:
        errors.append("Missing temporal edge sequence_start -> open_table")
    else:
        if e2.role != 'opens UI after placing':
            errors.append(f"open_table role: {e2.role!r}")

    return errors


def test_causal_descendants():
    """Test causal_descendants() traverses CausalEdge (not Prerequisite) edges."""
    text = """
concept A
  domain test
  description "root event"
  causes B via "A triggers B"

concept B
  domain test
  description "intermediate event"
  causes C via "B triggers C" [0.90]
  causes D via "B triggers D" [0.70]

concept C
  domain test
  description "terminal event C"

concept D
  domain test
  description "intermediate D"
  causes E via "D triggers E" [0.80]

concept E
  domain test
  description "deep terminal"

concept X
  domain test
  description "unreachable from A causally"
"""
    graph = parse(text)
    errors = []

    # causal_descendants('A') should reach B, C, D, E but not X or A itself
    desc_A = graph.causal_descendants('A')
    for expected in ('B', 'C', 'D', 'E'):
        if expected not in desc_A:
            errors.append(
                f"{expected} should be a causal descendant of A: {desc_A}")
    if 'A' in desc_A:
        errors.append("A should not be its own causal descendant")
    if 'X' in desc_A:
        errors.append(f"X should not be a causal descendant of A: {desc_A}")

    # causal_descendants('B') should reach C, D, E but not A
    desc_B = graph.causal_descendants('B')
    for expected in ('C', 'D', 'E'):
        if expected not in desc_B:
            errors.append(
                f"{expected} should be a causal descendant of B: {desc_B}")
    if 'A' in desc_B:
        errors.append("A should not be a causal descendant of B")

    # causal_descendants('E') is empty (terminal)
    desc_E = graph.causal_descendants('E')
    if desc_E:
        errors.append(
            f"E has no outgoing causal edges, expected empty: {desc_E}")

    # causal_descendants('X') is empty
    desc_X = graph.causal_descendants('X')
    if desc_X:
        errors.append(
            f"X has no outgoing causal edges, expected empty: {desc_X}")

    return errors


def test_phase_b_validation():
    """Test that validate() catches missing endpoints in new Phase B edge types."""
    errors = []

    # CausalEdge with missing target
    graph_c = parse("""
concept source_c
  domain test
  description "source"
  causes nonexistent_causal_target via "broken"
""")
    errs_c = graph_c.validate(check_types=False)
    missing_c = [e for e in errs_c
                 if isinstance(e, MissingPrerequisite) and 'CausalEdge' in e.message]
    if len(missing_c) != 1:
        errors.append(
            f"Expected 1 CausalEdge MissingPrerequisite, "
            f"got {len(missing_c)}: {errs_c}")

    # CompositionEdge with missing target
    graph_comp = parse("""
concept ingredient
  domain test
  description "ingredient"
  composes_into nonexistent_product via "broken composition"
""")
    errs_comp = graph_comp.validate(check_types=False)
    missing_comp = [e for e in errs_comp
                    if isinstance(e, MissingPrerequisite)
                    and 'CompositionEdge' in e.message]
    if len(missing_comp) != 1:
        errors.append(
            f"Expected 1 CompositionEdge MissingPrerequisite, "
            f"got {len(missing_comp)}: {errs_comp}")

    # InstanceEdge with missing supertype
    graph_i = parse("""
concept child_c
  domain test
  description "child"
  instance_of nonexistent_supertype via "broken instance"
""")
    errs_i = graph_i.validate(check_types=False)
    missing_i = [e for e in errs_i
                 if isinstance(e, MissingPrerequisite) and 'InstanceEdge' in e.message]
    if len(missing_i) != 1:
        errors.append(
            f"Expected 1 InstanceEdge MissingPrerequisite, "
            f"got {len(missing_i)}: {errs_i}")

    # TemporalEdge with missing target
    graph_t = parse("""
concept first_step
  domain test
  description "first step"
  precedes nonexistent_second_step via "broken temporal"
""")
    errs_t = graph_t.validate(check_types=False)
    missing_t = [e for e in errs_t
                 if isinstance(e, MissingPrerequisite) and 'TemporalEdge' in e.message]
    if len(missing_t) != 1:
        errors.append(
            f"Expected 1 TemporalEdge MissingPrerequisite, "
            f"got {len(missing_t)}: {errs_t}")

    return errors


def test_minecraft_domain_phase_b():
    """Test that minecraft.ctkg loads cleanly with Phase B edges present."""
    import os
    mc_path = os.path.join(
        os.path.dirname(__file__), 'domains', 'minecraft.ctkg')
    graph = parse_file(mc_path)

    errors = []

    # Phase A sanity: key concepts present
    expected_concepts = [
        'motor_forward', 'motor_attack', 'observe_rgb', 'observe_health',
        'log_detection', 'break_log', 'obtain_log', 'craft_planks',
        'craft_crafting_table', 'craft_wooden_pickaxe', 'mine_stone',
        'achieve_getting_wood', 'wooden_material', 'tool',
    ]
    for c in expected_concepts:
        if c not in graph.concepts:
            errors.append(f"Missing concept: {c}")

    # Phase B: causal edges — break_log causally produces obtain_log
    causal_targets_from_break = {
        e.target for e in graph.causal_edges if e.source == 'break_log'}
    if 'obtain_log' not in causal_targets_from_break:
        errors.append(
            f"break_log should causally produce obtain_log; "
            f"got: {causal_targets_from_break}")

    # causal_descendants of break_log should include obtain_log and craft_planks
    if 'obtain_log' in graph.concepts and 'craft_planks' in graph.concepts:
        desc = graph.causal_descendants('break_log')
        for expected in ('obtain_log', 'craft_planks'):
            if expected not in desc:
                errors.append(
                    f"{expected} should be a causal descendant of break_log: "
                    f"{desc}")

    # Phase B: instance_of — obtain_log is instance_of wooden_material
    instance_targets_from_obtain = {
        e.target for e in graph.instance_edges if e.source == 'obtain_log'}
    if 'wooden_material' not in instance_targets_from_obtain:
        errors.append(
            f"obtain_log should be instance_of wooden_material; "
            f"got: {instance_targets_from_obtain}")

    # analogous_concepts: obtain_log and craft_planks share wooden_material
    if 'obtain_log' in graph.concepts and 'craft_planks' in graph.concepts:
        analogs = graph.analogous_concepts('obtain_log')
        if 'craft_planks' not in analogs:
            errors.append(
                f"craft_planks should be analogous to obtain_log via "
                f"wooden_material; got: {analogs}")

    # Phase B: composition edges — craft_planks composes_into craft_crafting_table
    comp_targets_from_planks = {
        e.target for e in graph.composition_edges
        if e.source == 'craft_planks'}
    if not comp_targets_from_planks:
        errors.append("craft_planks should have at least one composition edge")
    else:
        expected_comp = {'craft_crafting_table', 'craft_sticks'}
        missing_comp = expected_comp - comp_targets_from_planks
        if missing_comp:
            errors.append(
                f"craft_planks missing composition targets: {missing_comp}")

    # Phase B: temporal edges — break_log precedes obtain_log
    temporal_sources = {e.source for e in graph.temporal_edges}
    if not temporal_sources:
        errors.append("minecraft.ctkg should have at least one temporal edge")

    # Full validation — no MissingPrerequisite errors expected
    val_errors = graph.validate(check_types=False)
    hard_errors = [e for e in val_errors if isinstance(e, MissingPrerequisite)]
    if hard_errors:
        errors.append(
            f"minecraft.ctkg has unexpected validation errors: {hard_errors}")

    return errors


# ---------------------------------------------------------------------------
# Phase VI: 2-category structure — NaturalTransformation + apply_nat_trans
# ---------------------------------------------------------------------------

def test_natural_transformation():
    """Gate test: apply_nat_trans('D', d(pow(x,2))) returns mul(2, x).

    Registers a differentiation NaturalTransformation D : Poly ⟹ Poly
    with the power rule, constant-fold rule, and ground NNO + pow(x,1)→x
    as components.  Verifies that apply_nat_trans reduces d(pow(x,2)) → mul(2,x).
    """
    from experiments.ctkg.graph import KnowledgeGraph, NaturalTransformation
    from experiments.symbolic_ai_v2.ctkg.core.term_algebra import atom, node, var
    from experiments.symbolic_ai_v2.ctkg.core.rewrite import RewriteRule

    errors = []

    # --------------- helpers ---------------
    def _pow(a, b):  return node('pow', a, b)
    def _mul(a, b):  return node('mul', a, b)
    def _pred(a):    return node('pred', a)
    def _d(a):       return node('d', a)
    x = atom('x')

    # Ground NNO rules: pred(n+1) → n for digits 1..9
    succ_map = {'0':'1','1':'2','2':'3','3':'4','4':'5',
                '5':'6','6':'7','7':'8','8':'9'}
    ground_nno = []
    for d_from, d_to in succ_map.items():
        ground_nno.append(RewriteRule(
            lhs=_pred(atom(d_to)), rhs=atom(d_from),
            algebra_name='pred', evidence=1,
        ))

    # pow(x,1) → x (inverse norm: after pred reduces, unpack pow(x,1))
    pow_x1_inv = RewriteRule(
        lhs=_pow(atom('x'), atom('1')), rhs=atom('x'),
        algebra_name='pow_inv', evidence=1,
    )

    # Power rule: d(pow(x,V0)) → mul(V0, pow(x, pred(V0)))
    power_rule = RewriteRule(
        lhs=_d(_pow(x, var('V0'))),
        rhs=_mul(var('V0'), _pow(x, _pred(var('V0')))),
        algebra_name='d', evidence=5,
    )

    components = [power_rule] + ground_nno + [pow_x1_inv]

    # --------------- build NatTrans ---------------
    nt = NaturalTransformation(
        name='D',
        source_functor='Poly',
        target_functor='Poly',
        components=components,
    )

    graph = KnowledgeGraph()
    graph.add_nat_trans(nt)

    # Gate: d(pow(x,2)) → mul(2, x)
    expr = _d(_pow(x, atom('2')))
    result = graph.apply_nat_trans('D', expr)
    expected = _mul(atom('2'), x)
    if result != expected:
        errors.append(
            f"apply_nat_trans('D', d(pow(x,2))): expected {expected}, got {result}")

    # OOD: d(pow(x,5)) → mul(5, pow(x,4))
    expr2 = _d(_pow(x, atom('5')))
    result2 = graph.apply_nat_trans('D', expr2)
    expected2 = _mul(atom('5'), _pow(x, atom('4')))
    if result2 != expected2:
        errors.append(
            f"apply_nat_trans('D', d(pow(x,5))): expected {expected2}, got {result2}")

    # Unknown name returns None
    if graph.apply_nat_trans('nonexistent', expr) is not None:
        errors.append("apply_nat_trans with unknown name should return None")

    # No-op expression returns None (no rule fires)
    no_op_expr = _mul(atom('3'), x)
    if graph.apply_nat_trans('D', no_op_expr) is not None:
        errors.append(
            f"apply_nat_trans on non-matching expr should return None, got non-None")

    # Adjunction: unit_nat_trans / counit_nat_trans fields exist
    from experiments.ctkg.graph import Adjunction
    adj = Adjunction(
        name='diff_integral',
        forward='differentiation',
        inverse='integration',
        unit_nat_trans='eta_diff_int',
        counit_nat_trans='eps_diff_int',
    )
    if adj.unit_nat_trans != 'eta_diff_int':
        errors.append("Adjunction.unit_nat_trans field not stored correctly")
    if adj.counit_nat_trans != 'eps_diff_int':
        errors.append("Adjunction.counit_nat_trans field not stored correctly")

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
        ('Logic domain', test_logic_domain),
        ('Sheaf compatible merge', test_sheaf_compatible_merge),
        ('Sheaf violation', test_sheaf_violation),
        ('Type compatibility', test_type_compatibility),
        ('Interface parsing', test_interface_parsing),
        ('Transfer probability parsing', test_transfer_probability_parsing),
        ('d-separation', test_d_separation),
        ('Entropy', test_entropy),
        ('Intervention', test_intervention),
        ('Mastery state', test_mastery_state),
        ('Epistemic tiers', test_epistemic_tiers),
        ('Challenge edges', test_challenge_edges),
        ('Overrides (Fido problem)', test_overrides_fido),
        ('Assumption-conditioned prereqs', test_assumption_conditioned_prereqs),
        ('what_if_not()', test_what_if_not),
        ('Ungrounded assumption', test_ungrounded_assumption),
        ('Tier parse error', test_tier_parse_error),
        # Phase B: new edge types
        ('Causal edges (causes)', test_causal_edges),
        ('Composition edges (composes_into)', test_composition_edges),
        ('Instance-of edges + analogous_concepts()', test_instance_of_and_analogous),
        ('Temporal edges (precedes)', test_temporal_edges),
        ('causal_descendants()', test_causal_descendants),
        ('Phase B validation (missing endpoints)', test_phase_b_validation),
        ('Minecraft domain (Phase B)', test_minecraft_domain_phase_b),
        # Phase VI: 2-category structure
        ('NaturalTransformation + apply_nat_trans (Phase VI)', test_natural_transformation),
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
