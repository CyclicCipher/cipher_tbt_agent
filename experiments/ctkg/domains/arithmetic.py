"""Arithmetic domain — the working subgraph with existing generators.

This maps to the generators in experiments/scratchpad/generators/:
  - QueryCountingGenerator (Stage 1)
  - CombinedCountingGenerator (Stage 2, count-up process with STOP)
  - SingleDigitArithmeticGenerator (Stage 3)
  - TwoDigitSingleArithmeticGenerator (Stage 4)
  - TwoDigitArithmeticGenerator (Stage 5)

These are the concepts we can currently train and test.
"""

from ..graph import Concept, KnowledgeGraph, Prerequisite


def build_arithmetic_graph() -> KnowledgeGraph:
    """Build the arithmetic subgraph with existing implemented generators."""
    g = KnowledgeGraph()

    # --- Concepts (implemented) ---

    g.add_concept(Concept(
        name='query_counting',
        description='Count DOTs or TENs in a shuffled sequence given a query',
        domain='arithmetic',
        input_type=['DOT_TEN_sequence', 'query_token'],
        output_type=['digit'],
        generator_class='QueryCountingGenerator',
        n_result=1,
        n_problems=100,
        status='implemented',
    ))

    g.add_concept(Concept(
        name='combined_counting',
        description=(
            'Count both DOTs and TENs via count-up process. '
            'Uses successor function: DOT 1 2 STOP... TEN 1 2 3 STOP...'
        ),
        domain='arithmetic',
        input_type=['DOT_TEN_sequence'],
        output_type=['count_up_sequence', 'count_up_sequence'],
        generator_class='CombinedCountingGenerator',
        n_result=20,
        n_problems=100,
        status='implemented',
    ))

    g.add_concept(Concept(
        name='single_digit_arithmetic',
        description='Single-digit +/- producing carry + ones',
        domain='arithmetic',
        input_type=['digit', 'op', 'digit'],
        output_type=['carry', 'digit'],
        generator_class='SingleDigitArithmeticGenerator',
        n_result=2,
        n_problems=155,
        is_atomic=True,
        supports_reverse=True,
        status='implemented',
    ))

    g.add_concept(Concept(
        name='two_digit_single_arithmetic',
        description='Two-digit +/- single-digit with column scratchpad',
        domain='arithmetic',
        input_type=['digit_pair', 'op', 'zero_padded_digit'],
        output_type=['column_scratchpad'],
        generator_class='TwoDigitSingleArithmeticGenerator',
        n_result=21,
        n_problems=1800,
        status='implemented',
    ))

    g.add_concept(Concept(
        name='two_digit_arithmetic',
        description='Two-digit +/- two-digit with column scratchpad',
        domain='arithmetic',
        input_type=['digit_pair', 'op', 'digit_pair'],
        output_type=['column_scratchpad'],
        generator_class='TwoDigitArithmeticGenerator',
        n_result=21,
        n_problems=12195,
        status='implemented',
    ))

    # --- Prerequisites ---

    g.add_prerequisite(Prerequisite(
        source='query_counting',
        target='combined_counting',
        role='Individual counting composes into dual counting',
        codomain_type=['digit'],
        domain_type=['digit'],
    ))

    g.add_prerequisite(Prerequisite(
        source='combined_counting',
        target='single_digit_arithmetic',
        role='Counting grounds digit semantics for arithmetic',
        codomain_type=['digit'],
        domain_type=['digit'],
    ))

    g.add_prerequisite(Prerequisite(
        source='single_digit_arithmetic',
        target='two_digit_single_arithmetic',
        role='Single-digit ops are the column operations',
        codomain_type=['carry', 'digit'],
        domain_type=['carry', 'digit'],
    ))

    g.add_prerequisite(Prerequisite(
        source='two_digit_single_arithmetic',
        target='two_digit_arithmetic',
        role='Bridge from one-variable column to two-variable column',
        codomain_type=['column_scratchpad'],
        domain_type=['column_scratchpad'],
    ))

    return g
