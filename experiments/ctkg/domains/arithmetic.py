"""Arithmetic domain — loads from arithmetic.ctkg.

This module provides build_arithmetic_graph() for backwards compatibility.
The authoritative definition is in arithmetic.ctkg.

Generator mappings (Layer 2, not stored in .ctkg):
  query_counting         → QueryCountingGenerator
  combined_counting      → CombinedCountingGenerator
  successor              → SuccessorGenerator
  predecessor            → PredecessorGenerator
  comparison             → ComparisonGenerator
  single_digit_addition  → SingleDigitArithmeticGenerator(op='+')
  single_digit_subtraction → SingleDigitArithmeticGenerator(op='-')
  two_digit_single_arithmetic → TwoDigitSingleArithmeticGenerator
  two_digit_arithmetic   → TwoDigitArithmeticGenerator
"""

import os
from ..parser import parse_file
from ..graph import KnowledgeGraph


def build_arithmetic_graph() -> KnowledgeGraph:
    """Build the arithmetic graph from the .ctkg file."""
    ctkg_path = os.path.join(os.path.dirname(__file__), 'arithmetic.ctkg')
    return parse_file(ctkg_path)
