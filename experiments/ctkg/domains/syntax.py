"""Syntax domains — loads from universal_syntax.ctkg and english_syntax.ctkg.

Provides build functions and the functor mapping between domains.
"""

import os
from ..parser import parse_file
from ..graph import KnowledgeGraph


def build_universal_syntax_graph() -> KnowledgeGraph:
    """Build the universal syntax graph from the .ctkg file."""
    ctkg_path = os.path.join(os.path.dirname(__file__), 'universal_syntax.ctkg')
    return parse_file(ctkg_path)


def build_english_syntax_graph() -> KnowledgeGraph:
    """Build the English syntax graph from the .ctkg file."""
    ctkg_path = os.path.join(os.path.dirname(__file__), 'english_syntax.ctkg')
    return parse_file(ctkg_path)


def build_merged_syntax_graph() -> KnowledgeGraph:
    """Build universal + English syntax as a merged graph."""
    universal = build_universal_syntax_graph()
    english = build_english_syntax_graph()
    violations = universal.sheaf_merge(english)
    if violations:
        raise ValueError(
            f"Sheaf violations merging syntax domains: {violations}")
    return universal
