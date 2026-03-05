"""Logic domain — loads from logic.ctkg.

This module provides build_logic_graph() for backwards compatibility.
The authoritative definition is in logic.ctkg.
"""

import os
from ..parser import parse_file
from ..graph import KnowledgeGraph


def build_logic_graph() -> KnowledgeGraph:
    """Build the logic graph from the .ctkg file."""
    ctkg_path = os.path.join(os.path.dirname(__file__), 'logic.ctkg')
    return parse_file(ctkg_path)
