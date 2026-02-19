"""CTKG domain graphs.

Each domain is defined in a .ctkg file using universal type primitives.
Python modules provide build functions for backwards compatibility.
"""

from .arithmetic import build_arithmetic_graph
from .logic import build_logic_graph
from .syntax import (
    build_universal_syntax_graph,
    build_english_syntax_graph,
    build_merged_syntax_graph,
)

__all__ = [
    'build_arithmetic_graph',
    'build_logic_graph',
    'build_universal_syntax_graph',
    'build_english_syntax_graph',
    'build_merged_syntax_graph',
]
