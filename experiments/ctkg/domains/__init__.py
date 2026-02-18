"""CTKG domain graphs.

Each domain is defined in a .ctkg file using universal type primitives.
Python modules provide build functions for backwards compatibility.
"""

from .arithmetic import build_arithmetic_graph

__all__ = ['build_arithmetic_graph']
