"""CTKG domain graphs.

Each domain module exports a build function that returns a KnowledgeGraph
populated with that domain's concepts and prerequisites.
"""

from .arithmetic import build_arithmetic_graph
from .full import build_full_graph

__all__ = ['build_arithmetic_graph', 'build_full_graph']
