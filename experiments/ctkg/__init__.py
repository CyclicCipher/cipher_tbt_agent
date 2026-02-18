# Category Theory Knowledge Graph (CTKG)
# See DESIGN.md for architecture and rationale.

from .graph import (
    Concept,
    Prerequisite,
    Functor,
    Adjunction,
    KnowledgeGraph,
    CurriculumStage,
    ValidationError,
    MissingPrerequisite,
    TypeMismatch,
    LargeFactTable,
    OrphanNode,
    CycleDetected,
    UnimplementedDependency,
)
from .parser import parse, parse_file, merge, ParseError
from .domains import build_arithmetic_graph, build_full_graph

__all__ = [
    'Concept', 'Prerequisite', 'Functor', 'Adjunction',
    'KnowledgeGraph', 'CurriculumStage',
    'ValidationError', 'MissingPrerequisite', 'TypeMismatch',
    'LargeFactTable', 'OrphanNode', 'CycleDetected', 'UnimplementedDependency',
    'parse', 'parse_file', 'merge', 'ParseError',
    'build_arithmetic_graph', 'build_full_graph',
]
