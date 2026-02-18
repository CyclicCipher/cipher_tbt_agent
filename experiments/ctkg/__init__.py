# Category Theory Knowledge Graph (CTKG)
# See DESIGN.md for architecture and rationale.

from .graph import (
    TypeDef,
    BUILTIN_TYPES,
    Concept,
    Prerequisite,
    Functor,
    Adjunction,
    Interface,
    KnowledgeGraph,
    CurriculumStage,
    ValidationError,
    MissingPrerequisite,
    TypeMismatch,
    LargeFactTable,
    OrphanNode,
    CycleDetected,
    UnimplementedDependency,
    UndefinedType,
    SheafViolation,
    types_compatible,
)
from .parser import parse, parse_file, merge, ParseError
from .domains import build_arithmetic_graph

__all__ = [
    'TypeDef', 'BUILTIN_TYPES',
    'Concept', 'Prerequisite', 'Functor', 'Adjunction',
    'Interface', 'KnowledgeGraph', 'CurriculumStage',
    'ValidationError', 'MissingPrerequisite', 'TypeMismatch',
    'LargeFactTable', 'OrphanNode', 'CycleDetected',
    'UnimplementedDependency', 'UndefinedType',
    'SheafViolation', 'types_compatible',
    'parse', 'parse_file', 'merge', 'ParseError',
    'build_arithmetic_graph',
]
