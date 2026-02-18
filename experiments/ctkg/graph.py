"""Category Theory Knowledge Graph — Core data structures and validation.

The CTKG is a DAG where nodes are concepts/skills and edges are prerequisite
relationships. It serves as:

1. Curriculum compiler — topological sort = valid training order; validation
   catches missing prerequisites before training runs.
2. Structured training data generator — graph structure determines training
   order, replay policy, and format consistency.

Usage:
    graph = KnowledgeGraph()
    graph.add_concept(Concept(name='counting', ...))
    graph.add_concept(Concept(name='addition', ...))
    graph.add_prerequisite(Prerequisite(source='counting', target='addition', ...))

    errors = graph.validate()
    if errors:
        for e in errors:
            print(f"CTKG ERROR: {e}")

    curriculum = graph.generate_curriculum(target='addition')
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Nodes (objects in the category)
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    """A node in the knowledge graph.

    Each concept represents a teachable skill. It carries metadata about
    what it teaches, how to verify learning, and its scratchpad format.
    """
    name: str
    description: str
    domain: str  # e.g., 'arithmetic', 'algebra', 'calculus', 'logic', 'science'

    # Type annotations — what this concept consumes and produces.
    # Types are string labels (advisory, used for validation warnings).
    input_type: List[str] = field(default_factory=list)
    output_type: List[str] = field(default_factory=list)

    # Scratchpad integration
    generator_class: Optional[str] = None  # ProblemGenerator subclass name
    n_result: Optional[int] = None         # fixed result length; None = variable

    # Verification
    pass_threshold: float = 0.95
    max_epochs: int = 100

    # Classification
    is_atomic: bool = False     # genuinely irreducible facts (<20 entries)
    n_problems: Optional[int] = None  # total problem count

    # Factorization order
    supports_reverse: bool = False

    # Epiplexity diagnostics (populated after training runs)
    empirical_epiplexity: Optional[float] = None
    epiplexity_threshold: float = 1.0

    # Implementation status
    status: str = 'planned'  # 'planned' | 'implemented' | 'verified'


# ---------------------------------------------------------------------------
# Edges (morphisms in the category)
# ---------------------------------------------------------------------------

@dataclass
class Prerequisite:
    """An edge in the knowledge graph.

    Represents "source is prerequisite for target", with metadata about
    how the source skill is used by the target.
    """
    source: str  # prerequisite concept
    target: str  # dependent concept
    role: str    # how source is used (human-readable)

    # Type constraints for validation
    codomain_type: List[str] = field(default_factory=list)  # source produces
    domain_type: List[str] = field(default_factory=list)    # target expects

    invertible: bool = False  # is this morphism reversible?


# ---------------------------------------------------------------------------
# Functors (structure-preserving maps between domains)
# ---------------------------------------------------------------------------

@dataclass
class Functor:
    """A structure-preserving map between domains.

    Maps concepts to concepts and prerequisites to prerequisites while
    preserving composition: if A → B in source domain, then
    F(A) → F(B) in target domain.
    """
    name: str
    source_domain: str
    target_domain: str
    concept_map: Dict[str, str] = field(default_factory=dict)  # source → target
    preserves: List[str] = field(default_factory=list)  # what it preserves


# ---------------------------------------------------------------------------
# Adjunctions (forward/inverse pairs)
# ---------------------------------------------------------------------------

@dataclass
class Adjunction:
    """A forward/inverse concept pair with round-trip verification.

    The unit and counit express that applying forward then inverse
    (or vice versa) recovers the original.
    """
    name: str
    forward: str   # concept name
    inverse: str   # concept name
    unit: str = ''    # round-trip expression: forward then inverse
    counit: str = ''  # round-trip expression: inverse then forward


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ValidationError:
    """Base for graph validation errors."""
    def __init__(self, message: str):
        self.message = message

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.message}"

    def __str__(self):
        return repr(self)


class MissingPrerequisite(ValidationError):
    """Edge references a concept not in the graph."""


class TypeMismatch(ValidationError):
    """Edge source output type doesn't match target input type."""


class LargeFactTable(ValidationError):
    """Node marked is_atomic but has >20 entries — likely decomposable."""


class OrphanNode(ValidationError):
    """Non-atomic internal node with no incoming prerequisites."""


class CycleDetected(ValidationError):
    """Circular dependency in the graph."""


class UnimplementedDependency(ValidationError):
    """Implemented concept depends on a planned (unimplemented) concept."""


# ---------------------------------------------------------------------------
# Curriculum stage (output of generate_curriculum)
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """One stage in a generated curriculum."""
    number: int
    concept: Concept
    replay_concepts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Category Theory Knowledge Graph.

    A DAG where nodes are concepts and edges are prerequisite relationships.
    Provides validation, topological sorting, and curriculum generation.
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.prerequisites: List[Prerequisite] = []
        self.functors: Dict[str, Functor] = {}
        self.adjunctions: Dict[str, Adjunction] = {}
        self._children: Dict[str, Set[str]] = {}  # parent -> children
        self._parents: Dict[str, Set[str]] = {}   # child -> parents

    def add_concept(self, concept: Concept) -> None:
        """Add a concept node to the graph."""
        self.concepts[concept.name] = concept
        self._children.setdefault(concept.name, set())
        self._parents.setdefault(concept.name, set())

    def add_prerequisite(self, prereq: Prerequisite) -> None:
        """Add a prerequisite edge to the graph."""
        self.prerequisites.append(prereq)
        self._children.setdefault(prereq.source, set()).add(prereq.target)
        self._parents.setdefault(prereq.target, set()).add(prereq.source)

    # -------------------------------------------------------------------
    # Validation (Use Case 1: Curriculum Compiler)
    # -------------------------------------------------------------------

    def validate(self, check_implementation: bool = False) -> List[ValidationError]:
        """Check all structural constraints.

        Args:
            check_implementation: If True, also flag implemented concepts
                that depend on planned (unimplemented) concepts.

        Returns:
            List of errors (empty = valid graph).
        """
        errors: List[ValidationError] = []
        concept_names = set(self.concepts.keys())

        # 1. Missing prerequisites: edges reference non-existent concepts
        for p in self.prerequisites:
            if p.source not in concept_names:
                errors.append(MissingPrerequisite(
                    f"Edge '{p.source}' -> '{p.target}': "
                    f"source '{p.source}' not in graph"))
            if p.target not in concept_names:
                errors.append(MissingPrerequisite(
                    f"Edge '{p.source}' -> '{p.target}': "
                    f"target '{p.target}' not in graph"))

        # 2. Type mismatches: codomain/domain annotations don't agree
        for p in self.prerequisites:
            if p.codomain_type and p.domain_type:
                if p.codomain_type != p.domain_type:
                    errors.append(TypeMismatch(
                        f"Edge '{p.source}' -> '{p.target}': "
                        f"codomain {p.codomain_type} != domain {p.domain_type}"))

        # 3. Large fact tables: atomic concept with >20 problems
        for c in self.concepts.values():
            if c.is_atomic and c.n_problems is not None and c.n_problems > 20:
                errors.append(LargeFactTable(
                    f"'{c.name}' marked is_atomic but has {c.n_problems} "
                    f"problems (>20). Likely decomposable."))

        # 4. Orphan nodes: non-atomic concept with children but no parents
        #    (i.e., an internal node that should have prerequisites but doesn't)
        for c in self.concepts.values():
            parents = self._parents.get(c.name, set())
            children = self._children.get(c.name, set())
            if not c.is_atomic and not parents and children:
                # Has dependents but no prerequisites — might be a legitimate
                # root concept. Only flag if it's not the only root.
                pass  # Root concepts are expected to have no parents.

        # 5. Cycle detection (via topological sort attempt)
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(CycleDetected(str(e)))

        # 6. Implementation dependency check
        if check_implementation:
            for c in self.concepts.values():
                if c.status != 'planned':
                    for parent_name in self._parents.get(c.name, set()):
                        parent = self.concepts.get(parent_name)
                        if parent and parent.status == 'planned':
                            errors.append(UnimplementedDependency(
                                f"'{c.name}' (status={c.status}) depends on "
                                f"'{parent_name}' (status=planned)"))

        return errors

    # -------------------------------------------------------------------
    # Graph traversal
    # -------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm. Returns concept names in valid training order.

        Deterministic: ties broken alphabetically.
        Raises ValueError if cycle detected.
        """
        in_degree = {name: 0 for name in self.concepts}
        for p in self.prerequisites:
            if p.target in in_degree:
                in_degree[p.target] += 1

        # Sorted for deterministic output
        queue = deque(sorted(
            name for name, deg in in_degree.items() if deg == 0
        ))
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for child in sorted(self._children.get(node, [])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.concepts):
            remaining = set(self.concepts) - set(result)
            raise ValueError(f"Cycle detected involving: {remaining}")

        return result

    def ancestors(self, name: str) -> Set[str]:
        """All transitive prerequisites of a concept."""
        visited: Set[str] = set()
        stack = list(self._parents.get(name, set()))
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self._parents.get(node, set()))
        return visited

    def descendants(self, name: str) -> Set[str]:
        """All concepts that transitively depend on this one."""
        visited: Set[str] = set()
        stack = list(self._children.get(name, set()))
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self._children.get(node, set()))
        return visited

    def frontier(self, learned: Set[str]) -> Set[str]:
        """Concepts whose prerequisites are all learned but aren't yet learned."""
        ready: Set[str] = set()
        for name in self.concepts:
            if name in learned:
                continue
            parents = self._parents.get(name, set())
            if parents <= learned:  # all parents learned (empty parents = root)
                ready.add(name)
        return ready

    def missing_for(self, name: str, learned: Set[str]) -> Set[str]:
        """Transitive prerequisites of `name` not yet learned."""
        return (self.ancestors(name) | self._parents.get(name, set())) - learned

    # -------------------------------------------------------------------
    # Curriculum generation (Use Case 2)
    # -------------------------------------------------------------------

    def generate_curriculum(
        self,
        target: Optional[str] = None,
        implemented_only: bool = False,
    ) -> List[CurriculumStage]:
        """Generate a training curriculum from the graph.

        Args:
            target: If set, only include ancestors of this concept + itself.
            implemented_only: If True, skip concepts with status='planned'.

        Returns:
            Ordered list of CurriculumStage objects.
        """
        order = self.topological_sort()

        # Filter to target's subgraph
        if target:
            relevant = self.ancestors(target) | {target}
            order = [n for n in order if n in relevant]

        # Filter to implemented concepts
        if implemented_only:
            order = [n for n in order if self.concepts[n].status != 'planned']

        stages: List[CurriculumStage] = []
        included = set()
        for i, name in enumerate(order):
            concept = self.concepts[name]
            # Replay from all ancestors that are in this curriculum
            replay = [n for n in order[:i] if n in self.ancestors(name)]
            stages.append(CurriculumStage(
                number=i + 1,
                concept=concept,
                replay_concepts=replay,
            ))
            included.add(name)

        return stages

    # -------------------------------------------------------------------
    # Subgraph extraction
    # -------------------------------------------------------------------

    def subgraph(self, names: Set[str]) -> 'KnowledgeGraph':
        """Extract a subgraph containing only the named concepts."""
        sub = KnowledgeGraph()
        for name in names:
            if name in self.concepts:
                sub.add_concept(self.concepts[name])
        for p in self.prerequisites:
            if p.source in names and p.target in names:
                sub.add_prerequisite(p)
        return sub

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def domains(self) -> Dict[str, List[str]]:
        """Group concepts by domain."""
        result: Dict[str, List[str]] = {}
        for c in self.concepts.values():
            result.setdefault(c.domain, []).append(c.name)
        return result

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        lines = [
            f"CTKG: {len(self.concepts)} concepts, "
            f"{len(self.prerequisites)} prerequisites"
        ]
        for domain, names in sorted(self.domains().items()):
            impl = sum(1 for n in names if self.concepts[n].status != 'planned')
            lines.append(f"  {domain}: {len(names)} concepts ({impl} implemented)")
        return '\n'.join(lines)

    def print_curriculum(
        self,
        target: Optional[str] = None,
        implemented_only: bool = False,
    ) -> None:
        """Pretty-print the generated curriculum."""
        stages = self.generate_curriculum(target, implemented_only)
        if not stages:
            print("(empty curriculum)")
            return
        for s in stages:
            status = s.concept.status
            gen = s.concept.generator_class or '(none)'
            replay = ', '.join(s.replay_concepts) if s.replay_concepts else '(none)'
            print(f"  Stage {s.number}: {s.concept.name} [{status}] "
                  f"gen={gen} replay=[{replay}]")
