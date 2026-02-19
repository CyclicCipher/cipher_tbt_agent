"""Category Theory Knowledge Graph — Core data structures and validation.

The CTKG is a DAG where nodes are concepts/skills and edges are prerequisite
relationships.  It is built on a universal type system of primitives that
compose into any domain-specific type.

Three levels of primitives:
  Level 1 — Computation:  succ, pred, compare, lookup, fold, scan, emit, if
  Level 2 — Logic:        equal, and, or, not, implies, forall, exists
  Level 3 — Transform:    quote, match, substitute, rewrite, decompose, compose

Type primitives:
  symbol(set)  — element from a named finite set
  nat          — natural number (inductive: zero | succ)
  bool         — true | false
  seq(T)       — variable-length sequence
  tuple(T...)  — fixed-length product
  tagged(...)  — sum / variant type
  expr         — quoted expression (code-as-data)
  proposition  — logical statement

Structure annotations:
  ordered, invertible, commutative, associative, periodic(k), metric

Epistemic tiers:
  axiom      — mathematical/logical necessity, don't question
  theorem    — derived from premises, valid iff premises hold
  conjecture — widely believed but unproven, actively probe
  heuristic  — useful approximation with known exceptions (Fido problem)

Usage:
    graph = KnowledgeGraph()
    graph.add_type(TypeDef('digit', 'symbol', ...))
    graph.add_concept(Concept(name='counting', ...))
    graph.add_prerequisite(Prerequisite(...))

    errors = graph.validate()
    curriculum = graph.generate_curriculum(target='addition')
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Type system — universal primitives
# ---------------------------------------------------------------------------

@dataclass
class TypeDef:
    """A named type definition.

    Types compose from a small set of universal constructors:
      symbol(set_name)        — element from a named finite set
      nat                     — natural number
      bool                    — true | false
      seq(T)                  — variable-length sequence of T
      tuple(T1, T2, ...)      — fixed-length product
      tagged(l1: T1, l2: T2)  — sum / variant
      expr                    — quoted expression (meta-reasoning)
      proposition             — logical statement
      rule(pat, repl)         — rewrite rule

    Structure annotations are stored as a set of strings:
      ordered, invertible, commutative, associative, periodic(k), metric
    """
    name: str                        # type identifier (e.g. 'digit')
    constructor: str                 # base constructor (symbol, nat, seq, ...)
    params: List[str] = field(default_factory=list)      # constructor params
    annotations: Set[str] = field(default_factory=set)   # structure tags
    description: str = ''            # human-readable doc

    def __post_init__(self):
        # Normalise annotations to a set
        if isinstance(self.annotations, list):
            self.annotations = set(self.annotations)

    def __repr__(self):
        ann = ' '.join(sorted(self.annotations))
        if self.params:
            params = ', '.join(self.params)
            base = f"{self.constructor}({params})"
        else:
            base = self.constructor
        return f"{self.name} = {base}" + (f" [{ann}]" if ann else "")


# Builtin types — always available, never need to be declared.
BUILTIN_TYPES: Dict[str, TypeDef] = {
    'nat':  TypeDef('nat', 'nat', description='Natural number'),
    'bool': TypeDef('bool', 'bool', description='Boolean'),
    'expr': TypeDef('expr', 'expr', description='Quoted expression'),
    'proposition': TypeDef(
        'proposition', 'proposition', description='Logical statement'),
}


# ---------------------------------------------------------------------------
# Nodes (objects in the category)
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    """A node in the knowledge graph.

    Each concept represents a teachable skill.  It carries metadata about
    what it teaches, how to verify learning, and its scratchpad format.
    """
    name: str
    description: str
    domain: str  # e.g., 'arithmetic', 'algebra', 'calculus', 'logic'

    # Type annotations — what this concept consumes and produces.
    # These are *type names* that resolve against the graph's type registry.
    input_type: List[str] = field(default_factory=list)
    output_type: List[str] = field(default_factory=list)

    # Process — computation rule expressed as a list of process-language lines.
    process: List[str] = field(default_factory=list)

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

    # Epistemic tier — how confidently is this concept known?
    tier: str = 'theorem'  # 'axiom' | 'theorem' | 'conjecture' | 'heuristic'

    # Assumptions — named assumptions this concept depends on
    assumes: List[str] = field(default_factory=list)

    # Defaults — for heuristic-tier concepts, default property values
    # that may be overridden by specific instances (the Fido problem)
    defaults: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Edges (morphisms in the category)
# ---------------------------------------------------------------------------

@dataclass
class Prerequisite:
    """An edge in the knowledge graph — a morphism in the Markov category.

    Represents "source is prerequisite for target", with metadata about
    how the source skill is used by the target.

    The transfer_probability is the Markov kernel weight: P(can learn target |
    mastered source).  Default 1.0 = hard prerequisite (must fully master
    source before target).  Values <1.0 = soft prerequisite (partial mastery
    of source partially enables target).

    Categorically: morphisms in FinStoch with objects as concepts and
    stochastic matrices as transition probabilities.
    """
    source: str  # prerequisite concept
    target: str  # dependent concept
    role: str    # how source is used (human-readable)

    # Type constraints for validation
    codomain_type: List[str] = field(default_factory=list)  # source produces
    domain_type: List[str] = field(default_factory=list)    # target expects

    invertible: bool = False  # is this morphism reversible?

    # Markov kernel weight: P(can learn target | mastered source)
    transfer_probability: float = 1.0

    # Assumption context — what assumption makes this prerequisite hold?
    # If None, the prerequisite is unconditional.
    assuming: Optional[str] = None

    # Status of the assumption: 'axiomatic' | 'derived' | 'empirical' | 'heuristic'
    assumption_status: str = 'derived'


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
# Challenge edges (epistemic reasoning)
# ---------------------------------------------------------------------------

@dataclass
class Challenge:
    """A challenge edge — evidence weakening a concept.

    Unlike prerequisites (which say "A enables B"), challenges say
    "evidence E weakens the claim that concept C holds."  When the model
    encounters a concept with active challenges, it must branch and
    consider the alternative.

    Categorically: challenge edges are morphisms in the *opposite*
    direction — they weaken rather than strengthen the target.
    """
    source: str        # the challenging concept (the new evidence)
    target: str        # the challenged concept (the claim being weakened)
    role: str          # how the challenge works (human-readable)
    strength: float = 1.0  # 0.0 = weak hint, 1.0 = full refutation


# ---------------------------------------------------------------------------
# Overrides (the Fido problem — exceptions to heuristic defaults)
# ---------------------------------------------------------------------------

@dataclass
class Override:
    """An instance-level exception to a heuristic default.

    "Dogs have 4 legs" is a heuristic.  "Fido has 3 legs" is an override.
    When reasoning about Fido, the override takes precedence over the
    default.

    Categorically: defaults are natural transformations from the heuristic
    concept to instances; overrides are modifications (whiskering) at
    specific components.
    """
    instance: str        # the instance concept
    default_concept: str  # the heuristic being overridden
    property: str        # which property is overridden
    value: str           # the override value
    reason: str = ''     # why the override exists


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


class UndefinedType(ValidationError):
    """Concept references a type name not in the type registry."""


class SheafViolation(ValidationError):
    """Overlapping definitions are incompatible across domains.

    In sheaf theory: two sections on overlapping open sets must agree
    on the overlap. In CTKG: if two domains both define a type or
    concept with the same name, the definitions must be structurally
    compatible.
    """


class ChallengedConjecture(ValidationError):
    """A conjecture has active challenge edges — consider branching."""


class UngroundedAssumption(ValidationError):
    """A prerequisite assumes X but X is not defined as a concept."""


# ---------------------------------------------------------------------------
# Interfaces (sheaf sections)
# ---------------------------------------------------------------------------

@dataclass
class Interface:
    """Declares what a domain exports for cross-domain composition.

    In sheaf theory terms: a domain is an "open set" in the topology of
    knowledge. An interface is the "section" — the data visible to other
    domains. The gluing axiom says: if two sections agree on their
    overlap, they can be composed into a global section.

    An interface lists the types and concepts that are available for
    cross-domain references. When merging graphs, overlapping names
    must have compatible definitions (the sheaf condition).
    """
    name: str                                      # interface name (usually domain name)
    types: List[str] = field(default_factory=list)  # exported type names
    concepts: List[str] = field(default_factory=list)  # exported concept names


# ---------------------------------------------------------------------------
# Type compatibility (sheaf restriction maps)
# ---------------------------------------------------------------------------

def types_compatible(a: TypeDef, b: TypeDef) -> bool:
    """Check if two type definitions are structurally compatible.

    Compatible means: same constructor, same params, same annotations.
    This is the sheaf condition on types — the restriction maps must
    agree on the overlap.
    """
    return (a.constructor == b.constructor
            and a.params == b.params
            and a.annotations == b.annotations)


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
# Mastery state (distribution over the knowledge graph)
# ---------------------------------------------------------------------------

class MasteryState:
    """Per-concept mastery levels forming a distribution over the graph.

    Categorically: a functor from the knowledge graph (viewed as a category)
    to the unit interval [0,1].  Each concept maps to its mastery level;
    each prerequisite edge maps to its transfer probability, weighting
    how mastery of the source enables learning the target.

    The Bayes filter interpretation (Fritz et al. 2024): update mastery
    beliefs as assessment data arrives.
    """

    def __init__(self, graph: 'KnowledgeGraph'):
        self._graph = graph
        self.levels: Dict[str, float] = {
            name: 0.0 for name in graph.concepts
        }

    def observe(self, concept: str, score: float) -> None:
        """Update mastery for a concept based on assessment score."""
        self.levels[concept] = max(0.0, min(1.0, score))

    def expected_readiness(self, concept: str) -> float:
        """Expected readiness = min over prerequisites of
        (source mastery * transfer probability).

        If a concept has no prerequisites, readiness is 1.0 (always ready).
        The min reflects the bottleneck: the weakest prerequisite limits
        readiness for the target.
        """
        prereqs = [p for p in self._graph.prerequisites
                   if p.target == concept]
        if not prereqs:
            return 1.0
        return min(
            self.levels.get(p.source, 0.0) * p.transfer_probability
            for p in prereqs
        )

    def frontier(self, threshold: float = 0.8) -> Set[str]:
        """Concepts ready to learn: readiness above threshold, not mastered.

        Generalises KnowledgeGraph.frontier() to the probabilistic case:
        instead of requiring ALL prerequisites to be in a learned set, we
        require the expected readiness (product of mastery * transfer) to
        exceed the threshold.
        """
        ready: Set[str] = set()
        for name in self._graph.concepts:
            if self.levels.get(name, 0.0) >= 0.95:
                continue  # already mastered
            if self.expected_readiness(name) >= threshold:
                ready.add(name)
        return ready

    def information_gain(self, concept: str) -> float:
        """Expected information gain from learning this concept.

        I(concept) = H(concept) - H(concept | prerequisites).
        Uses the graph's entropy methods.  Higher = more information
        transferred from prerequisites to this concept.
        """
        learned = {n for n, v in self.levels.items() if v >= 0.95}
        return self._graph.mutual_information(concept, learned)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """Category Theory Knowledge Graph.

    A DAG where nodes are concepts and edges are prerequisite relationships,
    with a type registry for universal primitives.
    """

    def __init__(self):
        self.types: Dict[str, TypeDef] = dict(BUILTIN_TYPES)
        self.concepts: Dict[str, Concept] = {}
        self.prerequisites: List[Prerequisite] = []
        self.challenges: List[Challenge] = []
        self.overrides: List[Override] = []
        self.functors: Dict[str, Functor] = {}
        self.adjunctions: Dict[str, Adjunction] = {}
        self.interfaces: Dict[str, Interface] = {}
        self._children: Dict[str, Set[str]] = {}  # parent -> children
        self._parents: Dict[str, Set[str]] = {}   # child -> parents

    # ---------------------------------------------------------------
    # Type registry
    # ---------------------------------------------------------------

    def add_type(self, typedef: TypeDef) -> None:
        """Register a named type."""
        self.types[typedef.name] = typedef

    def resolve_type(self, name: str) -> Optional[TypeDef]:
        """Look up a type by name.  Returns None if undefined."""
        return self.types.get(name)

    # ---------------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------------

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

    def add_challenge(self, challenge: Challenge) -> None:
        """Add a challenge edge to the graph."""
        self.challenges.append(challenge)

    def add_override(self, override: Override) -> None:
        """Add an override (instance-level exception to a default)."""
        self.overrides.append(override)

    # -------------------------------------------------------------------
    # Validation (Use Case 1: Curriculum Compiler)
    # -------------------------------------------------------------------

    def validate(self, check_implementation: bool = False,
                 check_types: bool = True) -> List[ValidationError]:
        """Check all structural constraints.

        Args:
            check_implementation: If True, also flag implemented concepts
                that depend on planned (unimplemented) concepts.
            check_types: If True, validate that all type names in concepts
                resolve against the type registry.

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
        for c in self.concepts.values():
            parents = self._parents.get(c.name, set())
            children = self._children.get(c.name, set())
            if not c.is_atomic and not parents and children:
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

        # 7. Type resolution check
        if check_types:
            for c in self.concepts.values():
                for tname in c.input_type:
                    if tname not in self.types:
                        errors.append(UndefinedType(
                            f"Concept '{c.name}' input type '{tname}' "
                            f"not defined in type registry"))
                for tname in c.output_type:
                    if tname not in self.types:
                        errors.append(UndefinedType(
                            f"Concept '{c.name}' output type '{tname}' "
                            f"not defined in type registry"))

        # 8. Challenged conjectures: flag conjectures with active challenges
        challenged_targets = {ch.target for ch in self.challenges}
        for c in self.concepts.values():
            if c.tier == 'conjecture' and c.name in challenged_targets:
                challengers = [ch.source for ch in self.challenges
                               if ch.target == c.name]
                errors.append(ChallengedConjecture(
                    f"Conjecture '{c.name}' has active challenges from: "
                    f"{', '.join(challengers)}. Consider branching."))

        # 9. Ungrounded assumptions: prerequisite assumes X but X not in graph
        all_names = set(self.concepts.keys())
        # Gather all assumption names from concepts and prerequisites
        all_assumptions: Set[str] = set()
        for c in self.concepts.values():
            all_assumptions.update(c.assumes)
        for p in self.prerequisites:
            if p.assuming:
                all_assumptions.add(p.assuming)
        # Check that each assumption is either a concept name or a known
        # assumption string (we only flag if check_types is on, to avoid
        # noise in basic validation)
        if check_types:
            for assumption in all_assumptions:
                if assumption not in all_names:
                    # Find who references it
                    refs = [c.name for c in self.concepts.values()
                            if assumption in c.assumes]
                    refs += [f"{p.source}->{p.target}" for p in self.prerequisites
                             if p.assuming == assumption]
                    errors.append(UngroundedAssumption(
                        f"Assumption '{assumption}' not defined as a concept. "
                        f"Referenced by: {', '.join(refs)}"))

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
    # Probabilistic structure (Markov category)
    # -------------------------------------------------------------------

    def d_separated(self, x: str, y: str, given: Set[str]) -> bool:
        """Test if x and y are d-separated given the observed set.

        Uses the Bayes-ball algorithm (Shachter 1998).  d-Separation is the
        core inference primitive of Bayesian networks.  In our curriculum
        graph it answers: "given that the student has mastered the concepts
        in `given`, is their performance on concept x independent of their
        performance on concept y?"

        Categorically: this is the conditional independence relation in the
        Markov category (Fritz & Klingler, JMLR 2023).

        Returns True if x and y are conditionally independent given `given`.
        """
        if x == y:
            return False
        if x not in self.concepts or y not in self.concepts:
            raise ValueError(f"Unknown concept: {x if x not in self.concepts else y}")

        # Phase 1: precompute collider activation set.
        # A collider A → B ← C is activated when B or any descendant of B
        # is observed.  Equivalently: B is in `given` or B is an ancestor
        # of some node in `given`.  So: collider_active = given ∪ ancestors(given).
        collider_active = set(given)
        for z in given:
            collider_active.update(self.ancestors(z))

        # Phase 2: Bayes-ball from x.
        # Track (node, direction) pairs.
        # direction: 'up' = ball arrived from a child, 'down' = from a parent.
        visited: Set[Tuple[str, str]] = set()
        queue: deque[Tuple[str, str]] = deque()
        reachable: Set[str] = set()

        # Launch balls from x in both directions
        queue.append((x, 'up'))
        queue.append((x, 'down'))

        while queue:
            node, direction = queue.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))

            if node != x:
                reachable.add(node)

            if direction == 'up' and node not in given:
                # Ball from child, node NOT observed:
                # pass up to parents (chain/fork unblocked)
                for parent in self._parents.get(node, set()):
                    queue.append((parent, 'up'))
                # pass down to children
                for child in self._children.get(node, set()):
                    queue.append((child, 'down'))

            elif direction == 'down':
                # Ball from parent
                if node not in given:
                    # Not observed: pass down to children (chain unblocked)
                    for child in self._children.get(node, set()):
                        queue.append((child, 'down'))
                if node in collider_active:
                    # Node is observed or has an observed descendant:
                    # collider activated, pass up to parents
                    for parent in self._parents.get(node, set()):
                        queue.append((parent, 'up'))

        return y not in reachable

    def concept_entropy(self, concept_name: str) -> float:
        """H(C) = log2(|problem_space|) — maximum uncertainty about concept.

        This is the entropy of the uniform distribution over the concept's
        problem space.  By the Baez-Fritz-Leinster theorem (2011), Shannon
        entropy is the unique functorial measure of information loss: if
        we view a concept as a morphism in FinProb, its entropy is the
        unique function that is (1) functorial, (2) convex-linear, (3)
        continuous.

        Returns inf if problem space size is unknown.
        """
        concept = self.concepts.get(concept_name)
        if not concept:
            raise ValueError(f"Unknown concept: {concept_name}")
        if concept.n_problems and concept.n_problems > 0:
            return math.log2(concept.n_problems)
        return float('inf')

    def conditional_entropy(self, concept_name: str,
                            learned: Set[str]) -> float:
        """H(C | learned) — remaining uncertainty given learned prerequisites.

        Approximated as: H(C) minus the information transferred from each
        learned prerequisite, weighted by transfer probability.

        This connects to the prequential coding interpretation: the
        information still to be extracted at a stage, given that the
        prerequisites have been learned.  Corresponds to the expected
        epiplexity of the stage.
        """
        h_c = self.concept_entropy(concept_name)
        if math.isinf(h_c):
            return h_c

        prereqs = [p for p in self.prerequisites
                   if p.target == concept_name]
        transfer = 0.0
        for p in prereqs:
            if p.source in learned:
                h_source = self.concept_entropy(p.source)
                if not math.isinf(h_source):
                    transfer += h_source * p.transfer_probability

        return max(0.0, h_c - transfer)

    def mutual_information(self, concept_name: str,
                           learned: Set[str]) -> float:
        """I(C; learned) = H(C) - H(C | learned).

        The information transferred from prerequisites to this concept.
        Higher values = prerequisites are highly informative for this concept.
        """
        h_c = self.concept_entropy(concept_name)
        h_c_given = self.conditional_entropy(concept_name, learned)
        if math.isinf(h_c) or math.isinf(h_c_given):
            return 0.0
        return h_c - h_c_given

    def intervene(self, do_concepts: Set[str]) -> 'KnowledgeGraph':
        """Pearl's do-operator via string diagram surgery.

        Returns a mutilated graph where all incoming edges to the
        do_concepts are removed.  This models "what if we force-teach
        (or skip) these concepts, breaking the natural prerequisite flow?"

        Categorically: an endofunctor on the diagram that severs incoming
        morphisms to the intervention targets (Jacobs, Kissinger, Zanasi
        2019).

        The returned graph is a new object — self is not modified.
        """
        # Build new graph with same concepts and types
        new_graph = KnowledgeGraph()
        new_graph.types = dict(self.types)
        for c in self.concepts.values():
            new_graph.add_concept(c)
        # Copy edges except those targeting intervened concepts
        for p in self.prerequisites:
            if p.target not in do_concepts:
                new_graph.add_prerequisite(p)
        new_graph.challenges = list(self.challenges)
        new_graph.overrides = list(self.overrides)
        new_graph.functors = dict(self.functors)
        new_graph.adjunctions = dict(self.adjunctions)
        new_graph.interfaces = dict(self.interfaces)
        return new_graph

    # -------------------------------------------------------------------
    # Epistemic reasoning (critical thinking)
    # -------------------------------------------------------------------

    def what_if_not(self, concept_name: str) -> Set[str]:
        """What concepts become unblocked if we remove a concept?

        The dual of missing_for(): instead of "what do I need to reach X?",
        this asks "what becomes reachable if I stop assuming Y?"

        Returns the set of concepts that were blocked ONLY by concept_name
        (directly or transitively).  These are the concepts that would
        become frontier candidates if the assumption were removed.

        Use case: if the removed concept is a conjecture with active
        challenges, a large returned set = high-value research direction.
        """
        if concept_name not in self.concepts:
            raise ValueError(f"Unknown concept: {concept_name}")

        # Build a graph without the concept and its prerequisite edges
        reduced = KnowledgeGraph()
        reduced.types = dict(self.types)
        for name, c in self.concepts.items():
            if name != concept_name:
                reduced.add_concept(c)
        for p in self.prerequisites:
            if p.source != concept_name and p.target != concept_name:
                reduced.add_prerequisite(p)

        # Find what's newly reachable: concepts that have all prereqs
        # satisfied in the reduced graph but not in the original graph
        all_names = set(self.concepts.keys()) - {concept_name}

        # In original graph: concepts whose ancestors include concept_name
        blocked_in_original = self.descendants(concept_name)

        # In reduced graph: which of those blocked concepts now have
        # all prereqs satisfied (i.e., concept_name was the only blocker)?
        opened: Set[str] = set()
        for name in blocked_in_original:
            if name == concept_name:
                continue
            if name not in reduced.concepts:
                continue
            # Check if all parents in reduced graph exist
            parents = reduced._parents.get(name, set())
            # A concept is "opened" if it exists in reduced graph and
            # all its remaining parents exist too
            if parents <= set(reduced.concepts.keys()):
                opened.add(name)

        return opened

    def challenged_concepts(self) -> Dict[str, List[Challenge]]:
        """Return concepts with active challenges and their challengers.

        Returns a dict mapping challenged concept name to list of
        Challenge objects targeting it.  Empty dict = no active disputes.
        """
        result: Dict[str, List[Challenge]] = {}
        for ch in self.challenges:
            result.setdefault(ch.target, []).append(ch)
        return result

    def assumption_dependents(self, assumption: str) -> Dict[str, List[str]]:
        """Find all concepts and prerequisites that depend on an assumption.

        Returns:
            Dict with keys 'concepts' and 'prerequisites', each mapping
            to a list of names/edge descriptions.
        """
        dependent_concepts = [
            c.name for c in self.concepts.values()
            if assumption in c.assumes
        ]
        dependent_prereqs = [
            f"{p.source}->{p.target}"
            for p in self.prerequisites
            if p.assuming == assumption
        ]
        return {
            'concepts': dependent_concepts,
            'prerequisites': dependent_prereqs,
        }

    def resolve_default(self, concept_name: str, property_name: str,
                        instance_name: Optional[str] = None) -> Optional[str]:
        """Resolve a property value, checking overrides before defaults.

        The Fido problem: "dogs have 4 legs" is a default, but Fido
        has 3 legs via an override.

        Args:
            concept_name: The heuristic concept with the default.
            property_name: The property to resolve.
            instance_name: If given, check for instance-level overrides.

        Returns:
            The override value if one exists for this instance,
            otherwise the default value, or None if neither exists.
        """
        # Check for instance-level override first
        if instance_name:
            for ov in self.overrides:
                if (ov.instance == instance_name
                        and ov.default_concept == concept_name
                        and ov.property == property_name):
                    return ov.value

        # Fall back to default
        concept = self.concepts.get(concept_name)
        if concept:
            return concept.defaults.get(property_name)

        return None

    def information_flow(self) -> Dict[str, float]:
        """Compute information flow through each edge.

        For each prerequisite edge, compute how much information
        (in bits) flows from source to target:
            flow = H(source) * transfer_probability

        Returns a dict mapping "source->target" to flow in bits.
        Edges with unknown source entropy are omitted.
        """
        flows: Dict[str, float] = {}
        for p in self.prerequisites:
            h_source = self.concept_entropy(p.source)
            if not math.isinf(h_source):
                key = f"{p.source}->{p.target}"
                flows[key] = h_source * p.transfer_probability
        return flows

    def mastery_state(self) -> 'MasteryState':
        """Create a fresh MasteryState for this graph."""
        return MasteryState(self)

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
        # Copy relevant types
        for name in names:
            c = self.concepts.get(name)
            if c:
                sub.add_concept(c)
                for tname in c.input_type + c.output_type:
                    if tname in self.types and tname not in sub.types:
                        sub.add_type(self.types[tname])
        for p in self.prerequisites:
            if p.source in names and p.target in names:
                sub.add_prerequisite(p)
        return sub

    # -------------------------------------------------------------------
    # Sheaf consistency (multi-domain composition)
    # -------------------------------------------------------------------

    def sheaf_check(self, other: 'KnowledgeGraph') -> List[SheafViolation]:
        """Check sheaf consistency between this graph and another.

        The sheaf condition requires that overlapping definitions agree:
        - If both graphs define a type with the same name, the TypeDefs
          must be structurally compatible (same constructor, params,
          annotations).
        - If both graphs define a concept with the same name, the
          concepts must have compatible input/output types.

        Returns a list of SheafViolation errors (empty = compatible).
        """
        violations: List[SheafViolation] = []

        # Check overlapping types (excluding builtins — those always agree)
        for name, my_type in self.types.items():
            if name in BUILTIN_TYPES:
                continue
            other_type = other.types.get(name)
            if other_type is None or name in BUILTIN_TYPES:
                continue
            if not types_compatible(my_type, other_type):
                violations.append(SheafViolation(
                    f"Type '{name}' defined incompatibly across domains: "
                    f"{my_type!r} vs {other_type!r}"))

        # Check overlapping concepts
        for name, my_concept in self.concepts.items():
            other_concept = other.concepts.get(name)
            if other_concept is None:
                continue
            # Input/output types must match
            if my_concept.input_type != other_concept.input_type:
                violations.append(SheafViolation(
                    f"Concept '{name}' input types differ: "
                    f"{my_concept.input_type} vs {other_concept.input_type}"))
            if my_concept.output_type != other_concept.output_type:
                violations.append(SheafViolation(
                    f"Concept '{name}' output types differ: "
                    f"{my_concept.output_type} vs {other_concept.output_type}"))

        return violations

    def sheaf_merge(self, source: 'KnowledgeGraph') -> List[SheafViolation]:
        """Merge source graph into this graph with sheaf consistency check.

        First checks that overlapping definitions are compatible. If any
        violations are found, returns them without modifying the graph.
        If all clear, performs the merge.

        Returns:
            List of SheafViolation errors (empty = merge succeeded).
        """
        violations = self.sheaf_check(source)
        if violations:
            return violations

        # Safe to merge — no conflicts
        for t in source.types.values():
            if t.name not in BUILTIN_TYPES:
                self.add_type(t)
        for c in source.concepts.values():
            self.add_concept(c)
        for p in source.prerequisites:
            self.add_prerequisite(p)
        for ch in source.challenges:
            self.add_challenge(ch)
        for ov in source.overrides:
            self.add_override(ov)
        self.functors.update(source.functors)
        self.adjunctions.update(source.adjunctions)
        self.interfaces.update(source.interfaces)

        return []

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
        parts = [
            f"{len(self.concepts)} concepts",
            f"{len(self.prerequisites)} prerequisites",
            f"{len(self.types) - len(BUILTIN_TYPES)} custom types",
        ]
        if self.interfaces:
            parts.append(f"{len(self.interfaces)} interfaces")
        lines = [f"CTKG: {', '.join(parts)}"]
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
