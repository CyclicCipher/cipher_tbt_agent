"""CatPlan core types — the internal representation.

A CatPlan domain is a category: types are objects, predicates are
morphisms to Prop, actions are typed state transitions.

This module defines the data structures. The parser (parser.py)
reads .catplan files into these structures. The planner (planner.py)
searches over states using these structures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Types (objects of the category)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Type:
    """A type in the CatPlan domain. Objects of the category."""
    name: str
    # Union types: Surface = Block | Table
    # If variants is non-empty, this is a union type.
    variants: tuple[str, ...] = ()

    def is_union(self) -> bool:
        return len(self.variants) > 0


# ---------------------------------------------------------------------------
# Predicates (morphisms to Prop)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predicate:
    """A typed predicate. A morphism from product of types to Prop."""
    name: str
    param_types: tuple[str, ...]  # names of the parameter types

    @property
    def arity(self) -> int:
        return len(self.param_types)


# ---------------------------------------------------------------------------
# Invariants (sheaf consistency conditions)
# ---------------------------------------------------------------------------

@dataclass
class Invariant:
    """A domain invariant — must hold in every valid state.

    If `condition` is not None, it's an evaluable ConditionExpr.
    If it's None, the invariant is stored as raw text only (not yet parseable).
    """
    description: str
    raw_text: str
    condition: 'ConditionExpr | None' = None


# ---------------------------------------------------------------------------
# Derived predicates (axioms — computed, not stored)
# ---------------------------------------------------------------------------

@dataclass
class DerivedPredicate:
    """A predicate computed from other predicates, not stored in state.

    Example: clear(b) = not(exists b2 : Block . on(b2, b))
    Recomputed after every action application.
    """
    name: str
    param_types: tuple[str, ...]
    # The body is a Condition tree (using the condition types below).
    # Evaluated for each ground instantiation of the parameters.
    body: 'ConditionExpr'


# ---------------------------------------------------------------------------
# Condition expressions (used in preconditions, derived predicates, invariants)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomCondition:
    """A simple predicate check: pred(args) or not(pred(args))."""
    predicate: str
    args: tuple[str, ...]
    negated: bool = False


@dataclass(frozen=True)
class ExistsCondition:
    """Existential quantifier: exists var : Type . body."""
    var_name: str
    var_type: str
    body: 'ConditionExpr'


@dataclass(frozen=True)
class ForallCondition:
    """Universal quantifier: forall var : Type . body."""
    var_name: str
    var_type: str
    body: 'ConditionExpr'


@dataclass(frozen=True)
class OrCondition:
    """Disjunction: cond1 or cond2 or ..."""
    conditions: tuple['ConditionExpr', ...]


@dataclass(frozen=True)
class AndCondition:
    """Conjunction: cond1 and cond2 and ..."""
    conditions: tuple['ConditionExpr', ...]


@dataclass(frozen=True)
class EqualityCondition:
    """Equality or inequality: a = b or a /= b."""
    left: str
    right: str
    negated: bool = False  # False = equality, True = inequality


@dataclass(frozen=True)
class CountCondition:
    """Counting quantifier: count(var : Type . body) op value.

    e.g., count(b : Block . holding(h, b)) <= 1
    """
    var_name: str
    var_type: str
    body: 'ConditionExpr'
    op: str   # '<=', '>=', '=', '<', '>'
    value: int


# Union type for all condition expressions.
ConditionExpr = (
    AtomCondition | ExistsCondition | ForallCondition |
    OrCondition | AndCondition | EqualityCondition | CountCondition
)


# ---------------------------------------------------------------------------
# Conditional effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionalEffect:
    """An effect that only fires when a condition holds.

    when condition : effect1, effect2, ...
    """
    condition: ConditionExpr
    effects: tuple['Effect', ...]


# ---------------------------------------------------------------------------
# Actions (morphisms between states)
# ---------------------------------------------------------------------------

@dataclass
class ActionParam:
    """A typed parameter of an action."""
    name: str
    type_name: str


@dataclass
class ActionDef:
    """An action definition: typed parameters + preconditions + effects.

    Preconditions and effects are stored as structured representations,
    not raw strings, so the planner can evaluate them.
    """
    name: str
    params: list[ActionParam]
    preconditions: list[ConditionExpr] = field(default_factory=list)
    effects: list['Effect'] = field(default_factory=list)
    conditional_effects: list[ConditionalEffect] = field(default_factory=list)


# Legacy alias — preconditions now use ConditionExpr directly.
Condition = AtomCondition


@dataclass(frozen=True)
class Effect:
    """An effect of an action."""
    predicate: str
    args: tuple[str, ...]
    set_to: bool = True  # True = assert, False = retract


# ---------------------------------------------------------------------------
# Domain (a category)
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    """A CatPlan domain — a category of types, predicates, and actions."""
    name: str
    types: dict[str, Type] = field(default_factory=dict)
    predicates: dict[str, Predicate] = field(default_factory=dict)
    derived: dict[str, DerivedPredicate] = field(default_factory=dict)
    invariants: list[Invariant] = field(default_factory=list)
    actions: dict[str, ActionDef] = field(default_factory=dict)
    # Phase D: categorical features
    composites: dict[str, 'CompositeAction'] = field(default_factory=dict)
    adjunctions: dict[str, 'Adjunction'] = field(default_factory=dict)
    initial_algebras: dict[str, 'InitialAlgebra'] = field(default_factory=dict)
    functors: dict[str, 'Functor'] = field(default_factory=dict)
    natural_transformations: dict[str, 'NaturalTransformation'] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Problems (specific instances)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectDecl:
    """An object in a problem instance."""
    name: str
    type_name: str


@dataclass(frozen=True)
class GroundAtom:
    """A ground (fully instantiated) predicate.

    e.g., on("a", "b") or clear("a").
    """
    predicate: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"


@dataclass(frozen=True)
class NegatedGoalAtom:
    """A goal condition that requires an atom to NOT be true."""
    predicate: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"not({self.predicate}({', '.join(self.args)}))"


@dataclass
class Problem:
    """A specific planning problem: objects + initial state + goal."""
    name: str
    domain_name: str
    objects: dict[str, ObjectDecl] = field(default_factory=dict)
    init: set[GroundAtom] = field(default_factory=set)
    goal: set[GroundAtom] = field(default_factory=set)
    neg_goal: set[NegatedGoalAtom] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Ground actions (for the planner)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundAction:
    """A fully instantiated action: action name + bound arguments.

    e.g., pick(hand, a) or stack(hand, a, b).
    """
    action_name: str
    args: tuple[str, ...]  # object names bound to action params

    def __str__(self) -> str:
        return f"{self.action_name}({', '.join(self.args)})"


# ---------------------------------------------------------------------------
# State (a frozenset of ground atoms — hashable for search)
# ---------------------------------------------------------------------------

State = frozenset[GroundAtom]


# ===========================================================================
# Phase D: Categorical features
# ===========================================================================

# ---------------------------------------------------------------------------
# D.1: Morphism composition (macro-operators)
# ---------------------------------------------------------------------------

@dataclass
class CompositeAction:
    """A composite action: a named sequence of sub-actions.

    compose pick_and_stack(h, b, target) = stack(h, b, target) . pick(h, b)

    The planner can use this as a single macro step. The composition is
    type-checked: postconditions of each step must satisfy preconditions
    of the next.
    """
    name: str
    params: list[ActionParam]
    # Each step is (action_name, param_mapping) where param_mapping maps
    # the sub-action's param names to this composite's param names.
    steps: list[tuple[str, dict[str, str]]]


# ---------------------------------------------------------------------------
# D.2: Adjunctions (inverse operations)
# ---------------------------------------------------------------------------

@dataclass
class Adjunction:
    """An adjunction F -| G: F is left adjoint to G.

    Solving F(x) = y is equivalent to x = G(y).
    The planner uses this to compute inverses directly instead of searching.

    left_action: the action name for F (e.g., "add")
    right_action: the action name for G (e.g., "sub")
    param_map: how F's parameters relate to G's parameters.
        e.g., if add(a, b) = c then sub(c, b) = a
        param_map = {"result": "first_arg", "second_arg": "second_arg"}
    """
    name: str
    left_action: str
    right_action: str
    # Maps: output_param_of_left -> input_param_of_right
    param_map: dict[str, str]


# ---------------------------------------------------------------------------
# D.3: Equalizers (single-equation solving)
# ---------------------------------------------------------------------------

@dataclass
class Equalizer:
    """The equalizer of f and g: find x where f(x) = g(x).

    morphism_f: name of first predicate/function
    morphism_g: name of second predicate/function
    source_type: the type over which to search
    """
    name: str
    morphism_f: str
    morphism_g: str
    source_type: str


# ---------------------------------------------------------------------------
# D.4: Pullbacks (simultaneous constraints)
# ---------------------------------------------------------------------------

@dataclass
class Pullback:
    """The pullback of f: A->C and g: B->C.

    Find the largest subset of A×B where f(a) = g(b).
    Constraint satisfaction: multiple constraints that must agree.
    """
    name: str
    constraints: list[tuple[str, str]]  # list of (predicate_name, target_value_or_pred)
    source_types: list[str]             # types being constrained


# ---------------------------------------------------------------------------
# D.5: Initial algebras (recursion and induction)
# ---------------------------------------------------------------------------

@dataclass
class InitialAlgebra:
    """An initial algebra: the smallest fixed point of an endofunctor.

    Gives recursion and induction for free.

    carrier_type: the recursive type (e.g., "Nat")
    zero: the base case constructor name
    succ: the recursive case constructor name
    """
    name: str
    carrier_type: str
    zero: str    # name of the zero/base action
    succ: str    # name of the successor/step action


# ---------------------------------------------------------------------------
# D.6: Kan extensions (generalization/extrapolation)
# ---------------------------------------------------------------------------

@dataclass
class KanExtension:
    """Left Kan extension of F along K.

    Extends a partial mapping to a complete mapping in the most general
    way consistent with the known data.

    base_domain: the domain where F is defined (training data)
    full_domain: the domain to extend to (full domain)
    known_map: dict mapping known inputs to known outputs
    """
    name: str
    base_type: str
    full_type: str
    known_map: dict[str, str]  # known input -> known output


# ---------------------------------------------------------------------------
# D.7: Galois connections (abstraction hierarchies)
# ---------------------------------------------------------------------------

@dataclass
class GaloisConnection:
    """A Galois connection between concrete and abstract domains.

    abstract: maps concrete states to abstract states
    concretize: maps abstract states to most general concrete states
    """
    name: str
    concrete_types: list[str]
    abstract_types: list[str]
    # Abstraction function: maps predicates in concrete to predicates in abstract
    abstraction_map: dict[str, str]


# ---------------------------------------------------------------------------
# D.8: Enriched predicates (continuous values)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnrichedAtom:
    """A predicate with a continuous value instead of boolean.

    e.g., distance(a, b) = 5.0, temperature(room) = 22.3
    """
    predicate: str
    args: tuple[str, ...]
    value: float

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)}) = {self.value}"


@dataclass(frozen=True)
class NumericEffect:
    """A numeric effect: modify a continuous predicate.

    op: 'assign', 'increase', 'decrease'
    """
    predicate: str
    args: tuple[str, ...]
    op: str        # 'assign', 'increase', 'decrease'
    value: float


@dataclass(frozen=True)
class NumericCondition:
    """A numeric precondition: compare a continuous predicate.

    e.g., temperature(room) > 100.0
    """
    predicate: str
    args: tuple[str, ...]
    op: str    # '<', '>', '<=', '>=', '='
    value: float


# ---------------------------------------------------------------------------
# D.9: Operads (multi-input composition)
# ---------------------------------------------------------------------------

@dataclass
class OperadicAction:
    """An action with multiple typed input slots and one output.

    Unlike regular actions which transform state, operadic actions
    combine multiple inputs to produce a result.

    e.g., combine(reagent1: Reagent, reagent2: Reagent) -> Product
    """
    name: str
    input_slots: list[ActionParam]  # multiple typed inputs
    output_type: str
    output_predicate: str           # predicate to assert on the result
    preconditions: list[ConditionExpr] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)


# ---------------------------------------------------------------------------
# D.10: Probabilistic morphisms (Markov category)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbabilisticEffect:
    """A probabilistic effect: one of several outcomes.

    Each outcome has a probability and a list of effects.
    Probabilities must sum to 1.
    """
    outcomes: tuple[tuple[float, tuple[Effect, ...]], ...]  # ((prob, (effects,...)), ...)


# ---------------------------------------------------------------------------
# Functors (domain transfer)
# ---------------------------------------------------------------------------

@dataclass
class Functor:
    """A structure-preserving map between domains.

    Maps types, predicates, and actions from source to target domain.
    A plan in the source domain translates to a plan in the target
    domain via the functor.
    """
    name: str
    source_domain: str
    target_domain: str
    type_map: dict[str, str]       # source_type -> target_type
    predicate_map: dict[str, str]  # source_pred -> target_pred
    action_map: dict[str, str]     # source_action -> target_action


# ---------------------------------------------------------------------------
# Natural transformations (operator schemas)
# ---------------------------------------------------------------------------

@dataclass
class NaturalTransformation:
    """An operator schema that works uniformly across types.

    Given functors F and G from a common source category, a natural
    transformation eta: F => G provides, for each object X, a morphism
    eta_X: F(X) -> G(X) that commutes with all morphisms in the source.

    In planning: an action template parameterized by type.
    e.g., "move" works for Block, Package, Piece — same schema.
    """
    name: str
    source_functor: str
    target_functor: str
    # For each type in the source category, the action that implements
    # the transformation for that type.
    components: dict[str, str]  # type_name -> action_name

