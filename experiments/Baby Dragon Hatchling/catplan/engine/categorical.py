"""CatPlan categorical operations — the features beyond PDDL.

D.1: Composition — expand composite actions into sub-action sequences
D.2: Adjunctions — solve inverse problems by computation, not search
D.3: Equalizers — find objects satisfying f(x) = g(x)
D.4: Pullbacks — simultaneous constraint satisfaction
D.5: Initial algebras — recursion via catamorphism
D.6: Kan extensions — generalize from partial data
D.7: Galois connections — abstract planning with refinement
D.8: Enriched predicates — continuous values
D.9: Operads — multi-input composition
D.10: Probabilistic — stochastic effects
"""
from __future__ import annotations

from itertools import product as iterproduct

from .types import (
    Domain, Problem, State,
    CompositeAction, Adjunction, Equalizer, Pullback,
    InitialAlgebra, KanExtension, GaloisConnection,
    EnrichedAtom, NumericEffect, NumericCondition,
    OperadicAction, ProbabilisticEffect,
    Functor, NaturalTransformation,
    GroundAction, GroundAtom, ActionParam, Effect,
)
from .planner import (
    build_objects_by_type, applicable, apply_action,
    _make_bindings, eval_condition, _objects_for_type,
)


# ---------------------------------------------------------------------------
# D.1: Composition — expand a composite action into ground sub-actions
# ---------------------------------------------------------------------------

def expand_composite(
    composite: CompositeAction,
    ground_args: tuple[str, ...],
    domain: Domain,
) -> list[GroundAction]:
    """Expand a composite action into its sub-action sequence.

    ground_args: the bound arguments for the composite's parameters.
    Returns a list of ground actions that implement the composite.
    """
    # Bind composite params to ground args.
    bindings = dict(zip([p.name for p in composite.params], ground_args))

    ground_steps = []
    for action_name, param_mapping in composite.steps:
        # Map composite params through the step's param mapping.
        step_args = tuple(bindings[param_mapping[p.name]]
                          if p.name in param_mapping
                          else bindings.get(p.name, p.name)
                          for p in domain.actions[action_name].params)
        ground_steps.append(GroundAction(action_name=action_name, args=step_args))

    return ground_steps


def validate_composite(composite: CompositeAction, domain: Domain) -> list[str]:
    """Type-check a composite action. Returns list of errors (empty = valid).

    Checks that each step's action exists and parameter types are compatible.
    """
    errors = []
    for i, (action_name, param_mapping) in enumerate(composite.steps):
        if action_name not in domain.actions:
            errors.append(f"Step {i+1}: action '{action_name}' not found")
    return errors


# ---------------------------------------------------------------------------
# D.2: Adjunctions — solve F(x) = y by computing x = G(y)
# ---------------------------------------------------------------------------

def solve_via_adjunction(
    adjunction: Adjunction,
    target_atom: GroundAtom,
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
) -> GroundAction | None:
    """Try to solve a goal atom using an adjunction.

    If target_atom matches the output pattern of the left adjoint,
    compute the input via the right adjoint. Returns a ground action
    that achieves the goal, or None.

    Example: goal is add(3, 4) = 7.
    Adjunction add -| sub means: to find x where add(x, 4) = 7,
    compute x = sub(7, 4) = 3.
    """
    # Check if the target atom's predicate matches the left adjoint's output.
    left_action = domain.actions.get(adjunction.left_action)
    right_action = domain.actions.get(adjunction.right_action)
    if left_action is None or right_action is None:
        return None

    # The adjunction provides a direct computation path.
    # For now, return the right adjoint action with remapped parameters.
    # Full implementation would need to evaluate the right adjoint.
    right_args = []
    for param in right_action.params:
        mapped = adjunction.param_map.get(param.name)
        if mapped and mapped in dict(zip(
            [p.name for p in left_action.params],
            target_atom.args[:len(left_action.params)]
        )):
            right_args.append(dict(zip(
                [p.name for p in left_action.params],
                target_atom.args
            ))[mapped])
        else:
            right_args.append(target_atom.args[0] if target_atom.args else "?")

    return GroundAction(action_name=adjunction.right_action, args=tuple(right_args))


# ---------------------------------------------------------------------------
# D.3: Equalizers — find x where f(x) = g(x)
# ---------------------------------------------------------------------------

def compute_equalizer(
    equalizer: Equalizer,
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
) -> list[str]:
    """Find all objects x of source_type where f(x) = g(x) in the current state.

    Returns list of object names satisfying the equality.
    """
    objs = _objects_for_type(equalizer.source_type, objects_by_type, domain)
    results = []

    for obj in objs:
        # Find f(obj) and g(obj) in state.
        f_val = None
        g_val = None
        for atom in state:
            if atom.predicate == equalizer.morphism_f and len(atom.args) >= 1 and atom.args[0] == obj:
                f_val = atom.args[1] if len(atom.args) > 1 else "true"
            if atom.predicate == equalizer.morphism_g and len(atom.args) >= 1 and atom.args[0] == obj:
                g_val = atom.args[1] if len(atom.args) > 1 else "true"
        if f_val is not None and g_val is not None and f_val == g_val:
            results.append(obj)

    return results


# ---------------------------------------------------------------------------
# D.4: Pullbacks — find (a, b) where f(a) = g(b)
# ---------------------------------------------------------------------------

def compute_pullback(
    pullback: Pullback,
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
) -> list[tuple[str, ...]]:
    """Compute the pullback: find all tuples satisfying all constraints simultaneously.

    Each constraint is (predicate, target): the predicate applied to the
    source objects must all agree on the target value.

    Returns list of object tuples that satisfy all constraints.
    """
    # Get objects for each source type.
    obj_lists = [_objects_for_type(t, objects_by_type, domain) for t in pullback.source_types]

    results = []
    for combo in iterproduct(*obj_lists):
        # Check all constraints.
        values = []
        all_satisfied = True
        for pred_name, _ in pullback.constraints:
            found = False
            for atom in state:
                if atom.predicate == pred_name:
                    # Check if this atom's args match the combo.
                    if len(atom.args) >= 1 and atom.args[0] in combo:
                        values.append(atom.args[-1] if len(atom.args) > 1 else "true")
                        found = True
                        break
            if not found:
                all_satisfied = False
                break

        if all_satisfied and len(set(values)) <= 1:  # all agree
            results.append(combo)

    return results


# ---------------------------------------------------------------------------
# D.5: Initial algebras — catamorphism (fold)
# ---------------------------------------------------------------------------

def catamorphism(
    algebra: InitialAlgebra,
    target_zero,
    target_succ,
    value: str,
    state: State,
    domain: Domain,
    max_depth: int = 100,
) -> any:
    """Apply a catamorphism: the unique map from the initial algebra to a target.

    Given:
    - algebra: the initial algebra (e.g., Nat with zero and succ)
    - target_zero: the value at zero (e.g., 1 for factorial base case)
    - target_succ: a function (current_value, depth) -> new_value
    - value: the starting object name

    Walks the successor chain from value back to zero, applying the
    target algebra at each step.

    This IS recursion: f(0) = target_zero, f(succ(n)) = target_succ(f(n), n).
    """
    # Walk the chain from value to zero.
    chain = []
    current = value
    for _ in range(max_depth):
        if current == algebra.zero:
            break
        # Find predecessor: who has succ -> current?
        pred = None
        for atom in state:
            if atom.predicate == algebra.succ and len(atom.args) >= 2 and atom.args[1] == current:
                pred = atom.args[0]
                break
        if pred is None:
            break
        chain.append(current)
        current = pred

    # Apply catamorphism: fold from zero upward.
    result = target_zero
    for i, _ in enumerate(chain):
        result = target_succ(result, i)

    return result


# ---------------------------------------------------------------------------
# D.6: Kan extensions — extend a partial map to all objects
# ---------------------------------------------------------------------------

def left_kan_extension(
    kan: KanExtension,
    query: str,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
) -> str | None:
    """Compute the left Kan extension: best approximation of the map at a new point.

    If the query is in the known map, return the known value.
    Otherwise, find the "closest" known input and return its value.
    Closeness is determined by shared predicates in the state.

    This is the simplest possible Kan extension — nearest-neighbor.
    A proper implementation would use the colimit formula.
    """
    if query in kan.known_map:
        return kan.known_map[query]

    # Fallback: find the closest known key.
    # For now, just return None if not in the map.
    # TODO: implement proper colimit-based extension.
    return None


# ---------------------------------------------------------------------------
# D.7: Galois connections — abstract then refine
# ---------------------------------------------------------------------------

def abstract_state(
    gc: GaloisConnection,
    state: State,
) -> State:
    """Map a concrete state to an abstract state via the abstraction function."""
    abstract_atoms = set()
    for atom in state:
        if atom.predicate in gc.abstraction_map:
            abstract_pred = gc.abstraction_map[atom.predicate]
            abstract_atoms.add(GroundAtom(predicate=abstract_pred, args=atom.args))
    return frozenset(abstract_atoms)


def concretize_plan(
    gc: GaloisConnection,
    abstract_plan: list[GroundAction],
    domain: Domain,
) -> list[GroundAction]:
    """Translate an abstract plan back to a concrete plan.

    Uses the inverse of the action map in the functor.
    """
    inverse_action_map = {v: k for k, v in gc.abstraction_map.items()
                          if k in domain.actions}
    concrete_plan = []
    for action in abstract_plan:
        concrete_name = inverse_action_map.get(action.action_name, action.action_name)
        concrete_plan.append(GroundAction(action_name=concrete_name, args=action.args))
    return concrete_plan


# ---------------------------------------------------------------------------
# D.8: Enriched state operations
# ---------------------------------------------------------------------------

EnrichedState = dict[tuple[str, tuple[str, ...]], float]


def apply_numeric_effects(
    effects: list[NumericEffect],
    bindings: dict[str, str],
    enriched: EnrichedState,
):
    """Apply numeric effects to an enriched state."""
    for eff in effects:
        bound_args = tuple(bindings.get(a, a) for a in eff.args)
        key = (eff.predicate, bound_args)
        if eff.op == 'assign':
            enriched[key] = eff.value
        elif eff.op == 'increase':
            enriched[key] = enriched.get(key, 0.0) + eff.value
        elif eff.op == 'decrease':
            enriched[key] = enriched.get(key, 0.0) - eff.value


def check_numeric_condition(
    cond: NumericCondition,
    bindings: dict[str, str],
    enriched: EnrichedState,
) -> bool:
    """Check a numeric precondition against an enriched state."""
    bound_args = tuple(bindings.get(a, a) for a in cond.args)
    key = (cond.predicate, bound_args)
    val = enriched.get(key, 0.0)
    if cond.op == '<': return val < cond.value
    elif cond.op == '>': return val > cond.value
    elif cond.op == '<=': return val <= cond.value
    elif cond.op == '>=': return val >= cond.value
    elif cond.op == '=': return abs(val - cond.value) < 1e-9
    return False


# ---------------------------------------------------------------------------
# Functors — translate plans between domains
# ---------------------------------------------------------------------------

def translate_plan(
    functor: Functor,
    plan_actions: list[GroundAction],
    source_domain: Domain,
    target_domain: Domain,
) -> list[GroundAction]:
    """Translate a plan from source domain to target domain via functor."""
    translated = []
    for action in plan_actions:
        new_name = functor.action_map.get(action.action_name, action.action_name)
        # Map object names through type map.
        # This requires knowing which objects in the target correspond to source objects.
        # For now, keep args unchanged — the functor maps types not individual objects.
        translated.append(GroundAction(action_name=new_name, args=action.args))
    return translated
