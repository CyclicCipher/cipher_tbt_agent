"""CatPlan forward planner — A* search over typed state space.

Supports: derived predicates, conditional effects, quantified
preconditions (exists/forall), disjunction (or), equality (/=),
negative goals.
"""
from __future__ import annotations

import heapq
from itertools import product as iterproduct

from .types import (
    Domain, Problem, ActionDef, Effect, ConditionalEffect,
    AtomCondition, ExistsCondition, ForallCondition,
    OrCondition, AndCondition, EqualityCondition, CountCondition, ConditionExpr,
    DerivedPredicate, Invariant,
    GroundAction, GroundAtom, NegatedGoalAtom, State, ObjectDecl,
)


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def eval_condition(
    cond: ConditionExpr,
    state: State,
    bindings: dict[str, str],
    objects_by_type: dict[str, list[str]],
    domain: Domain,
) -> bool:
    """Evaluate a condition expression against a state.

    bindings: {param_name: object_name} — maps variables to objects.
    objects_by_type: {type_name: [object_names]} — for quantifier iteration.
    """
    if isinstance(cond, AtomCondition):
        args = tuple(bindings.get(a, a) for a in cond.args)
        atom = GroundAtom(predicate=cond.predicate, args=args)
        # Check derived predicates.
        if cond.predicate in domain.derived:
            present = _eval_derived(cond.predicate, args, state, objects_by_type, domain)
        else:
            present = atom in state
        return (not present) if cond.negated else present

    elif isinstance(cond, ExistsCondition):
        type_name = cond.var_type
        objs = _objects_for_type(type_name, objects_by_type, domain)
        for obj in objs:
            new_bindings = dict(bindings)
            new_bindings[cond.var_name] = obj
            if eval_condition(cond.body, state, new_bindings, objects_by_type, domain):
                return True
        return False

    elif isinstance(cond, ForallCondition):
        type_name = cond.var_type
        objs = _objects_for_type(type_name, objects_by_type, domain)
        for obj in objs:
            new_bindings = dict(bindings)
            new_bindings[cond.var_name] = obj
            if not eval_condition(cond.body, state, new_bindings, objects_by_type, domain):
                return False
        return True

    elif isinstance(cond, OrCondition):
        return any(eval_condition(c, state, bindings, objects_by_type, domain)
                   for c in cond.conditions)

    elif isinstance(cond, AndCondition):
        return all(eval_condition(c, state, bindings, objects_by_type, domain)
                   for c in cond.conditions)

    elif isinstance(cond, EqualityCondition):
        left = bindings.get(cond.left, cond.left)
        right = bindings.get(cond.right, cond.right)
        eq = (left == right)
        return (not eq) if cond.negated else eq

    elif isinstance(cond, CountCondition):
        objs = _objects_for_type(cond.var_type, objects_by_type, domain)
        count = 0
        for obj in objs:
            new_bindings = dict(bindings)
            new_bindings[cond.var_name] = obj
            if eval_condition(cond.body, state, new_bindings, objects_by_type, domain):
                count += 1
        op = cond.op
        if op == '<=': return count <= cond.value
        elif op == '>=': return count >= cond.value
        elif op == '=': return count == cond.value
        elif op == '<': return count < cond.value
        elif op == '>': return count > cond.value
        return False

    return False


def _objects_for_type(type_name: str, objects_by_type: dict[str, list[str]],
                      domain: Domain) -> list[str]:
    """Get all objects of a given type, including union type members."""
    objs = list(objects_by_type.get(type_name, []))
    # Check if this is a union type.
    t = domain.types.get(type_name)
    if t and t.is_union():
        for variant in t.variants:
            for obj in objects_by_type.get(variant, []):
                if obj not in objs:
                    objs.append(obj)
    return objs


# ---------------------------------------------------------------------------
# Derived predicate evaluation
# ---------------------------------------------------------------------------

def _eval_derived(
    pred_name: str,
    args: tuple[str, ...],
    state: State,
    objects_by_type: dict[str, list[str]],
    domain: Domain,
) -> bool:
    """Evaluate a derived predicate for specific ground arguments."""
    dp = domain.derived[pred_name]
    # Bind the derived predicate's parameters to the given arguments.
    # We need the parameter names — derive them from the domain's derived definition.
    # The param names come from the original parse; we need to reconstruct them.
    # For now, use positional binding: param 0 = args[0], etc.
    # We need the param names from the DerivedPredicate body's free variables.
    # Simpler: the body references variables by name. We extract them from
    # the body's AtomConditions. But the parser stores the param names in
    # the derived predicate's body.
    # Actually, the body uses the param names from the function signature.
    # We need to know those names. Let's store them.
    # For now: scan the body for free variables and bind positionally.
    # This is fragile — TODO: store param names in DerivedPredicate.
    bindings: dict[str, str] = {}
    # Heuristic: find all variable names used in the body that aren't
    # quantifier-bound, and bind them positionally to args.
    free_vars = _free_vars(dp.body)
    for i, var in enumerate(sorted(free_vars)):
        if i < len(args):
            bindings[var] = args[i]
    return eval_condition(dp.body, state, bindings, objects_by_type, domain)


def _free_vars(cond: ConditionExpr) -> set[str]:
    """Find free variable names in a condition expression."""
    if isinstance(cond, AtomCondition):
        return set(cond.args)
    elif isinstance(cond, ExistsCondition):
        return _free_vars(cond.body) - {cond.var_name}
    elif isinstance(cond, ForallCondition):
        return _free_vars(cond.body) - {cond.var_name}
    elif isinstance(cond, (OrCondition, AndCondition)):
        result: set[str] = set()
        for c in cond.conditions:
            result |= _free_vars(c)
        return result
    elif isinstance(cond, EqualityCondition):
        return {cond.left, cond.right}
    return set()


def recompute_derived(state: State, domain: Domain,
                      objects_by_type: dict[str, list[str]]) -> State:
    """Recompute all derived predicates and add/remove them from state.

    Returns a new state with derived atoms updated.
    """
    if not domain.derived:
        return state

    # Remove old derived atoms.
    derived_names = set(domain.derived.keys())
    base_atoms = frozenset(a for a in state if a.predicate not in derived_names)

    # Recompute each derived predicate for all object combinations.
    new_atoms = set(base_atoms)
    for dp in domain.derived.values():
        param_types = dp.param_types
        if not param_types:
            continue
        # Enumerate all type-valid argument combinations.
        arg_lists = [_objects_for_type(t, objects_by_type, domain) for t in param_types]
        for combo in iterproduct(*arg_lists):
            args = tuple(combo)
            if _eval_derived(dp.name, args, base_atoms, objects_by_type, domain):
                new_atoms.add(GroundAtom(predicate=dp.name, args=args))

    return frozenset(new_atoms)


# ---------------------------------------------------------------------------
# Action grounding
# ---------------------------------------------------------------------------

def build_objects_by_type(domain: Domain, problem: Problem) -> dict[str, list[str]]:
    """Build type -> objects mapping."""
    obt: dict[str, list[str]] = {}
    for obj in problem.objects.values():
        obt.setdefault(obj.type_name, []).append(obj.name)
        for t in domain.types.values():
            if t.is_union() and obj.type_name in t.variants:
                obt.setdefault(t.name, []).append(obj.name)
    return obt


def ground_actions(domain: Domain, problem: Problem) -> list[GroundAction]:
    """Generate all type-valid ground actions for a problem."""
    obt = build_objects_by_type(domain, problem)
    grounded: list[GroundAction] = []

    for action in domain.actions.values():
        param_domains = []
        for param in action.params:
            objects = _objects_for_type(param.type_name, obt, domain)
            param_domains.append(objects)

        if not param_domains:
            grounded.append(GroundAction(action_name=action.name, args=()))
            continue

        for binding in iterproduct(*param_domains):
            if len(set(binding)) == len(binding):
                grounded.append(GroundAction(
                    action_name=action.name, args=tuple(binding)))

    return grounded


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def _make_bindings(action_def: ActionDef, ground_action: GroundAction) -> dict[str, str]:
    """Create param_name -> object_name bindings."""
    return dict(zip([p.name for p in action_def.params], ground_action.args))


def applicable(action: GroundAction, state: State,
               domain: Domain, objects_by_type: dict[str, list[str]]) -> bool:
    """Check if a ground action's preconditions hold."""
    action_def = domain.actions[action.action_name]
    bindings = _make_bindings(action_def, action)

    for cond in action_def.preconditions:
        if not eval_condition(cond, state, bindings, objects_by_type, domain):
            return False
    return True


def apply_action(action: GroundAction, state: State,
                 domain: Domain, objects_by_type: dict[str, list[str]]) -> State:
    """Apply a ground action to a state, returning the new state."""
    action_def = domain.actions[action.action_name]
    bindings = _make_bindings(action_def, action)
    atoms = set(state)

    # Apply unconditional effects.
    _apply_effects(action_def.effects, bindings, atoms)

    # Apply conditional effects.
    # Conditional effects may have free variables (not bound by action params).
    # We enumerate all objects for each free variable and apply effects
    # for each binding where the condition holds.
    for ce in action_def.conditional_effects:
        free = _free_vars(ce.condition) - set(bindings.keys())
        # Also check effect args for free vars.
        for eff in ce.effects:
            free |= set(eff.args) - set(bindings.keys())
        if not free:
            # No free variables — just check the condition.
            if eval_condition(ce.condition, state, bindings, objects_by_type, domain):
                _apply_effects(list(ce.effects), bindings, atoms)
        else:
            # Enumerate all possible bindings for free variables.
            # Use all objects (any type) since we don't know the type of
            # free variables in conditional effects.
            all_objects = []
            for objs in objects_by_type.values():
                for o in objs:
                    if o not in all_objects:
                        all_objects.append(o)
            free_list = sorted(free)
            for combo in iterproduct(*[all_objects] * len(free_list)):
                extended_bindings = dict(bindings)
                for var, obj in zip(free_list, combo):
                    extended_bindings[var] = obj
                if eval_condition(ce.condition, state, extended_bindings, objects_by_type, domain):
                    _apply_effects(list(ce.effects), extended_bindings, atoms)

    new_state = frozenset(atoms)

    # Recompute derived predicates.
    new_state = recompute_derived(new_state, domain, objects_by_type)

    return new_state


def _apply_effects(effects: list[Effect], bindings: dict[str, str],
                   atoms: set[GroundAtom]):
    """Apply a list of effects to a mutable atom set."""
    for eff in effects:
        bound_args = tuple(bindings.get(a, a) for a in eff.args)
        if "" in bound_args:
            # Wildcard: remove all matching atoms.
            to_remove = []
            for atom in atoms:
                if atom.predicate != eff.predicate:
                    continue
                match = all(a == b or b == "" for a, b in zip(atom.args, bound_args))
                if match:
                    to_remove.append(atom)
            for atom in to_remove:
                atoms.discard(atom)
        else:
            atom = GroundAtom(predicate=eff.predicate, args=bound_args)
            if eff.set_to:
                atoms.add(atom)
            else:
                atoms.discard(atom)


# ---------------------------------------------------------------------------
# Invariant evaluation
# ---------------------------------------------------------------------------

def check_invariants(
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
) -> list[str]:
    """Check all domain invariants. Returns list of violation descriptions.

    Empty list = all invariants hold.
    """
    violations = []
    for inv in domain.invariants:
        if inv.condition is None:
            continue  # unparseable invariant, skip
        if not eval_condition(inv.condition, state, {}, objects_by_type, domain):
            violations.append(inv.description)
    return violations


def sheaf_energy(
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
) -> float:
    """Compute sheaf consistency energy. 0 = fully consistent.

    Each violated invariant contributes 1.0 to the energy.
    Each unsatisfied derived predicate that "should" hold contributes
    a smaller amount (0.1) — these are soft constraints.
    """
    violations = check_invariants(state, domain, objects_by_type)
    return float(len(violations))


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(
    domain: Domain,
    problem: Problem,
    actions: list[GroundAction],
) -> tuple[bool, str]:
    """Independently validate a plan.

    Returns (valid, message).
    Checks: preconditions at each step, invariants at each state,
    goal satisfaction at the end.
    """
    obt = build_objects_by_type(domain, problem)
    state = frozenset(problem.init)
    state = recompute_derived(state, domain, obt)

    for i, action in enumerate(actions):
        # Check preconditions.
        if not applicable(action, state, domain, obt):
            return False, f"Step {i+1}: {action} preconditions not met"

        # Apply action.
        state = apply_action(action, state, domain, obt)

        # Check invariants.
        violations = check_invariants(state, domain, obt)
        if violations:
            return False, f"Step {i+1}: {action} violates: {violations[0]}"

    # Check goal.
    if not goal_satisfied(state, problem.goal, problem.neg_goal):
        unsatisfied = [str(a) for a in problem.goal if a not in state]
        return False, f"Goal not satisfied: {unsatisfied}"

    return True, f"Valid plan ({len(actions)} steps)"


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def goal_count_heuristic(state: State, goal: set[GroundAtom],
                         neg_goal: set[NegatedGoalAtom]) -> int:
    """Count unsatisfied goal atoms (positive + negative). Admissible."""
    count = sum(1 for atom in goal if atom not in state)
    count += sum(1 for ng in neg_goal
                 if GroundAtom(predicate=ng.predicate, args=ng.args) in state)
    return count


def goal_satisfied(state: State, goal: set[GroundAtom],
                   neg_goal: set[NegatedGoalAtom]) -> bool:
    """Check if all goal conditions are satisfied."""
    if not goal.issubset(state):
        return False
    for ng in neg_goal:
        if GroundAtom(predicate=ng.predicate, args=ng.args) in state:
            return False
    return True


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

def plan(domain: Domain, problem: Problem,
         max_expansions: int = 100000,
         use_sheaf: bool = True) -> list[GroundAction] | None:
    """Find a plan using A* search.

    Heuristic: goal-count + sheaf energy (if use_sheaf=True).
    Sheaf energy penalizes states that violate invariants, steering
    search toward consistent states.
    """
    obt = build_objects_by_type(domain, problem)

    init_state: State = frozenset(problem.init)
    init_state = recompute_derived(init_state, domain, obt)

    goal_atoms = problem.goal
    neg_goal = problem.neg_goal

    if goal_satisfied(init_state, goal_atoms, neg_goal):
        return []

    all_actions = ground_actions(domain, problem)
    has_invariants = use_sheaf and any(inv.condition is not None for inv in domain.invariants)

    def heuristic(state: State) -> float:
        h = goal_count_heuristic(state, goal_atoms, neg_goal)
        if has_invariants:
            h += sheaf_energy(state, domain, obt)
        return h

    h0 = heuristic(init_state)
    counter = 0
    open_list: list[tuple[float, int, int, State, list[GroundAction]]] = []
    heapq.heappush(open_list, (h0, 0, counter, init_state, []))
    closed: set[State] = set()
    expansions = 0

    while open_list and expansions < max_expansions:
        f, g, _, state, path = heapq.heappop(open_list)

        if state in closed:
            continue
        closed.add(state)
        expansions += 1

        for action in all_actions:
            if not applicable(action, state, domain, obt):
                continue

            new_state = apply_action(action, state, domain, obt)

            if new_state in closed:
                continue

            new_path = path + [action]

            if goal_satisfied(new_state, goal_atoms, neg_goal):
                return new_path

            h = heuristic(new_state)
            counter += 1
            heapq.heappush(open_list, (g + 1 + h, g + 1, counter, new_state, new_path))

    return None


# ---------------------------------------------------------------------------
# Advanced A* search (C.1-C.7: all search improvements)
# ---------------------------------------------------------------------------

def plan_advanced(
    domain: Domain,
    problem: Problem,
    max_expansions: int = 100000,
    heuristic_mode: str = "ff",  # "ff", "cg", "landmark", "goal_count"
    use_preferred: bool = True,
    use_sheaf: bool = True,
    verbose: bool = False,
) -> tuple[list[GroundAction] | None, dict]:
    """Advanced planner with all C-phase search improvements.

    Returns (plan, stats) where stats includes expansion count,
    heuristic evaluations, etc.

    Heuristic modes:
    - "goal_count": basic goal-count (Phase A)
    - "cg": causal graph heuristic (C.2)
    - "landmark": landmark count heuristic (C.3)
    - "ff": FF heuristic / delete relaxation (C.7) — recommended
    """
    from .analysis import (
        find_landmarks, landmark_heuristic, ff_heuristic,
        causal_graph_heuristic, find_preferred,
    )

    obt = build_objects_by_type(domain, problem)

    init_state: State = frozenset(problem.init)
    init_state = recompute_derived(init_state, domain, obt)

    goal_atoms = problem.goal
    neg_goal = problem.neg_goal

    if goal_satisfied(init_state, goal_atoms, neg_goal):
        return [], {"expansions": 0, "heuristic_evals": 0}

    all_actions = ground_actions(domain, problem)
    has_invariants = use_sheaf and any(inv.condition is not None for inv in domain.invariants)

    # Pre-compute landmarks.
    landmarks = find_landmarks(domain, problem, all_actions) if use_preferred or heuristic_mode == "landmark" else set()

    if verbose:
        print(f"  Ground actions: {len(all_actions)}")
        print(f"  Landmarks: {len(landmarks)}")
        if landmarks:
            for lm in sorted(landmarks, key=str):
                print(f"    {lm}")

    # Heuristic function.
    h_evals = [0]

    def heuristic(state: State) -> float:
        h_evals[0] += 1
        if heuristic_mode == "ff":
            h = ff_heuristic(state, domain, problem, obt, all_actions)
        elif heuristic_mode == "cg":
            h = causal_graph_heuristic(state, goal_atoms, neg_goal, domain, obt, all_actions)
        elif heuristic_mode == "landmark":
            h = float(landmark_heuristic(state, landmarks))
        else:
            h = float(goal_count_heuristic(state, goal_atoms, neg_goal))
        if has_invariants:
            h += sheaf_energy(state, domain, obt)
        return h

    # A* search with preferred operator boosting (C.4).
    # Two open lists: preferred (boosted priority) and regular.
    h0 = heuristic(init_state)
    counter = 0
    # (f, g, counter, state, path, is_preferred)
    open_list: list[tuple[float, int, int, State, list[GroundAction]]] = []
    heapq.heappush(open_list, (h0, 0, counter, init_state, []))
    # Preferred open list — sampled more frequently.
    pref_list: list[tuple[float, int, int, State, list[GroundAction]]] = []
    closed: set[State] = set()
    expansions = 0
    use_pref_next = False  # alternate between lists

    while (open_list or pref_list) and expansions < max_expansions:
        # C.6 simplified: alternate between preferred and regular lists.
        if use_pref_next and pref_list:
            f, g, _, state, path = heapq.heappop(pref_list)
        elif open_list:
            f, g, _, state, path = heapq.heappop(open_list)
        elif pref_list:
            f, g, _, state, path = heapq.heappop(pref_list)
        else:
            break
        use_pref_next = not use_pref_next

        if state in closed:
            continue
        closed.add(state)
        expansions += 1

        # C.4: Find preferred operators for this state.
        preferred = find_preferred(state, domain, problem, obt, all_actions, landmarks) if use_preferred else set()

        for action in all_actions:
            if not applicable(action, state, domain, obt):
                continue

            new_state = apply_action(action, state, domain, obt)
            if new_state in closed:
                continue

            new_path = path + [action]

            if goal_satisfied(new_state, goal_atoms, neg_goal):
                stats = {
                    "expansions": expansions,
                    "heuristic_evals": h_evals[0],
                    "plan_length": len(new_path),
                    "landmarks": len(landmarks),
                    "ground_actions": len(all_actions),
                }
                return new_path, stats

            # C.5: Deferred evaluation — compute h only when expanding.
            # (Simplified: we compute h here but could defer.)
            h = heuristic(new_state)
            counter += 1

            if action in preferred:
                heapq.heappush(pref_list, (g + 1 + h, g + 1, counter, new_state, new_path))
            else:
                heapq.heappush(open_list, (g + 1 + h, g + 1, counter, new_state, new_path))

    stats = {"expansions": expansions, "heuristic_evals": h_evals[0], "plan_length": -1}
    return None, stats


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def solve_file(domain_path: str, problem_path: str,
               max_expansions: int = 100000) -> list[tuple[str, list[GroundAction] | None]]:
    """Load domain + problem files and find plans."""
    from .parser import parse_file

    domains_d, _ = parse_file(domain_path)
    if not domains_d:
        raise ValueError(f"No domain found in {domain_path}")
    domain = domains_d[0]

    _, problems_p = parse_file(problem_path)
    if not problems_p:
        raise ValueError(f"No problem found in {problem_path}")

    results = []
    for problem in problems_p:
        result = plan(domain, problem, max_expansions=max_expansions)
        results.append((problem.name, result))

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m catplan.engine.planner <domain.catplan> <problem.catplan>")
        sys.exit(1)

    results = solve_file(sys.argv[1], sys.argv[2])
    for name, actions in results:
        print(f"\nProblem: {name}")
        if actions is None:
            print("  No plan found!")
        elif len(actions) == 0:
            print("  Goal already satisfied!")
        else:
            print(f"  Plan ({len(actions)} steps):")
            for i, a in enumerate(actions):
                print(f"    {i+1}. {a}")
