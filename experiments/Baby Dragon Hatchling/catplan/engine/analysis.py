"""CatPlan analysis — causal graph, landmarks, FF heuristic.

Search improvements borrowed from the PDDL planning ecosystem,
adapted to CatPlan's typed categorical structure.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product as iterproduct

from .types import (
    Domain, Problem, ActionDef, Effect, ConditionalEffect,
    AtomCondition, ExistsCondition, ForallCondition,
    OrCondition, AndCondition, EqualityCondition, CountCondition,
    ConditionExpr, GroundAction, GroundAtom, NegatedGoalAtom, State,
)
from .planner import (
    build_objects_by_type, ground_actions, applicable,
    recompute_derived, goal_satisfied, _objects_for_type,
    _make_bindings, eval_condition,
)


# ---------------------------------------------------------------------------
# C.1: Causal graph extraction
# ---------------------------------------------------------------------------

def _predicates_read(cond: ConditionExpr) -> set[str]:
    """Extract all predicate names referenced in a condition."""
    if isinstance(cond, AtomCondition):
        return {cond.predicate}
    elif isinstance(cond, (ExistsCondition, ForallCondition)):
        return _predicates_read(cond.body)
    elif isinstance(cond, CountCondition):
        return _predicates_read(cond.body)
    elif isinstance(cond, (OrCondition, AndCondition)):
        result: set[str] = set()
        for c in cond.conditions:
            result |= _predicates_read(c)
        return result
    elif isinstance(cond, EqualityCondition):
        return set()
    return set()


def _predicates_written(action: ActionDef) -> set[str]:
    """Extract all predicate names modified by an action's effects."""
    written: set[str] = set()
    for eff in action.effects:
        written.add(eff.predicate)
    for ce in action.conditional_effects:
        for eff in ce.effects:
            written.add(eff.predicate)
    return written


def build_causal_graph(domain: Domain) -> dict[str, set[str]]:
    """Build the causal graph: predicate → set of predicates it affects.

    An edge from P to Q means there exists an action that reads P
    (in preconditions or conditional effect conditions) and writes Q
    (in effects). "Changing P can lead to changing Q."

    Also includes derived predicate dependencies: if derived predicate D
    reads predicate P, then P → D.
    """
    # Collect edges: read_pred → written_pred for each action.
    edges: dict[str, set[str]] = defaultdict(set)

    for action in domain.actions.values():
        reads: set[str] = set()
        for cond in action.preconditions:
            reads |= _predicates_read(cond)
        for ce in action.conditional_effects:
            reads |= _predicates_read(ce.condition)

        writes = _predicates_written(action)

        for r in reads:
            for w in writes:
                if r != w:
                    edges[r].add(w)
        # Self-loops for predicates that are both read and written.
        for w in writes:
            if w in reads:
                edges[w].add(w)

    # Derived predicate dependencies.
    for dp in domain.derived.values():
        deps = _predicates_read(dp.body)
        for dep in deps:
            edges[dep].add(dp.name)

    return dict(edges)


def tarjan_scc(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's SCC algorithm. Returns SCCs in topological order.

    Each SCC is a list of predicate names that are mutually dependent.
    SCCs are ordered so that if SCC_A depends on SCC_B, B comes first.
    """
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    # Collect all nodes.
    all_nodes: set[str] = set(graph.keys())
    for targets in graph.values():
        all_nodes |= targets

    def strongconnect(v: str):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, ()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in sorted(all_nodes):
        if v not in index:
            strongconnect(v)

    return sccs


def causal_graph_summary(domain: Domain) -> dict:
    """Build causal graph and return summary statistics."""
    cg = build_causal_graph(domain)
    sccs = tarjan_scc(cg)

    total_edges = sum(len(targets) for targets in cg.values())
    nontrivial = [s for s in sccs if len(s) > 1 or (len(s) == 1 and s[0] in cg.get(s[0], set()))]

    return {
        "nodes": sorted(set(cg.keys()) | {n for s in cg.values() for n in s}),
        "edges": total_edges,
        "sccs": sccs,
        "nontrivial_sccs": nontrivial,
        "graph": cg,
    }


# ---------------------------------------------------------------------------
# C.2: Causal graph heuristic
# ---------------------------------------------------------------------------

def causal_graph_heuristic(
    state: State,
    goal: set[GroundAtom],
    neg_goal: set[NegatedGoalAtom],
    domain: Domain,
    objects_by_type: dict[str, list[str]],
    all_actions: list[GroundAction],
) -> float:
    """Estimate goal distance using causal graph decomposition.

    For each unsatisfied goal predicate, estimate the cost of achieving it
    by counting how many actions touch that predicate and have unsatisfied
    preconditions. This is a relaxed estimate — it considers each goal
    predicate independently (ignoring interactions between them).

    More accurate than goal-count because it considers whether the
    preconditions for achieving a goal are themselves satisfied.
    """
    h = 0.0

    for goal_atom in goal:
        if goal_atom in state:
            continue
        # Find the cheapest action that achieves this goal atom.
        min_cost = _relaxed_cost(goal_atom, True, state, domain, objects_by_type, all_actions)
        h += min_cost

    for ng in neg_goal:
        atom = GroundAtom(predicate=ng.predicate, args=ng.args)
        if atom not in state:
            continue
        min_cost = _relaxed_cost(atom, False, state, domain, objects_by_type, all_actions)
        h += min_cost

    return h


def _relaxed_cost(
    target_atom: GroundAtom,
    want_true: bool,
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
    all_actions: list[GroundAction],
    depth: int = 0,
    max_depth: int = 3,
) -> float:
    """Estimate cost to make target_atom true (or false).

    Finds actions that achieve the target, counts their unsatisfied
    preconditions recursively (up to max_depth).
    """
    if depth >= max_depth:
        return 1.0

    best = float('inf')

    for action in all_actions:
        action_def = domain.actions[action.action_name]
        bindings = _make_bindings(action_def, action)

        # Check if this action achieves the target.
        achieves = False
        for eff in action_def.effects:
            bound_args = tuple(bindings.get(a, a) for a in eff.args)
            eff_atom = GroundAtom(predicate=eff.predicate, args=bound_args)
            if eff_atom == target_atom and eff.set_to == want_true:
                achieves = True
                break
        if not achieves:
            for ce in action_def.conditional_effects:
                for eff in ce.effects:
                    bound_args = tuple(bindings.get(a, a) for a in eff.args)
                    eff_atom = GroundAtom(predicate=eff.predicate, args=bound_args)
                    if eff_atom == target_atom and eff.set_to == want_true:
                        achieves = True
                        break
                if achieves:
                    break

        if not achieves:
            continue

        # Count unsatisfied preconditions.
        cost = 1.0  # 1 for the action itself
        for cond in action_def.preconditions:
            if not eval_condition(cond, state, bindings, objects_by_type, domain):
                cost += 1.0  # simplified: each unsatisfied precondition adds 1

        best = min(best, cost)

    return best if best < float('inf') else 1.0


# ---------------------------------------------------------------------------
# C.3: Landmark analysis
# ---------------------------------------------------------------------------

def find_landmarks(
    domain: Domain,
    problem: Problem,
    all_actions: list[GroundAction] | None = None,
) -> set[GroundAtom]:
    """Find landmark atoms: atoms that MUST become true in any valid plan.

    Uses the relaxed planning graph method:
    1. Build a relaxed planning graph (ignore delete effects) from init.
    2. For each goal atom, trace back through the RPG to find atoms that
       are the ONLY achiever at their level. Those are landmarks.

    Simplified version: an atom is a landmark if it appears as a
    precondition of every action that achieves some goal atom.
    """
    obt = build_objects_by_type(domain, problem)
    if all_actions is None:
        all_actions = ground_actions(domain, problem)

    init_state = frozenset(problem.init)
    init_state = recompute_derived(init_state, domain, obt)

    landmarks: set[GroundAtom] = set()

    # For each goal atom, find what must be true before any achiever can fire.
    for goal_atom in problem.goal:
        if goal_atom in init_state:
            continue  # already true, not a landmark

        # Find all actions that achieve this goal atom.
        achiever_preconditions: list[set[GroundAtom]] = []

        for action in all_actions:
            action_def = domain.actions[action.action_name]
            bindings = _make_bindings(action_def, action)

            achieves = False
            for eff in action_def.effects:
                bound_args = tuple(bindings.get(a, a) for a in eff.args)
                if GroundAtom(predicate=eff.predicate, args=bound_args) == goal_atom and eff.set_to:
                    achieves = True
                    break

            if not achieves:
                continue

            # Collect this achiever's required atoms.
            required: set[GroundAtom] = set()
            for cond in action_def.preconditions:
                if isinstance(cond, AtomCondition) and not cond.negated:
                    bound_args = tuple(bindings.get(a, a) for a in cond.args)
                    required.add(GroundAtom(predicate=cond.predicate, args=bound_args))
            achiever_preconditions.append(required)

        if not achiever_preconditions:
            continue

        # Atoms that appear in ALL achievers' preconditions are landmarks.
        common = achiever_preconditions[0]
        for preconds in achiever_preconditions[1:]:
            common = common & preconds

        # Only atoms not in init are interesting landmarks.
        for atom in common:
            if atom not in init_state:
                landmarks.add(atom)

    # Goal atoms themselves are landmarks.
    for goal_atom in problem.goal:
        if goal_atom not in init_state:
            landmarks.add(goal_atom)

    return landmarks


def landmark_heuristic(
    state: State,
    landmarks: set[GroundAtom],
) -> int:
    """Count unachieved landmarks. Admissible heuristic."""
    return sum(1 for lm in landmarks if lm not in state)


# ---------------------------------------------------------------------------
# C.7: FF heuristic (delete relaxation)
# ---------------------------------------------------------------------------

def _build_relaxed_planning_graph(
    state: State,
    domain: Domain,
    objects_by_type: dict[str, list[str]],
    all_actions: list[GroundAction],
    goal: set[GroundAtom],
    max_layers: int = 50,
) -> tuple[list[set[GroundAtom]], list[list[GroundAction]]]:
    """Build a relaxed planning graph (RPG) ignoring delete effects.

    Returns (atom_layers, action_layers) where:
    - atom_layers[i] = set of atoms reachable at layer i
    - action_layers[i] = actions applicable at layer i
    """
    current_atoms = set(state)
    atom_layers: list[set[GroundAtom]] = [set(current_atoms)]
    action_layers: list[list[GroundAction]] = []

    for layer in range(max_layers):
        # Find applicable actions (in relaxed state — more atoms than real).
        relaxed_state = frozenset(current_atoms)
        new_actions: list[GroundAction] = []
        new_atoms: set[GroundAtom] = set()

        for action in all_actions:
            if applicable(action, relaxed_state, domain, objects_by_type):
                action_def = domain.actions[action.action_name]
                bindings = _make_bindings(action_def, action)

                # Collect ADD effects only (ignore deletes — that's the relaxation).
                for eff in action_def.effects:
                    if eff.set_to:
                        bound_args = tuple(bindings.get(a, a) for a in eff.args)
                        atom = GroundAtom(predicate=eff.predicate, args=bound_args)
                        if atom not in current_atoms:
                            new_atoms.add(atom)
                            new_actions.append(action)

                # Conditional effects (add only).
                for ce in action_def.conditional_effects:
                    if eval_condition(ce.condition, relaxed_state, bindings, objects_by_type, domain):
                        for eff in ce.effects:
                            if eff.set_to:
                                bound_args = tuple(bindings.get(a, a) for a in eff.args)
                                atom = GroundAtom(predicate=eff.predicate, args=bound_args)
                                if atom not in current_atoms:
                                    new_atoms.add(atom)

        if not new_atoms:
            break  # fixpoint reached

        action_layers.append(new_actions)
        current_atoms |= new_atoms
        atom_layers.append(set(current_atoms))

        # Check if goal is reachable.
        if goal.issubset(current_atoms):
            break

    return atom_layers, action_layers


def ff_heuristic(
    state: State,
    domain: Domain,
    problem: Problem,
    objects_by_type: dict[str, list[str]],
    all_actions: list[GroundAction],
) -> float:
    """FF heuristic: length of the relaxed plan.

    Build a relaxed planning graph (ignore delete effects), then
    extract a relaxed plan by backward chaining from the goal.
    h = number of actions in the relaxed plan.

    This is the most widely used PDDL heuristic.
    """
    goal = problem.goal

    # Quick check: goal already satisfied?
    if goal.issubset(state):
        return 0.0

    atom_layers, action_layers = _build_relaxed_planning_graph(
        state, domain, objects_by_type, all_actions, goal,
    )

    # Check if goal is reachable.
    all_atoms = atom_layers[-1] if atom_layers else set()
    if not goal.issubset(all_atoms):
        return float('inf')  # unreachable

    # Extract relaxed plan by backward chaining.
    # For each goal atom not in the initial state, find the earliest
    # layer where an action achieves it, and add that action.
    remaining_goals = set(goal) - set(state)
    relaxed_plan_size = 0

    for layer_idx in range(len(action_layers) - 1, -1, -1):
        if not remaining_goals:
            break
        for action in action_layers[layer_idx]:
            if not remaining_goals:
                break
            action_def = domain.actions[action.action_name]
            bindings = _make_bindings(action_def, action)

            achieved = set()
            for eff in action_def.effects:
                if eff.set_to:
                    bound_args = tuple(bindings.get(a, a) for a in eff.args)
                    atom = GroundAtom(predicate=eff.predicate, args=bound_args)
                    if atom in remaining_goals:
                        achieved.add(atom)

            if achieved:
                remaining_goals -= achieved
                relaxed_plan_size += 1
                # Add this action's preconditions as new subgoals.
                for cond in action_def.preconditions:
                    if isinstance(cond, AtomCondition) and not cond.negated:
                        bound_args = tuple(bindings.get(a, a) for a in cond.args)
                        prec_atom = GroundAtom(predicate=cond.predicate, args=bound_args)
                        if prec_atom not in state:
                            remaining_goals.add(prec_atom)

    return float(relaxed_plan_size + len(remaining_goals))


# ---------------------------------------------------------------------------
# C.4: Preferred operators
# ---------------------------------------------------------------------------

def find_preferred(
    state: State,
    domain: Domain,
    problem: Problem,
    objects_by_type: dict[str, list[str]],
    all_actions: list[GroundAction],
    landmarks: set[GroundAtom] | None = None,
) -> set[GroundAction]:
    """Find preferred operators: actions that achieve a goal or landmark.

    Preferred actions are expanded first in the search.
    """
    preferred: set[GroundAction] = set()
    targets = set(problem.goal) - state
    if landmarks:
        targets |= (landmarks - state)

    for action in all_actions:
        if not applicable(action, state, domain, objects_by_type):
            continue
        action_def = domain.actions[action.action_name]
        bindings = _make_bindings(action_def, action)

        for eff in action_def.effects:
            if eff.set_to:
                bound_args = tuple(bindings.get(a, a) for a in eff.args)
                atom = GroundAtom(predicate=eff.predicate, args=bound_args)
                if atom in targets:
                    preferred.add(action)
                    break

    return preferred
