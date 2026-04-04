"""CatPlan performance optimizations.

E.1: Bitset state representation
E.2: Successor generator (precondition index)
E.3: Symmetry breaking
E.4: IDA* and anytime search
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product as iterproduct

from .types import (
    Domain, Problem, ActionDef, Effect,
    AtomCondition, ConditionExpr,
    GroundAction, GroundAtom, State, NegatedGoalAtom,
)
from .planner import (
    build_objects_by_type, applicable, apply_action,
    recompute_derived, goal_satisfied, _make_bindings,
    eval_condition, sheaf_energy, goal_count_heuristic,
    _objects_for_type,
)


# ---------------------------------------------------------------------------
# E.1: Bitset state representation
# ---------------------------------------------------------------------------

class AtomIndex:
    """Maps ground atoms to integer IDs for bitset representation.

    Built once per problem. All atoms that could ever appear are
    enumerated from the initial state + all action effects.
    """

    def __init__(self, domain: Domain, problem: Problem):
        obt = build_objects_by_type(domain, problem)

        # Collect all possible ground atoms.
        all_atoms: set[GroundAtom] = set(problem.init)
        all_atoms |= problem.goal

        # Add atoms from all possible action effects.
        for action_def in domain.actions.values():
            param_domains = []
            for param in action_def.params:
                param_domains.append(_objects_for_type(param.type_name, obt, domain))
            if not param_domains:
                continue
            for binding in iterproduct(*param_domains):
                bindings = dict(zip([p.name for p in action_def.params], binding))
                for eff in action_def.effects:
                    args = tuple(bindings.get(a, a) for a in eff.args if a)
                    if args:
                        all_atoms.add(GroundAtom(predicate=eff.predicate, args=args))

        # Derived predicate atoms.
        for dp in domain.derived.values():
            arg_lists = [_objects_for_type(t, obt, domain) for t in dp.param_types]
            for combo in iterproduct(*arg_lists):
                all_atoms.add(GroundAtom(predicate=dp.name, args=tuple(combo)))

        self._atom_to_id: dict[GroundAtom, int] = {}
        self._id_to_atom: list[GroundAtom] = []
        for atom in sorted(all_atoms, key=str):
            idx = len(self._id_to_atom)
            self._atom_to_id[atom] = idx
            self._id_to_atom.append(atom)

        self.n_atoms = len(self._id_to_atom)

    def atom_to_id(self, atom: GroundAtom) -> int | None:
        return self._atom_to_id.get(atom)

    def id_to_atom(self, idx: int) -> GroundAtom:
        return self._id_to_atom[idx]

    def state_to_bitset(self, state: State) -> frozenset[int]:
        """Convert a state to a set of atom IDs."""
        return frozenset(self._atom_to_id[a] for a in state if a in self._atom_to_id)

    def bitset_to_state(self, bitset: frozenset[int]) -> State:
        """Convert a bitset back to a state."""
        return frozenset(self._id_to_atom[i] for i in bitset)


# ---------------------------------------------------------------------------
# E.2: Successor generator (precondition index)
# ---------------------------------------------------------------------------

class SuccessorGenerator:
    """Index that maps predicates to actions that need them.

    Instead of checking all ground actions for applicability,
    only check actions whose positive preconditions overlap with
    the current state's predicates.

    This is Fast Downward's successor generator adapted for CatPlan.
    """

    def __init__(self, domain: Domain, problem: Problem):
        self.domain = domain
        self.obt = build_objects_by_type(domain, problem)

        # Ground all actions.
        self.all_actions = self._ground_all(domain, problem)

        # Build index: predicate_name -> list of ground actions that
        # have this predicate in a positive precondition.
        self._pred_index: dict[str, list[int]] = defaultdict(list)
        # Also: index by (predicate, first_arg) for more precise matching.
        self._pred_arg_index: dict[tuple[str, str], list[int]] = defaultdict(list)

        for i, action in enumerate(self.all_actions):
            action_def = domain.actions[action.action_name]
            bindings = _make_bindings(action_def, action)
            preds_needed: set[str] = set()

            for cond in action_def.preconditions:
                if isinstance(cond, AtomCondition) and not cond.negated:
                    preds_needed.add(cond.predicate)
                    bound_args = tuple(bindings.get(a, a) for a in cond.args)
                    if bound_args:
                        self._pred_arg_index[(cond.predicate, bound_args[0])].append(i)

            for pred in preds_needed:
                self._pred_index[pred].append(i)

        # Deduplicate index lists.
        for k in self._pred_index:
            self._pred_index[k] = sorted(set(self._pred_index[k]))
        for k in self._pred_arg_index:
            self._pred_arg_index[k] = sorted(set(self._pred_arg_index[k]))

    def _ground_all(self, domain, problem):
        from .planner import ground_actions
        return ground_actions(domain, problem)

    def applicable_actions(self, state: State) -> list[GroundAction]:
        """Find applicable actions using the index.

        Instead of checking all N ground actions, only check those
        whose precondition predicates are present in the state.
        """
        # Collect candidate action indices from the state's predicates.
        candidates: set[int] = set()
        state_preds = set()
        for atom in state:
            state_preds.add(atom.predicate)
            # Use the more precise (pred, first_arg) index.
            key = (atom.predicate, atom.args[0]) if atom.args else (atom.predicate, "")
            if key in self._pred_arg_index:
                candidates.update(self._pred_arg_index[key])

        # Also add actions indexed by predicate only (for those not in arg index).
        for pred in state_preds:
            if pred in self._pred_index:
                candidates.update(self._pred_index[pred])

        # Check actual applicability only for candidates.
        result = []
        for i in candidates:
            action = self.all_actions[i]
            if applicable(action, state, self.domain, self.obt):
                result.append(action)
        return result


# ---------------------------------------------------------------------------
# E.3: Symmetry breaking
# ---------------------------------------------------------------------------

def find_symmetric_objects(domain: Domain, problem: Problem) -> list[set[str]]:
    """Find groups of interchangeable objects.

    Two objects are symmetric if they have the same type and appear
    in exactly the same predicates in the initial state (modulo
    swapping their names).

    Returns a list of equivalence classes.
    """
    # Group objects by type.
    type_groups: dict[str, list[str]] = defaultdict(list)
    for obj in problem.objects.values():
        type_groups[obj.type_name].append(obj.name)

    # For each type group, check which objects have identical predicate patterns.
    symmetric_groups: list[set[str]] = []

    for type_name, objects in type_groups.items():
        if len(objects) < 2:
            continue

        # Build a "signature" for each object: the set of predicates
        # it appears in, with its position replaced by a placeholder.
        signatures: dict[str, frozenset] = {}
        for obj in objects:
            sig_parts: list[tuple] = []
            for atom in problem.init:
                if obj in atom.args:
                    # Replace obj with placeholder '*' in the atom.
                    normalized_args = tuple('*' if a == obj else a for a in atom.args)
                    sig_parts.append((atom.predicate, normalized_args))
            signatures[obj] = frozenset(sig_parts)

        # Group by signature.
        sig_to_objs: dict[frozenset, set[str]] = defaultdict(set)
        for obj, sig in signatures.items():
            sig_to_objs[sig].add(obj)

        for group in sig_to_objs.values():
            if len(group) >= 2:
                symmetric_groups.append(group)

    return symmetric_groups


def break_symmetry(
    actions: list[GroundAction],
    symmetric_groups: list[set[str]],
) -> list[GroundAction]:
    """Prune symmetric actions.

    If objects a and b are symmetric, and we have both pick(hand, a) and
    pick(hand, b) as candidates, keep only the one with the lexically
    smaller object name. This halves the branching factor for symmetric objects.
    """
    if not symmetric_groups:
        return actions

    # Build lookup: object -> canonical representative (lexically smallest).
    canonical: dict[str, str] = {}
    for group in symmetric_groups:
        rep = min(group)
        for obj in group:
            canonical[obj] = rep

    # Filter: keep an action only if all its args are canonical
    # (or the arg isn't in any symmetric group).
    pruned = []
    seen_canonical: set[tuple] = set()
    for action in actions:
        # Map args to canonical forms.
        canon_args = tuple(canonical.get(a, a) for a in action.args)
        key = (action.action_name, canon_args)
        if key not in seen_canonical:
            seen_canonical.add(key)
            pruned.append(action)

    return pruned


# ---------------------------------------------------------------------------
# E.4: Iterative deepening A* (IDA*)
# ---------------------------------------------------------------------------

def plan_ida_star(
    domain: Domain,
    problem: Problem,
    max_depth: int = 50,
) -> list[GroundAction] | None:
    """IDA* search — memory-efficient alternative to A*.

    Uses iterative deepening with the goal-count heuristic.
    Memory usage is O(depth) instead of O(expanded states).
    Better for problems with deep solutions and limited memory.
    """
    obt = build_objects_by_type(domain, problem)
    init_state = frozenset(problem.init)
    init_state = recompute_derived(init_state, domain, obt)

    goal_atoms = problem.goal
    neg_goal = problem.neg_goal

    if goal_satisfied(init_state, goal_atoms, neg_goal):
        return []

    from .planner import ground_actions
    all_actions = ground_actions(domain, problem)

    def h(state):
        return goal_count_heuristic(state, goal_atoms, neg_goal)

    bound = h(init_state)

    for iteration in range(max_depth):
        result, new_bound = _ida_search(
            init_state, [], 0, bound,
            domain, obt, all_actions, goal_atoms, neg_goal, h,
        )
        if result is not None:
            return result
        if new_bound == float('inf'):
            return None
        bound = new_bound

    return None


def _ida_search(
    state: State,
    path: list[GroundAction],
    g: int,
    bound: float,
    domain: Domain,
    obt: dict,
    all_actions: list[GroundAction],
    goal: set[GroundAtom],
    neg_goal: set[NegatedGoalAtom],
    h_fn,
) -> tuple[list[GroundAction] | None, float]:
    """Recursive IDA* helper."""
    f = g + h_fn(state)
    if f > bound:
        return None, f

    if goal_satisfied(state, goal, neg_goal):
        return list(path), f

    min_bound = float('inf')

    for action in all_actions:
        if not applicable(action, state, domain, obt):
            continue
        new_state = apply_action(action, state, domain, obt)
        path.append(action)
        result, new_bound = _ida_search(
            new_state, path, g + 1, bound,
            domain, obt, all_actions, goal, neg_goal, h_fn,
        )
        path.pop()
        if result is not None:
            return result, new_bound
        min_bound = min(min_bound, new_bound)

    return None, min_bound


# ---------------------------------------------------------------------------
# Optimized planner combining all E-phase improvements
# ---------------------------------------------------------------------------

def plan_optimized(
    domain: Domain,
    problem: Problem,
    max_expansions: int = 100000,
    use_symmetry: bool = True,
    verbose: bool = False,
) -> tuple[list[GroundAction] | None, dict]:
    """Optimized planner with successor generator and symmetry breaking.

    Combines E.2 (successor generator) and E.3 (symmetry breaking)
    with the best heuristic from Phase C (CG or FF depending on problem size).
    """
    import heapq

    obt = build_objects_by_type(domain, problem)
    init_state = frozenset(problem.init)
    init_state = recompute_derived(init_state, domain, obt)

    goal_atoms = problem.goal
    neg_goal = problem.neg_goal

    if goal_satisfied(init_state, goal_atoms, neg_goal):
        return [], {"expansions": 0}

    # Build successor generator (E.2).
    succ_gen = SuccessorGenerator(domain, problem)

    # Find symmetric objects (E.3).
    sym_groups = find_symmetric_objects(domain, problem) if use_symmetry else []

    if verbose:
        print(f"  Total ground actions: {len(succ_gen.all_actions)}")
        print(f"  Symmetric groups: {sym_groups}")

    has_invariants = any(inv.condition is not None for inv in domain.invariants)

    def heuristic(state):
        h = goal_count_heuristic(state, goal_atoms, neg_goal)
        if has_invariants:
            h += sheaf_energy(state, domain, obt)
        return h

    h0 = heuristic(init_state)
    counter = 0
    open_list = []
    heapq.heappush(open_list, (h0, 0, counter, init_state, []))
    closed: set[State] = set()
    expansions = 0
    succ_gen_hits = 0
    succ_gen_total = 0

    while open_list and expansions < max_expansions:
        f, g, _, state, path = heapq.heappop(open_list)

        if state in closed:
            continue
        closed.add(state)
        expansions += 1

        # E.2: Use successor generator instead of checking all actions.
        candidates = succ_gen.applicable_actions(state)
        succ_gen_total += len(succ_gen.all_actions)
        succ_gen_hits += len(candidates)

        # E.3: Symmetry breaking.
        if sym_groups:
            candidates = break_symmetry(candidates, sym_groups)

        for action in candidates:
            new_state = apply_action(action, state, domain, obt)
            if new_state in closed:
                continue

            new_path = path + [action]

            if goal_satisfied(new_state, goal_atoms, neg_goal):
                stats = {
                    "expansions": expansions,
                    "candidates_per_expansion": succ_gen_hits / max(expansions, 1),
                    "pruning_ratio": 1.0 - succ_gen_hits / max(succ_gen_total, 1),
                }
                return new_path, stats

            h = heuristic(new_state)
            counter += 1
            heapq.heappush(open_list, (g + 1 + h, g + 1, counter, new_state, new_path))

    stats = {
        "expansions": expansions,
        "candidates_per_expansion": succ_gen_hits / max(expansions, 1),
        "pruning_ratio": 1.0 - succ_gen_hits / max(succ_gen_total, 1),
    }
    return None, stats
