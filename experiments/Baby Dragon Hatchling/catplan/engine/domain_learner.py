"""Domain learning — discover a CatPlan domain from demonstrations.

Given demonstration trajectories from a world simulator, discover:
1. Types (object categories) — group by predicate signature
2. Predicates (relevant properties) — filter constants
3. Actions (operator schemas) — preconditions + effects from transitions
4. Numeric preconditions — discover threshold conditions on numeric predicates
5. Invariants (conservation laws) — properties conserved across transitions
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import sys, os
_CATPLAN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _CATPLAN_DIR not in sys.path:
    sys.path.insert(0, _CATPLAN_DIR)

from worlds.base import Observation, Transition, Demonstration
from .types import (
    Type, Predicate, ActionParam, ActionDef, Invariant, Domain,
    AtomCondition, Effect,
)


# ---------------------------------------------------------------------------
# Step 1: Type discovery
# ---------------------------------------------------------------------------

def discover_types(demos: list[Demonstration]) -> dict[str, set[str]]:
    """Group objects by predicate signature (which predicates they appear in)."""
    object_sigs: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for demo in demos:
        for trans in demo.transitions:
            for obs in [trans.before, trans.after]:
                for pred, args, val in obs.facts:
                    for pos, obj in enumerate(args):
                        object_sigs[obj].add((pred, pos))

    # Group by signature.
    sig_to_objects: dict[frozenset, set[str]] = defaultdict(set)
    for obj, sig in object_sigs.items():
        sig_to_objects[frozenset(sig)].add(obj)

    # Merge types with high overlap (>80% Jaccard similarity).
    # This fixes the over-splitting problem.
    sig_list = list(sig_to_objects.items())
    merged: list[tuple[frozenset, set[str]]] = []
    used = set()

    for i, (sig_i, objs_i) in enumerate(sig_list):
        if i in used:
            continue
        group_sig = set(sig_i)
        group_objs = set(objs_i)
        for j, (sig_j, objs_j) in enumerate(sig_list):
            if j <= i or j in used:
                continue
            intersection = set(sig_i) & set(sig_j)
            union = set(sig_i) | set(sig_j)
            if len(union) > 0 and len(intersection) / len(union) > 0.6:
                group_sig |= set(sig_j)
                group_objs |= objs_j
                used.add(j)
        used.add(i)
        merged.append((frozenset(group_sig), group_objs))

    # Name types from their most distinctive predicate.
    types: dict[str, set[str]] = {}
    used_names: set[str] = set()
    for sig, objects in sorted(merged, key=lambda x: -len(x[1])):
        pred_names = sorted({p for p, _ in sig})
        # Pick the most distinctive predicate name.
        name = pred_names[0] if pred_names else "Entity"
        name = name.capitalize()
        base = name
        counter = 2
        while name in used_names:
            name = f"{base}{counter}"
            counter += 1
        used_names.add(name)
        types[name] = objects

    return types


# ---------------------------------------------------------------------------
# Step 2: Predicate discovery
# ---------------------------------------------------------------------------

def discover_predicates(
    demos: list[Demonstration],
    types: dict[str, set[str]],
) -> dict[str, Predicate]:
    """Find predicates, their types, and whether they're boolean or numeric."""
    obj_to_type: dict[str, str] = {}
    for type_name, objects in types.items():
        for obj in objects:
            obj_to_type[obj] = type_name

    pred_info: dict[str, dict] = {}

    for demo in demos:
        for trans in demo.transitions:
            for obs in [trans.before, trans.after]:
                for pred, args, val in obs.facts:
                    if pred not in pred_info:
                        param_types = tuple(obj_to_type.get(a, "Any") for a in args)
                        pred_info[pred] = {
                            "arity": len(args),
                            "param_types": param_types,
                            "is_numeric": isinstance(val, (int, float)) and not isinstance(val, bool),
                            "demo_count": set(),
                            "values_seen": set(),
                        }
                    pred_info[pred]["demo_count"].add(id(demo))
                    if isinstance(val, (bool, int, float)):
                        v = round(val, 3) if isinstance(val, float) else val
                        pred_info[pred]["values_seen"].add(v)

    predicates: dict[str, Predicate] = {}
    for pred_name, info in pred_info.items():
        if len(info["demo_count"]) >= 2 or len(info["values_seen"]) > 1:
            predicates[pred_name] = Predicate(
                name=pred_name,
                param_types=info["param_types"],
            )

    return predicates


# ---------------------------------------------------------------------------
# Step 3: Operator extraction
# ---------------------------------------------------------------------------

def discover_actions(
    demos: list[Demonstration],
    types: dict[str, set[str]],
    predicates: dict[str, Predicate],
) -> dict[str, ActionDef]:
    """Discover action schemas from transitions."""
    obj_to_type: dict[str, str] = {}
    for type_name, objects in types.items():
        for obj in objects:
            obj_to_type[obj] = type_name

    action_transitions: dict[str, list[Transition]] = defaultdict(list)
    for demo in demos:
        for trans in demo.transitions:
            action_transitions[trans.action].append(trans)

    actions: dict[str, ActionDef] = {}

    for action_name, transitions in action_transitions.items():
        if len(transitions) < 2:
            continue

        # Discover parameter types: use MAJORITY vote across all transitions
        # (not just the first one) to avoid overfitting.
        param_type_votes: dict[int, Counter] = defaultdict(Counter)
        for trans in transitions:
            for i, arg in enumerate(trans.action_args):
                param_type_votes[i][obj_to_type.get(arg, "Any")] += 1

        n_params = len(transitions[0].action_args)
        params = []
        for i in range(n_params):
            if i in param_type_votes:
                best_type = param_type_votes[i].most_common(1)[0][0]
            else:
                best_type = "Any"
            params.append(ActionParam(name=f"p{i}", type_name=best_type))

        # Boolean preconditions.
        bool_preconds = _discover_bool_preconditions(transitions, params)

        # Numeric preconditions (NEW).
        num_preconds = _discover_numeric_preconditions(transitions, params, predicates)

        # Effects.
        effects = _discover_effects(transitions, params)

        # Numeric preconditions and effects are stored as metadata
        # (logged but not used for planning). The planner operates on
        # boolean predicates only. Numeric constraints are informational.
        # TODO: Phase D.8 will add proper enriched predicate support.

        actions[action_name] = ActionDef(
            name=action_name,
            params=params,
            preconditions=bool_preconds,
            effects=effects,
        )

    return actions


def _generalize_args(args: tuple[str, ...], action_args: tuple[str, ...]) -> tuple[str, ...]:
    """Replace action args with parameter names."""
    arg_set = set(action_args)
    arg_list = list(action_args)
    return tuple(
        f"p{arg_list.index(a)}" if a in arg_set else a
        for a in args
    )


def _discover_bool_preconditions(
    transitions: list[Transition],
    params: list[ActionParam],
) -> list[AtomCondition]:
    """Find boolean predicates true before EVERY application."""
    all_before: list[set[tuple[str, tuple[str, ...]]]] = []

    for trans in transitions:
        arg_set = set(trans.action_args)
        before: set[tuple[str, tuple[str, ...]]] = set()
        for pred, args, val in trans.before.facts:
            if val is not True:
                continue
            if any(a in arg_set for a in args):
                gen = _generalize_args(args, trans.action_args)
                before.add((pred, gen))
        all_before.append(before)

    if not all_before:
        return []

    # Intersection across ALL transitions.
    common = all_before[0]
    for bp in all_before[1:]:
        common = common & bp

    return [
        AtomCondition(predicate=pred, args=args, negated=False)
        for pred, args in sorted(common)
    ]


def _discover_numeric_preconditions(
    transitions: list[Transition],
    params: list[ActionParam],
    predicates: dict[str, Predicate],
) -> list[AtomCondition]:
    """Discover numeric threshold preconditions.

    If a numeric predicate's value is ALWAYS above (or below) a threshold
    when the action is applied, that's a numeric precondition.

    e.g., free_valence(p0) > 0 whenever bond(p0, p1) happens.
    """
    # For each numeric predicate involving action args, collect the
    # before-values across all transitions.
    numeric_profiles: dict[tuple[str, tuple[str, ...]], list[float]] = defaultdict(list)

    for trans in transitions:
        arg_set = set(trans.action_args)
        for pred, args, val in trans.before.facts:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if not any(a in arg_set for a in args):
                continue
            gen = _generalize_args(args, trans.action_args)
            numeric_profiles[(pred, gen)].append(float(val))

    # For each profile, check if there's a consistent lower or upper bound.
    # We return these as invariant annotations (stored in the action's metadata).
    # For now, encode as AtomCondition with a special naming convention.
    preconds = []
    for (pred, gen_args), values in numeric_profiles.items():
        min_val = min(values)
        max_val = max(values)

        # If the minimum observed value is strictly positive,
        # the precondition is "pred(args) > 0".
        if min_val > 0 and 0 not in values:
            # Store as a pseudo-condition: pred_positive(args).
            # The planner will need to interpret this.
            preconds.append(AtomCondition(
                predicate=f"{pred}_positive",
                args=gen_args,
                negated=False,
            ))

    return preconds


def _discover_effects(
    transitions: list[Transition],
    params: list[ActionParam],
) -> list[Effect]:
    """Find boolean predicates that consistently change."""
    add_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    del_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    total = len(transitions)

    for trans in transitions:
        before_true = set()
        after_true = set()
        arg_set = set(trans.action_args)

        for pred, args, val in trans.before.facts:
            if val is True and any(a in arg_set for a in args):
                before_true.add((pred, _generalize_args(args, trans.action_args)))
        for pred, args, val in trans.after.facts:
            if val is True and any(a in arg_set for a in args):
                after_true.add((pred, _generalize_args(args, trans.action_args)))

        for item in after_true - before_true:
            add_counts[item] += 1
        for item in before_true - after_true:
            del_counts[item] += 1

    threshold = total * 0.7
    effects = []

    for (pred, args), count in add_counts.items():
        if count >= threshold:
            effects.append(Effect(predicate=pred, args=args, set_to=True))

    for (pred, args), count in del_counts.items():
        if count >= threshold:
            effects.append(Effect(predicate=pred, args=args, set_to=False))

    # Constructive effects: detect result predicates with computed values.
    # These are N-ary predicates where the last argument is NOT an action
    # parameter but varies systematically with the inputs.
    result_patterns: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    for trans in transitions:
        arg_set = set(trans.action_args)
        for pred, args, val in trans.after.facts:
            if val is not True:
                continue
            # Check: some args are action params, some are not.
            param_args = [a for a in args if a in arg_set]
            non_param_args = [a for a in args if a not in arg_set]
            if param_args and non_param_args and len(args) >= 3:
                gen_args = _generalize_args(args, trans.action_args)
                result_patterns[pred].append((gen_args, non_param_args[0]))

    for pred, patterns in result_patterns.items():
        if len(patterns) >= threshold:
            # This predicate has a constructive effect.
            # The generalized args will have some 'p0', 'p1' and some
            # specific result values. Use a wildcard for the result.
            sample_args = patterns[0][0]
            # Replace the non-parameter positions with a result marker.
            result_args = []
            for a in sample_args:
                if a.startswith("p"):
                    result_args.append(a)
                else:
                    result_args.append("_result")
            # Check if we already have this effect.
            eff_key = (pred, tuple(result_args))
            already = any(e.predicate == pred for e in effects)
            if not already:
                effects.append(Effect(
                    predicate=pred,
                    args=tuple(result_args),
                    set_to=True,
                ))

    return effects


def _discover_numeric_effects(
    transitions: list[Transition],
    params: list[ActionParam],
    predicates: dict[str, Predicate],
) -> list[Effect]:
    """Discover consistent numeric changes (e.g., free_valence decreases by 1)."""
    # For each numeric predicate involving action args, track the delta.
    deltas: dict[tuple[str, tuple[str, ...]], list[float]] = defaultdict(list)

    for trans in transitions:
        arg_set = set(trans.action_args)
        before_vals: dict[tuple[str, tuple[str, ...]], float] = {}
        after_vals: dict[tuple[str, tuple[str, ...]], float] = {}

        for pred, args, val in trans.before.facts:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if any(a in arg_set for a in args):
                    gen = _generalize_args(args, trans.action_args)
                    before_vals[(pred, gen)] = float(val)

        for pred, args, val in trans.after.facts:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if any(a in arg_set for a in args):
                    gen = _generalize_args(args, trans.action_args)
                    after_vals[(pred, gen)] = float(val)

        for key in before_vals:
            if key in after_vals:
                delta = after_vals[key] - before_vals[key]
                if abs(delta) > 0.001:
                    deltas[key].append(delta)

    # If a delta is consistent (same value in >70% of transitions), it's an effect.
    effects = []
    total = len(transitions)
    for (pred, args), delta_list in deltas.items():
        if len(delta_list) < total * 0.5:
            continue
        # Check if deltas are consistent.
        from collections import Counter
        rounded = [round(d, 2) for d in delta_list]
        most_common_delta, count = Counter(rounded).most_common(1)[0]
        if count >= len(delta_list) * 0.7:
            # Consistent numeric effect.
            # Encode as a special effect annotation.
            direction = "decrease" if most_common_delta < 0 else "increase"
            effects.append(Effect(
                predicate=f"{pred}_{direction}_{abs(most_common_delta):.0f}",
                args=args,
                set_to=True,  # marker
            ))

    return effects


# ---------------------------------------------------------------------------
# Step 4: Invariant discovery
# ---------------------------------------------------------------------------

def discover_invariants(
    demos: list[Demonstration],
    predicates: dict[str, Predicate],
) -> list[Invariant]:
    """Find conserved numeric quantities."""
    invariants = []

    numeric_preds: set[str] = set()
    for demo in demos:
        for trans in demo.transitions:
            for pred, args, val in trans.before.facts:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    numeric_preds.add(pred)

    for pred_name in sorted(numeric_preds):
        totals: list[float] = []
        for demo in demos:
            for trans in demo.transitions:
                for obs in [trans.before, trans.after]:
                    total = 0.0
                    for p, a, v in obs.facts:
                        if p == pred_name and isinstance(v, (int, float)):
                            total += v
                    totals.append(total)

        if totals:
            first = totals[0]
            if all(abs(t - first) < 0.01 for t in totals):
                invariants.append(Invariant(
                    description=f"conservation of {pred_name}: total = {first:.1f}",
                    raw_text=f"sum({pred_name}) = {first:.1f}",
                    condition=None,
                ))

    return invariants


# ---------------------------------------------------------------------------
# Step 5: Assemble domain
# ---------------------------------------------------------------------------

def learn_domain(demos: list[Demonstration], domain_name: str = "Discovered") -> Domain:
    """Full pipeline: demonstrations -> CatPlan domain."""
    types_map = discover_types(demos)
    predicates = discover_predicates(demos, types_map)
    actions = discover_actions(demos, types_map, predicates)
    invariants = discover_invariants(demos, predicates)

    domain = Domain(name=domain_name)
    for type_name in types_map:
        domain.types[type_name] = Type(name=type_name)
    for pred in predicates.values():
        domain.predicates[pred.name] = pred
    for action in actions.values():
        domain.actions[action.name] = action
    domain.invariants = invariants

    return domain, types_map


def domain_to_catplan(domain: Domain) -> str:
    """Serialize to .catplan text."""
    lines = [f"category {domain.name} where", ""]

    for t in sorted(domain.types.values(), key=lambda x: x.name):
        lines.append(f"  type {t.name}")
    lines.append("")

    for p in sorted(domain.predicates.values(), key=lambda x: x.name):
        type_chain = " -> ".join(p.param_types) + " -> Prop"
        lines.append(f"  pred {p.name} : {type_chain}")
    lines.append("")

    for inv in domain.invariants:
        lines.append(f'  invariant "{inv.description}"')
        if inv.raw_text:
            lines.append(f"    {inv.raw_text}")
    if domain.invariants:
        lines.append("")

    for action in sorted(domain.actions.values(), key=lambda x: x.name):
        param_str = ", ".join(f"{p.name} : {p.type_name}" for p in action.params)
        lines.append(f"  action {action.name}({param_str})")
        for cond in action.preconditions:
            if isinstance(cond, AtomCondition):
                neg = "not(" if cond.negated else ""
                neg_close = ")" if cond.negated else ""
                args_str = ", ".join(cond.args)
                lines.append(f"    require {neg}{cond.predicate}({args_str}){neg_close}")
        for eff in action.effects:
            neg = "not(" if not eff.set_to else ""
            neg_close = ")" if not eff.set_to else ""
            args_str = ", ".join(eff.args)
            lines.append(f"    effect {neg}{eff.predicate}({args_str}){neg_close}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end: learn domain, then solve a problem
# ---------------------------------------------------------------------------

def learn_and_solve(
    demos: list[Demonstration],
    problem_init: Observation,
    problem_goal: set[tuple[str, tuple[str, ...]]],
    domain_name: str = "Discovered",
) -> tuple[Domain, list | None]:
    """Learn a domain from demos, then plan to reach a goal.

    problem_init: observation of the initial state
    problem_goal: set of (predicate, args) that must be true

    Returns (domain, plan) where plan is a list of (action, args) or None.
    """
    from .types import Problem, ObjectDecl, GroundAtom
    from .planner import plan_advanced
    from .heuristic_selector import select_heuristic

    # Learn domain.
    domain, types_map = learn_domain(demos, domain_name)

    # Build problem from observations.
    # For novel objects not in training, assign types by matching their
    # predicate signature against the learned type signatures.
    objects: dict[str, ObjectDecl] = {}
    obj_to_type: dict[str, str] = {}
    for type_name, objs in types_map.items():
        for obj in objs:
            obj_to_type[obj] = type_name

    # Build predicate signatures for each learned type.
    type_sigs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for type_name, objs in types_map.items():
        for demo in demos:
            for trans in demo.transitions:
                for obs_data in [trans.before, trans.after]:
                    for pred, args, val in obs_data.facts:
                        for pos, arg in enumerate(args):
                            if arg in objs:
                                type_sigs[type_name].add((pred, pos))

    # Build predicate signatures for novel objects from the init observation.
    novel_obj_sigs: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for pred, args, val in problem_init.facts:
        for pos, arg in enumerate(args):
            novel_obj_sigs[arg].add((pred, pos))

    # Assign novel objects to the best-matching learned type.
    for obj, sig in novel_obj_sigs.items():
        if obj in obj_to_type:
            continue
        best_type = None
        best_overlap = -1
        for type_name, type_sig in type_sigs.items():
            overlap = len(sig & type_sig)
            if overlap > best_overlap:
                best_overlap = overlap
                best_type = type_name
        if best_type:
            obj_to_type[obj] = best_type

    init_atoms: set[GroundAtom] = set()
    for pred, args, val in problem_init.facts:
        if val is True:
            init_atoms.add(GroundAtom(predicate=pred, args=args))
        for arg in args:
            if arg not in objects:
                t = obj_to_type.get(arg, list(domain.types.keys())[0] if domain.types else "Any")
                objects[arg] = ObjectDecl(name=arg, type_name=t)

    goal_atoms: set[GroundAtom] = set()
    for pred, args in problem_goal:
        goal_atoms.add(GroundAtom(predicate=pred, args=args))
        for arg in args:
            if arg not in objects:
                t = obj_to_type.get(arg, list(domain.types.keys())[0] if domain.types else "Any")
                objects[arg] = ObjectDecl(name=arg, type_name=t)

    problem = Problem(
        name="novel_problem",
        domain_name=domain_name,
        objects=objects,
        init=init_atoms,
        goal=goal_atoms,
    )

    # Select heuristic and plan.
    heuristic = select_heuristic(domain, problem)
    result, stats = plan_advanced(domain, problem, heuristic_mode=heuristic)

    return domain, result, stats


if __name__ == "__main__":
    sys.path.insert(0, _CATPLAN_DIR)
    from worlds.chemistry import ChemistryWorld

    print("=" * 60)
    print("  END-TO-END TEST: Learn chemistry, solve novel problem")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Phase 1: Generate training demonstrations.
    # The learner sees random bond/break on a SMALL set of atoms.
    # ---------------------------------------------------------------

    print("\n--- Phase 1: Training demonstrations ---")
    training_world = ChemistryWorld([
        ("h1", "H"), ("h2", "H"), ("o1", "O"),
    ])
    train_demos = training_world.generate_demonstrations(n=50, n_steps=5, seed=42)
    print(f"Generated {len(train_demos)} demos from H2O world")

    # Also add demos from a slightly different atom set to reduce overfitting.
    training_world2 = ChemistryWorld([
        ("h3", "H"), ("h4", "H"), ("o2", "O"), ("o3", "O"),
    ])
    train_demos += training_world2.generate_demonstrations(n=50, n_steps=5, seed=99)
    print(f"Total: {len(train_demos)} demos")

    # ---------------------------------------------------------------
    # Phase 2: Learn the domain.
    # ---------------------------------------------------------------

    print("\n--- Phase 2: Domain learning ---")
    domain, types_map = learn_domain(train_demos, "Chemistry")
    print(f"Types: {list(domain.types.keys())}")
    print(f"Predicates: {list(domain.predicates.keys())}")
    print(f"Actions: {list(domain.actions.keys())}")
    print(f"Invariants: {[inv.description for inv in domain.invariants]}")
    for a in domain.actions.values():
        print(f"\n  {a.name}:")
        print(f"    params: {[(p.name, p.type_name) for p in a.params]}")
        print(f"    preconditions: {[str(c) for c in a.preconditions]}")
        print(f"    effects: {[str(e) for e in a.effects]}")

    # ---------------------------------------------------------------
    # Phase 3: NOVEL PROBLEM — never seen in training.
    #
    # Task: Given 4 hydrogen atoms and 2 oxygen atoms (none bonded),
    # form TWO water molecules (h5-o4-h6 and h7-o5-h8).
    #
    # This is novel because:
    # - These specific atoms (h5-h8, o4-o5) never appeared in training
    # - Forming two molecules requires 4 bond actions in the right order
    # - The planner must use the discovered domain to figure this out
    # ---------------------------------------------------------------

    print("\n--- Phase 3: Solve NOVEL problem ---")
    print("Task: form 2 water molecules from 4H + 2O (never seen in training)")

    novel_world = ChemistryWorld([
        ("h5", "H"), ("h6", "H"), ("h7", "H"), ("h8", "H"),
        ("o4", "O"), ("o5", "O"),
    ])
    init_obs = novel_world.observe()

    # Goal: two water molecules.
    goal = {
        ("bonded", ("h5", "o4")),
        ("bonded", ("h6", "o4")),
        ("bonded", ("h7", "o5")),
        ("bonded", ("h8", "o5")),
    }

    domain, result, stats = learn_and_solve(
        train_demos, init_obs, goal, "Chemistry"
    )

    if result is None:
        print("\n  FAILED: No plan found!")
        print(f"  Stats: {stats}")
    else:
        print(f"\n  SOLVED in {len(result)} steps ({stats['expansions']} expansions):")
        for i, action in enumerate(result):
            print(f"    {i+1}. {action}")

        # Verify by executing in the actual world.
        print("\n--- Verification ---")
        for action in result:
            novel_world.execute(action.action_name, action.args)
        final_obs = novel_world.observe()
        all_bonded = True
        for pred, args in goal:
            val = final_obs.get(pred, args)
            status = "OK" if val else "MISSING"
            print(f"  {pred}({', '.join(args)}) = {val} [{status}]")
            if not val:
                all_bonded = False
        print(f"\n  {'SUCCESS' if all_bonded else 'FAILURE'}: "
              f"{'All' if all_bonded else 'Not all'} bonds formed!")
