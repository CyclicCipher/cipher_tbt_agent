"""Automatic heuristic selection based on domain structure.

Instead of hardcoding which heuristic to use, analyze the causal graph
and problem structure to pick the best one.

Decision rules (from our benchmarks):

1. If the causal graph has ONE large SCC (>60% of predicates):
   -> CG heuristic. The domain is tightly coupled; CG decomposes it.
   Example: blocks world (on/clear/holding/empty form one SCC).

2. If the causal graph has MANY small SCCs or a shallow DAG:
   -> FF heuristic. Independent subgoals; relaxed planning captures this.
   Example: logistics (each package is independent).

3. If the causal graph has FEW edges (sparse dependencies):
   -> Goal-count. Most actions are always applicable; expensive heuristics
   waste time computing what goal-count gives for free.
   Example: arithmetic (every action only needs value(x,x)).

4. If the problem has invariants:
   -> Add sheaf energy to whatever heuristic is selected.
"""
from __future__ import annotations

from .types import Domain, Problem
from .analysis import build_causal_graph, tarjan_scc


def select_heuristic(domain: Domain, problem: Problem) -> str:
    """Analyze domain structure and return the best heuristic name.

    Returns one of: "goal_count", "cg", "ff".
    """
    cg = build_causal_graph(domain)
    sccs = tarjan_scc(cg)

    # Collect all predicate names.
    all_preds = set(cg.keys())
    for targets in cg.values():
        all_preds |= targets
    n_preds = max(len(all_preds), 1)

    # Count edges.
    n_edges = sum(len(targets) for targets in cg.values())

    # Find the largest SCC.
    largest_scc_size = max((len(s) for s in sccs), default=0)
    largest_scc_ratio = largest_scc_size / n_preds

    # Count nontrivial SCCs (size > 1 or self-loop).
    nontrivial = [s for s in sccs if len(s) > 1
                  or (len(s) == 1 and s[0] in cg.get(s[0], set()))]

    # Edge density: edges per predicate.
    edge_density = n_edges / n_preds if n_preds > 0 else 0

    # Count ground actions (rough proxy for branching factor).
    from .planner import ground_actions
    n_actions = len(ground_actions(domain, problem))

    # Decision logic.
    #
    # Rule 1: Large dominant SCC -> CG heuristic.
    # CG exploits the decomposition of tightly-coupled variables.
    if largest_scc_ratio > 0.5 and edge_density > 2.0:
        return "cg"

    # Rule 2: Sparse graph or extreme action count -> goal-count.
    # Cheap heuristic when expensive ones don't help.
    if edge_density <= 1.0 or n_actions > 100:
        return "goal_count"

    # Rule 3: Small problem (few actions) -> goal-count.
    # The overhead of FF/CG isn't worth it for small problems.
    if n_actions < 30:
        return "goal_count"

    # Rule 4: Multiple small SCCs, moderate density -> FF heuristic.
    # FF captures subgoal independence via relaxed planning.
    if len(nontrivial) >= 1 and edge_density > 1.0:
        return "ff"

    # Default: goal-count (safest, cheapest).
    return "goal_count"


def select_heuristic_with_explanation(domain: Domain, problem: Problem) -> tuple[str, str]:
    """Like select_heuristic but also returns explanation."""
    cg = build_causal_graph(domain)
    sccs = tarjan_scc(cg)

    all_preds = set(cg.keys())
    for targets in cg.values():
        all_preds |= targets
    n_preds = max(len(all_preds), 1)
    n_edges = sum(len(targets) for targets in cg.values())
    largest_scc_size = max((len(s) for s in sccs), default=0)
    largest_scc_ratio = largest_scc_size / n_preds
    edge_density = n_edges / n_preds if n_preds > 0 else 0

    nontrivial = [s for s in sccs if len(s) > 1
                  or (len(s) == 1 and s[0] in cg.get(s[0], set()))]

    from .planner import ground_actions
    n_actions = len(ground_actions(domain, problem))

    facts = (f"{n_preds} predicates, {n_edges} causal edges, "
             f"density={edge_density:.1f}, "
             f"{len(sccs)} SCCs ({len(nontrivial)} nontrivial), "
             f"largest SCC={largest_scc_size} ({largest_scc_ratio:.0%}), "
             f"{n_actions} ground actions")

    if largest_scc_ratio > 0.5 and edge_density > 2.0:
        return "cg", f"CG: dominant SCC covers {largest_scc_ratio:.0%}. {facts}"
    if edge_density <= 1.0 or n_actions > 100:
        return "goal_count", f"Goal-count: sparse graph or large action space. {facts}"
    if n_actions < 30:
        return "goal_count", f"Goal-count: small problem ({n_actions} actions). {facts}"
    if len(nontrivial) >= 1 and edge_density > 1.0:
        return "ff", f"FF: multiple nontrivial SCCs, moderate density. {facts}"
    return "ff", f"FF (default): {facts}"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from engine.parser import parse_file

    test_cases = [
        ("blocks", "domains/blocks/domain.catplan", "domains/blocks/tier2_hanoi3.catplan"),
        ("logistics", "domains/logistics/domain.catplan", "domains/logistics/tier2_multihop.catplan"),
        ("arithmetic", "domains/arithmetic/domain.catplan", "domains/arithmetic/tier1_simple.catplan"),
    ]

    for name, domain_file, prob_file in test_cases:
        domains, _ = parse_file(domain_file)
        _, problems = parse_file(prob_file)
        domain = domains[0]
        problem = problems[0]

        heuristic, explanation = select_heuristic_with_explanation(domain, problem)
        print(f"{name:12s}: {heuristic:12s} — {explanation}")
