"""
Abductive inference: given an observed output, find the most likely explanation.

Two complementary search strategies are combined:

1. **FOLD_RIGHT carry inversion** (exact, algebraic):
   For a FOLD_RIGHT process rule with initial_carry k, the inverse rule has
   initial_carry -k.  Applying the inverse to the observed output digits yields
   the most likely input.  The succ rule (k=+1) and pred rule (k=-1) are mutual
   inverses; this relationship is discovered from the `process_rules` list — no
   operator names are hardcoded.

   Algorithm (for each process rule r):
     1. Find the inverse rule: process_rules entry with initial_carry = -r.initial_carry.
     2. Apply the inverse rule to observed_output_digits → candidate input digits.
     3. Verify: forward-apply r to candidate input → must reproduce observed output.
     4. Compute cost = -confidence of any morphism in mg associated with r.op_atom.

2. **MorphismGraph backward BFS / A* search** (type-level, general):
   For rules without an algebraic inverse (non-FOLD_RIGHT or no inverse-carry
   rule in the corpus), search backward through the MorphismGraph:
     1. Score each object O by how well its intent_weights match the observed
        output tokens (cosine-like overlap).
     2. Priority queue over (cost, object_id); expand in_morphisms.
     3. Each morphism A → O contributes an explanation with:
           op_atom   = morphism.morph_type
           cost      = -morphism.confidence + (1 - match_score)
        input_digits cannot be recovered at the type level, so the result
        records `input_digits=[]` and marks the explanation as graph-derived.
     4. Stop when max_results candidates are found or the queue is exhausted.

Both strategies sort by cost (ascending = most likely).  FOLD_RIGHT results
are preferred (lower base cost) because they provide exact digit sequences.

Reference: Jacobs, Kissinger & Zanasi (2019) — "Causal inference by string
diagram surgery." MSCS.

See CTKG_ARCHITECTURE.md §Abduce for the full specification.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    ProcessRule,
    apply_process_rule,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AbductionResult:
    """One explanation candidate from the abduction search.

    Attributes
    ----------
    op_atom:
        The operator atom responsible for the transformation (e.g. 'succ').
    input_digits:
        Inferred input digit sequence (MSB first).
        Empty list for graph-derived results (type level only — exact digits
        cannot be recovered from the MorphismGraph).
    explanation_prefix:
        Complete prefix that would produce the observed output:
        [op_atom] + input_digits + ['eq'].
        Empty list for graph-derived results.
    cost:
        Lower = more likely.  Equals -confidence of the corresponding morphism
        in the MorphismGraph (0.0 if no matching morphism is found).
    graph_derived:
        True if this result was produced by backward MorphismGraph BFS rather
        than algebraic FOLD_RIGHT carry inversion.  Graph-derived results
        provide type-level explanations only (input_digits is empty).
    """

    op_atom: str
    input_digits: list[str] = field(default_factory=list)
    explanation_prefix: list[str] = field(default_factory=list)
    cost: float = 0.0
    graph_derived: bool = False

    def __repr__(self) -> str:
        tag = "[graph]" if self.graph_derived else "[exact]"
        return (
            f"AbductionResult{tag}(op={self.op_atom!r}, "
            f"input={''.join(self.input_digits)!r}, "
            f"cost={self.cost:.3f})"
        )


# ---------------------------------------------------------------------------
# Internal: MorphismGraph backward BFS / A* (type-level)
# ---------------------------------------------------------------------------

def _intent_match(
    obj_concept,
    observed_tokens: list[str],
) -> float:
    """Cosine-like overlap between an object's intent_weights and observed tokens.

    Parameters
    ----------
    obj_concept:
        A DistributionalConcept with an `intent_weights` dict.
    observed_tokens:
        The observed output token sequence.

    Returns
    -------
    Score ∈ [0, 1].  1.0 = all observed tokens are fully covered by the
    concept's intent distribution.
    """
    weights = getattr(obj_concept, "intent_weights", {})
    if not weights or not observed_tokens:
        return 0.0
    total_w = sum(weights.values())
    if total_w <= 0.0:
        return 0.0
    score = sum(weights.get(tok, 0.0) for tok in observed_tokens)
    # Normalise: divide by both the intent mass and the number of tokens so
    # that the score is bounded by the per-token average intent weight.
    return score / (total_w * len(observed_tokens))


def _graph_abduce(
    mg: MorphismGraph,
    observed_tokens: list[str],
    max_results: int = 5,
    max_hops: int = 4,
) -> list[AbductionResult]:
    """Backward BFS / A* over the MorphismGraph.

    Finds morphisms A → O where O's intent distribution best covers the
    observed output tokens.  Cost = -morphism.confidence + (1 - match_score).

    Parameters
    ----------
    mg:
        The MorphismGraph.
    observed_tokens:
        The observed output token sequence.
    max_results:
        Maximum number of AbductionResult objects to return.
    max_hops:
        Maximum number of morphism hops to follow backward.

    Returns
    -------
    List of AbductionResult objects (graph_derived=True) sorted by cost.
    """
    if not observed_tokens:
        return []

    # Phase 1: score all objects by intent match
    obj_match: dict[int, float] = {}
    for obj in mg.objects(active_only=False):
        score = _intent_match(obj.concept, observed_tokens)
        if score > 0.0:
            obj_match[obj.obj_id] = score

    if not obj_match:
        return []

    # Phase 2: backward BFS priority queue
    # heap entry: (cost, obj_id, hops_remaining, morph_type, morph_confidence)
    heap: list[tuple[float, int, int, str, float]] = []
    for obj_id, match_score in obj_match.items():
        seed_cost = 1.0 - match_score  # cost 0.0 = perfect match
        heapq.heappush(heap, (seed_cost, obj_id, max_hops, "", 0.0))

    visited: set[tuple[int, int]] = set()  # (obj_id, hops_remaining)
    results: list[AbductionResult] = []

    while heap and len(results) < max_results:
        cost, obj_id, hops_left, via_morph_type, via_conf = heapq.heappop(heap)

        state = (obj_id, hops_left)
        if state in visited:
            continue
        visited.add(state)

        # Each in_morphism A → obj_id is one candidate explanation
        for m in mg.in_morphisms(obj_id, include_identity=False):
            morph_type = m.morph_type or f"morph_{m.morph_id}"
            morph_cost = cost - m.confidence  # lower confidence = higher cost
            results.append(
                AbductionResult(
                    op_atom=morph_type,
                    input_digits=[],
                    explanation_prefix=[],
                    cost=morph_cost,
                    graph_derived=True,
                )
            )
            # Optionally continue backward if hops remain
            if hops_left > 1:
                heapq.heappush(heap, (morph_cost, m.source, hops_left - 1,
                                      morph_type, m.confidence))

    results.sort(key=lambda r: r.cost)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Abduction
# ---------------------------------------------------------------------------

def abduce(
    process_rules: list[ProcessRule],
    mg: MorphismGraph,
    observed_output_digits: list[str],
    max_results: int = 5,
    graph_fallback: bool = True,
) -> list[AbductionResult]:
    """Abductively infer the most likely (operator, input) for an observed output.

    Combines two complementary strategies:

    1. **FOLD_RIGHT carry inversion** (exact, algebraic) — for rules with a
       known inverse carry value in `process_rules`.
    2. **MorphismGraph backward BFS** (type-level, general) — for rules with
       no algebraic inverse.  Activated when `graph_fallback=True` and Strategy 1
       produces fewer than `max_results` results.

    Parameters
    ----------
    process_rules:
        List of ProcessRule objects (from discover_processes).
    mg:
        The MorphismGraph (used to look up morphism confidence for costing
        and for graph-search abduction).
    observed_output_digits:
        The observed output token sequence (digits only, MSB first).
    max_results:
        Maximum number of AbductionResult objects to return.
    graph_fallback:
        When True (default), supplement carry-inversion results with
        graph-search results when the algebraic path finds fewer than
        `max_results` candidates.

    Returns
    -------
    List of AbductionResult objects sorted by cost (ascending = most likely).
    Exact carry-inversion results (graph_derived=False) sort before graph-derived
    results (graph_derived=True) at equal cost.  At most `max_results` results.
    Returns [] if no valid explanation is found.
    """
    if not observed_output_digits:
        return []

    # Build a lookup: initial_carry → ProcessRule (for FOLD_RIGHT inversion)
    rule_by_carry: dict[int, ProcessRule] = {}
    if process_rules:
        rule_by_carry = {r.initial_carry: r for r in process_rules}

    # Build morphism confidence lookup: op_atom → max confidence in mg
    _morph_conf: dict[str, float] = {}
    for m in mg.morphisms(include_identity=False):
        for rule in (process_rules or []):
            if rule.op_atom in m.morph_type or m.morph_type == rule.op_atom:
                cur = _morph_conf.get(rule.op_atom, None)
                if cur is None or m.confidence > cur:
                    _morph_conf[rule.op_atom] = m.confidence

    results: list[AbductionResult] = []
    rules_with_inverse: set[str] = set()

    # Strategy 1: FOLD_RIGHT carry inversion (exact, algebraic)
    for rule in (process_rules or []):
        inv_carry = -rule.initial_carry
        inv_rule = rule_by_carry.get(inv_carry)
        if inv_rule is None:
            continue  # no algebraic inverse — fall through to graph search

        rules_with_inverse.add(rule.op_atom)

        # Apply inverse rule to observed output to get candidate input
        candidate_input = apply_process_rule(inv_rule, observed_output_digits)
        if candidate_input is None:
            continue

        # Verify: forward-apply rule to candidate input → must equal observed
        verification = apply_process_rule(rule, candidate_input)
        if verification != observed_output_digits:
            continue

        # Cost = -morphism confidence (0.0 if no morphism found)
        conf = _morph_conf.get(rule.op_atom, 0.0)
        cost = -conf

        explanation_prefix = [rule.op_atom] + candidate_input + ["eq"]
        results.append(
            AbductionResult(
                op_atom=rule.op_atom,
                input_digits=candidate_input,
                explanation_prefix=explanation_prefix,
                cost=cost,
                graph_derived=False,
            )
        )

    # Strategy 2: MorphismGraph backward BFS / A* (type-level, general)
    # Only activated when there are process rules to invert — if no rules are
    # known at all, there is no basis for abduction.
    if graph_fallback and process_rules and len(results) < max_results:
        remaining = max_results - len(results)
        graph_results = _graph_abduce(mg, observed_output_digits, max_results=remaining)
        results.extend(graph_results)

    # Sort: exact (graph_derived=False) preferred over graph-derived at equal cost
    results.sort(key=lambda r: (r.cost, int(r.graph_derived)))
    return results[:max_results]
