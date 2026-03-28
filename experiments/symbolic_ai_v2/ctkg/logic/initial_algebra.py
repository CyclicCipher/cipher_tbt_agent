"""
Initial algebra discovery — universal property test.

An F-algebra for endofunctor F(X) = 1 + X is a set A with:
  - a distinguished element z: 1 → A  (zero)
  - an endomorphism s: A → A          (successor)

The initial F-algebra (the NNO) is the one that maps uniquely into
every other F-algebra. Concretely: for any (A, a, f), there exists
a UNIQUE h: N → A such that h(z) = a and h(s(n)) = f(h(n)).

Discovery algorithm:
1. Find all candidate algebras: (start_node, successor_map) pairs
   where successor_map is a partial function on a set of nodes.
2. For each candidate, test the universal property against all other
   candidates: can we build a unique h that commutes?
3. The candidate satisfying the universal property for the most
   targets is the best NNO.

Sources of candidates:
- Chains from algebra.py (Phase 2)
- Cycles within SCCs from algebra.py
- Co-occurrence chains: sequences A→B→C where A→B and B→C are the
  strongest forward co-occurrence edges

No domain-specific code. Operates on abstract graph structure only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, NodeId, COOCCURRENCE, TRANSITION,
)
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Algebra:
    """An F-algebra: a set with a zero element and a successor map."""
    zero: NodeId
    succ: dict[NodeId, NodeId]   # partial function: node → successor

    @property
    def carrier(self) -> frozenset[NodeId]:
        """All nodes in the algebra (domain ∪ codomain ∪ {zero})."""
        nodes = {self.zero}
        nodes.update(self.succ.keys())
        nodes.update(self.succ.values())
        return frozenset(nodes)

    @property
    def chain_length(self) -> int:
        """Length of the chain starting from zero."""
        n = 0
        current = self.zero
        visited = set()
        while current in self.succ and current not in visited:
            visited.add(current)
            current = self.succ[current]
            n += 1
        return n

    def walk(self, steps: int) -> NodeId | None:
        """Walk `steps` successor applications from zero. Returns None if chain too short."""
        current = self.zero
        for _ in range(steps):
            nxt = self.succ.get(current)
            if nxt is None:
                return None
            current = nxt
        return current


@dataclass
class UniversalPropertyResult:
    """Result of testing the universal property for one candidate."""
    algebra: Algebra
    targets_tested: int
    targets_satisfied: int     # how many (a, f) pairs admit a unique h
    score: float               # targets_satisfied / targets_tested


# ---------------------------------------------------------------------------
# Candidate algebra extraction
# ---------------------------------------------------------------------------

def _compute_pmi(
    hippo: Hippocampus,
    kg: KnowledgeGraph,
    since_index: int = 0,
    max_observations: int = 2000,
) -> dict[tuple[NodeId, NodeId], float]:
    """Compute pointwise mutual information for co-occurring token pairs.

    PMI(A,B) = log(P(A,B) / (P(A) * P(B)))

    High PMI = specifically associated. Low/negative = co-occur by chance.
    This naturally distinguishes successor pairs (3→4: high PMI) from
    hub co-occurrences (3→space: low PMI because space appears everywhere).

    Returns {(src, tgt): PMI} for all directed pairs (causal order).
    """
    import math

    all_obs = hippo.all_observations()
    obs_slice = all_obs[max(0, since_index):]
    if len(obs_slice) > max_observations:
        obs_slice = obs_slice[-max_observations:]
    if not obs_slice:
        return {}

    n_obs = len(obs_slice)

    # Count: how many observations each token appears in.
    token_count: dict[NodeId, int] = {}
    # Count: how many observations each directed pair appears in.
    pair_count: dict[tuple[NodeId, NodeId], int] = {}

    for obs in obs_slice:
        nids = list(dict.fromkeys(obs.token_nids))  # deduplicate preserving order
        nid_set = set(nids)
        for nid in nid_set:
            token_count[nid] = token_count.get(nid, 0) + 1
        # Directed pairs (causal order: i before j).
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                key = (nids[i], nids[j])
                pair_count[key] = pair_count.get(key, 0) + 1

    # Compute PMI.
    pmi: dict[tuple[NodeId, NodeId], float] = {}
    for (a, b), count_ab in pair_count.items():
        count_a = token_count.get(a, 0)
        count_b = token_count.get(b, 0)
        if count_a == 0 or count_b == 0:
            continue
        p_ab = count_ab / n_obs
        p_a = count_a / n_obs
        p_b = count_b / n_obs
        if p_ab > 0 and p_a * p_b > 0:
            pmi[(a, b)] = math.log(p_ab / (p_a * p_b))

    return pmi


def _find_cooccurrence_chains(
    kg: KnowledgeGraph,
    min_chain_length: int = 3,
    max_chains: int = 20,
    hippo: Hippocampus | None = None,
) -> list[Algebra]:
    """Find chains in the co-occurrence graph via best-PMI-neighbor.

    Uses pointwise mutual information (not raw edge weight) to find
    specifically associated token pairs. PMI naturally filters hubs
    (tokens that co-occur with everything have low PMI with each).

    Falls back to raw edge weight if hippo is not provided.
    """
    # Compute PMI-based best neighbors if hippo available.
    use_pmi = hippo is not None and hippo.observation_count() > 0

    if use_pmi:
        pmi = _compute_pmi(hippo, kg)

        # Build a PMI-weighted directed graph. For chain discovery, we want
        # the longest path where every step has positive PMI. We use a
        # greedy longest-path search instead of mutual-best-neighbor, because
        # mutual-best fails when structural tokens (FEEDBACK, SCORE) have
        # higher PMI with content tokens than the successor pairs do.
        #
        # Algorithm: for each node, try extending a chain forward by picking
        # the highest-PMI neighbor that also has co-occurrence edge. Track
        # visited to avoid cycles. Keep the longest chain found.
        pmi_fwd: dict[NodeId, list[tuple[NodeId, float]]] = {}
        for (a, b), score in pmi.items():
            if score <= 0:
                continue
            # Only consider edges that also exist in the co-occurrence graph.
            if kg.edge(a, b) is None:
                continue
            pmi_fwd.setdefault(a, []).append((b, score))
        # Sort each node's neighbors by PMI descending.
        for nid in pmi_fwd:
            pmi_fwd[nid].sort(key=lambda x: -x[1])

        # Find longest chains via greedy forward extension from each node.
        best_chains: list[list[NodeId]] = []
        global_visited: set[NodeId] = set()

        for start in sorted(pmi_fwd.keys()):
            if start in global_visited:
                continue
            chain = [start]
            visited = {start}
            current = start
            while current in pmi_fwd:
                # Pick highest-PMI unvisited neighbor.
                found = False
                for nxt, score in pmi_fwd[current]:
                    if nxt not in visited:
                        chain.append(nxt)
                        visited.add(nxt)
                        current = nxt
                        found = True
                        break
                if not found:
                    break
            if len(chain) >= min_chain_length:
                best_chains.append(chain)
                global_visited.update(chain)
            if len(best_chains) >= max_chains:
                break

        algebras: list[Algebra] = []
        for chain in best_chains:
            succ = {chain[i]: chain[i + 1] for i in range(len(chain) - 1)}
            algebras.append(Algebra(zero=chain[0], succ=succ))
        return algebras
    else:
        # Fallback: raw edge weight.
        best_fwd = {}
        for nid in kg._nodes:
            best_target = None
            best_weight = 0.0
            for edge in kg._outgoing.get(nid, ()):
                if edge.role != COOCCURRENCE:
                    continue
                if edge.weight > best_weight:
                    best_weight = edge.weight
                    best_target = edge.target
            if best_target is not None and best_weight > 0.05:
                best_fwd[nid] = (best_target, best_weight)

        best_bwd = {}
        for nid in kg._nodes:
            best_source = None
            best_weight = 0.0
            for edge in kg._incoming.get(nid, ()):
                if edge.role != COOCCURRENCE:
                    continue
                if edge.weight > best_weight:
                    best_weight = edge.weight
                    best_source = edge.source
            if best_source is not None and best_weight > 0.05:
                best_bwd[nid] = (best_source, best_weight)

    # Find chains of mutual best neighbors.
    used: set[NodeId] = set()
    algebras: list[Algebra] = []

    # Find chain starts: nodes with no mutual-best predecessor.
    starts: list[NodeId] = []
    for nid in kg._nodes:
        if nid in used:
            continue
        # Walk backward to find the chain start.
        current = nid
        visited_back = set()
        while current in best_bwd:
            prev, _ = best_bwd[current]
            # Check mutual: prev's best forward is current.
            if prev in best_fwd and best_fwd[prev][0] == current:
                if prev in visited_back:
                    break  # cycle
                visited_back.add(current)
                current = prev
            else:
                break
        if current not in used:
            starts.append(current)

    for start in starts:
        if start in used:
            continue
        # Walk forward via mutual best neighbors.
        chain = [start]
        visited = {start}
        current = start
        while current in best_fwd:
            nxt, _ = best_fwd[current]
            # Check mutual: nxt's best backward is current.
            if nxt in best_bwd and best_bwd[nxt][0] == current:
                if nxt in visited:
                    break  # cycle hit
                chain.append(nxt)
                visited.add(nxt)
                current = nxt
            else:
                break

        if len(chain) >= min_chain_length:
            succ = {chain[i]: chain[i + 1] for i in range(len(chain) - 1)}
            algebras.append(Algebra(zero=chain[0], succ=succ))
            used.update(chain)

        if len(algebras) >= max_chains:
            break

    return algebras


def _find_transition_chains(
    kg: KnowledgeGraph,
    min_chain_length: int = 3,
    max_chains: int = 20,
) -> list[Algebra]:
    """Find chains in the transition graph via best-forward-neighbor.

    Same algorithm as co-occurrence chains but restricted to TRANSITION edges.
    """
    best_fwd: dict[NodeId, tuple[NodeId, float]] = {}
    for nid in kg._nodes:
        best_target = None
        best_weight = 0.0
        for edge in kg._outgoing.get(nid, ()):
            if edge.role != TRANSITION:
                continue
            if edge.weight > best_weight:
                best_weight = edge.weight
                best_target = edge.target
        if best_target is not None and best_weight > 0.05:
            best_fwd[nid] = (best_target, best_weight)

    best_bwd: dict[NodeId, tuple[NodeId, float]] = {}
    for nid in kg._nodes:
        best_source = None
        best_weight = 0.0
        for edge in kg._incoming.get(nid, ()):
            if edge.role != TRANSITION:
                continue
            if edge.weight > best_weight:
                best_weight = edge.weight
                best_source = edge.source
        if best_source is not None and best_weight > 0.05:
            best_bwd[nid] = (best_source, best_weight)

    used: set[NodeId] = set()
    algebras: list[Algebra] = []

    for nid in sorted(kg._nodes.keys()):
        if nid in used:
            continue
        current = nid
        visited_back = set()
        while current in best_bwd:
            prev, _ = best_bwd[current]
            if prev in best_fwd and best_fwd[prev][0] == current:
                if prev in visited_back:
                    break
                visited_back.add(current)
                current = prev
            else:
                break

        chain = [current]
        visited = {current}
        cur = current
        while cur in best_fwd:
            nxt, _ = best_fwd[cur]
            if nxt in best_bwd and best_bwd[nxt][0] == cur:
                if nxt in visited:
                    break
                chain.append(nxt)
                visited.add(nxt)
                cur = nxt
            else:
                break

        if len(chain) >= min_chain_length:
            succ = {chain[i]: chain[i + 1] for i in range(len(chain) - 1)}
            algebras.append(Algebra(zero=chain[0], succ=succ))
            used.update(chain)

        if len(algebras) >= max_chains:
            break

    return algebras


# ---------------------------------------------------------------------------
# Universal property test
# ---------------------------------------------------------------------------

def _test_universal_property(
    candidate: Algebra,
    target: Algebra,
) -> int:
    """Test how far a unique h: candidate → target can be defined.

    h must satisfy:
      h(candidate.zero) = target.zero
      h(candidate.succ(n)) = target.succ(h(n))

    Returns the number of candidate nodes that h successfully maps.
    0 means even zero can't be mapped (contradicts target structure).
    Equal to candidate.chain_length + 1 means full mapping.

    For finite targets without cycles, h is defined up to the target's
    chain length. The candidate that achieves the longest h for the
    most targets is the best NNO.
    """
    h: dict[NodeId, NodeId] = {}
    h[candidate.zero] = target.zero
    mapped = 1  # zero is mapped

    current = candidate.zero
    target_current = target.zero

    visited = set()
    while current in candidate.succ:
        if current in visited:
            break
        visited.add(current)

        next_c = candidate.succ[current]
        next_t = target.succ.get(target_current)

        if next_t is None:
            break  # target exhausted — h defined up to here

        if next_c in h:
            if h[next_c] != next_t:
                break  # contradiction
        else:
            h[next_c] = next_t

        mapped += 1
        current = next_c
        target_current = next_t

    return mapped


def _score_candidate(
    candidate: Algebra,
    all_algebras: list[Algebra],
) -> UniversalPropertyResult:
    """Score a candidate by total mapping coverage across all targets.

    For each target, compute how many nodes of the candidate can be
    mapped via h. The score is the total mapped nodes divided by the
    total possible (candidate length × number of targets). This favors
    long candidates that map deeply into many targets.
    """
    tested = 0
    total_mapped = 0
    candidate_len = candidate.chain_length + 1  # +1 for zero

    for target in all_algebras:
        if target.zero == candidate.zero and target.succ == candidate.succ:
            continue
        tested += 1
        mapped = _test_universal_property(candidate, target)
        total_mapped += mapped

    max_possible = candidate_len * max(tested, 1)
    score = total_mapped / max_possible if max_possible > 0 else 0.0

    return UniversalPropertyResult(
        algebra=candidate,
        targets_tested=tested,
        targets_satisfied=total_mapped,
        score=score,
    )


# ---------------------------------------------------------------------------
# Main: discover initial algebras
# ---------------------------------------------------------------------------

def discover_initial_algebras(
    kg: KnowledgeGraph,
    hippo: Hippocampus,
    since_index: int = 0,
    min_chain_length: int = 3,
    min_score: float = 0.5,
) -> dict[str, Any]:
    """Discover initial algebras (NNO candidates) via the universal property.

    1. Collect candidate algebras from co-occurrence and transition chains.
    2. Test each candidate against all others.
    3. The candidate satisfying the universal property for the most
       targets is the best NNO.
    4. Materialize the winner as a distinguished chain in the KG.

    Called by Consolidation.consolidate().
    """
    # Collect all candidate algebras.
    cooccur_algebras = _find_cooccurrence_chains(
        kg, min_chain_length=min_chain_length, hippo=hippo,
    )
    transition_algebras = _find_transition_chains(
        kg, min_chain_length=min_chain_length,
    )

    all_algebras = cooccur_algebras + transition_algebras
    if len(all_algebras) < 2:
        return {
            "candidates_cooccur": len(cooccur_algebras),
            "candidates_transition": len(transition_algebras),
            "initial_algebras_found": 0,
            "best_score": 0.0,
            "best_chain_length": 0,
        }

    # Score each candidate.
    results: list[UniversalPropertyResult] = []
    for candidate in all_algebras:
        result = _score_candidate(candidate, all_algebras)
        results.append(result)

    # Find the best candidate(s).
    results.sort(key=lambda r: (-r.score, -r.algebra.chain_length))
    best = results[0] if results else None

    initial_found = 0
    best_score = 0.0
    best_length = 0

    if best is not None and best.score >= min_score:
        initial_found = 1
        best_score = best.score
        best_length = best.algebra.chain_length

        # Materialize the NNO chain.
        chain_nodes = [best.algebra.zero]
        current = best.algebra.zero
        visited = set()
        while current in best.algebra.succ and current not in visited:
            visited.add(current)
            current = best.algebra.succ[current]
            chain_nodes.append(current)

        nno_key = ("__nno__", tuple(chain_nodes))
        if nno_key not in kg._value_to_node:
            nno_nid = kg.get_or_create(nno_key)
            node = kg.node(nno_nid)
            if node is not None:
                node.resting = 0.6

            # Cocone: chain members → NNO node.
            for member_nid in chain_nodes:
                if member_nid in kg._nodes:
                    edge = kg.get_or_create_edge(
                        member_nid, nno_nid, role=COOCCURRENCE,
                    )
                    edge.strengthen(0.4)

            # Strengthen the successor edges along the chain.
            for i in range(len(chain_nodes) - 1):
                src = chain_nodes[i]
                tgt = chain_nodes[i + 1]
                if src in kg._nodes and tgt in kg._nodes:
                    edge = kg.get_or_create_edge(src, tgt, role=COOCCURRENCE)
                    edge.strengthen(0.3)

    # Count how many candidates passed the threshold.
    passing = sum(1 for r in results if r.score >= min_score)

    # Populate _discovered_succ from ALL chains.
    # When multiple chains disagree on a node's successor, keep the pair
    # with the highest PMI (most specifically associated). This resolves
    # conflicts like 0→next_is vs 0→1 in favor of 0→1 (higher content PMI).
    succ_candidates: dict[NodeId, list[tuple[NodeId, float]]] = {}
    for alg in all_algebras:
        for src, tgt in alg.succ.items():
            if src not in kg._nodes or tgt not in kg._nodes:
                continue
            pmi_score = kg._pmi.get((src, tgt), 0.0)
            succ_candidates.setdefault(src, []).append((tgt, pmi_score))

    succ_pairs_added = 0
    kg._discovered_succ.clear()
    for src, tgt_list in succ_candidates.items():
        # Pick the target with the highest PMI to the source.
        best_tgt = max(tgt_list, key=lambda x: x[1])[0]
        kg._discovered_succ[src] = best_tgt
        succ_pairs_added += 1

    return {
        "candidates_cooccur": len(cooccur_algebras),
        "candidates_transition": len(transition_algebras),
        "initial_algebras_found": passing,
        "best_score": best_score,
        "best_chain_length": best_length,
        "succ_pairs_added": succ_pairs_added,
    }
