"""
Phase 2: probabilistic FCA, hierarchical multi-scale, productivity regularisation.

Distributional FCA on the Hankel co-occurrence matrix H produced by Phase 1.

Algorithm (agglomerative):
1. **Pre-filter:** keep only the top-K contexts by raw support (default K=80).
   Rare contexts are statistical noise and dramatically inflate k — FCA is O(k³)
   in the worst case, so k must be kept small.
2. Start with one concept per (retained) context row.  Each concept's distribution
   is the row-normalised probability vector P(atom | context).
3. **Heap agglomeration (O(k² log k)):** compute all pairwise JSDs once, push into a
   min-heap.  Pop the best pair; if both clusters still active, merge; push new pairs
   for the merged cluster.  Repeat until no JSD below merge_threshold.
4. Apply MDL productivity pruning: drop concepts with net-negative MDL benefit.
5. Run independently for each radius level r in r_levels.
6. Return one ConceptLattice per radius level, in the same order as r_levels.

Complexity:
    k  = number of retained contexts (capped at max_contexts)
    k² pairs computed once, k merges × k new pairs → O(k²) heap operations,
    O(k² log k) total.  With max_contexts=80 this is ~6400 log(80) ≈ 28K ops —
    fast enough for interactive use.

Productivity score:
    productivity(C) = support(C) * H(C)  /  (H(C) + 1)
    where H(C) = Shannon entropy of C's centroid distribution.
    High entropy = uniform = low productivity.
    High support * moderate entropy = productive generalisation.

MDL penalty:
    A concept is kept iff  support(C) * log2(support(C)) > lambda_productivity
    (i.e. the description length benefit outweighs the cost of a new concept symbol).

Jensen-Shannon divergence:
    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)  where m = 0.5*(p+q).
    JSD ∈ [0, log(2)] ≈ [0, 0.693].

See CTKG_ARCHITECTURE.md §Phase 2 for the full specification.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.special import rel_entr  # KL term without log-of-zero

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount, AtomValue, ContextKey
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import (
    ConceptLattice,
    DistributionalConcept,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors.

    Both p and q must be non-negative and should sum to ≈1.
    Returns a value in [0, log(2)] ≈ [0, 0.693].
    """
    m = 0.5 * (p + q)
    kl_pm = np.sum(rel_entr(p, m))
    kl_qm = np.sum(rel_entr(q, m))
    return float(0.5 * (kl_pm + kl_qm))


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector (base 2, in bits)."""
    p_pos = p[p > 0]
    return float(-np.sum(p_pos * np.log2(p_pos)))


def _productivity_score(support: float, centroid: np.ndarray) -> float:
    """Productivity = support * H / (H + 1)."""
    h = _entropy(centroid)
    return support * h / (h + 1.0)


def _mdl_benefit(support: float) -> float:
    """MDL benefit ≈ support * log2(support) bits saved by concept compression."""
    if support <= 0:
        return 0.0
    return support * math.log2(support)


# ---------------------------------------------------------------------------
# _Cluster: internal mutable representation during agglomeration
# ---------------------------------------------------------------------------

class _Cluster:
    """Working representation of a concept during agglomeration."""

    __slots__ = ('cid', 'centroid', 'contexts', 'context_weights', 'support')

    def __init__(
        self,
        cid: int,
        centroid: np.ndarray,
        contexts: list[ContextKey],
        context_weights: dict[ContextKey, float],
        support: float,
    ) -> None:
        self.cid = cid
        self.centroid = centroid
        self.contexts = contexts
        self.context_weights = context_weights
        self.support = support

    @classmethod
    def from_row(
        cls,
        cid: int,
        context: ContextKey,
        distribution: np.ndarray,
        support: float,
    ) -> "_Cluster":
        return cls(
            cid=cid,
            centroid=distribution.copy(),
            contexts=[context],
            context_weights={context: support},
            support=support,
        )

    def merge_with(self, other: "_Cluster", new_cid: int) -> "_Cluster":
        """Support-weighted merge of two clusters."""
        total = self.support + other.support
        if total == 0:
            new_centroid = 0.5 * (self.centroid + other.centroid)
        else:
            new_centroid = (
                self.support * self.centroid + other.support * other.centroid
            ) / total
        s = new_centroid.sum()
        if s > 0:
            new_centroid = new_centroid / s
        return _Cluster(
            cid=new_cid,
            centroid=new_centroid,
            contexts=self.contexts + other.contexts,
            context_weights={**self.context_weights, **other.context_weights},
            support=total,
        )


# ---------------------------------------------------------------------------
# Heap-based agglomeration — O(k² log k)
# ---------------------------------------------------------------------------

def _agglomerate(
    clusters: list[_Cluster],
    merge_threshold: float,
) -> list[_Cluster]:
    """Greedily merge the closest pair (by JSD) until no pair is below threshold.

    Uses a min-heap so pairs are processed in order of increasing JSD.
    Stale heap entries (one member already merged) are discarded on pop.

    Complexity: O(k² log k) — k² initial pairs, at most k merges × k new pairs.
    """
    if len(clusters) <= 1:
        return clusters

    next_cid = max(c.cid for c in clusters) + 1
    # Track active cluster IDs for quick staleness check
    active: dict[int, _Cluster] = {c.cid: c for c in clusters}

    # Initialise heap with all pairwise JSDs
    heap: list[tuple[float, int, int]] = []
    cids = list(active.keys())
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ci, cj = active[cids[i]], active[cids[j]]
            d = _jsd(ci.centroid, cj.centroid)
            if d < merge_threshold:
                heapq.heappush(heap, (d, ci.cid, cj.cid))

    while heap:
        d, cid_a, cid_b = heapq.heappop(heap)
        # Skip stale entries (one or both clusters already merged away)
        if cid_a not in active or cid_b not in active:
            continue
        if d >= merge_threshold:
            break

        # Merge
        ca = active.pop(cid_a)
        cb = active.pop(cid_b)
        merged = ca.merge_with(cb, next_cid)
        next_cid += 1
        active[merged.cid] = merged

        # Push new pairs for the merged cluster
        for other_cid, other in active.items():
            if other_cid == merged.cid:
                continue
            new_d = _jsd(merged.centroid, other.centroid)
            if new_d < merge_threshold:
                heapq.heappush(heap, (new_d, merged.cid, other_cid))

    return list(active.values())


# ---------------------------------------------------------------------------
# MDL pruning
# ---------------------------------------------------------------------------

def _apply_mdl_pruning(
    clusters: list[_Cluster],
    lambda_productivity: float,
) -> list[_Cluster]:
    """Drop clusters whose MDL benefit is below the productivity penalty."""
    kept = []
    for c in clusters:
        benefit = _mdl_benefit(c.support)
        prod = _productivity_score(c.support, c.centroid)
        penalty = lambda_productivity * (1.0 - prod)
        if benefit >= penalty:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Build ConceptLattice from _Cluster list
# ---------------------------------------------------------------------------

def _build_lattice(
    clusters: list[_Cluster],
    atoms: list[AtomValue],
    radius: int,
    subtype_threshold: float,
) -> ConceptLattice:
    """Convert _Cluster objects to a ConceptLattice, sorted by descending support."""
    concepts = []
    for c in sorted(clusters, key=lambda x: -x.support):
        intent = {atoms[j]: float(c.centroid[j]) for j in range(len(atoms))}
        concept = DistributionalConcept(
            concept_id=c.cid,
            centroid_vector=c.centroid.copy(),
            extent_weights=dict(c.context_weights),
            intent_weights=intent,
            support=c.support,
            member_contexts=list(c.contexts),
        )
        concepts.append(concept)
    return ConceptLattice(
        radius=radius,
        concepts=concepts,
        atoms=atoms,
        subtype_threshold=subtype_threshold,
    )


# ---------------------------------------------------------------------------
# Single-level FCA
# ---------------------------------------------------------------------------

def _fca_one_level(
    hankel: HankelCount,
    r: int,
    lambda_productivity: float,
    merge_threshold: float,
    subtype_threshold: float,
    max_contexts: int,
    min_support: float,
) -> ConceptLattice:
    """Run distributional FCA on H for a single radius level.

    Parameters
    ----------
    max_contexts:
        Hard cap on k (number of context rows fed to FCA).
        Contexts are ranked by raw support and only the top-max_contexts are kept.
        This is the primary complexity control: FCA is O(k² log k).
    min_support:
        Discard contexts with raw count < min_support before ranking.
        Removes statistical noise from rare contexts.
    """
    contexts_all, atoms, H = hankel.matrix(r=r)
    n_ctx, n_atoms = H.shape

    if n_ctx == 0:
        return ConceptLattice(radius=r, concepts=[], atoms=atoms,
                              subtype_threshold=subtype_threshold)

    # Raw support per context
    raw_support: list[tuple[float, int]] = []  # (support, row_index)
    for i, ctx in enumerate(contexts_all):
        raw = hankel._counts.get(ctx, {})
        s = float(sum(raw.values()))
        if s >= min_support:
            raw_support.append((s, i))

    if not raw_support:
        # All contexts below min_support — keep singletons from the top-5
        raw_support = [(1.0, i) for i in range(min(5, n_ctx))]

    # Rank by support, cap at max_contexts
    raw_support.sort(key=lambda x: -x[0])
    raw_support = raw_support[:max_contexts]

    # Build clusters for retained contexts
    clusters: list[_Cluster] = []
    for cid, (support, row_idx) in enumerate(raw_support):
        ctx = contexts_all[row_idx]
        row = H[row_idx].copy()
        clusters.append(_Cluster.from_row(cid=cid, context=ctx,
                                           distribution=row, support=support))

    # Heap-based agglomeration
    clusters = _agglomerate(clusters, merge_threshold=merge_threshold)

    # MDL pruning
    clusters = _apply_mdl_pruning(clusters, lambda_productivity=lambda_productivity)

    # If everything pruned, return singletons from original retained set
    if not clusters:
        clusters = []
        for cid, (support, row_idx) in enumerate(raw_support):
            ctx = contexts_all[row_idx]
            row = H[row_idx].copy()
            clusters.append(_Cluster.from_row(cid=cid, context=ctx,
                                               distribution=row, support=support))

    return _build_lattice(clusters, atoms, r, subtype_threshold)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_concepts(
    hankel: HankelCount,
    r_levels: list[int],
    lambda_productivity: float = 0.1,
    merge_threshold: float = 0.15,
    subtype_threshold: float = 0.05,
    max_contexts: int = 80,
    min_support: float = 3.0,
) -> list[ConceptLattice]:
    """Run distributional FCA on the Hankel matrix and return one ConceptLattice
    per radius level.

    Parameters
    ----------
    hankel:
        Trained HankelCount object.
    r_levels:
        Radius levels to run FCA at (e.g. [1, 2, 3]).
    lambda_productivity:
        MDL regularisation strength.  Higher = more aggressive pruning.
    merge_threshold:
        JSD threshold for merging two concepts.  Lower = finer concepts.
    subtype_threshold:
        Soft inclusion threshold for the lattice ordering.
    max_contexts:
        Hard cap on k before FCA.  Primary complexity control.
        FCA is O(k² log k), so k=80 → ~28K ops; k=500 → ~1.1M ops.
    min_support:
        Minimum raw count for a context row to be included.
        Contexts with < min_support observations are filtered before FCA.

    Returns
    -------
    list of ConceptLattice, one per r_level, in the same order as r_levels.
    """
    lattices = []
    for r in r_levels:
        lattice = _fca_one_level(
            hankel=hankel,
            r=r,
            lambda_productivity=lambda_productivity,
            merge_threshold=merge_threshold,
            subtype_threshold=subtype_threshold,
            max_contexts=max_contexts,
            min_support=min_support,
        )
        lattice.compute_ordering()
        lattices.append(lattice)
    return lattices


# ---------------------------------------------------------------------------
# Composition-based productivity score (architecture §Phase 2)
# ---------------------------------------------------------------------------

def composition_productivity(
    corpus_grammar,          # Graph grammar / SEQUITUR Grammar object
    lattice: ConceptLattice,
    typed_corpus: list[list[int]],
) -> dict[int, float]:
    """Compute composition-based productivity for each concept.

    Architecture definition:
        productivity(C) = |novel_compositions_containing_C|
                        / |all_compositions_containing_C|

    A composition is "novel" if the pair (C, other_concept) appears in the
    grammar's rules but was NOT seen as a raw adjacent pair in the corpus.
    This tests whether C appears in *productive* compositions (new combinations)
    rather than just fixed collocations.

    Parameters
    ----------
    corpus_grammar:
        Grammar object from graph_grammar.corpus_grammar().
    lattice:
        Current ConceptLattice.
    typed_corpus:
        List of typed sequences (concept IDs per token position).

    Returns
    -------
    Mapping concept_id → productivity in [0, 1].
    """
    from experiments.symbolic_ai_v2.ctkg.learning.graph_grammar import (
        _is_terminal, _is_nonterminal, _terminal_id, _nonterminal_id,
    )

    # Build set of raw adjacent pairs observed directly in the typed corpus
    raw_adjacent: set[tuple[int, int]] = set()
    for seq in typed_corpus:
        for i in range(len(seq) - 1):
            raw_adjacent.add((seq[i], seq[i + 1]))

    # Walk grammar rules to find all compositions containing each concept
    # A rule R → [A, B] means concept A is composed with concept B
    concept_ids = {c.concept_id for c in lattice.concepts}
    all_comps: dict[int, set[tuple[int, int]]] = defaultdict(set)
    novel_comps: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for rule in corpus_grammar.rules.values():
        body = rule.body
        for i in range(len(body) - 1):
            sym_a = body[i]
            sym_b = body[i + 1]
            # Resolve to terminal concept IDs where possible
            cid_a = _terminal_id(sym_a) if _is_terminal(sym_a) else None
            cid_b = _terminal_id(sym_b) if _is_terminal(sym_b) else None
            if cid_a is None or cid_b is None:
                continue
            if cid_a not in concept_ids or cid_b not in concept_ids:
                continue
            pair = (cid_a, cid_b)
            all_comps[cid_a].add(pair)
            all_comps[cid_b].add(pair)
            if pair not in raw_adjacent:
                novel_comps[cid_a].add(pair)
                novel_comps[cid_b].add(pair)

    result: dict[int, float] = {}
    for c in lattice.concepts:
        cid = c.concept_id
        n_all = len(all_comps.get(cid, set()))
        n_novel = len(novel_comps.get(cid, set()))
        if n_all == 0:
            # Concept never appears in any grammar rule — low productivity
            result[cid] = 0.0
        else:
            result[cid] = n_novel / n_all
    return result


def all_concepts_productive(
    productivity_scores: dict[int, float],
    threshold: float = 0.1,
    top_k: int = 100,
) -> bool:
    """Return True iff the top-k concepts all have productivity >= threshold.

    Architecture §Phase 2 convergence criterion (b):
        productivity scores of top-100 concepts are all above threshold.
    """
    scores = sorted(productivity_scores.values(), reverse=True)[:top_k]
    if not scores:
        return True
    return all(s >= threshold for s in scores)


# ---------------------------------------------------------------------------
# Tree-context FCA (architecture §Phase 2 iteration n — after Phase 4)
# ---------------------------------------------------------------------------

def discover_concepts_from_tree(
    corpus_grammar,
    typed_corpus: list[list[int]],
    lattice: ConceptLattice,
    lambda_productivity: float = 0.1,
    merge_threshold: float = 0.15,
    subtype_threshold: float = 0.05,
    max_contexts: int = 80,
    min_support: float = 2.0,
) -> ConceptLattice:
    """Run FCA using compression-tree-position contexts (EM iteration ≥2).

    Architecture §Phase 2 iteration n:
        Replace flat k-gram contexts with compression-tree-position contexts:
            context = (rule_id, position_within_rule_body)
        for each terminal in the grammar.

    H_tree[tree_context, atom] = P(atom | tree_context) is built by walking
    the grammar rules and recording which concept_id appears at each
    (rule_id, position) slot, weighted by rule frequency.

    Parameters
    ----------
    corpus_grammar:
        Grammar from graph_grammar.corpus_grammar().
    typed_corpus:
        Typed sequences (concept IDs) — used to compute rule frequencies.
    lattice:
        Current ConceptLattice (atoms list required for output).
    lambda_productivity, merge_threshold, subtype_threshold, max_contexts,
    min_support:
        Same as discover_concepts.

    Returns
    -------
    New ConceptLattice using tree-position contexts.
    """
    from experiments.symbolic_ai_v2.ctkg.learning.graph_grammar import (
        _is_terminal, _terminal_id,
    )

    atoms = lattice.atoms
    atom_idx = {a: j for j, a in enumerate(atoms)}
    n_atoms = len(atoms)
    if n_atoms == 0:
        return lattice

    # Build H_tree: context_key → distribution over atoms
    # context_key = f"R{rule_id}:p{position}"
    counts: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for rule in corpus_grammar.rules.values():
        rid = rule.rule_id
        for pos, sym in enumerate(rule.body):
            if not _is_terminal(sym):
                continue
            cid = _terminal_id(sym)
            # Map concept_id back to atom via lattice
            c = lattice.by_id(cid)
            if c is None:
                continue
            # Distribute weight across the concept's top atoms
            ctx_key = f"R{rid}:p{pos}"
            for atom, w in c.intent_weights.items():
                j = atom_idx.get(atom, -1)
                if j >= 0 and w > 0.0:
                    counts[ctx_key][j] += w

    if not counts:
        return lattice

    # Convert to _Cluster list
    clusters: list[_Cluster] = []
    support_rank = sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))
    support_rank = [(ctx, d) for ctx, d in support_rank
                    if sum(d.values()) >= min_support][:max_contexts]

    for cid, (ctx_key, dist_dict) in enumerate(support_rank):
        vec = np.zeros(n_atoms, dtype=float)
        for j, v in dist_dict.items():
            if j < n_atoms:
                vec[j] = v
        total = vec.sum()
        if total > 0:
            vec /= total
        support = float(sum(dist_dict.values()))
        clusters.append(_Cluster.from_row(
            cid=cid, context=ctx_key, distribution=vec, support=support
        ))

    if not clusters:
        return lattice

    clusters = _agglomerate(clusters, merge_threshold=merge_threshold)
    clusters = _apply_mdl_pruning(clusters, lambda_productivity=lambda_productivity)

    if not clusters:
        return lattice

    new_lattice = _build_lattice(clusters, atoms, lattice.radius, subtype_threshold)
    new_lattice.compute_ordering()
    return new_lattice
