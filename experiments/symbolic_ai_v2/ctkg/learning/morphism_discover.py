"""
Phase 3: morphism discovery via distributional FCA on H_morph.

H_morph[pair(concept_A_id, concept_B_id), bridge_hash] = count of times
concept_A and concept_B co-occur with bridge_hash as the intervening typed
sub-sequence pattern.

The bridge_hash is the canonical encoding of the type-sequence between A and B:
a sorted tuple of (distance, concept_id) pairs, analogous to the WL hash in
Phase 1 but operating on concept IDs rather than atom values.

Run FCA on H_morph (structurally identical to Phase 2 `fca_discover`) to
cluster co-occurrence pairs into morphism types.  Each discovered morphism-type
cluster is inserted into the MorphismGraph as an edge.

See CTKG_ARCHITECTURE.md §Phase 3 for the full specification.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import (
    discover_concepts,
    _jsd,
    _agglomerate,
    _apply_mdl_pruning,
    _Cluster,
)
from experiments.symbolic_ai_v2.ctkg.learning.graph_grammar import (
    assign_types,
    typed_corpus_from_lattice,
)
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import (
    ConceptLattice,
    ConceptId,
)
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import (
    MorphismGraph,
    CTKGObject,
)


# ---------------------------------------------------------------------------
# Bridge hash
# ---------------------------------------------------------------------------

def _bridge_hash(
    typed_seq: list[ConceptId],
    i: int,
    j: int,
) -> str:
    """Canonical key for the typed sub-sequence between positions i and j.

    For a pair (concept_A at position i, concept_B at position j > i), the
    bridge is the sequence typed_seq[i+1 : j].

    The key encodes the bridge as:
        'len={L}|{offset},{cid}|...'
    where offset = 1..L and cid is the concept id at each bridge position.
    An empty bridge (adjacent atoms) has key 'len=0'.
    """
    bridge = typed_seq[i + 1: j]
    if not bridge:
        return "len=0"
    parts = [f"{k + 1},{cid}" for k, cid in enumerate(bridge)]
    return f"len={len(bridge)}|" + "|".join(parts)


# ---------------------------------------------------------------------------
# HankelMorphCount
# ---------------------------------------------------------------------------

class HankelMorphCount:
    """Co-occurrence matrix H_morph[pair(A, B), bridge] = count.

    Scans each typed sequence for all (A, B) pairs within `max_gap` positions
    and records the bridge between them.

    Parameters
    ----------
    max_gap:
        Maximum distance between A and B (inclusive of endpoints).
        At max_gap=4, we capture pairs within 4 steps = bridge length ≤ 2.
        Default 6 (covers most arithmetic fact patterns).
    """

    def __init__(self, max_gap: int = 6) -> None:
        self.max_gap = max_gap
        # _counts[(A_cid, B_cid)][bridge_hash] = count
        self._counts: dict[tuple[ConceptId, ConceptId], dict[str, int]] = \
            defaultdict(lambda: defaultdict(int))
        self._pair_vocab: set[tuple[ConceptId, ConceptId]] = set()
        self._bridge_vocab: set[str] = set()

    def update(self, typed_sequence: list[ConceptId]) -> None:
        """Scan one typed sequence and update co-occurrence counts."""
        n = len(typed_sequence)
        for i in range(n):
            for j in range(i + 1, min(n, i + self.max_gap + 1)):
                a_cid = typed_sequence[i]
                b_cid = typed_sequence[j]
                bridge = _bridge_hash(typed_sequence, i, j)
                pair = (a_cid, b_cid)
                self._counts[pair][bridge] += 1
                self._pair_vocab.add(pair)
                self._bridge_vocab.add(bridge)

    def update_batch(self, typed_corpus: list[list[ConceptId]]) -> None:
        for seq in typed_corpus:
            self.update(seq)

    def all_pairs(self) -> list[tuple[ConceptId, ConceptId]]:
        return list(self._counts.keys())

    def bridges(self) -> list[str]:
        return sorted(self._bridge_vocab)

    def get_distribution(
        self,
        pair: tuple[ConceptId, ConceptId],
    ) -> dict[str, float]:
        """P(bridge | pair co-occurs) — row-normalised."""
        raw = self._counts.get(pair, {})
        total = sum(raw.values())
        if total == 0:
            return {}
        return {b: cnt / total for b, cnt in raw.items()}

    def matrix(self) -> tuple[list[str], list[str], np.ndarray]:
        """Return (pair_keys, bridge_keys, H_morph) as dense float64 array.

        pair_keys: list of 'A,B' strings (row labels)
        bridge_keys: list of bridge hashes (column labels)
        H_morph: shape (n_pairs, n_bridges), row-normalised
        """
        pairs = self.all_pairs()
        bridges = self.bridges()
        bridge_idx = {b: j for j, b in enumerate(bridges)}

        pair_keys = [f"{a},{b}" for a, b in pairs]
        H = np.zeros((len(pairs), len(bridges)), dtype=np.float64)
        for i, pair in enumerate(pairs):
            raw = self._counts[pair]
            total = sum(raw.values())
            if total == 0:
                continue
            for bridge, cnt in raw.items():
                j = bridge_idx.get(bridge)
                if j is not None:
                    H[i, j] = cnt / total

        return pair_keys, bridges, H

    def raw_support(self, pair: tuple[ConceptId, ConceptId]) -> float:
        return float(sum(self._counts.get(pair, {}).values()))


# ---------------------------------------------------------------------------
# Morphism discovery: FCA on H_morph → MorphismGraph edges
# ---------------------------------------------------------------------------

def discover_morphisms(
    corpus: list[list[str]],
    hankel: HankelCount,
    lattice: ConceptLattice,
    r: int = 1,
    merge_threshold: float = 0.15,
    lambda_productivity: float = 0.1,
    max_pairs: int = 200,
    min_pair_support: float = 3.0,
    max_gap: int = 6,
) -> MorphismGraph:
    """Discover morphisms from the corpus and return a populated MorphismGraph.

    Steps:
    1. Type the corpus using `lattice` at radius `r`.
    2. Build H_morph by scanning all (A, B) pairs within max_gap.
    3. Run FCA on H_morph to cluster similar co-occurrence patterns.
    4. Each cluster = one morphism type; insert into MorphismGraph.
    5. Objects in the graph = one per concept in lattice.top_concepts(20).

    Parameters
    ----------
    corpus:
        Raw string sequences.
    hankel:
        Trained HankelCount from Phase 1.
    lattice:
        ConceptLattice from Phase 2 (radius r).
    r:
        Radius used for type assignment.
    merge_threshold:
        JSD threshold for morphism FCA.
    lambda_productivity:
        MDL regularisation for morphism pruning.
    max_pairs:
        Hard cap on number of (A,B) pairs fed to FCA (complexity control).
    min_pair_support:
        Minimum raw count for a pair to be included.
    max_gap:
        Maximum gap between A and B in the typed sequence.

    Returns
    -------
    MorphismGraph with objects (one per concept) and morphisms (discovered types).
    """
    # Step 1: type the corpus
    typed_corpus = typed_corpus_from_lattice(corpus, hankel, lattice, r=r)

    # Step 2: build H_morph
    hmc = HankelMorphCount(max_gap=max_gap)
    hmc.update_batch(typed_corpus)

    # Step 3: FCA on H_morph
    pair_keys, bridge_keys, H_morph = hmc.matrix()
    pairs = hmc.all_pairs()

    # Filter by support and cap
    support_rank = sorted(
        range(len(pairs)),
        key=lambda i: -hmc.raw_support(pairs[i]),
    )
    support_rank = [
        i for i in support_rank
        if hmc.raw_support(pairs[i]) >= min_pair_support
    ][:max_pairs]

    if not support_rank:
        # No pairs — return graph with objects only
        mg = _build_graph_objects(lattice)
        return mg

    # Build clusters from filtered pairs
    clusters: list[_Cluster] = []
    for cid, i in enumerate(support_rank):
        pair = pairs[i]
        row = H_morph[i].copy()
        support = hmc.raw_support(pair)
        cluster = _Cluster.from_row(
            cid=cid,
            context=f"{pair[0]},{pair[1]}",
            distribution=row,
            support=support,
        )
        clusters.append(cluster)

    # Agglomerate
    clusters = _agglomerate(clusters, merge_threshold=merge_threshold)
    clusters = _apply_mdl_pruning(clusters, lambda_productivity=lambda_productivity)

    if not clusters:
        clusters = [
            _Cluster.from_row(
                cid=i,
                context=f"{pairs[support_rank[i]][0]},{pairs[support_rank[i]][1]}",
                distribution=H_morph[support_rank[i]].copy(),
                support=hmc.raw_support(pairs[support_rank[i]]),
            )
            for i in range(min(3, len(support_rank)))
        ]

    # Step 4: build MorphismGraph
    mg = _build_graph_objects(lattice)
    concept_to_obj = {obj.concept.concept_id: obj.obj_id for obj in mg.objects()}

    for morph_cid, cluster in enumerate(clusters):
        # The cluster's contexts are 'A,B' pair strings
        # Determine dominant source and target from the member pairs
        src_counts: dict[ConceptId, float] = defaultdict(float)
        tgt_counts: dict[ConceptId, float] = defaultdict(float)

        for ctx_str in cluster.contexts:
            parts = ctx_str.split(",")
            if len(parts) == 2:
                try:
                    a_cid = int(parts[0])
                    b_cid = int(parts[1])
                    weight = cluster.context_weights.get(ctx_str, 1.0)
                    src_counts[a_cid] += weight
                    tgt_counts[b_cid] += weight
                except ValueError:
                    continue

        if not src_counts or not tgt_counts:
            continue

        src_cid = max(src_counts, key=lambda x: src_counts[x])
        tgt_cid = max(tgt_counts, key=lambda x: tgt_counts[x])

        src_obj_id = concept_to_obj.get(src_cid)
        tgt_obj_id = concept_to_obj.get(tgt_cid)
        if src_obj_id is None or tgt_obj_id is None:
            continue

        # Dominant bridge label (top bridge hash by weight in centroid)
        top_bridge_idx = int(np.argmax(cluster.centroid))
        bridge_label = bridge_keys[top_bridge_idx] if top_bridge_idx < len(bridge_keys) else "?"

        mg.add_morphism(
            source_id=src_obj_id,
            target_id=tgt_obj_id,
            body=[src_obj_id, tgt_obj_id],
            evidence=int(cluster.support),
            morph_type=f"MORPH_{morph_cid}:{bridge_label[:20]}",
            confidence=0.0,
        )

    return mg


def _build_graph_objects(lattice: ConceptLattice) -> MorphismGraph:
    """Create one CTKGObject per concept in the top-20 of the lattice."""
    mg = MorphismGraph()
    for concept in lattice.top_concepts(20):
        mg.add_object(concept)
    return mg
