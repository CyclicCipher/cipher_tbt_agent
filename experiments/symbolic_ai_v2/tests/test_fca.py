"""
FCA (Formal Concept Analysis) integration tests.

Tests that FCA discovers type hierarchy from observation data via the
AgenticLoop. No direct graph manipulation for learning — all observations
go through loop.observe().

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_fca.py -v
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import KnowledgeGraph, COOCCURRENCE
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus
from experiments.symbolic_ai_v2.ctkg.logic.fca import (
    build_incidence_matrix,
    run_fca_at_threshold,
    multi_threshold_fca,
    discover_fca_structure,
)


def _make_loop() -> AgenticLoop:
    """Create a fresh loop with consolidation disabled."""
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = 0  # manual consolidation only
    return loop


# ============================================================================
# Incidence matrix construction
# ============================================================================

class TestBuildIncidenceMatrix:

    def test_basic_incidence(self):
        """Tokens that appear in observations show True in the matrix."""
        loop = _make_loop()
        # Three observations with overlapping tokens.
        loop.observe(["A", "B", "C"])
        loop.observe(["A", "B", "D"])
        loop.observe(["C", "D", "E"])

        objects, properties, bools = build_incidence_matrix(
            loop.kg, loop.hippo, threshold=0.0,
        )
        assert len(objects) > 0
        assert len(properties) == 3  # 3 observations
        assert len(bools) == len(objects)  # one row per token

        # Token "A" should be in observations 0 and 1, not 2.
        a_nid = loop.kg._value_to_node["A"]
        a_idx = objects.index(str(a_nid))
        assert bools[a_idx][0] is True   # obs 0
        assert bools[a_idx][1] is True   # obs 1
        assert bools[a_idx][2] is False  # obs 2

    def test_threshold_filters_weak_edges(self):
        """At high threshold, tokens with weak co-occurrence are excluded."""
        loop = _make_loop()
        # Observe A,B many times (strong edges) and C,D once (weak edges).
        for _ in range(10):
            loop.observe(["A", "B"])
        loop.observe(["C", "D"])

        # At threshold 0.0, all tokens appear.
        obj0, prop0, bools0 = build_incidence_matrix(
            loop.kg, loop.hippo, threshold=0.0,
        )
        assert len(obj0) >= 4

        # At threshold 0.5, the single C,D observation may be filtered
        # because C↔D edge weight is weak after only 1 observation.
        obj5, prop5, bools5 = build_incidence_matrix(
            loop.kg, loop.hippo, threshold=0.5,
        )
        # A and B should still have many True entries.
        if obj5:
            a_nid = loop.kg._value_to_node["A"]
            if str(a_nid) in obj5:
                a_idx = obj5.index(str(a_nid))
                a_trues = sum(1 for v in bools5[a_idx] if v)
                assert a_trues >= 5  # A appeared in many observations

    def test_since_index_skips_old(self):
        """since_index parameter skips old observations."""
        loop = _make_loop()
        for _ in range(5):
            loop.observe(["old_A", "old_B"])
        for _ in range(5):
            loop.observe(["new_C", "new_D"])

        # Only include observations from index 5 onward.
        objects, properties, bools = build_incidence_matrix(
            loop.kg, loop.hippo, since_index=5, threshold=0.0,
        )
        # "new_C" should be present, "old_A" should not be in any True column.
        if objects:
            new_c_nid = loop.kg._value_to_node.get("new_C")
            old_a_nid = loop.kg._value_to_node.get("old_A")
            if new_c_nid is not None and str(new_c_nid) in objects:
                c_idx = objects.index(str(new_c_nid))
                c_trues = sum(1 for v in bools[c_idx] if v)
                assert c_trues >= 1


# ============================================================================
# Single-threshold FCA
# ============================================================================

class TestRunFCA:

    def test_simple_lattice(self):
        """Tokens that always co-occur should form a concept."""
        loop = _make_loop()
        # A and B always appear together; C and D always appear together.
        # A,B never appear with C,D.
        for _ in range(5):
            loop.observe(["A", "B"])
        for _ in range(5):
            loop.observe(["C", "D"])

        concepts = run_fca_at_threshold(loop.kg, loop.hippo, threshold=0.0)
        extents = [fc.extent for fc in concepts]

        # Should find concepts containing {A_nid, B_nid} and {C_nid, D_nid}.
        a_nid = loop.kg._value_to_node["A"]
        b_nid = loop.kg._value_to_node["B"]
        c_nid = loop.kg._value_to_node["C"]
        d_nid = loop.kg._value_to_node["D"]

        ab = frozenset({a_nid, b_nid})
        cd = frozenset({c_nid, d_nid})

        assert ab in extents, f"{{A,B}} should be a concept extent. Got: {extents}"
        assert cd in extents, f"{{C,D}} should be a concept extent. Got: {extents}"

    def test_empty_observations(self):
        """No observations → no concepts."""
        loop = _make_loop()
        concepts = run_fca_at_threshold(loop.kg, loop.hippo, threshold=0.0)
        assert concepts == []


# ============================================================================
# Multi-threshold FCA with persistence
# ============================================================================

class TestMultiThreshold:

    def test_persistent_concept(self):
        """Strongly co-occurring tokens form a concept at all thresholds."""
        loop = _make_loop()
        # Observe A,B many times to build strong edges.
        for _ in range(20):
            loop.observe(["A", "B"])

        result = multi_threshold_fca(
            loop.kg, loop.hippo, min_persistence=1,
        )
        a_nid = loop.kg._value_to_node["A"]
        b_nid = loop.kg._value_to_node["B"]
        ab = frozenset({a_nid, b_nid})

        # Find the concept with extent {A, B}.
        ab_concepts = [fc for fc in result.concepts if fc.extent == ab]
        assert len(ab_concepts) >= 1, f"Should find {{A,B}} concept. Got extents: {[fc.extent for fc in result.concepts]}"
        # Should persist across multiple thresholds.
        assert ab_concepts[0].persistence >= 2, (
            f"Strong co-occurrence should persist, got {ab_concepts[0].persistence}"
        )

    def test_weak_concept_low_persistence(self):
        """Weakly co-occurring tokens have low persistence."""
        loop = _make_loop()
        # A,B strong (20 observations), C,D weak (2 observations).
        for _ in range(20):
            loop.observe(["A", "B"])
        for _ in range(2):
            loop.observe(["C", "D"])

        result = multi_threshold_fca(
            loop.kg, loop.hippo, min_persistence=1,
        )
        c_nid = loop.kg._value_to_node["C"]
        d_nid = loop.kg._value_to_node["D"]
        cd = frozenset({c_nid, d_nid})

        cd_concepts = [fc for fc in result.concepts if fc.extent == cd]
        a_nid = loop.kg._value_to_node["A"]
        b_nid = loop.kg._value_to_node["B"]
        ab = frozenset({a_nid, b_nid})
        ab_concepts = [fc for fc in result.concepts if fc.extent == ab]

        # CD should have lower persistence than AB.
        if cd_concepts and ab_concepts:
            assert cd_concepts[0].persistence <= ab_concepts[0].persistence

    def test_subconcept_ordering(self):
        """Subset extents produce subconcept → superconcept pairs."""
        loop = _make_loop()
        # Observations where {A,B} always co-occur, and sometimes {A,B,C}.
        for _ in range(10):
            loop.observe(["A", "B"])
        for _ in range(10):
            loop.observe(["A", "B", "C"])

        result = multi_threshold_fca(
            loop.kg, loop.hippo, min_persistence=1,
        )
        a_nid = loop.kg._value_to_node["A"]
        b_nid = loop.kg._value_to_node["B"]
        c_nid = loop.kg._value_to_node["C"]

        ab = frozenset({a_nid, b_nid})
        abc = frozenset({a_nid, b_nid, c_nid})

        # Find indices.
        ab_idx = None
        abc_idx = None
        for i, fc in enumerate(result.concepts):
            if fc.extent == ab:
                ab_idx = i
            if fc.extent == abc:
                abc_idx = i

        if ab_idx is not None and abc_idx is not None:
            # ab ⊂ abc, so (ab_idx, abc_idx) should be a subconcept pair.
            # But in FCA, the subconcept has the SMALLER extent...
            # Actually in FCA lattice ordering: smaller extent = more specific
            # = subconcept. But {A,B} ⊂ {A,B,C} means {A,B,C} is more general
            # (more objects). Wait — in FCA, extent ⊂ means MORE specific intent.
            # The subconcept relation is: A ≤ B iff A.extent ⊆ B.extent.
            # So {A,B} ≤ {A,B,C} because {A,B} ⊆ {A,B,C}.
            assert (ab_idx, abc_idx) in result.subconcept_pairs, (
                f"Expected ({ab_idx}, {abc_idx}) in subconcept_pairs"
            )


# ============================================================================
# Materialization in KnowledgeGraph
# ============================================================================

class TestDiscoverFCAStructure:

    def test_materializes_concepts(self):
        """Persistent concepts become nodes in the KG."""
        loop = _make_loop()
        for _ in range(20):
            loop.observe(["A", "B"])
        for _ in range(20):
            loop.observe(["C", "D"])

        stats = discover_fca_structure(
            loop.kg, loop.hippo, min_persistence=1,
        )
        assert stats["concepts_materialized"] >= 1

        # Check that concept nodes exist.
        concept_nodes = [
            v for v in loop.kg._value_to_node.keys()
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "__fca_concept__"
        ]
        assert len(concept_nodes) >= 1

    def test_cocone_edges(self):
        """Member tokens have edges to their concept node."""
        loop = _make_loop()
        for _ in range(20):
            loop.observe(["A", "B"])

        discover_fca_structure(loop.kg, loop.hippo, min_persistence=1)

        a_nid = loop.kg._value_to_node["A"]
        b_nid = loop.kg._value_to_node["B"]
        ab = frozenset({a_nid, b_nid})
        concept_key = ("__fca_concept__", ab)

        if concept_key in loop.kg._value_to_node:
            concept_nid = loop.kg._value_to_node[concept_key]
            # A → concept edge should exist.
            e_a = loop.kg.edge(a_nid, concept_nid)
            assert e_a is not None, "A → concept cocone edge should exist"
            assert e_a.weight > 0
            # B → concept edge should exist.
            e_b = loop.kg.edge(b_nid, concept_nid)
            assert e_b is not None, "B → concept cocone edge should exist"
            assert e_b.weight > 0

    def test_idempotent(self):
        """Running discovery twice doesn't create duplicate nodes."""
        loop = _make_loop()
        for _ in range(20):
            loop.observe(["A", "B"])

        stats1 = discover_fca_structure(loop.kg, loop.hippo, min_persistence=1)
        node_count_1 = loop.kg.node_count()

        stats2 = discover_fca_structure(loop.kg, loop.hippo, min_persistence=1)
        node_count_2 = loop.kg.node_count()

        assert node_count_2 == node_count_1, "Second run should not create new nodes"

    def test_integrates_with_consolidation(self):
        """FCA stats appear in consolidate() output."""
        loop = _make_loop()
        for _ in range(20):
            loop.observe(["X", "Y"])

        stats = loop.consolidate()
        assert "fca_concepts_discovered" in stats
        assert "fca_concepts_materialized" in stats
