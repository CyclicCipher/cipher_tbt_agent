"""
Sheaf Laplacian tests.

Tests that the sheaf consistency measure correctly identifies
agreement/disagreement in the graph's activation patterns.

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_sheaf.py -v
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, COOCCURRENCE, TRANSITION,
)
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.ctkg.logic.sheaf import (
    compute_sheaf_energy,
    compute_sheaf_consistency,
    discover_sheaf_structure,
)


# ============================================================================
# Per-edge energy
# ============================================================================

class TestSheafEnergy:

    def test_agreeing_activations_low_energy(self):
        """Two nodes with equal activation and positive edge → low energy."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        edge.weight = 0.8

        activations = {a: 1.0, b: 1.0}
        diag = compute_sheaf_energy(kg, activations)
        assert diag.global_energy < 0.01, (
            f"Equal activations + excitatory edge should have near-zero energy, got {diag.global_energy}"
        )

    def test_disagreeing_activations_high_energy(self):
        """One active, one inactive with positive edge → high energy."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        edge.weight = 0.8

        activations = {a: 1.0}  # b not active
        diag = compute_sheaf_energy(kg, activations)
        assert diag.global_energy > 0.5, (
            f"Disagreeing activations should have high energy, got {diag.global_energy}"
        )

    def test_inhibitory_both_active_high_energy(self):
        """Inhibitory edge with both nodes active → high energy."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        edge.weight = -0.8

        activations = {a: 1.0, b: 1.0}
        diag = compute_sheaf_energy(kg, activations)
        assert diag.global_energy > 0.5, (
            f"Inhibitory edge with both active should have high energy, got {diag.global_energy}"
        )

    def test_inhibitory_one_inactive_low_energy(self):
        """Inhibitory edge with one node inactive → low energy."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        edge.weight = -0.8

        activations = {a: 1.0}  # b inactive
        diag = compute_sheaf_energy(kg, activations)
        assert diag.global_energy < 0.01

    def test_empty_graph_zero_energy(self):
        """No edges → zero energy."""
        kg = KnowledgeGraph()
        kg.get_or_create("A")
        diag = compute_sheaf_energy(kg, {})
        assert diag.global_energy == 0.0
        assert diag.global_consistency == 1.0

    def test_worst_edges_sorted(self):
        """worst_edges is sorted by inconsistency (highest first)."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        c = kg.get_or_create("C")

        e1 = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        e1.weight = 0.9  # strong excitatory
        e2 = kg.get_or_create_edge(a, c, role=COOCCURRENCE)
        e2.weight = 0.3  # weak excitatory

        # A active, B and C inactive → both edges inconsistent,
        # but A→B more so because higher weight.
        activations = {a: 1.0}
        diag = compute_sheaf_energy(kg, activations)

        if len(diag.worst_edges) >= 2:
            assert diag.worst_edges[0][2] >= diag.worst_edges[1][2]


# ============================================================================
# Consistency over snapshots
# ============================================================================

class TestSheafConsistency:

    def test_consistent_observations(self):
        """Observations that match the edge structure → high consistency."""
        loop = AgenticLoop(KnowledgeGraph())
        loop.CONSOLIDATION_INTERVAL = 0

        # Always observe A,B together → strong edge, consistent.
        for _ in range(20):
            loop.observe(["A", "B"])

        stats = compute_sheaf_consistency(loop.kg, loop.hippo)
        assert stats["snapshots_analyzed"] == 20
        assert stats["mean_consistency"] > 0.5, (
            f"Consistent observations should give high consistency, got {stats['mean_consistency']}"
        )

    def test_empty_hippo(self):
        """No snapshots → perfect consistency (nothing to disagree about)."""
        kg = KnowledgeGraph()
        from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus
        hippo = Hippocampus()
        stats = compute_sheaf_consistency(kg, hippo)
        assert stats["mean_consistency"] == 1.0

    def test_integrates_with_consolidation(self):
        """Sheaf stats appear in consolidate() output."""
        loop = AgenticLoop(KnowledgeGraph())
        loop.CONSOLIDATION_INTERVAL = 0
        for _ in range(5):
            loop.observe(["X", "Y"])

        stats = loop.consolidate()
        assert "sheaf_mean_consistency" in stats
        assert "sheaf_mean_energy" in stats
        assert "sheaf_snapshots_analyzed" in stats
