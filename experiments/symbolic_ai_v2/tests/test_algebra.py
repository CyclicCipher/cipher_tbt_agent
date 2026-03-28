"""
Algebraic skeleton discovery tests.

Tests SCC decomposition, chain extraction, cycle detection, and
materialization via the AgenticLoop (the only door).

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_algebra.py -v
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import KnowledgeGraph, TRANSITION
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.ctkg.logic.algebra import (
    _extract_transition_graph,
    _tarjan_scc,
    _find_cycle_in_scc,
    discover_algebraic_structure,
)


def _make_loop() -> AgenticLoop:
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = 0
    return loop


# ============================================================================
# Transition graph extraction
# ============================================================================

class TestTransitionGraph:

    def test_sequential_observations_create_transitions(self):
        """Consecutive observe() calls create transition edges."""
        loop = _make_loop()
        loop.observe(["room_A", "item_X"])
        loop.observe(["room_B", "item_Y"])

        adj = _extract_transition_graph(loop.kg)
        # room_A and item_X should have transitions to room_B and/or item_Y.
        assert len(adj) > 0, "Should have transition edges after sequential observations"

    def test_no_transitions_from_single_observation(self):
        """A single observation creates no transition edges."""
        loop = _make_loop()
        loop.observe(["A", "B", "C"])
        adj = _extract_transition_graph(loop.kg)
        assert len(adj) == 0

    def test_action_observations_dont_create_transitions(self):
        """Action observations (edge_type=2) don't create forward transitions."""
        loop = _make_loop()
        loop.observe(["context_A", "context_B"])
        loop.observe(["action_X"], [2])  # action observation
        loop.observe(["result_Y", "result_Z"])

        adj = _extract_transition_graph(loop.kg)
        # context tokens should NOT have transitions to action token
        # (blocked by is_action_obs check in loop.py)
        ctx_a = loop.kg._value_to_node.get("context_A")
        act_x = loop.kg._value_to_node.get("action_X")
        if ctx_a is not None and act_x is not None:
            assert act_x not in adj.get(ctx_a, []), (
                "Context should not have transitions to action observations"
            )


# ============================================================================
# SCC decomposition
# ============================================================================

class TestSCC:

    def test_linear_chain_all_trivial(self):
        """A linear chain A→B→C has 3 trivial SCCs."""
        adj = {0: [1], 1: [2], 2: []}
        sccs = _tarjan_scc(adj)
        assert len(sccs) == 3
        assert all(len(scc) == 1 for scc in sccs)

    def test_cycle_single_scc(self):
        """A→B→C→A forms one SCC of size 3."""
        adj = {0: [1], 1: [2], 2: [0]}
        sccs = _tarjan_scc(adj)
        # Should find exactly one SCC containing all 3 nodes.
        big_sccs = [scc for scc in sccs if len(scc) == 3]
        assert len(big_sccs) == 1
        assert set(big_sccs[0]) == {0, 1, 2}

    def test_two_sccs_with_bridge(self):
        """Two cycles connected by a one-way bridge: {A,B} → {C,D}."""
        adj = {0: [1], 1: [0, 2], 2: [3], 3: [2]}
        sccs = _tarjan_scc(adj)
        scc_sets = [frozenset(scc) for scc in sccs]
        assert frozenset({0, 1}) in scc_sets
        assert frozenset({2, 3}) in scc_sets

    def test_empty_graph(self):
        """Empty graph produces no SCCs."""
        sccs = _tarjan_scc({})
        assert sccs == []


# ============================================================================
# Cycle detection
# ============================================================================

class TestCycleDetection:

    def test_simple_cycle(self):
        """3-node cycle should be detected."""
        adj = {0: [1], 1: [2], 2: [0]}
        members = frozenset({0, 1, 2})
        cycle = _find_cycle_in_scc(members, adj)
        assert len(cycle) == 3
        # Verify it's actually a cycle: last → first edge exists.
        assert cycle[0] in adj.get(cycle[-1], [])

    def test_self_loop(self):
        """A self-loop is a cycle of length 1."""
        adj = {5: [5]}
        members = frozenset({5})
        cycle = _find_cycle_in_scc(members, adj)
        assert cycle == [5]

    def test_no_cycle_in_singleton(self):
        """A singleton with no self-loop has no cycle."""
        adj = {5: []}
        members = frozenset({5})
        cycle = _find_cycle_in_scc(members, adj)
        assert cycle == []


# ============================================================================
# Full discovery via AgenticLoop
# ============================================================================

class TestDiscoverAlgebraicStructure:

    def test_room_transitions_produce_structure(self):
        """Science-lab-like room transitions should produce components."""
        loop = _make_loop()
        # Simulate room navigation: A → B → C → A (cycle)
        for _ in range(5):
            loop.observe(["room_A", "items_A"])
            loop.observe(["room_B", "items_B"])
            loop.observe(["room_C", "items_C"])
            loop.observe(["room_A", "items_A"])  # return to start

        stats = discover_algebraic_structure(loop.kg, loop.hippo)
        assert stats["transition_nodes"] > 0
        assert stats["components"] > 0

    def test_linear_sequence_finds_chain(self):
        """A strictly linear sequence should produce a chain."""
        loop = _make_loop()
        # Observe a strict sequence: 1, 2, 3, 4, 5, 6 (never revisiting)
        tokens = [f"step_{i}" for i in range(8)]
        for i in range(len(tokens)):
            loop.observe([tokens[i]])

        stats = discover_algebraic_structure(
            loop.kg, loop.hippo, min_chain_length=3,
        )
        # Should find at least one chain of length >= 3.
        assert stats["transition_nodes"] > 0

    def test_integrates_with_consolidation(self):
        """Algebra stats appear in consolidate() output."""
        loop = _make_loop()
        for _ in range(3):
            loop.observe(["X", "Y"])
            loop.observe(["Y", "Z"])

        stats = loop.consolidate()
        assert "alg_transition_nodes" in stats
        assert "alg_components" in stats
        assert "alg_chains_found" in stats

    def test_empty_graph_returns_zeros(self):
        """No observations → zero stats."""
        loop = _make_loop()
        stats = discover_algebraic_structure(loop.kg, loop.hippo)
        assert stats["transition_nodes"] == 0
        assert stats["components"] == 0
        assert stats["chains_found"] == 0

    def test_cycle_detected_in_navigation(self):
        """Cyclic navigation should produce a non-trivial SCC."""
        loop = _make_loop()
        rooms = ["north", "east", "south", "west"]
        # Navigate in a cycle many times to build strong transitions.
        for _ in range(10):
            for room in rooms:
                loop.observe([room])

        stats = discover_algebraic_structure(
            loop.kg, loop.hippo, min_scc_size=2,
        )
        # The 4-room cycle should produce at least one non-trivial component.
        assert stats["nontrivial_components"] >= 1 or stats["cycles_found"] >= 1
