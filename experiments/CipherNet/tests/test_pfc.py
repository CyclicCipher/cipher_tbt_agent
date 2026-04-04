"""PFC capability tests.

Tests every function the PFC should provide. These tests define
what SUCCESS looks like — the implementation must pass all of them.

The PFC is not just for multi-step math. It provides:
1. Working memory — hold values across time steps
2. Gating — control when WM updates vs maintains
3. Multi-step computation — chain operations, feed results forward
4. Task switching — change which operation to apply mid-sequence
5. Inhibition — suppress dominant but wrong responses
6. Goal maintenance — hold a goal steady while executing substeps
7. Error monitoring — detect when results don't match expectations
8. Sequencing — maintain and advance through a multi-step plan
9. Attention/selection — route inputs to the right operation column
10. Distractor resistance — maintain WM despite irrelevant inputs
"""
from __future__ import annotations

import sys
import os

_CIPHER_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_CIPHER_ROOT, "src"))
sys.path.insert(0, _CIPHER_ROOT)

import pytest
from graph import Graph
from prior_loader import load_priors
from displacement import ManifoldColumn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def brain():
    """A fresh brain with ANS + PFC priors and arithmetic columns."""
    graph, prior_nodes = load_priors()

    # Create arithmetic columns.
    add_col = ManifoldColumn(graph, "add")
    add_col.learn(1, 2, 3)
    add_col.learn(8, 7, 15)
    add_col.learn(20, 13, 33)

    mul_col = ManifoldColumn(graph, "mul")
    mul_col.learn(3, 4, 12)
    mul_col.learn(2, 5, 10)
    mul_col.learn(7, 3, 21)
    mul_col.learn(3, -2, -6)
    mul_col.learn(-4, 3, -12)
    mul_col.learn(-3, -4, 12)
    mul_col.learn(0, 7, 0)
    mul_col.learn(5, 0, 0)

    sub_col = ManifoldColumn(graph, "sub")
    sub_col.learn(10, 3, 7)
    sub_col.learn(20, 8, 12)
    sub_col.learn(5, 5, 0)

    return {
        "graph": graph,
        "priors": prior_nodes,
        "add": add_col,
        "mul": mul_col,
        "sub": sub_col,
    }


# ---------------------------------------------------------------------------
# Test 1: Working memory — hold a value across time steps
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    """The PFC must hold a value in a WM stripe and retrieve it later."""

    def test_store_and_retrieve(self, brain):
        """Store a value in WM0, advance several time steps, retrieve it."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        # Activate WM0 L23 with a value (simulate storing 10).
        wm0_l23 = pfc["wm0:L23"]
        g.activate(wm0_l23, 1.0)

        # Simulate several time steps of decay.
        for _ in range(5):
            g.step()

        # WM0 L23 has a self-loop (weight 0.95) and a gate edge.
        # With gate closed (gate node not active), it should hold.
        node = g.get_node(wm0_l23)
        assert node.activation > 0.3, (
            f"WM0 should maintain activation after 5 steps, got {node.activation:.3f}"
        )

    def test_non_wm_decays_faster(self, brain):
        """A non-PFC node should decay much faster than a WM stripe."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        # Activate both a WM node and the add column's output.
        wm0_l23 = pfc["wm0:L23"]
        add_output = brain["add"].output

        g.activate(wm0_l23, 1.0)
        g.activate(add_output, 1.0)

        for _ in range(10):
            g.step()

        wm_act = g.get_node(wm0_l23).activation
        add_act = g.get_node(add_output).activation

        assert wm_act > add_act, (
            f"WM stripe ({wm_act:.3f}) should retain more than "
            f"non-WM node ({add_act:.3f}) after decay"
        )

    def test_goal_stripe_most_persistent(self, brain):
        """Stripe 2 (goal) should be the most persistent of all stripes."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        for stripe in ["wm0:L23", "wm1:L23", "wm2:L23"]:
            g.activate(pfc[stripe], 1.0)

        for _ in range(20):
            g.step()

        act0 = g.get_node(pfc["wm0:L23"]).activation
        act1 = g.get_node(pfc["wm1:L23"]).activation
        act2 = g.get_node(pfc["wm2:L23"]).activation

        assert act2 >= act0, (
            f"Goal stripe ({act2:.3f}) should be >= WM0 ({act0:.3f})"
        )
        assert act2 >= act1, (
            f"Goal stripe ({act2:.3f}) should be >= WM1 ({act1:.3f})"
        )


# ---------------------------------------------------------------------------
# Test 2: Gating — control when WM updates
# ---------------------------------------------------------------------------

class TestGating:
    """Gates control whether WM stripes update or maintain."""

    def test_gate_open_allows_update(self, brain):
        """When the BG Go pathway fires, the thalamic relay opens, and WM updates."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]
        bg = brain["priors"]["basal_ganglia"]
        thal = brain["priors"]["thalamus"]

        wm0_l4 = pfc["wm0:L4"]
        wm0_l23 = pfc["wm0:L23"]

        # Fire the Go pathway for stripe 0.
        g.activate(bg["d1_go_0"], 1.0)
        # Provide input.
        g.activate(wm0_l4, 0.8)

        # Run steps to propagate: D1 Go → GPi (inhibited) → relay (disinhibited) → WM L23.
        for _ in range(3):
            g.step()

        # With gate open, L23 should have received input.
        act = g.get_node(wm0_l23).activation
        assert act > 0, (
            f"With BG Go active, input should reach WM L23, got {act:.3f}"
        )

    def test_gate_structure_exists(self, brain):
        """Verify BG GPi connects to thalamus relay, relay connects to PFC via GATE edges."""
        g = brain["graph"]
        bg = brain["priors"]["basal_ganglia"]
        thal = brain["priors"]["thalamus"]
        pfc = brain["priors"]["pfc"]

        # GPi should gate the thalamic relay.
        gpi0_edges = g.edges_from(bg["gpi_0"])
        gpi0_targets = {e.target for e in gpi0_edges}
        assert thal["relay_0"] in gpi0_targets, "GPi_0 should connect to thalamic relay_0"

        # Thalamic relay should gate PFC WM L23.
        relay0_edges = g.edges_from(thal["relay_0"])
        relay0_targets = {e.target for e in relay0_edges}
        assert pfc["wm0:L23"] in relay0_targets, "Thalamic relay_0 should connect to PFC WM0:L23"


# ---------------------------------------------------------------------------
# Test 3: Multi-step computation
# ---------------------------------------------------------------------------

class TestMultiStep:
    """Chain operations: 3 + 7 + 1 = 11 requires two addition steps."""

    def test_two_step_addition(self, brain):
        """Compute 3 + 7 = 10, then 10 + 1 = 11."""
        add = brain["add"]

        # Step 1: 3 + 7 = ?
        r1 = add.predict({"a": 3, "b": 7}, "c")
        assert r1 is not None and abs(r1 - 10) < 0.01, f"3+7 should be 10, got {r1}"

        # Step 2: 10 + 1 = ?
        r2 = add.predict({"a": r1, "b": 1}, "c")
        assert r2 is not None and abs(r2 - 11) < 0.01, f"10+1 should be 11, got {r2}"

    def test_three_step_addition(self, brain):
        """Compute 3 + 7 + 1 + 4 = 15."""
        add = brain["add"]
        result = 3
        for operand in [7, 1, 4]:
            result = add.predict({"a": result, "b": operand}, "c")
            assert result is not None, f"Addition failed at operand {operand}"
        assert abs(result - 15) < 0.01, f"3+7+1+4 should be 15, got {result}"

    def test_mixed_operations(self, brain):
        """Compute 3 + 4 * 2 = 11 (multiply first, then add)."""
        add = brain["add"]
        mul = brain["mul"]

        # Correct order: multiply first.
        r1 = mul.predict({"a": 4, "b": 2}, "c")
        assert r1 is not None and abs(r1 - 8) < 0.01

        r2 = add.predict({"a": 3, "b": r1}, "c")
        assert r2 is not None and abs(r2 - 11) < 0.01, f"3+4*2 should be 11, got {r2}"


# ---------------------------------------------------------------------------
# Test 4: Task switching
# ---------------------------------------------------------------------------

class TestTaskSwitching:
    """Switch between different operations within a sequence."""

    def test_add_then_multiply(self, brain):
        """Compute (3 + 4) * 2 = 14."""
        add, mul = brain["add"], brain["mul"]

        r1 = add.predict({"a": 3, "b": 4}, "c")
        r2 = mul.predict({"a": r1, "b": 2}, "c")
        assert abs(r2 - 14) < 0.01, f"(3+4)*2 should be 14, got {r2}"

    def test_multiply_then_subtract(self, brain):
        """Compute 5 * 6 - 10 = 20."""
        mul, sub = brain["mul"], brain["sub"]

        r1 = mul.predict({"a": 5, "b": 6}, "c")
        r2 = sub.predict({"a": r1, "b": 10}, "c")
        assert abs(r2 - 20) < 0.01, f"5*6-10 should be 20, got {r2}"

    def test_alternating_operations(self, brain):
        """2 + 3 * 4 + 5 * 6 = 2 + 12 + 30 = 44 (left to right for simplicity)."""
        add, mul = brain["add"], brain["mul"]

        # Step 1: 2 + 3 = 5
        r = add.predict({"a": 2, "b": 3}, "c")
        # Step 2: 5 * 4 = 20
        r = mul.predict({"a": r, "b": 4}, "c")
        # Step 3: 20 + 5 = 25
        r = add.predict({"a": r, "b": 5}, "c")
        # Step 4: 25 * 6 = 150
        r = mul.predict({"a": r, "b": 6}, "c")
        assert abs(r - 150) < 0.01, f"Expected 150, got {r}"


# ---------------------------------------------------------------------------
# Test 5: Inhibition
# ---------------------------------------------------------------------------

class TestInhibition:
    """The inhibitor node should create competition between stripes."""

    def test_inhibitor_connected(self, brain):
        """Verify the inhibitor receives from and projects to all stripes."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        inhib = pfc["inhibitor"]
        outgoing = {e.target for e in g.edges_from(inhib)}
        incoming = {e.source for e in g.edges_to(inhib)}

        # Inhibitor should project to all WM L23 nodes.
        for stripe in ["wm0:L23", "wm1:L23", "wm2:L23"]:
            assert pfc[stripe] in outgoing, f"Inhibitor should project to {stripe}"

        # Inhibitor should receive from all WM L5 nodes.
        for stripe in ["wm0:L5", "wm1:L5", "wm2:L5"]:
            assert pfc[stripe] in incoming, f"Inhibitor should receive from {stripe}"

    def test_inhibitory_weights_negative(self, brain):
        """Inhibitor → stripe edges should have negative weights."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        inhib = pfc["inhibitor"]
        for edge in g.edges_from(inhib):
            assert edge.weight < 0, (
                f"Inhibitor edge to node {edge.target} should be negative, got {edge.weight}"
            )


# ---------------------------------------------------------------------------
# Test 6: Goal maintenance
# ---------------------------------------------------------------------------

class TestGoalMaintenance:
    """The task/goal stripe should maintain its value across many steps."""

    def test_goal_survives_long_sequence(self, brain):
        """Set a goal, run 50 decay steps, goal should still be active."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        goal = pfc["wm2:L23"]
        g.activate(goal, 1.0)

        for _ in range(50):
            g.step()

        act = g.get_node(goal).activation
        assert act > 0.1, (
            f"Goal should survive 50 steps, got activation {act:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Error monitoring
# ---------------------------------------------------------------------------

class TestErrorMonitoring:
    """The monitor should receive from WM stripes and connect to gates."""

    def test_monitor_receives_wm_outputs(self, brain):
        """Monitor L4 should receive from WM stripe outputs."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        monitor_in = pfc["monitor:L4"]
        incoming_sources = {e.source for e in g.edges_to(monitor_in)}

        assert pfc["wm0:L5"] in incoming_sources, "Monitor should receive from WM0"
        assert pfc["wm1:L5"] in incoming_sources, "Monitor should receive from WM1"

    def test_monitor_output_reaches_bg(self, brain):
        """Monitor output should connect to BG Go nodes (to trigger updates on error)."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]
        bg = brain["priors"]["basal_ganglia"]

        monitor_out = pfc["monitor:L5"]
        outgoing_targets = {e.target for e in g.edges_from(monitor_out)}

        assert bg["d1_go_0"] in outgoing_targets, "Monitor should reach BG D1 Go 0"
        assert bg["d1_go_1"] in outgoing_targets, "Monitor should reach BG D1 Go 1"


# ---------------------------------------------------------------------------
# Test 8: Sequencing
# ---------------------------------------------------------------------------

class TestSequencing:
    """The sequencer should observe stripe states and control gates."""

    def test_sequencer_observes_stripes(self, brain):
        """Sequencer input should receive from all WM stripe outputs."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        seq_in = pfc["sequencer:L4"]
        incoming_sources = {e.source for e in g.edges_to(seq_in)}

        for stripe in ["wm0:L5", "wm1:L5", "wm2:L5"]:
            assert pfc[stripe] in incoming_sources, (
                f"Sequencer should observe {stripe}"
            )

    def test_sequencer_controls_bg(self, brain):
        """Sequencer output should connect to BG Go nodes (controls which gate opens)."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]
        bg = brain["priors"]["basal_ganglia"]

        seq_out = pfc["sequencer:L5"]
        outgoing_targets = {e.target for e in g.edges_from(seq_out)}

        assert bg["d1_go_0"] in outgoing_targets, "Sequencer should reach BG D1 Go 0"
        assert bg["d1_go_1"] in outgoing_targets, "Sequencer should reach BG D1 Go 1"


# ---------------------------------------------------------------------------
# Test 9: Attention/routing
# ---------------------------------------------------------------------------

class TestAttention:
    """The system must route values to the correct operation column."""

    def test_route_to_addition(self, brain):
        """Given '+' context, the system should use the add column."""
        add = brain["add"]
        result = add.predict({"a": 100, "b": 200}, "c")
        assert result is not None and abs(result - 300) < 0.01

    def test_route_to_multiplication(self, brain):
        """Given '*' context, the system should use the mul column."""
        mul = brain["mul"]
        result = mul.predict({"a": 100, "b": 200}, "c")
        assert result is not None and abs(result - 20000) < 0.01

    def test_wrong_column_gives_wrong_answer(self, brain):
        """Using the wrong column should give the wrong answer."""
        add = brain["add"]
        # 3 * 4 through addition gives 7, not 12.
        result = add.predict({"a": 3, "b": 4}, "c")
        assert abs(result - 12) > 0.01, "Addition column should NOT give multiplication answer"


# ---------------------------------------------------------------------------
# Test 10: Distractor resistance
# ---------------------------------------------------------------------------

class TestDistractorResistance:
    """WM should maintain its value even when other nodes are activated."""

    def test_wm_survives_other_activations(self, brain):
        """Activate WM0, then activate many other nodes, WM0 should persist."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        wm0 = pfc["wm0:L23"]
        g.activate(wm0, 1.0)

        # Activate many other nodes as "distractors".
        for nid in range(min(32, g.node_count())):
            if nid != wm0:
                g.activate(nid, 0.5)

        for _ in range(5):
            g.step()

        act = g.get_node(wm0).activation
        assert act > 0.3, (
            f"WM0 should resist distractors, got {act:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 11: Graph structural integrity
# ---------------------------------------------------------------------------

class TestGraphStructure:
    """Verify the graph is correctly wired."""

    def test_total_node_count(self, brain):
        """ANS(8) + PFC(21) + BG(14) + Thalamus(9) + OutputCortex(50) + 3 columns(6 each) = 120."""
        g = brain["graph"]
        expected = 8 + 21 + 14 + 9 + 50 + 18  # ANS + PFC + BG + Thal + Output + 3 columns
        assert g.node_count() == expected, f"Expected {expected} nodes, got {g.node_count()}"

    def test_pfc_subgraph_size(self, brain):
        g = brain["graph"]
        pfc_nodes = g.nodes_in_subgraph("pfc")
        assert len(pfc_nodes) == 21, f"PFC should have 21 nodes, got {len(pfc_nodes)}"

    def test_bg_subgraph_size(self, brain):
        g = brain["graph"]
        bg_nodes = g.nodes_in_subgraph("basal_ganglia")
        assert len(bg_nodes) == 14, f"BG should have 14 nodes, got {len(bg_nodes)}"

    def test_thalamus_subgraph_size(self, brain):
        g = brain["graph"]
        thal_nodes = g.nodes_in_subgraph("thalamus")
        assert len(thal_nodes) == 9, f"Thalamus should have 9 nodes, got {len(thal_nodes)}"

    def test_ans_connected_to_pfc(self, brain):
        """ANS output should connect to PFC monitor."""
        g = brain["graph"]
        ans = brain["priors"]["ans"]
        pfc = brain["priors"]["pfc"]

        ans_out = ans["output_a_larger"]
        pfc_monitor = pfc["monitor:L4"]

        edges = g.edges_from(ans_out)
        targets = {e.target for e in edges}
        assert pfc_monitor in targets, "ANS output_a_larger should connect to PFC monitor"

    def test_all_stripes_have_self_loops(self, brain):
        """Every WM L23 node should have a self-recurrent edge."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        for stripe in ["wm0:L23", "wm1:L23", "wm2:L23"]:
            nid = pfc[stripe]
            self_edges = [e for e in g.edges_from(nid) if e.target == nid]
            assert len(self_edges) > 0, f"{stripe} should have a self-loop"
            assert self_edges[0].weight > 0.8, (
                f"{stripe} self-loop should have high weight for persistence, "
                f"got {self_edges[0].weight}"
            )

    def test_all_stripes_have_gate_edges(self, brain):
        """Every WM L23 node should receive a GATE edge from the thalamic relay."""
        g = brain["graph"]
        pfc = brain["priors"]["pfc"]

        from graph import GATE
        for stripe in ["wm0:L23", "wm1:L23", "wm2:L23"]:
            nid = pfc[stripe]
            gate_edges = [e for e in g.edges_to(nid) if e.edge_type == GATE]
            assert len(gate_edges) > 0, f"{stripe} should have a GATE edge from thalamus"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
