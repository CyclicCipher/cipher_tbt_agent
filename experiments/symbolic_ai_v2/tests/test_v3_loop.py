"""
Tests for the v3 neocortical graph — minimal viable loop.

Tests the single process (spread → compare → update) on:
1. Basic activation and decay.
2. Co-occurrence edge creation from observations.
3. Transition edge creation between timesteps.
4. Learning: spread predicts, Hebbian updates strengthen/weaken.
5. Action selection via activation spread.
6. Science lab integration: does the system learn room adjacency?

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_v3_loop.py -v
"""
from __future__ import annotations

import os
import sys
import random

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, COOCCURRENCE, TRANSITION, ACTIVATION_THRESHOLD,
)
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus


# ============================================================================
# Graph basics
# ============================================================================

class TestGraphBasics:

    def test_get_or_create_identity(self):
        """Same value → same node."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("hello")
        b = kg.get_or_create("hello")
        assert a == b
        assert kg.node_count() == 1

    def test_different_values_different_nodes(self):
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        assert a != b
        assert kg.node_count() == 2

    def test_activate_and_decay(self):
        kg = KnowledgeGraph()
        nid = kg.get_or_create("X")
        kg.activate(nid, 1.0)
        assert kg.node(nid).activation == 1.0
        kg.decay()
        assert kg.node(nid).activation < 1.0
        assert kg.node(nid).activation > 0.0  # one step isn't enough to kill it

    def test_decay_to_zero(self):
        """Repeated decay eventually kills activation."""
        kg = KnowledgeGraph()
        nid = kg.get_or_create("X")
        kg.activate(nid, 1.0)
        for _ in range(100):
            kg.decay()
        assert kg.node(nid).activation == 0.0

    def test_active_nodes(self):
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        c = kg.get_or_create("C")
        kg.activate(a)
        kg.activate(b)
        active = kg.active_nodes()
        assert a in active
        assert b in active
        assert c not in active


# ============================================================================
# Edges
# ============================================================================

class TestEdges:

    def test_edge_creation(self):
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b)
        assert edge.source == a
        assert edge.target == b
        assert edge.weight == 0.0

    def test_edge_identity(self):
        """Same (src, tgt) → same edge object."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        e1 = kg.get_or_create_edge(a, b)
        e2 = kg.get_or_create_edge(a, b)
        assert e1 is e2

    def test_directed(self):
        """A→B and B→A are different edges."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        e1 = kg.get_or_create_edge(a, b)
        e2 = kg.get_or_create_edge(b, a)
        assert e1 is not e2


# ============================================================================
# Spread (= prediction)
# ============================================================================

class TestSpread:

    def test_spread_propagates(self):
        """Active source with positive edge → target gets predicted."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b)
        edge.weight = 1.0
        kg.activate(a)
        predicted = kg.spread()
        assert b in predicted
        assert predicted[b] > 0

    def test_spread_zero_weight(self):
        """Edge with weight 0 → no spread."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        kg.get_or_create_edge(a, b)  # weight = 0
        kg.activate(a)
        predicted = kg.spread()
        assert b not in predicted

    def test_spread_inactive_source(self):
        """Inactive source → no spread even with positive edge."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b)
        edge.weight = 1.0
        # a is not activated
        predicted = kg.spread()
        assert b not in predicted


# ============================================================================
# Learn (= Hebbian update)
# ============================================================================

class TestLearn:

    def test_confirmed_edge_strengthens(self):
        """Predicted and observed → edge weight increases."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b)
        edge.weight = 0.5
        kg.activate(a)
        predicted = {b: 0.5}  # we predicted B
        actual = {b: 1.0}     # B was observed
        kg.learn(predicted, actual)
        assert edge.weight > 0.5

    def test_wrong_prediction_weakens(self):
        """Predicted but not observed → TRANSITION edge weight decreases.

        Co-occurrence edges are NOT weakened by absence (absence from one
        observation doesn't disprove co-occurrence). Only transition edges
        — temporal predictions — are weakened when wrong.
        """
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=TRANSITION)
        edge.weight = 0.5
        kg.activate(a)
        predicted = {b: 0.5}  # we predicted B
        actual = {}            # B was NOT observed
        kg.learn(predicted, actual, prev_active={a: 1.0})
        assert edge.weight < 0.5

    def test_surprise_creates_edge(self):
        """Observed but not predicted → new edge created (abduction)."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        kg.activate(a)
        predicted = {}         # predicted nothing
        actual = {b: 1.0}     # but B appeared
        kg.learn(predicted, actual)
        edge = kg.edge(a, b)
        assert edge is not None
        assert edge.weight > 0


# ============================================================================
# Loop integration
# ============================================================================

class TestLoop:

    def test_observe_creates_nodes(self):
        """Observing tokens creates nodes in the KG."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.observe(["AT_corridor", "SEE_wrench", "ENERGY_sated"])
        assert kg.node_count() == 3

    def test_observe_creates_cooccurrence_edges(self):
        """Tokens with the same edge type get forward co-occurrence edges."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        # Explicitly set all tokens to edge_type 0 so they're in the same group.
        loop.observe(["A", "B", "C"], [0, 0, 0])
        # Causal masking: A→B, A→C, B→C (forward only, no reverse).
        assert kg.edge_count() >= 2  # at least A→B and B→C

    def test_observe_creates_transition_edges(self):
        """Tokens from step t to step t+1 get transition edges."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.observe(["AT_corridor"])
        loop.observe(["AT_closet"])
        # AT_corridor → AT_closet should have a transition edge
        nid_corridor = kg.get_or_create("AT_corridor")
        nid_closet = kg.get_or_create("AT_closet")
        edge = kg.edge(nid_corridor, nid_closet)
        assert edge is not None
        assert edge.role == TRANSITION

    def test_repeated_observation_strengthens(self):
        """Seeing the same transition twice makes the edge stronger."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        for _ in range(5):
            loop.observe(["AT_corridor"])
            loop.observe(["AT_closet"])
        nid_corridor = kg.get_or_create("AT_corridor")
        nid_closet = kg.get_or_create("AT_closet")
        edge = kg.edge(nid_corridor, nid_closet)
        assert edge is not None
        assert edge.weight > 0.1  # strengthened over multiple observations

    def test_hippo_stores_snapshots(self):
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.observe(["A", "B"])
        loop.observe(["C", "D"])
        assert loop.hippo.episode_count() == 2

    def test_act_always_returns_something(self):
        """EFE-based selection always picks an action (epistemic drive)."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.observe(["AT_corridor", "SEE_wrench"])
        result = loop.act(["go_west", "take_wrench"])
        # With EFE, unknown actions have epistemic value — always selects.
        assert result is not None

    def test_act_after_learning_prefers_pragmatic(self):
        """After learning that an action leads to a preferred state,
        the model prefers that action over unknown alternatives."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        # Set a preference: we WANT to be AT_closet.
        loop.set_preferred("AT_closet", 1.0)
        # Train: observe state, then observe action token, then new state.
        for _ in range(20):
            loop.observe(["AT_corridor"])
            loop.observe(["go_west"])
            loop.observe(["AT_closet"])
        # Now when we're at corridor, go_west has pragmatic value (leads
        # to preferred AT_closet). go_east/go_north are unknown.
        loop.observe(["AT_corridor"])
        result = loop.act(["go_west", "go_east", "go_north"])
        assert result == "go_west", (
            f"Agent should prefer go_west (leads to preferred state), got {result}"
        )


# ============================================================================
# Science lab integration
# ============================================================================

class TestScienceLabIntegration:

    def test_learn_room_adjacency(self):
        """After 20 round-trips, the model learns corridor↔closet."""
        from experiments.symbolic_ai_v2.environments.science_lab import ScienceLabEnv

        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()

        # Train: 20 round-trips corridor → supply_closet → corridor
        for _ in range(20):
            obs = env.observe()
            tokens = [t[0] for t in obs]
            etypes = [t[1] for t in obs]
            loop.observe(tokens, etypes)

            # Action: go_west
            loop.observe(["go_west"])
            env.act("go_west")

            obs = env.observe()
            tokens = [t[0] for t in obs]
            etypes = [t[1] for t in obs]
            loop.observe(tokens, etypes)

            # Action: go_east (back)
            loop.observe(["go_east"])
            env.act("go_east")

        # Test: from corridor, does the model predict supply_closet tokens
        # after go_west?
        obs = env.observe()
        tokens = [t[0] for t in obs]
        loop.observe(tokens)
        loop.observe(["go_west"])

        # Check: AT_supply_closet node should have activation from spread.
        nid_closet = kg.get_or_create("AT_supply_closet")
        # After observing go_west from corridor, spread should activate
        # AT_supply_closet (learned transition).
        predicted = kg.spread()
        assert nid_closet in predicted, (
            f"Model failed to predict AT_supply_closet after go_west from corridor. "
            f"Predicted: {[kg.label_for_node(n) for n in list(predicted.keys())[:10]]}"
        )

    def test_affordance_take(self):
        """After taking items several times, model learns take → HOLD."""
        from experiments.symbolic_ai_v2.environments.science_lab import ScienceLabEnv

        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()

        # Repeatedly take and drop keycard_blue
        for _ in range(10):
            obs = env.observe()
            loop.observe([t[0] for t in obs])

            loop.observe(["take_keycard_blue"])
            env.act("take_keycard_blue")

            obs = env.observe()
            loop.observe([t[0] for t in obs])

            loop.observe(["drop_keycard_blue"])
            env.act("drop_keycard_blue")

        # Test: after take_keycard_blue, HOLD_keycard_blue should be predicted
        obs = env.observe()
        loop.observe([t[0] for t in obs])
        loop.observe(["take_keycard_blue"])

        nid_hold = kg.get_or_create("HOLD_keycard_blue")
        predicted = kg.spread()
        assert nid_hold in predicted, (
            "Model failed to predict HOLD_keycard_blue after take_keycard_blue"
        )
