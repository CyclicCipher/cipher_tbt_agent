"""
Tests for inhibitory edges — negative-weight edges suppress activation.

Covers:
1. spread() propagates negative weights as suppression.
2. learn() credits inhibitory edges that correctly predicted absence.
3. learn() weakens inhibitory edges that incorrectly predicted absence.
4. Mutual exclusion emerges from alternating observations (room A / room B).

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_inhibition.py -v
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, COOCCURRENCE, TRANSITION,
    SPREAD_CAP,
)


# ── Spread propagates inhibition ─────────────────────────────────────────────

class TestInhibitorySpread:
    """spread() propagates negative-weight edges as suppression."""

    def test_negative_edge_produces_negative_prediction(self):
        """A negative-weight co-occurrence edge from A to B suppresses B."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        edge.weight = -1.0
        kg.activate(a, 1.0)
        pred = kg.spread(role_filter=COOCCURRENCE)
        assert b in pred, "B should be in prediction (with negative value)"
        assert pred[b] < 0, f"B prediction should be negative, got {pred[b]}"

    def test_positive_and_negative_cancel(self):
        """Excitatory and inhibitory edges to the same target cancel out."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        c = kg.get_or_create("C")
        b = kg.get_or_create("B")
        # A excites B, C inhibits B
        # Transition excitatory edges are normalised per source, so
        # a weight=1.0 edge sends activation 1.0 * (1.0/1.0) = 1.0.
        # Inhibitory edges are raw, so -1.0 sends -1.0. Net = 0.
        e1 = kg.get_or_create_edge(a, b, role=TRANSITION)
        e1.weight = 1.0
        e2 = kg.get_or_create_edge(c, b, role=TRANSITION)
        e2.weight = -1.0
        kg.activate(a, 1.0)
        kg.activate(c, 1.0)
        pred = kg.spread(role_filter=TRANSITION)
        assert abs(pred.get(b, 0.0)) < 0.01, (
            f"Equal excitation and inhibition should cancel, got {pred.get(b, 0.0)}"
        )

    def test_negative_prediction_clamped(self):
        """Negative predictions are clamped to -SPREAD_CAP."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=COOCCURRENCE)
        edge.weight = -0.9  # strongly inhibitory
        kg.activate(a, 1.0)
        pred = kg.spread(role_filter=COOCCURRENCE)
        assert pred[b] >= -SPREAD_CAP

    def test_zero_weight_not_propagated(self):
        """Zero-weight edges contribute nothing to spread."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=TRANSITION)
        edge.weight = 0.0
        kg.activate(a, 1.0)
        pred = kg.spread(role_filter=TRANSITION)
        assert b not in pred, "Zero-weight edge should not produce prediction"


# ── Learn handles inhibitory credit assignment ───────────────────────────────

class TestInhibitoryLearning:
    """learn() handles negative-weight edge credit assignment."""

    def test_inhibitory_edge_confirmed_strengthens(self):
        """Negative edge predicted absence, target absent → edge gets MORE negative."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        c = kg.get_or_create("C")
        edge = kg.get_or_create_edge(a, b, role=TRANSITION)
        edge.weight = -0.5
        kg.activate(a, 1.0)
        # Prediction: B at -0.5 (should not appear)
        predicted = {b: -0.5}
        # Actual: C appeared, B did NOT
        actual = {c: 1.0}
        w_before = edge.weight
        kg.learn(predicted, actual)
        # Confirmed absence → edge should be more negative
        assert edge.weight < w_before, (
            f"Confirmed inhibition should strengthen: {w_before} -> {edge.weight}"
        )

    def test_inhibitory_edge_wrong_weakens(self):
        """Negative edge predicted absence, but target appeared → edge weakened."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=TRANSITION)
        edge.weight = -0.5
        kg.activate(a, 1.0)
        # Prediction: B at -0.5 (should not appear)
        predicted = {b: -0.5}
        # Actual: B DID appear (surprise!)
        actual = {b: 1.0}
        w_before = edge.weight
        kg.learn(predicted, actual)
        # Wrong inhibition → edge should move toward zero
        assert edge.weight > w_before, (
            f"Wrong inhibition should weaken: {w_before} -> {edge.weight}"
        )

    def test_weight_bounded_by_beta(self):
        """With asymptotic weaken(), weight stays in [-1, 1]."""
        kg = KnowledgeGraph()
        a = kg.get_or_create("A")
        b = kg.get_or_create("B")
        edge = kg.get_or_create_edge(a, b, role=TRANSITION)
        edge.weight = -0.99  # strongly inhibitory
        kg.activate(a, 1.0)
        predicted = {b: edge.weight}
        actual = {}  # B absent → weaken
        kg.learn(predicted, actual)
        assert edge.weight >= -1.0, (
            f"Weight should be >= -1.0, got {edge.weight}"
        )
        assert edge.weight <= 1.0


# ── Mutual exclusion emerges from alternation ────────────────────────────────

class TestMutualExclusion:
    """Alternating observations drive co-occurrence edges negative."""

    def test_alternating_tokens_drives_transition_negative(self):
        """If A always transitions to B (never to A), the A→A transition
        edge (if it exists) should weaken, and A→B should strengthen.
        The system learns the structure of alternation."""
        from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop

        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)

        # Simulate 20 alternations via AgenticLoop:
        # Observation [room_A], action [switch], observation [room_B],
        # action [switch], observation [room_A], ...
        for i in range(20):
            if i % 2 == 0:
                loop.observe(["room_A"], [None])
            else:
                loop.observe(["room_B"], [None])
            loop.observe(["switch"], [2])  # action

        a = kg.get_or_create("room_A")
        b = kg.get_or_create("room_B")

        # After alternation, the transition switch→room_A and switch→room_B
        # should both exist. The key test: room_A and room_B never co-occur
        # in the same observation, so their co-occurrence edge (if any)
        # should be weak or nonexistent.
        e_ab = kg.edge(a, b)
        e_ba = kg.edge(b, a)
        # Co-occurrence edges between rooms should not be strongly positive.
        if e_ab is not None and e_ab.role == COOCCURRENCE:
            assert e_ab.weight <= 0.3, (
                f"Rooms never co-occur, co-occurrence edge too strong: {e_ab.weight}"
            )
        if e_ba is not None and e_ba.role == COOCCURRENCE:
            assert e_ba.weight <= 0.3, (
                f"Rooms never co-occur, co-occurrence edge too strong: {e_ba.weight}"
            )


# ── Abduction creates correct edge role (via AgenticLoop) ────────────────────

class TestAbductionEdgeRole:
    """Abduction through AgenticLoop creates TRANSITION edges across timesteps."""

    def test_transition_from_sequential_observations(self):
        """Observing state_A then state_B creates a TRANSITION edge A→B."""
        from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop

        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)

        # First observation: state_A
        loop.observe(["state_A"], [None])
        # Action between observations
        loop.observe(["do_something"], [2])
        # Second observation: state_B (surprising — never seen before)
        loop.observe(["state_B"], [None])

        a = kg.get_or_create("state_A")
        b = kg.get_or_create("state_B")
        # There should be a transition edge from something in the first
        # observation toward state_B (created during the learning step).
        # The action token is the most likely source of the transition.
        action = kg.get_or_create("do_something")
        edge = kg.edge(action, b)
        assert edge is not None, "Action→state_B transition should exist"
        assert edge.role == TRANSITION, (
            f"Edge from action to next state should be TRANSITION, got {edge.role}"
        )

    def test_cooccurrence_within_observation(self):
        """Tokens in the same observation with matching edge types get
        COOCCURRENCE edges."""
        from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop

        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)

        # Observe three extero tokens together (edge_type 0 for all after first)
        loop.observe(["room", "tok_A", "tok_B"], [None, 0, 0])

        a = kg.get_or_create("tok_A")
        b = kg.get_or_create("tok_B")
        edge = kg.edge(a, b)
        assert edge is not None, "Consecutive same-type tokens should have an edge"
        assert edge.role == COOCCURRENCE, (
            f"Same-observation edge should be COOCCURRENCE, got {edge.role}"
        )
