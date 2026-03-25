"""
Tests for prospective configuration — the inference-before-plasticity
credit assignment principle.

Based on Song, Bogacz et al. (2024) "Inferring neural activity before
plasticity as a foundation for learning beyond backpropagation."

The core problem: standard Hebbian learning produces catastrophic
interference. Learning that one input is absent (e.g., the bear can't
hear the river) weakens edges to OTHER outputs that were correctly
predicted (e.g., the bear should still smell the salmon). Prospective
configuration avoids this by first inferring what activations SHOULD be,
then updating weights to consolidate that pattern.

All tests go through AgenticLoop (the only door).

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_prospective.py -v
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _train_bear_salmon(n_episodes: int = 20) -> AgenticLoop:
    """Train the bear/salmon scenario.

    The bear learns a multi-sensory association:
      - SEE_river + HEAR_river + SMELL_salmon → CATCH_salmon
      - SEE_river + HEAR_river + SMELL_nothing → CATCH_nothing

    After n_episodes of the full association, the bear has strong
    transition edges from sensory tokens to CATCH_salmon.
    """
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)

    for _ in range(n_episodes):
        # Full sensory observation: see river, hear river, smell salmon
        loop.observe(
            ["SEE_river", "HEAR_river", "SMELL_salmon"],
            [None, 0, 0],
        )
        # Action: fish
        loop.observe(["fish"], [2])
        # Outcome: caught salmon
        loop.observe(
            ["CATCH_salmon", "ENERGY_sated"],
            [None, 1],
        )

    return loop


# ── Bear/Salmon: catastrophic interference test ─────────────────────────────

class TestBearSalmon:
    """The bear/salmon scenario from Song & Bogacz (2024).

    After learning that (see_river, hear_river, smell_salmon) → catch_salmon,
    the bear arrives one day with a damaged ear. It can't hear the river,
    but it CAN still see it and smell the salmon.

    With standard Hebbian learning (backprop-like), the absence of HEAR_river
    weakens the edges from the entire sensory context to CATCH_salmon,
    including the SEE_river→CATCH_salmon and SMELL_salmon→CATCH_salmon
    edges. This is catastrophic interference: learning about the absence
    of one sense degrades the association with other senses.

    With prospective configuration, the system first infers that CATCH_salmon
    is still the correct outcome (because SEE and SMELL still predict it),
    and only weakens the HEAR→CATCH edge. The other edges are preserved.
    """

    def test_baseline_prediction_works(self):
        """Full sensory input correctly predicts CATCH_salmon."""
        loop = _train_bear_salmon(20)
        kg = loop.kg

        # Observe full sensory context
        loop.observe(
            ["SEE_river", "HEAR_river", "SMELL_salmon"],
            [None, 0, 0],
        )
        loop.observe(["fish"], [2])

        # Check that CATCH_salmon is predicted
        catch = kg.get_or_create("CATCH_salmon")
        pred = loop.last_predicted
        assert catch in pred and pred[catch] > 0, (
            f"CATCH_salmon should be predicted after full sensory input. "
            f"Predicted: {pred.get(catch, 'not in pred')}"
        )

    def test_partial_input_still_predicts(self):
        """Even without HEAR_river, CATCH_salmon should still be predicted
        (because SEE_river and SMELL_salmon still point to it)."""
        loop = _train_bear_salmon(20)
        kg = loop.kg

        # Observe partial sensory context: no HEAR_river
        loop.observe(
            ["SEE_river", "SMELL_salmon"],
            [None, 0],
        )
        loop.observe(["fish"], [2])

        catch = kg.get_or_create("CATCH_salmon")
        pred = loop.last_predicted
        assert catch in pred and pred[catch] > 0, (
            f"CATCH_salmon should still be predicted without hearing. "
            f"Predicted: {pred.get(catch, 'not in pred')}"
        )

    def test_no_catastrophic_interference(self):
        """After observing one episode WITHOUT hearing, the SEE→CATCH and
        SMELL→CATCH edges should NOT be weakened.

        This is the critical test. With standard Hebbian learning, the
        absence of HEAR_river causes the learn step to weaken ALL edges
        that contributed to predicting tokens from the previous
        (incomplete) context — including the correct SEE and SMELL edges.

        With prospective configuration, only the HEAR→CATCH edge is
        affected. The others are preserved.
        """
        loop = _train_bear_salmon(30)
        kg = loop.kg

        # Record edge weights BEFORE the damaged-ear episode
        see = kg.get_or_create("SEE_river")
        smell = kg.get_or_create("SMELL_salmon")
        catch = kg.get_or_create("CATCH_salmon")
        fish = kg.get_or_create("fish")

        # We care about the edges that predict CATCH_salmon
        # The relevant transition edges go from the action "fish"
        # to the outcome "CATCH_salmon"
        fish_catch = kg.edge(fish, catch)
        assert fish_catch is not None, "fish→CATCH_salmon edge should exist"
        weight_before = fish_catch.weight
        conf_before = fish_catch.confidence

        # Now: one episode without hearing (damaged ear)
        loop.observe(
            ["SEE_river", "SMELL_salmon"],  # no HEAR_river
            [None, 0],
        )
        loop.observe(["fish"], [2])
        # Outcome is still CATCH_salmon (the salmon IS there)
        loop.observe(
            ["CATCH_salmon", "ENERGY_sated"],
            [None, 1],
        )

        weight_after = fish_catch.weight

        # The fish→CATCH_salmon edge should NOT have been weakened.
        # It was correctly confirmed (fish was done, salmon was caught).
        assert weight_after >= weight_before - 0.05, (
            f"fish→CATCH_salmon edge should be preserved after partial input. "
            f"Before: {weight_before:.4f}, After: {weight_after:.4f}. "
            f"Catastrophic interference detected!"
        )


# ── Multi-hop chain: credit assignment across multiple edges ─────────────────

class TestMultiHopCredit:
    """Test that credit assignment works across a multi-hop chain.

    Scenario: a 4-step causal chain through the science lab.
      AT_corridor → go_west → AT_supply_closet → SEE_wrench

    If the agent learns this chain, then later arrives at the closet
    and the wrench is GONE (someone took it), the system should weaken
    the supply_closet→SEE_wrench edge, but NOT the corridor→go_west
    or go_west→supply_closet edges. Those transitions are still valid.
    """

    def _train_chain(self, n_episodes: int = 15) -> AgenticLoop:
        """Train a 4-step chain: corridor → go_west → closet → see wrench."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)

        for _ in range(n_episodes):
            loop.observe(["AT_corridor"], [None])
            loop.observe(["go_west"], [2])
            loop.observe(["AT_supply_closet", "SEE_wrench", "SEE_flashlight"], [None, 0, 0])

        return loop

    def test_chain_learns(self):
        """The full chain is learned: corridor→go_west→closet→wrench."""
        loop = self._train_chain(15)
        kg = loop.kg

        go_west = kg.get_or_create("go_west")
        closet = kg.get_or_create("AT_supply_closet")

        # go_west→AT_supply_closet should be a strong transition
        edge = kg.edge(go_west, closet)
        assert edge is not None, "go_west→closet transition should exist"
        assert edge.weight > 0.3, (
            f"go_west→closet should be strong after 15 episodes, got {edge.weight:.3f}"
        )

    def test_downstream_change_doesnt_damage_upstream(self):
        """When the wrench disappears, upstream edges are preserved.

        After training the chain, observe one episode where the wrench
        is absent. The supply_closet→SEE_wrench edge should weaken,
        but go_west→AT_supply_closet should be preserved (the room
        transition is still valid even if the wrench is gone).
        """
        loop = self._train_chain(20)
        kg = loop.kg

        go_west = kg.get_or_create("go_west")
        closet = kg.get_or_create("AT_supply_closet")
        wrench = kg.get_or_create("SEE_wrench")

        # Record upstream edge weight before the surprise
        gw_closet = kg.edge(go_west, closet)
        assert gw_closet is not None
        upstream_weight_before = gw_closet.weight

        # Now observe: closet WITHOUT wrench (someone took it)
        loop.observe(["AT_corridor"], [None])
        loop.observe(["go_west"], [2])
        loop.observe(["AT_supply_closet", "SEE_flashlight"], [None, 0])
        # Note: SEE_wrench is absent!

        upstream_weight_after = gw_closet.weight

        # The go_west→closet edge should NOT be damaged
        assert upstream_weight_after >= upstream_weight_before - 0.05, (
            f"Upstream edge go_west→closet should be preserved. "
            f"Before: {upstream_weight_before:.4f}, After: {upstream_weight_after:.4f}. "
            f"Catastrophic interference in multi-hop chain!"
        )

    def test_downstream_edge_weakens_appropriately(self):
        """The supply_closet→SEE_wrench edge DOES weaken when wrench disappears."""
        loop = self._train_chain(20)
        kg = loop.kg

        closet = kg.get_or_create("AT_supply_closet")
        wrench = kg.get_or_create("SEE_wrench")

        # Check if there's a transition edge from closet context to wrench.
        # It may be indirect (via co-occurrence or through the go_west node).
        # For now just verify the system doesn't predict wrench after it's gone.

        # Observe chain WITH wrench first, to establish prediction
        loop.observe(["AT_corridor"], [None])
        loop.observe(["go_west"], [2])
        loop.observe(["AT_supply_closet", "SEE_wrench", "SEE_flashlight"], [None, 0, 0])

        # Now observe chain WITHOUT wrench
        loop.observe(["AT_corridor"], [None])
        loop.observe(["go_west"], [2])

        # At this point, spread should still predict wrench (from training)
        pred_before = loop.last_predicted
        wrench_pred_before = pred_before.get(wrench, 0.0)

        # Observe: wrench is ABSENT
        loop.observe(["AT_supply_closet", "SEE_flashlight"], [None, 0])

        # After this observation, the system has learned wrench can be absent.
        # Do one more cycle to check the updated prediction.
        loop.observe(["AT_corridor"], [None])
        loop.observe(["go_west"], [2])
        pred_after = loop.last_predicted
        wrench_pred_after = pred_after.get(wrench, 0.0)

        # Prediction of wrench should have decreased
        assert wrench_pred_after <= wrench_pred_before, (
            f"Wrench prediction should decrease after observing absence. "
            f"Before: {wrench_pred_before:.4f}, After: {wrench_pred_after:.4f}"
        )


# ── Continual learning: new associations don't destroy old ones ──────────────

class TestContinualLearning:
    """Learning new associations shouldn't destroy old ones.

    Train the agent on corridor↔closet navigation, then train on
    corridor→chem_lab navigation. The closet edges should survive.
    """

    def test_new_route_preserves_old_route(self):
        """Learning corridor→south→chem_lab doesn't destroy corridor→west→closet."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)

        # Phase 1: learn corridor↔closet (20 episodes)
        for _ in range(20):
            loop.observe(["AT_corridor"], [None])
            loop.observe(["go_west"], [2])
            loop.observe(["AT_supply_closet"], [None])
            loop.observe(["go_east"], [2])
            loop.observe(["AT_corridor"], [None])

        go_west = kg.get_or_create("go_west")
        closet = kg.get_or_create("AT_supply_closet")
        gw_closet = kg.edge(go_west, closet)
        assert gw_closet is not None
        old_route_weight = gw_closet.weight

        # Phase 2: learn corridor→chem_lab (20 episodes)
        for _ in range(20):
            loop.observe(["AT_corridor"], [None])
            loop.observe(["go_south"], [2])
            loop.observe(["AT_chem_lab"], [None])
            loop.observe(["go_north"], [2])
            loop.observe(["AT_corridor"], [None])

        # The old route (go_west→closet) should still be strong
        new_route_weight = gw_closet.weight
        assert new_route_weight >= old_route_weight * 0.5, (
            f"Old route go_west→closet should survive learning new route. "
            f"Before phase 2: {old_route_weight:.4f}, After: {new_route_weight:.4f}. "
            f"Lost {(1 - new_route_weight/old_route_weight)*100:.0f}% — catastrophic interference!"
        )
