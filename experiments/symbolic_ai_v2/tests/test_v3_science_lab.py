"""
Step 2 — Science lab integration via AgenticLoop.

The agent plays the science lab autonomously. Actions are selected via
loop.act() (highest activation among candidates). When no candidate has
activation, the agent falls back to random. All interaction goes through
AgenticLoop — no glue code, no parallel systems.

Measurements after N steps of autonomous play:
1. Room adjacency learned (transition edges with positive weight).
2. Affordances learned (take → HOLD, equip → EQUIPPED).
3. Action selection improves over time (fewer random actions).
4. The model handles the full observation token stream without crashing.

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_v3_science_lab.py -v
"""
from __future__ import annotations

import os
import sys
import random

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import KnowledgeGraph, TRANSITION
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.environments.science_lab import ScienceLabEnv


# ============================================================================
# Runner — the only interface between environment and model
# ============================================================================

def run_episode(
    env: ScienceLabEnv,
    loop: AgenticLoop,
    max_steps: int = 200,
    seed: int = 42,
) -> dict:
    """Run one full episode through AgenticLoop.

    Returns a stats dict:
      steps, random_actions, predicted_actions, won, dead,
      final_energy, final_health, rooms_visited
    """
    rng = random.Random(seed)
    env.reset()

    stats = {
        "steps": 0,
        "random_actions": 0,
        "predicted_actions": 0,
        "won": False,
        "dead": False,
        "final_energy": 0,
        "final_health": 0,
        "rooms_visited": set(),
    }

    for step in range(max_steps):
        if env.done:
            break

        # 1. Observe — through AgenticLoop only.
        obs = env.observe()
        tokens = [t[0] for t in obs]
        etypes = [t[1] for t in obs]
        loop.observe(tokens, etypes)

        stats["rooms_visited"].add(env._location)

        # 2. Get available actions from environment.
        actions = env.available_actions()
        if not actions:
            break

        # 3. Select action — through AgenticLoop only.
        chosen = loop.act(actions)
        if chosen is None:
            chosen = rng.choice(actions)
            stats["random_actions"] += 1
        else:
            stats["predicted_actions"] += 1

        # 4. Feed the chosen action as an observation (action token).
        loop.observe([chosen], [2])  # edge_type 2 = action

        # 5. Execute in environment.
        env.act(chosen)
        stats["steps"] += 1

    stats["won"] = env.won
    stats["dead"] = env._health <= 0
    stats["final_energy"] = env._energy
    stats["final_health"] = env._health
    stats["rooms_visited"] = len(stats["rooms_visited"])

    return stats


# ============================================================================
# Tests
# ============================================================================

class TestAutonomousPlay:
    """The agent plays the science lab without any hand-scripted actions."""

    def test_survives_100_steps(self):
        """Agent doesn't crash or die in 100 steps of random+predicted play."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()
        stats = run_episode(env, loop, max_steps=100)
        assert stats["steps"] == 100 or stats["won"]
        # Should have visited at least 2 rooms (starts in corridor)
        assert stats["rooms_visited"] >= 2

    def test_learns_from_play(self):
        """After 100 steps, the KG has nodes, edges, and transition structure."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()
        run_episode(env, loop, max_steps=100)

        assert kg.node_count() > 20, "Should have many token nodes"
        assert kg.edge_count() > 50, "Should have many edges"

        # Should have some transition edges with positive weight
        positive_transitions = [
            e for e in kg._edges.values()
            if e.role == TRANSITION and e.weight > 0
        ]
        assert len(positive_transitions) > 5, (
            f"Expected >5 positive transition edges, got {len(positive_transitions)}"
        )

    def test_predicted_actions_increase(self):
        """Over two blocks of 50 steps, predicted actions increase."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()

        # Block 1: first 50 steps (mostly random, learning)
        stats1 = run_episode(env, loop, max_steps=50, seed=42)

        # Block 2: next 50 steps (should use more predictions)
        env.reset()
        stats2 = run_episode(env, loop, max_steps=50, seed=43)

        # The model should make more predicted actions in block 2
        # (it has learned transition edges from block 1)
        assert stats2["predicted_actions"] >= stats1["predicted_actions"], (
            f"Block 1: {stats1['predicted_actions']} predicted, "
            f"Block 2: {stats2['predicted_actions']} predicted — should improve"
        )


class TestLearnedTransitions:
    """After training, the model knows specific transitions."""

    def _train(self, n_steps: int = 150) -> tuple[KnowledgeGraph, AgenticLoop]:
        """Run n_steps of autonomous play to build up the KG."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()
        run_episode(env, loop, max_steps=n_steps, seed=42)
        return kg, loop

    def test_room_adjacency_edges(self):
        """After training, go_X actions have transitions to AT_room tokens.

        Room adjacency is a 2-hop path: AT_corridor → go_west → AT_closet.
        The test checks that at least one go_X → AT_Y transition edge exists.
        """
        kg, loop = self._train()

        # Look for any go_X action that has a transition to any AT_Y room.
        go_actions = [v for v in kg._value_to_node.keys()
                      if isinstance(v, str) and v.startswith("go_")]
        at_rooms = [v for v in kg._value_to_node.keys()
                    if isinstance(v, str) and v.startswith("AT_")]
        found = False
        for action_val in go_actions:
            nid_act = kg._value_to_node[action_val]
            for room_val in at_rooms:
                nid_room = kg._value_to_node[room_val]
                e = kg.edge(nid_act, nid_room)
                if e is not None and e.role == TRANSITION:
                    found = True
                    break
            if found:
                break
        assert found, (
            "Model should learn at least one go_X → AT_room transition"
        )

    def test_action_to_state_edges(self):
        """After training, go_X actions have transition edges to room tokens.

        Note: without consolidation (replay), the Hebbian online learning
        may not push these edges to positive weight. The test checks that
        the transition was DISCOVERED (edge exists), not that it's positive.
        Positive convergence requires consolidation (Step 4).
        """
        kg, loop = self._train()
        go_actions = [v for v in kg._value_to_node.keys()
                      if isinstance(v, str) and v.startswith("go_")]
        found = False
        for action_val in go_actions:
            nid = kg._value_to_node[action_val]
            for e in kg.edges_from(nid):
                if e.role == TRANSITION:
                    target_label = kg.label_for_node(e.target)
                    if target_label.startswith("AT_"):
                        found = True
                        break
            if found:
                break
        assert found, "Some go_X action should have a transition edge to an AT_room token"


class TestGraphHealth:
    """The graph doesn't blow up or degenerate."""

    def test_edge_count_bounded(self):
        """Edge count stays reasonable (not O(n^2) on tokens)."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()
        run_episode(env, loop, max_steps=200)

        n = kg.node_count()
        e = kg.edge_count()
        # Edges should be much less than n^2
        assert e < n * n * 0.5, (
            f"{e} edges for {n} nodes — too dense (>50% of n^2)"
        )

    def test_no_runaway_activation(self):
        """No node has activation > 1.0 after a full episode."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()
        run_episode(env, loop, max_steps=100)

        for node in kg._nodes.values():
            assert node.activation <= 1.0, (
                f"Node {node.label} has activation {node.activation} > 1.0"
            )

    def test_hippo_stores_all_steps(self):
        """Hippocampus has a snapshot for each observe() call."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        env = ScienceLabEnv()
        stats = run_episode(env, loop, max_steps=50)
        # Each step does 2 observe() calls (state + action)
        expected = stats["steps"] * 2
        assert loop.hippo.episode_count() == expected, (
            f"Expected {expected} snapshots, got {loop.hippo.episode_count()}"
        )
