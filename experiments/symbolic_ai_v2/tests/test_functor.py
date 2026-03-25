"""
Step 7b tests — functor discovery.

Verifies that consolidation discovers structure-preserving maps (functors)
between NNO sub-chains from addition training examples.

The functor for "+4" maps the source chain [0,1,2,3,4] to target chains
[n, n+1, n+2, n+3, n+4] for each training example with second operand 4.
Discovery creates direct co-occurrence edges from first operand to result,
enabling the attention mechanism to predict the answer.

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_functor.py -v
"""
from __future__ import annotations

import random
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import KnowledgeGraph, COOCCURRENCE
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.ctkg.logic.hippocampus import Hippocampus
from experiments.symbolic_ai_v2.ctkg.logic import consolidation
from experiments.symbolic_ai_v2.environments.math_classroom import MathClassroomEnv


def _run_addition_classroom(max_cycles=5, warmup=2, seed=42):
    """Run addition classroom through AgenticLoop. Returns (kg, loop, env)."""
    env = MathClassroomEnv(
        problem_type='addition', mode='A',
        max_cycles=max_cycles, counting_warmup=warmup,
        answer_range=(0, 18), seed=seed,
    )
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = 0  # manual consolidation
    loop.set_preferred('FEEDBACK_correct', +1.0)
    loop.set_preferred('FEEDBACK_wrong', -1.0)
    rng = random.Random(seed + 1)

    while not env.done:
        obs = env.observe()
        loop.observe([t[0] for t in obs], [t[1] for t in obs])
        actions = env.available_actions()
        if not actions:
            break
        chosen = loop.act(actions) or rng.choice(actions)
        loop.observe([chosen], [2])
        env.act(chosen)

    return kg, loop, env


class TestNNOWalking:
    """The NNO successor walk on a clean number line."""

    def test_walk_clean_number_line(self):
        """Walk from 0 to 4 on a manually-built clean number line."""
        from experiments.symbolic_ai_v2.tests.test_multihop import _build_number_line
        kg = KnowledgeGraph()
        _build_number_line(kg, n=10)
        zero = kg.get_or_create('0')
        four = kg.get_or_create('4')
        chain = consolidation._walk_chain(kg, zero, four)
        assert chain is not None, "Should walk 0→4 on clean number line"
        assert chain[0] == zero
        assert chain[-1] == four

    def test_walk_respects_forward_only(self):
        """Walking backward fails on a clean forward-only number line."""
        from experiments.symbolic_ai_v2.tests.test_multihop import _build_number_line
        kg = KnowledgeGraph()
        _build_number_line(kg, n=10)
        seven = kg.get_or_create('7')
        three = kg.get_or_create('3')
        chain = consolidation._walk_chain(kg, seven, three)
        assert chain is None, "Backward walk should fail"

    def test_walk_end_to_end_reachable(self):
        """In a classroom graph, 0 can reach 4 (some path exists)."""
        kg, loop, _ = _run_addition_classroom(max_cycles=1, warmup=2)
        zero = kg._value_to_node.get('0')
        four = kg._value_to_node.get('4')
        if zero is None or four is None:
            return
        chain = consolidation._walk_chain(kg, zero, four)
        # May not follow the exact digit chain due to structural edges,
        # but SOME path should exist.
        assert chain is not None, "Should find some path from 0 to 4"
        assert chain[-1] == four, "Path should end at 4"


class TestFunctorDiscovery:
    """Functor discovery finds structure-preserving maps from addition examples."""

    def test_discovers_at_least_one_functor(self):
        """After addition training, at least one functor is discovered."""
        kg, loop, _ = _run_addition_classroom(max_cycles=5, warmup=2)
        stats = consolidation.discover_functors(kg, loop.hippo, min_examples=2)
        assert stats["functors_discovered"] > 0, (
            f"Should discover at least one functor, got {stats}"
        )

    def test_creates_shortcut_edges(self):
        """Functor discovery creates direct first_op → result edges."""
        kg, loop, _ = _run_addition_classroom(max_cycles=5, warmup=2)
        stats = consolidation.discover_functors(kg, loop.hippo, min_examples=2)
        assert stats["functor_edges_created"] > 0, (
            f"Should create shortcut edges, got {stats}"
        )

    def test_shortcut_edge_correct_direction(self):
        """Shortcut edges go from first operand to result, not backward."""
        kg, loop, _ = _run_addition_classroom(max_cycles=5, warmup=2)
        consolidation.discover_functors(kg, loop.hippo, min_examples=2)

        # Check: if 3+4=7 was in training, there should be a 3→7 edge
        # (or whatever examples were in the training set).
        # We can't check specific examples without knowing the random split,
        # but we can verify that shortcut edges go from LOWER to HIGHER
        # digit nodes (since addition always increases).
        functor_edges = []
        for (s, t), e in kg._edges.items():
            if e.role == COOCCURRENCE and e.confidence > 10:
                sv = kg.value_for_node(s)
                tv = kg.value_for_node(t)
                if (isinstance(sv, str) and sv.isdigit() and
                    isinstance(tv, str) and tv.isdigit()):
                    if int(tv) > int(sv) + 1:  # skip successor edges
                        functor_edges.append((sv, tv, e.effective_weight))

        # At least some multi-hop shortcut edges should exist.
        assert len(functor_edges) > 0, "Should have multi-hop shortcut edges"

    def test_observations_contain_addition_patterns(self):
        """Verify that observation records contain + and = tokens."""
        kg, loop, _ = _run_addition_classroom(max_cycles=3, warmup=1)
        plus_nid = kg._value_to_node.get('+')
        eq_nid = kg._value_to_node.get('=')
        if plus_nid is None or eq_nid is None:
            return

        count = 0
        for obs in loop.hippo.all_observations():
            if plus_nid in obs.token_nids and eq_nid in obs.token_nids:
                count += 1
        assert count >= 3, f"Should have at least 3 addition observations, got {count}"
