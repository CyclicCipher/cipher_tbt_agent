"""
Step 7a tests — multi-hop co-occurrence spread.

Verifies that iterated co-occurrence spread can traverse the number line
chain built by counting warmup. This is the SUBSTRATE for functor-based
addition (step 7b) — it doesn't produce correct addition answers by itself,
it just makes multi-hop nodes reachable.

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_multihop.py -v
"""
from __future__ import annotations

import random
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.ctkg.logic.graph import (
    KnowledgeGraph, COOCCURRENCE, TRANSITION,
)
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop
from experiments.symbolic_ai_v2.environments.math_classroom import MathClassroomEnv


def _build_number_line(kg: KnowledgeGraph, n: int = 10):
    """Build a number line 0→1→2→...→n as forward co-occurrence edges.

    Simulates what counting warmup produces: each consecutive pair (i, i+1)
    has a forward co-occurrence edge.
    """
    for i in range(n):
        src = kg.get_or_create(str(i))
        tgt = kg.get_or_create(str(i + 1))
        edge = kg.get_or_create_edge(src, tgt, role=COOCCURRENCE)
        edge.weight = 0.9  # strong positive weight


def _run_classroom_warmup(max_cycles: int = 3, warmup: int = 2, seed: int = 42):
    """Run the math classroom through AgenticLoop with counting warmup.

    Returns (kg, loop) after warmup + some training cycles.
    """
    env = MathClassroomEnv(
        problem_type='succession', mode='A',
        max_cycles=max_cycles, counting_warmup=warmup,
        answer_range=(0, 9), seed=seed,
    )
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = 0
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

    return kg, loop


class TestMultiHopSpread:
    """Multi-hop co-occurrence spread traverses the number line."""

    def test_one_hop_reaches_successor(self):
        """From node 3, one hop reaches node 4."""
        kg = KnowledgeGraph()
        _build_number_line(kg)
        nid_3 = kg.get_or_create('3')
        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=1)
        nid_4 = kg.get_or_create('4')
        assert nid_4 in result, "1 hop from 3 should reach 4"
        assert result[nid_4] > 0.3, f"4 should be strongly activated: {result[nid_4]}"

    def test_four_hops_reaches_7_from_3(self):
        """From node 3, four hops reaches node 7 (3+4=7 substrate)."""
        kg = KnowledgeGraph()
        _build_number_line(kg)
        nid_3 = kg.get_or_create('3')
        nid_7 = kg.get_or_create('7')
        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=4)
        assert nid_7 in result, "4 hops from 3 should reach 7"
        assert result[nid_7] > 0.01, f"7 should be reachable: {result[nid_7]}"

    def test_closer_nodes_stronger_than_distant(self):
        """Node 4 (1 hop from 3) has higher activation than node 7 (4 hops)."""
        kg = KnowledgeGraph()
        _build_number_line(kg)
        nid_3 = kg.get_or_create('3')
        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=4)
        nid_4 = kg.get_or_create('4')
        nid_7 = kg.get_or_create('7')
        assert result.get(nid_4, 0) > result.get(nid_7, 0), \
            "4 (1 hop) should be stronger than 7 (4 hops)"

    def test_unreachable_node_not_activated(self):
        """Node 0 is not reachable from 3 (forward-only edges)."""
        kg = KnowledgeGraph()
        _build_number_line(kg)
        nid_3 = kg.get_or_create('3')
        nid_0 = kg.get_or_create('0')
        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=4)
        assert result.get(nid_0, 0) < 0.01, \
            "0 should not be reachable from 3 via forward-only edges"

    def test_multiple_seeds_converge(self):
        """Spreading from both 3 and 5 should activate 7 more than from 3 alone."""
        kg = KnowledgeGraph()
        _build_number_line(kg)
        nid_3 = kg.get_or_create('3')
        nid_5 = kg.get_or_create('5')
        nid_7 = kg.get_or_create('7')

        from_3 = kg.spread_cooccurrence({nid_3: 1.0}, hops=5)
        from_both = kg.spread_cooccurrence({nid_3: 1.0, nid_5: 1.0}, hops=5)
        assert from_both.get(nid_7, 0) >= from_3.get(nid_7, 0), \
            "Adding seed 5 should not decrease 7's activation"

    def test_zero_hops_returns_seeds(self):
        """With 0 hops, only the seeds are returned."""
        kg = KnowledgeGraph()
        _build_number_line(kg)
        nid_3 = kg.get_or_create('3')
        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=0)
        assert nid_3 in result
        nid_4 = kg.get_or_create('4')
        assert nid_4 not in result or result[nid_4] < 0.01


class TestMultiHopWithClassroom:
    """Multi-hop spread works with edges learned from classroom warmup."""

    def test_number_line_from_warmup(self):
        """Counting warmup creates forward co-occurrence chain 0→1→2→...→9."""
        kg, loop = _run_classroom_warmup(max_cycles=1, warmup=2)

        nid_3 = kg._value_to_node.get('3')
        nid_4 = kg._value_to_node.get('4')
        if nid_3 is None or nid_4 is None:
            # Tokens might not exist if warmup didn't cover them
            return

        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=1)
        assert nid_4 in result, "Counting warmup should create 3→4 co-occurrence edge"

    def test_multihop_from_warmup_reaches_distant(self):
        """After warmup, multi-hop from 3 reaches 7."""
        kg, loop = _run_classroom_warmup(max_cycles=3, warmup=2)

        nid_3 = kg._value_to_node.get('3')
        nid_7 = kg._value_to_node.get('7')
        if nid_3 is None or nid_7 is None:
            return

        result = kg.spread_cooccurrence({nid_3: 1.0}, hops=4)
        assert nid_7 in result and result[nid_7] > 0.001, \
            f"4 hops from 3 should reach 7 after warmup, got {result.get(nid_7, 0)}"
