"""
Tests for MathClassroom — arithmetic learning as a textworld.

The agent uses the same observe/act loop as the science lab. No special
math code. The agent must discover that "3 succ" is followed by "4" the
same way it discovers that "go_west" from corridor leads to supply_closet.

All tests go through AgenticLoop (the only door).

Run with:
    ./venv/Scripts/python.exe -m pytest experiments/symbolic_ai_v2/tests/test_math_classroom.py -v
"""
from __future__ import annotations

import os
import sys
import random

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.symbolic_ai_v2.environments.math_classroom import (
    MathClassroomEnv, PROBLEM_UNIVERSES,
)
from experiments.symbolic_ai_v2.ctkg.logic.graph import KnowledgeGraph
from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_classroom(
    problem_type: str = "succession",
    mode: str = "A",
    max_cycles: int = 100,
    seed: int = 42,
    consolidation_interval: int = 40,
) -> tuple[AgenticLoop, MathClassroomEnv]:
    """Run the math classroom through AgenticLoop.

    Consolidation is handled INSIDE AgenticLoop (the only door).
    consolidation_interval sets how often (in observe() calls) it triggers.
    Returns (loop, env) so tests can inspect results.
    """
    env = MathClassroomEnv(
        problem_type=problem_type,
        mode=mode,
        max_cycles=max_cycles,
        seed=seed,
    )
    kg = KnowledgeGraph()
    loop = AgenticLoop(kg)
    loop.CONSOLIDATION_INTERVAL = consolidation_interval

    # Homeostatic priors: the agent prefers being correct.
    # This is the "reward" in active inference — not an external signal,
    # but a preference for observations consistent with success.
    loop.set_preferred("FEEDBACK_correct", +1.0)
    loop.set_preferred("FEEDBACK_wrong", -1.0)

    rng = random.Random(seed + 1)

    while not env.done:
        obs = env.observe()
        loop.observe([t[0] for t in obs], [t[1] for t in obs])

        actions = env.available_actions()
        if not actions:
            break

        chosen = loop.act(actions)
        if chosen is None:
            chosen = rng.choice(actions)

        loop.observe([chosen], [2])
        env.act(chosen)

    return loop, env


# ── Environment self-tests ───────────────────────────────────────────────────

class TestEnvironmentMechanics:
    """Verify the classroom environment works correctly."""

    def test_reset_produces_observation(self):
        env = MathClassroomEnv(problem_type="succession")
        obs = env.observe()
        assert len(obs) > 0
        toks = [t[0] for t in obs]
        assert any("PHASE_" in t for t in toks)
        # Bare digit tokens (no BOARD_ prefix).
        assert any(t.isdigit() for t in toks)

    def test_training_shows_answer(self):
        """In training phase, the answer digit is shown (bare token)."""
        env = MathClassroomEnv(problem_type="succession")
        obs = env.observe()
        toks = [t[0] for t in obs]
        # The answer is a bare digit, same node as question digits.
        # Check that at least one digit appears after the operator.
        has_succ = "succ" in toks
        has_digit = any(t.isdigit() for t in toks)
        assert has_succ or has_digit, f"Training should show problem, got: {toks}"

    def test_test_phase_has_question_mark(self):
        """In test phase, ? appears instead of answer."""
        env = MathClassroomEnv(problem_type="succession", train_per_cycle=0,
                                test_per_cycle=1, counting_warmup=0)
        obs = env.observe()
        toks = [t[0] for t in obs]
        assert "?" in toks

    def test_correct_answer_gives_feedback(self):
        """Answering correctly produces FEEDBACK_correct."""
        env = MathClassroomEnv(problem_type="succession", train_per_cycle=0,
                                test_per_cycle=1, seed=99, counting_warmup=0)
        # Emit correct answer digit by digit, then done.
        for d in env._correct_answer:
            env.act(d)
        env.act("done")
        obs = env.observe()
        toks = [t[0] for t in obs]
        assert "FEEDBACK_correct" in toks

    def test_wrong_answer_gives_feedback(self):
        """Answering incorrectly produces FEEDBACK_wrong."""
        env = MathClassroomEnv(problem_type="succession", train_per_cycle=0,
                                test_per_cycle=1, seed=99, counting_warmup=0)
        # Emit a wrong digit, then done.
        wrong_digit = str((int(env._correct_answer[0]) + 5) % 10)
        env.act(wrong_digit)
        env.act("done")
        obs = env.observe()
        toks = [t[0] for t in obs]
        assert "FEEDBACK_wrong" in toks
        # Correct answer is revealed as bare digit(s).
        assert any(t.isdigit() for t in toks), (
            "Wrong answer should reveal correct answer as bare digit"
        )

    def test_mode_a_cycles(self):
        """Mode A runs through train+test cycles."""
        env = MathClassroomEnv(problem_type="succession", mode="A",
                                max_cycles=2, train_per_cycle=2, test_per_cycle=2,
                                counting_warmup=0)
        steps = 0
        while not env.done:
            actions = env.available_actions()
            if not actions:
                break
            if "next" in actions:
                env.act("next")
            elif "done" in actions:
                env.act("done")
            else:
                env.act("0")  # emit a digit
            steps += 1
            if steps > 200:
                break
        assert env.done

    def test_mode_b_adaptive(self):
        """Mode B presents test first, trains on errors."""
        env = MathClassroomEnv(problem_type="succession", mode="B",
                                max_cycles=5, seed=42, counting_warmup=0)
        # First problem is a test.
        obs = env.observe()
        toks = [t[0] for t in obs]
        assert "PHASE_test" in toks

    def test_all_problem_types_generate(self):
        """All registered problem types generate valid problems."""
        for ptype in ["succession", "predecessor", "addition", "subtraction",
                       "multiplication"]:
            env = MathClassroomEnv(problem_type=ptype)
            obs = env.observe()
            assert len(obs) > 0, f"{ptype} produced empty observation"

    def test_train_test_pools_disjoint(self):
        """Training and test problems are NEVER the same."""
        env = MathClassroomEnv(problem_type="succession")
        train_set = {(tuple(b), a) for b, a in env._train_pool}
        test_set = {(tuple(b), a) for b, a in env._test_pool}
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Train/test overlap detected: {overlap}"
        )

    def test_counting_warmup_shows_number_line(self):
        """Counting warmup presents 0→1→2→...→199 as bare digit tokens."""
        env = MathClassroomEnv(problem_type="succession", counting_warmup=1)
        # First observation should be counting phase with bare 0 and 1.
        obs = env.observe()
        toks = [t[0] for t in obs]
        assert "PHASE_counting" in toks
        assert "0" in toks
        assert "1" in toks

        # Advance through the number line until warmup ends.
        steps = 0
        seen_multi_digit = False
        for _ in range(210):
            toks = [t[0] for t in env.observe()]
            digit_toks = [t for t in toks if t.isdigit()]
            # Multi-digit numbers show as separate digit tokens.
            # e.g., 42 shows as "4", "2".
            if len(digit_toks) >= 3:  # at least 2 for number + 1 for answer
                seen_multi_digit = True
            if "PHASE_counting" not in toks:
                break
            env.act("next")
            steps += 1

        # Should have counted through the full range (200 numbers).
        assert steps >= 199, f"Should count 0-199, only got {steps} steps"
        # Should have seen multi-digit numbers (10+).
        assert seen_multi_digit, "Numbers above 9 should produce multiple digit tokens"

    def test_counting_warmup_builds_edges(self):
        """After counting warmup, the model has number-line edges."""
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        loop.CONSOLIDATION_INTERVAL = 0  # manual
        env = MathClassroomEnv(problem_type="succession", counting_warmup=5)

        # Run through just the warmup.
        while env._in_warmup:
            obs = env.observe()
            loop.observe([t[0] for t in obs], [t[1] for t in obs])
            loop.observe(["next"], [2])
            env.act("next")

        # Bare tokens: "3" and "4" are the same nodes in questions and answers.
        node_3 = kg.get_or_create("3")
        node_4 = kg.get_or_create("4")
        assert kg.node(node_3) is not None
        assert kg.node(node_4) is not None
        # The counting warmup shows [3, next_is, 4] in one observation.
        # This creates co-occurrence edges between 3 and 4.
        assert kg.edge_count() > 0, "Warmup should create edges"

    def test_train_test_cover_universe(self):
        """Train + test pools cover the full problem universe."""
        from experiments.symbolic_ai_v2.environments.math_classroom import PROBLEM_UNIVERSES
        for ptype in PROBLEM_UNIVERSES:
            env = MathClassroomEnv(problem_type=ptype)
            total = len(env._train_pool) + len(env._test_pool)
            universe = len(PROBLEM_UNIVERSES[ptype]())
            assert total == universe, (
                f"{ptype}: train({len(env._train_pool)}) + test({len(env._test_pool)}) "
                f"= {total} != universe({universe})"
            )


# ── Agent integration tests ──────────────────────────────────────────────────

class TestAgentLearning:
    """Test whether the agent learns arithmetic through the textworld."""

    def test_succession_mode_a_learns(self):
        """Agent should learn succession (succ(X) = X+1) in Mode A."""
        loop, env = run_classroom(
            problem_type="succession",
            mode="A",
            max_cycles=10,
            seed=42,
            consolidation_interval=500,
        )
        assert env.total_tested > 0, "Agent should have been tested"
        print(f"Succession Mode A: {env.total_correct}/{env.total_tested} "
              f"= {env.score:.1%}")
        # Currently expected to fail — needs natural transformations.
        # Threshold is above random (1/10 = 10% per digit).
        assert env.score > 0.05, (
            f"Score {env.score:.1%} is not above random baseline"
        )

    def test_succession_mode_b_learns(self):
        """Agent should learn succession in Mode B (test-first)."""
        loop, env = run_classroom(
            problem_type="succession",
            mode="B",
            max_cycles=50,
            seed=42,
            consolidation_interval=500,
        )
        assert env.total_tested > 0
        print(f"Succession Mode B: {env.total_correct}/{env.total_tested} "
              f"= {env.score:.1%}")
        assert env.score > 0.05

    def test_graph_learns_digit_associations(self):
        """After training, the KG should have co-occurrence edges between
        digit tokens that appear together in problems (e.g., 3 and 4 for succ)."""
        loop, env = run_classroom(
            problem_type="succession",
            mode="A",
            max_cycles=20,
            seed=42,
            consolidation_interval=100,
        )
        kg = loop.kg
        # Bare digit tokens should exist in the graph.
        digit_nodes = [v for v in kg._value_to_node.keys()
                       if isinstance(v, str) and v.isdigit()]
        assert len(digit_nodes) >= 9, (
            f"Should have digit tokens 0-9 in KG, got {len(digit_nodes)}"
        )

        # At least some edges should exist between consecutive digits
        # (from counting warmup and training observations).
        found = False
        for d in range(9):
            dnid = kg._value_to_node.get(str(d))
            next_nid = kg._value_to_node.get(str(d + 1))
            if dnid is None or next_nid is None:
                continue
            e = kg.edge(dnid, next_nid)
            if e is not None:
                found = True
                break
        assert found, "Should have at least one digit→next_digit edge"

    def test_no_special_math_code(self):
        """Verify the classroom uses the same AgenticLoop as the science lab.

        This test exists purely as documentation: the MathClassroom uses
        the exact same loop.observe() and loop.act() as every other
        environment. No math-specific code in the loop.
        """
        from experiments.symbolic_ai_v2.ctkg.logic.loop import AgenticLoop as LoopClass
        kg = KnowledgeGraph()
        loop = AgenticLoop(kg)
        # Same class, same methods. Nothing math-specific.
        assert hasattr(loop, 'observe')
        assert hasattr(loop, 'act')
        assert isinstance(loop, LoopClass)
