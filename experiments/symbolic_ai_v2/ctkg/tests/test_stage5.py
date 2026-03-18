"""Stage 5 validation tests.

Covers:
  - EpisodicStore  (8 tests)
  - SelfModel      (8 tests)
  - TheoryOfMind   (8 tests)
  - ActiveInference (8 tests)
  - AgentLoop      (8 tests)
  - Integration    (5 tests: cross-module + TextWorldEnv smoke test)

Total: 45 tests.
"""

from __future__ import annotations

import sys
import os
import math

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import numpy as np

# Stage 5 modules
from experiments.symbolic_ai_v2.ctkg.core.episodic_store import (
    EpisodicStore,
    EpisodicEvent,
)
from experiments.symbolic_ai_v2.ctkg.core.self_model import (
    SelfModel,
    MetaConcept,
)
from experiments.symbolic_ai_v2.ctkg.core.theory_of_mind import (
    TheoryOfMind,
    BeliefState,
)
from experiments.symbolic_ai_v2.ctkg.inference.active_inference import (
    PolicyScore,
    epistemic_value,
    pragmatic_value,
    score_policy,
    select_action,
)
from experiments.symbolic_ai_v2.ctkg.agent.loop import (
    AgentLoop,
    StepInfo,
    RunSummary,
    CONTEXT_MAX,
)

# Supporting modules
from experiments.symbolic_ai_v2.ctkg.core.concept_lattice import (
    DistributionalConcept,
)
from experiments.symbolic_ai_v2.ctkg.core.morphism_graph import MorphismGraph
from experiments.symbolic_ai_v2.ctkg.learning.hankel_count import HankelCount
from experiments.symbolic_ai_v2.ctkg.learning.fca_discover import discover_concepts
from experiments.symbolic_ai_v2.ctkg.learning.morphism_discover import (
    discover_morphisms,
)
from experiments.symbolic_ai_v2.ctkg.learning.process_discover import (
    discover_processes,
)
from experiments.symbolic_ai_v2.ctkg.inference.predict import Predictor
from experiments.symbolic_ai_v2.corpus.digit_math_generator import (
    digit_succ_pred_split,
)
from experiments.symbolic_ai_v2.environments.textworld import TextWorldEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(concept_id: int, atoms: dict[str, float]) -> DistributionalConcept:
    total = sum(atoms.values()) or 1.0
    centroid = np.zeros(max(10, len(atoms) + 1))
    for i, (a, w) in enumerate(atoms.items()):
        if i < len(centroid):
            centroid[i] = w / total
    return DistributionalConcept(
        concept_id=concept_id,
        centroid_vector=centroid,
        extent_weights={},
        intent_weights={a: w / total for a, w in atoms.items()},
        support=sum(atoms.values()),
    )


def _make_mg_varied() -> MorphismGraph:
    """Build a MorphismGraph with varied confidence/evidence for SelfModel tests."""
    mg = MorphismGraph()
    c0 = _make_concept(0, {"a": 1.0})
    c1 = _make_concept(1, {"b": 1.0})
    c2 = _make_concept(2, {"c": 1.0})
    o0 = mg.add_object(c0, label="A")
    o1 = mg.add_object(c1, label="B")
    o2 = mg.add_object(c2, label="C")
    # HIGH_CONFIDENCE morphism: A->B with confidence 0.8
    mg.add_morphism(o0.obj_id, o1.obj_id, morph_type="AB", evidence=15, confidence=0.8)
    # CONTESTED morphism: B->C with confidence -0.7
    mg.add_morphism(o1.obj_id, o2.obj_id, morph_type="BC", evidence=3, confidence=-0.7)
    # DEEPLY_EVIDENCED: A->C with evidence 20
    mg.add_morphism(o0.obj_id, o2.obj_id, morph_type="AC", evidence=20, confidence=0.1)
    return mg


@pytest.fixture(scope="module")
def small_predictor():
    """A Predictor trained on succ/pred for 0-99."""
    _, train, _ = digit_succ_pred_split(train_max=99, test_min=100, test_max=199)
    hc = HankelCount(r_max=1)
    hc.update_batch(train)
    lattices = discover_concepts(
        hankel=hc, r_levels=[1],
        lambda_productivity=0.1, merge_threshold=0.15, min_support=2.0,
    )
    lattice = lattices[0]
    mg = discover_morphisms(train, hc, lattice, r=1)
    process_rules = discover_processes(train, op_atoms=["succ", "pred"])
    return Predictor(
        hankel=hc, lattice=lattice, morphism_graph=mg,
        process_rules=process_rules, k_neighbours=5, r=1,
    )


# ===========================================================================
# TestEpisodicStore
# ===========================================================================

class TestEpisodicStore:

    def test_store_above_threshold(self):
        """Events with PE > threshold are stored."""
        store = EpisodicStore(surprise_threshold=0.5)
        evt = store.add_event(0, ["tok"], 0.8)
        assert evt is not None
        assert len(store) == 1

    def test_reject_below_threshold(self):
        """Events with PE <= threshold are NOT stored."""
        store = EpisodicStore(surprise_threshold=0.5)
        evt = store.add_event(0, ["tok"], 0.3)
        assert evt is None
        assert len(store) == 0

    def test_reject_at_threshold(self):
        """Boundary: PE == threshold is rejected (strict >)."""
        store = EpisodicStore(surprise_threshold=0.5)
        evt = store.add_event(0, ["tok"], 0.5)
        assert evt is None

    def test_max_events_enforced_on_add(self):
        """Buffer never exceeds max_events."""
        store = EpisodicStore(surprise_threshold=0.0, max_events=3)
        for i in range(10):
            store.add_event(i, [str(i)], 0.9 - i * 0.01)
        assert len(store) <= 3

    def test_prune_respects_limit(self):
        """prune(n) leaves at most n events."""
        store = EpisodicStore(surprise_threshold=0.0, max_events=None)
        for i in range(20):
            store.add_event(i, [str(i)], 0.6 + i * 0.01)
        removed = store.prune(max_events=5)
        assert len(store) == 5
        assert removed == 15

    def test_prune_removes_lowest_surprise(self):
        """prune() removes the lowest-PE events first."""
        store = EpisodicStore(surprise_threshold=0.0, max_events=None)
        store.add_event(0, ["low"], 0.6)
        store.add_event(1, ["high"], 0.9)
        store.prune(max_events=1)
        assert len(store) == 1
        assert store.get_recent(1)[0].prediction_error == pytest.approx(0.9)

    def test_get_recent_newest_first(self):
        """get_recent() returns events newest first."""
        store = EpisodicStore(surprise_threshold=0.0)
        for i in range(5):
            store.add_event(i, [str(i)], 0.7)
        recent = store.get_recent(3)
        assert recent[0].step > recent[-1].step

    def test_replay_batch_weighted_high_surprise(self):
        """replay_batch() should return high-surprise events more often."""
        store = EpisodicStore(surprise_threshold=0.0)
        store.add_event(0, ["low"], 0.51)    # low surprise
        for i in range(1, 6):
            store.add_event(i, ["high", str(i)], 0.99)  # high surprise
        batch = store.replay_batch(n=50)
        high_count = sum(1 for e in batch if e.tokens[0] == "high")
        # high-surprise events should dominate
        assert high_count > 30

    def test_consolidate_merges_duplicates(self):
        """consolidate() removes duplicate sequences."""
        store = EpisodicStore(surprise_threshold=0.0)
        for i in range(4):
            store.add_event(i, ["same", "tok"], 0.7 + i * 0.05)
        store.add_event(4, ["other"], 0.8)
        removed = store.consolidate()
        assert removed == 3            # 4 duplicates -> 1 representative
        assert len(store) == 2

    def test_empty_store_returns_empty_batch(self):
        """replay_batch() on empty store returns []."""
        store = EpisodicStore()
        assert store.replay_batch(n=5) == []


# ===========================================================================
# TestSelfModel
# ===========================================================================

class TestSelfModel:

    def test_meta_concepts_empty_before_update(self):
        """meta_concepts() returns [] before update() is called."""
        sm = SelfModel()
        assert sm.meta_concepts() == []

    def test_update_returns_list(self):
        """update() returns a list of MetaConcept."""
        mg = _make_mg_varied()
        sm = SelfModel()
        result = sm.update(mg)
        assert isinstance(result, list)
        assert all(isinstance(mc, MetaConcept) for mc in result)

    def test_high_confidence_group(self):
        """HIGH_CONFIDENCE group contains morphism with conf 0.8."""
        mg = _make_mg_varied()
        sm = SelfModel(confidence_high=0.5)
        sm.update(mg)
        hc = sm.concept_for("HIGH_CONFIDENCE")
        assert hc is not None
        assert len(hc.morph_ids) >= 1

    def test_contested_group(self):
        """CONTESTED group contains morphism with conf -0.7."""
        mg = _make_mg_varied()
        sm = SelfModel(confidence_low=-0.5)
        sm.update(mg)
        contested = sm.concept_for("CONTESTED")
        assert contested is not None
        assert len(contested.morph_ids) >= 1

    def test_deeply_composed_group(self):
        """DEEPLY_COMPOSED group contains morphism with body length > 2."""
        mg = MorphismGraph()
        c0 = _make_concept(0, {"a": 1.0})
        c1 = _make_concept(1, {"b": 1.0})
        c2 = _make_concept(2, {"c": 1.0})
        o0 = mg.add_object(c0)
        o1 = mg.add_object(c1)
        o2 = mg.add_object(c2)
        # Composite morphism A → B → C (body length 3)
        mg.add_morphism(o0.obj_id, o2.obj_id, body=[o0.obj_id, o1.obj_id, o2.obj_id],
                        morph_type="composite", evidence=5, confidence=0.3)
        # Shallow morphism A → B (body length 2)
        mg.add_morphism(o0.obj_id, o1.obj_id, morph_type="shallow", evidence=5, confidence=0.3)
        sm = SelfModel(min_body_depth=2)
        sm.update(mg)
        dc = sm.concept_for("DEEPLY_COMPOSED")
        assert dc is not None
        assert len(dc.morph_ids) >= 1

    def test_singleton_detection(self):
        """SINGLETON morphisms are those with unique (src, tgt)."""
        mg = _make_mg_varied()
        sm = SelfModel()
        sm.update(mg)
        singleton = sm.concept_for("SINGLETON")
        # All three morphisms in _make_mg_varied have unique (src, tgt) pairs
        assert singleton is not None
        assert len(singleton.morph_ids) == 3

    def test_reflective_correction_high_confidence(self):
        """HIGH_CONFIDENCE morphism gets +0.1 correction."""
        mg = _make_mg_varied()
        sm = SelfModel(confidence_high=0.5)
        sm.update(mg)
        hc = sm.concept_for("HIGH_CONFIDENCE")
        assert hc is not None
        mid = hc.morph_ids[0]
        delta = sm.reflective_correction(mg, mid)
        assert delta == pytest.approx(0.1)

    def test_reflective_correction_contested(self):
        """CONTESTED morphism gets -0.1 correction."""
        mg = _make_mg_varied()
        sm = SelfModel(confidence_low=-0.5)
        sm.update(mg)
        contested = sm.concept_for("CONTESTED")
        assert contested is not None
        mid = contested.morph_ids[0]
        delta = sm.reflective_correction(mg, mid)
        assert delta == pytest.approx(-0.1)

    def test_reflective_correction_neutral(self):
        """Morphism in no special group gets 0.0 correction."""
        mg = MorphismGraph()
        c0 = _make_concept(0, {"x": 1.0})
        c1 = _make_concept(1, {"y": 1.0})
        o0 = mg.add_object(c0)
        o1 = mg.add_object(c1)
        m = mg.add_morphism(o0.obj_id, o1.obj_id, confidence=0.1, evidence=5)
        sm = SelfModel(confidence_high=0.5, confidence_low=-0.5, min_body_depth=2)
        sm.update(mg)
        delta = sm.reflective_correction(mg, m.morph_id)
        assert delta == pytest.approx(0.0)

    def test_isomorphism_check(self):
        """Rename tokens; same structure should give same number of meta-concepts."""
        mg1 = _make_mg_varied()
        sm1 = SelfModel()
        mc1 = sm1.update(mg1)

        # Rename atoms: same structure, different labels
        mg2 = MorphismGraph()
        c0 = _make_concept(0, {"X": 1.0})
        c1 = _make_concept(1, {"Y": 1.0})
        c2 = _make_concept(2, {"Z": 1.0})
        o0 = mg2.add_object(c0)
        o1 = mg2.add_object(c1)
        o2 = mg2.add_object(c2)
        mg2.add_morphism(o0.obj_id, o1.obj_id, evidence=15, confidence=0.8)
        mg2.add_morphism(o1.obj_id, o2.obj_id, evidence=3, confidence=-0.7)
        mg2.add_morphism(o0.obj_id, o2.obj_id, evidence=20, confidence=0.1)
        sm2 = SelfModel()
        mc2 = sm2.update(mg2)
        # Same number of meta-concepts
        assert len(mc1) == len(mc2)


# ===========================================================================
# TestTheoryOfMind
# ===========================================================================

class TestTheoryOfMind:

    def test_observe_increments_count(self):
        """observe_action increments count for agent."""
        tom = TheoryOfMind()
        tom.observe_action("npc_a", "go_north")
        tom.observe_action("npc_a", "go_north")
        belief = tom.get_belief("npc_a")
        assert belief is not None
        assert belief.action_counts["go_north"] == 2

    def test_action_probs_sum_to_one(self):
        """action_probs distribution sums to 1."""
        tom = TheoryOfMind()
        for a in ["go_north", "go_south", "take_key"]:
            tom.observe_action("npc", a)
        probs = tom.action_probs("npc")
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_predict_action_returns_argmax(self):
        """predict_action returns the most frequently observed action."""
        tom = TheoryOfMind()
        for _ in range(5):
            tom.observe_action("npc", "go_north")
        for _ in range(2):
            tom.observe_action("npc", "go_south")
        assert tom.predict_action("npc") == "go_north"

    def test_unknown_agent_returns_empty(self):
        """predict_action returns '' for unknown agent."""
        tom = TheoryOfMind()
        assert tom.predict_action("ghost") == ""

    def test_unknown_agent_probs_empty(self):
        """action_probs returns {} for unknown agent."""
        tom = TheoryOfMind()
        assert tom.action_probs("ghost") == {}

    def test_window_evicts_old(self):
        """Window eviction: old actions disappear from the distribution."""
        tom = TheoryOfMind(window=3)
        # First fill with 'go_north'
        for _ in range(3):
            tom.observe_action("npc", "go_north")
        # Then overflow with 'go_south'
        for _ in range(3):
            tom.observe_action("npc", "go_south")
        probs = tom.action_probs("npc")
        # go_north should have been evicted
        assert probs.get("go_north", 0.0) == pytest.approx(0.0)
        assert probs.get("go_south", 0.0) == pytest.approx(1.0)

    def test_two_agent_independence(self):
        """Two agents maintain independent belief states."""
        tom = TheoryOfMind()
        tom.observe_action("alice", "go_north")
        tom.observe_action("bob", "go_south")
        assert tom.predict_action("alice") == "go_north"
        assert tom.predict_action("bob") == "go_south"

    def test_known_agents_sorted(self):
        """known_agents() returns sorted list."""
        tom = TheoryOfMind()
        tom.observe_action("charlie", "x")
        tom.observe_action("alice", "x")
        tom.observe_action("bob", "x")
        assert tom.known_agents() == ["alice", "bob", "charlie"]

    def test_repeated_observe_converges(self):
        """After many observations, dominant action approaches probability 1."""
        tom = TheoryOfMind(window=10)
        for _ in range(10):
            tom.observe_action("npc", "go_north")
        probs = tom.action_probs("npc")
        assert probs.get("go_north", 0.0) == pytest.approx(1.0)


# ===========================================================================
# TestActiveInference
# ===========================================================================

class TestActiveInference:

    def test_score_returns_policy_score_list(self, small_predictor):
        """score_policy returns list of PolicyScore."""
        ctx = ["succ", "5", "eq"]
        actions = ["6", "7", "8"]
        scores = score_policy(small_predictor, ctx, actions, ["6"])
        assert isinstance(scores, list)
        assert all(isinstance(s, PolicyScore) for s in scores)

    def test_score_returns_correct_length(self, small_predictor):
        """score_policy returns one score per action."""
        actions = ["6", "7", "8", "9"]
        scores = score_policy(small_predictor, ["succ", "3", "eq"], actions, ["4"])
        assert len(scores) == 4

    def test_select_returns_string(self, small_predictor):
        """select_action returns a string."""
        ctx = ["succ", "5", "eq"]
        action = select_action(small_predictor, ctx, ["6", "7"], ["6"])
        assert isinstance(action, str)
        assert action in ["6", "7"]

    def test_select_empty_actions_returns_empty(self, small_predictor):
        """select_action with empty list returns ''."""
        result = select_action(small_predictor, ["succ", "5", "eq"], [], ["6"])
        assert result == ""

    def test_epistemic_value_nonneg(self, small_predictor):
        """Epistemic value is always >= 0."""
        ev = epistemic_value(small_predictor, ["succ", "3", "eq"], "4")
        assert ev >= 0.0

    def test_pragmatic_value_in_range(self, small_predictor):
        """Pragmatic value is in [0, 1]."""
        pv = pragmatic_value(
            small_predictor, ["succ", "3", "eq"], "4", ["4"]
        )
        assert 0.0 <= pv <= 1.0

    def test_pragmatic_value_empty_goal(self, small_predictor):
        """Empty goal_tokens returns 0.0 pragmatic value."""
        pv = pragmatic_value(small_predictor, ["succ", "3", "eq"], "4", [])
        assert pv == 0.0

    def test_scores_sorted_by_G(self, small_predictor):
        """score_policy result is sorted by G ascending."""
        actions = ["4", "5", "6", "9"]
        scores = score_policy(
            small_predictor, ["succ", "3", "eq"], actions, ["4"]
        )
        G_vals = [s.G for s in scores]
        assert G_vals == sorted(G_vals)

    def test_goal_relevant_action_preferred(self, small_predictor):
        """The correct successor digit should score better (lower G) than far-off digit."""
        # succ(3) = 4; so '4' should be preferred over '9'
        actions = ["4", "9"]
        scores = score_policy(
            small_predictor, ["succ", "3", "eq"], actions, ["4"]
        )
        score_4 = next(s for s in scores if s.action == "4")
        score_9 = next(s for s in scores if s.action == "9")
        assert score_4.G <= score_9.G


# ===========================================================================
# TestAgentLoop
# ===========================================================================

class TestAgentLoop:

    def _make_loop(self, predictor=None, goal=None, random_until=0):
        env = TextWorldEnv(seed=0)
        store = EpisodicStore(surprise_threshold=0.3)
        return AgentLoop(
            env=env,
            predictor=predictor,
            episodic_store=store,
            goal_tokens=goal or ["HOLD_gem"],
            random_until=random_until,
            seed=0,
        )

    def test_step_increments_counter(self):
        """step() increments the step counter."""
        loop = self._make_loop()
        info = loop.step()
        assert info.step == 0
        info2 = loop.step()
        assert info2.step == 1

    def test_step_returns_step_info(self):
        """step() returns a StepInfo."""
        loop = self._make_loop()
        info = loop.step()
        assert isinstance(info, StepInfo)

    def test_step_info_has_tokens(self):
        """StepInfo.tokens is non-empty on first step."""
        loop = self._make_loop()
        info = loop.step()
        assert len(info.tokens) > 0

    def test_run_returns_run_summary(self):
        """run() returns a RunSummary."""
        loop = self._make_loop()
        summary = loop.run(max_steps=5)
        assert isinstance(summary, RunSummary)

    def test_run_stops_at_max_steps(self):
        """run() stops after max_steps if not done."""
        loop = self._make_loop()
        summary = loop.run(max_steps=7)
        assert summary.n_steps <= 7

    def test_run_pe_history_length_matches_steps(self):
        """RunSummary.pe_history has one entry per step."""
        loop = self._make_loop()
        summary = loop.run(max_steps=10)
        assert len(summary.pe_history) == summary.n_steps

    def test_context_tokens_capped(self):
        """context_tokens() never exceeds CONTEXT_MAX."""
        loop = self._make_loop()
        for _ in range(30):
            loop.step()
        assert len(loop.context_tokens()) <= CONTEXT_MAX

    def test_no_predictor_uses_random(self):
        """Without predictor, loop acts randomly (G is nan)."""
        loop = self._make_loop(predictor=None)
        summary = loop.run(max_steps=10)
        # All G values are nan when random
        for g in summary.G_history:
            assert math.isnan(g)

    def test_run_no_crash_100_steps(self):
        """AgentLoop runs 100 steps without raising any exception."""
        loop = self._make_loop()
        loop.run(max_steps=100)   # must not raise

    def test_reset_clears_state(self):
        """reset() zeroes step count and context."""
        loop = self._make_loop()
        loop.run(max_steps=5)
        loop.reset()
        assert loop._step_count == 0
        assert loop.context_tokens() == []


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:

    def test_textworld_env_imports(self):
        """TextWorldEnv can be instantiated (import stub resolved)."""
        env = TextWorldEnv(seed=42)
        obs = env.observe()
        assert len(obs) > 0
        assert obs[0][1] is None    # first token has None etype

    def test_episodic_store_fills_from_loop(self):
        """EpisodicStore receives events from AgentLoop run."""
        env = TextWorldEnv(seed=0)
        store = EpisodicStore(surprise_threshold=0.0)  # store everything
        loop = AgentLoop(env, predictor=None, episodic_store=store,
                         goal_tokens=["HOLD_gem"], seed=0)
        loop.run(max_steps=20)
        # Should have stored events equal to steps
        assert len(store) == loop._step_count

    def test_self_model_with_predictor_mg(self, small_predictor):
        """SelfModel.update() runs on the Predictor's MorphismGraph."""
        mg = small_predictor._morphism_graph
        sm = SelfModel()
        concepts = sm.update(mg)
        # Should find at least one meta-concept in a real graph
        assert isinstance(concepts, list)

    def test_loop_with_predictor_no_crash(self, small_predictor):
        """AgentLoop with fitted predictor runs 20 steps without error."""
        env = TextWorldEnv(seed=1)
        store = EpisodicStore()
        loop = AgentLoop(
            env=env,
            predictor=small_predictor,
            episodic_store=store,
            goal_tokens=["HOLD_gem"],
            random_until=0,
            seed=1,
        )
        summary = loop.run(max_steps=20)
        assert summary.n_steps <= 20

    def test_theory_of_mind_tracks_loop_actions(self):
        """TheoryOfMind can track actions observed during a loop run."""
        env = TextWorldEnv(seed=2)
        store = EpisodicStore(surprise_threshold=0.0)
        tom = TheoryOfMind(window=50)
        loop = AgentLoop(env, predictor=None, episodic_store=store,
                         goal_tokens=["HOLD_gem"], seed=2)

        # Run loop manually, recording actions in TOM
        for _ in range(15):
            info = loop.step()
            tom.observe_action("agent", info.action)
            if info.done:
                break

        # TOM should know about the 'agent'
        assert "agent" in tom.known_agents()
        probs = tom.action_probs("agent")
        assert abs(sum(probs.values()) - 1.0) < 1e-9
