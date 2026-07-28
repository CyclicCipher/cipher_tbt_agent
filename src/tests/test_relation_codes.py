"""test_relation_codes.py — the KEY problem: relations as OVERLAP-BEARING SDRs instead of exact-match keys.

H3 measured the bottleneck: task states were exact-match frozensets of quantised relations, so "the block one cell from the
pad" and "the block ON the pad" were as unrelated as either was to a wall across the board. Nothing a configuration taught
transferred to the configuration beside it, and a configuration never visited was worth exactly zero — which is why the task
drive was starved (mean 0.02 learned successors of the state the agent was standing in).

THE FIX, AND THE SHAPE IT WAS FORCED INTO BY MEASUREMENT. A relation is a DISPLACEMENT — a metric quantity — and a metric
needs a code whose overlap falls off with distance (`reference_sdr_regime_and_phase_codes`: SDRs carry IDENTITY well, a
metric needs more). The grid code gives exactly that. But a configuration must be scored relation BY relation and never as
one union: measured, a dozen relations unioned leaves 86% of the bits active, at which point every configuration looks like
every other and the overlap that was the whole point is destroyed. So a configuration is a bit LIST with multiplicity, and
the learner is the existing SDR-linear `ValueCritic` — `value` sums the active entries (= Σ over relations of that
relation's value) and `learn` shares the error across them. No new learner, no union.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.operator import eye                       # noqa: E402

FLAT = (1.0, 0.0, 0.0, 1.0)                        # the identity rotation, as `_quantise` renders it
PAD = 5


def _agent() -> Agent:
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a._movers.add(6)
    return a


def _scene(a: Agent, block_x: float) -> frozenset:
    """A block at `block_x` and a pad at PAD, both on the same row — one relation that matters, and it varies continuously."""
    a.clear_scene()
    a.place_object(6, ((float(block_x), 4.0), eye(2)))
    a.place_object(7, ((float(PAD), 4.0), eye(2)))
    return a.task_state()


def _overlap(a: Agent, u, v) -> float:
    col = a._scene_col()
    x, y = set(col.relation_code((u, FLAT))), set(col.relation_code((v, FLAT)))
    return len(x & y) / len(x | y)


def test_a_relation_code_overlaps_by_metric_distance():
    """What an exact-match key could not do: two relations one cell apart SHARE most of their bits, three cells apart share
    some, and unrelated ones share none. The overlap IS the generalisation — everything below follows from it."""
    a = _agent()
    assert _overlap(a, (3, 2), (3, 2)) == 1.0, "a relation must match itself exactly"
    near, mid, far = _overlap(a, (3, 2), (4, 2)), _overlap(a, (3, 2), (6, 2)), _overlap(a, (3, 2), (-7, 5))
    assert near > mid > far, f"overlap must fall off with distance, got {near:.2f} / {mid:.2f} / {far:.2f}"
    assert near > 0.5 and far == 0.0, f"one cell apart must be mostly shared and unrelated wholly disjoint ({near:.2f}, {far:.2f})"


def test_a_configuration_is_scored_relation_by_relation_and_never_unioned():
    """WHY THE OBVIOUS IMPLEMENTATION IS WRONG, kept because it is the trap. Unioning a configuration's relations into one
    SDR saturates it — measured at 86% of bits active for a dozen relations — and a saturated code cannot tell two
    configurations apart. Multiplicity is what preserves the per-relation sum: the same bit appears once per relation that
    carries it, so the critic reads Σ over relations rather than a set."""
    a = _agent()
    configuration = _scene(a, 3)
    bits = a._configuration_bits(configuration)
    n_relations = sum(len(relations) for _oid, relations in configuration)
    assert n_relations >= 2, "the fixture must have several relations"
    assert len(bits) > len(set(bits)), "bits must carry MULTIPLICITY — a set-union is the saturating version"
    assert len(bits) == n_relations * len(a._scene_col()._rel_enc.encode((0, 0)).active), (
        "exactly one code per relation, concatenated")


def test_value_generalises_to_configurations_that_were_never_visited():
    """THE FIX, MEASURED. Only the block-ON-the-pad configuration is ever rewarded. Every other configuration below has never
    been seen, and under exact-match keys every one of them was worth exactly 0. They now carry a graded value that falls off
    with the block's distance from the pad — a potential field over configuration space, learned from ONE rewarded example.
    That gradient is what a planner can descend."""
    a = _agent()
    paid = _scene(a, PAD)
    for _ in range(30):
        a.task_reward.learn(a._configuration_bits(paid), 1.0)
    values = [a._relation_value(_scene(a, x)) for x in (PAD, PAD + 1, PAD + 2, PAD + 4)]
    assert values[0] > 0.9, f"the rewarded configuration must be worth ~1, got {values[0]:.3f}"
    assert values == sorted(values, reverse=True), f"value must fall off with distance, got {[round(v, 3) for v in values]}"
    assert values[1] > 0.5 * values[0], "one cell off the pad must retain most of the value — that is the generalisation"
    assert min(values) > 0.0, "and NO never-visited configuration may be worth zero, which is what the exact key gave"


def test_a_subgoal_is_ranked_by_R_and_not_by_V():
    """A REAL ERROR THE GENERALISING R EXPOSED. `V = M·R` is the discounted STREAM from a state, so when a reward is not
    consumed on arrival, standing NEXT to the payoff scores higher than standing on it — measured here: V(beside) = 1.565
    against V(on) = 0.999, because from beside it you collect the pad's reward AND your own. Ranking subgoals by V therefore
    proposes staying put.

    It read correctly only while R was exactly zero everywhere except the paying configuration — i.e. it worked by accident,
    and only until the reward generalised. A subgoal is a TARGET STATE (`reference_hypothesis_generation`), so it is ranked
    by how good it is to BE there; `V` keeps its own job as the rollout's distance-aware leaf estimate."""
    a = _agent()
    beside, on = _scene(a, PAD - 2), _scene(a, PAD)
    col = a._task_col()
    for _ in range(200):
        col.learn_transition(beside, "push", on)
    for _ in range(30):
        a.task_reward.learn(a._configuration_bits(on), 1.0)
    rewards = a._task_rewards()
    assert col.state_value(rewards, beside) > col.state_value(rewards, on), (
        "the V inversion this guards against must still be real — otherwise the test is guarding nothing")
    assert a._relation_value(on) > a._relation_value(beside), "R ranks them the right way round"
    a._last_task = beside
    want = a._task_subgoal()
    assert want is not None and want[0] == on, f"the subgoal must be the pad configuration, got {want}"


def test_relation_codes_are_memoised():
    """A live run timed OUT at >115s before this: a relation's code is a pure function of its quantised displacement and is
    read at every rollout LEAF, so recomputing it meant thousands of numpy encodes per step. Recorded because the 2-minute
    law is what caught it, and a cache in an inner loop is not an optimisation here but the difference between running and
    not."""
    a = _agent()
    col = a._scene_col()
    col.relation_code(((3, 2), FLAT))
    assert col._rel_cache, "the code must be cached"
    assert col.relation_code(((3, 2), FLAT)) is col._rel_cache[((3, 2), FLAT)], "and re-read from the cache, not rebuilt"
