"""test_task_value.py — the planner reads the TASK column's `V = M·R`, and exactly what that does and does not buy.

H2 built the task region and left it read by nothing. This wires it into the rollout's LEAF HEURISTIC — what a world-state
is worth when the goal lies beyond the horizon — alongside the positional `ValueCritic`. The two answer over different state
spaces: a linear value over grid features provably cannot represent a relational V*
(`project_linear_value_cannot_hold_sokoban`), while the same linearity is no ceiling over a space whose states ARE relations.

CORTEX STAYS VALUE-FREE. The task column holds the map `M`; `R` — which configurations paid — is learned outside it by the
same delta rule as every other contingency here (`task_reward`, a `GoalMemory` over configurations). `V = M·R` is precisely
that split, which is why the SR is the right shape for a value the neocortex is not allowed to hold.

THE MEASURED LIMIT, AND IT IS THE POINT OF THIS FILE. Wiring it changed no behaviour on any current benchmark, and the reason
is structural rather than a tuning failure: the task value ranks CONFIGURATIONS, while a rollout's leaves differ mostly in
where the AGENT is standing. Measured on CollectAll — over 119 consecutive steps, the four one-step successors had IDENTICAL
task value every single time, because walking changes no object's relations. It discriminates precisely when a move changes
the configuration (a push), and is flat otherwise BY DESIGN, since position-freeness is what H2 was for.

So this is the right value on the wrong decision variable for a primitive-action rollout. Its natural consumer is subgoal
selection — WHICH configuration to aim for — which is the plan's H3, and this file is the evidence for why that is the next
step rather than a further tweak to the leaf.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.operator import eye                       # noqa: E402
from tasks.core import GameAction                  # noqa: E402
from tasks.games.collectall import CollectAll      # noqa: E402
from tasks.harness import Environment              # noqa: E402

ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


def _agent() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _pad_world():
    """A block and a pad, with the two configurations that matter — the block off the pad, and on it — and a task graph in
    which the push connects them."""
    a = _agent()
    a._movers.add(6)
    a.place_object(6, ((3.0, 4.0), eye(2)))
    a.place_object(7, ((5.0, 4.0), eye(2)))
    off = a.task_state()
    a.clear_scene()
    a.place_object(6, ((5.0, 4.0), eye(2)))
    a.place_object(7, ((5.0, 4.0), eye(2)))
    on = a.task_state()
    col = a._task_col()
    for _ in range(200):
        col.learn_transition(off, "push", on)
    for _ in range(30):
        a.task_reward.learn(a._configuration_bits(on), 1.0)   # R is learned over RELATION bits now
    return a, col, off, on


def test_value_falls_off_with_distance_to_a_paying_configuration():
    """`V = M·R` over the task graph: occupancy discounted by how far a configuration is from one that paid. This is the SR's
    whole point as a planner — value for ANY reward function is a dot product against the map, so a goal that moves re-values
    the space without any of it being learned again."""
    a = _agent()
    col = a._task_col()
    chain = ["s0", "s1", "s2", "s3"]
    for _ in range(300):
        for i in range(3):
            col.learn_transition(chain[i], "go", chain[i + 1])
    values = [col.state_value({"s3": 1.0}, s) for s in chain]
    assert values == sorted(values), f"value must rise toward the payer, got {[round(v, 3) for v in values]}"
    gamma = col.graph.gamma                                  # each step back is worth exactly one discount factor — the
    for near, far in zip(values[1:], values[:-1]):           # assertion the mechanism licenses, rather than a chosen bar
        assert abs(far - gamma * near) < 1e-2, (
            f"value must decay by exactly gamma per step; got {[round(v, 3) for v in values]} at gamma={gamma}")


def test_it_ranks_a_configuration_that_pays_above_one_that_does_not():
    """The relational judgement a positional critic cannot make: the block ON the pad is worth more than the block beside it,
    and nothing about the two world-states differs except one object's relation to another."""
    _a, col, off, on = _pad_world()
    rewards = {on: 1.0}
    assert col.state_value(rewards, on) > col.state_value(rewards, off), "on the pad must outrank off it"


def test_the_reward_vector_is_learned_OUTSIDE_the_column():
    """ARCHITECTURE's rule that the neocortex never sees a value, checked rather than asserted: the column holds only the map,
    and `R` lives in `task_reward` — the same delta-rule learner used for every other contingency, no new machinery."""
    a, col, _off, on = _pad_world()
    assert a._relation_value(on) > 0.0, "the paying configuration must be learned by the delta rule"
    assert not hasattr(col.graph, "rewards"), "the column's frame must hold no reward of its own"
    assert col.state_value({}, on) == 0.0, "with no reward vector there is no value — the map alone is value-free"


def test_the_planner_actually_reads_it():
    """The wiring itself: the rollout's leaf heuristic is the positional critic PLUS the relational task value, so a plan
    whose leaves differ in configuration is scored by the task column. Read off `value_of`, which is what `plan` passes."""
    a, _col, _off, on = _pad_world()
    a.clear_scene()
    a.place_object(6, ((5.0, 4.0), eye(2)))
    a.place_object(7, ((5.0, 4.0), eye(2)))
    assert a._task_value_of(a.world_state()) > 0.0, "the paying configuration must be worth something to the planner"
    assert a.value_of(a.world_state()) >= a._task_value_of(a.world_state()), "and `value_of` must include that term"


def test_it_is_FLAT_across_moves_that_change_no_configuration():
    """THE LIMIT, measured, and the reason this does not improve any benchmark. Walking changes no object's relations, so
    every one-step successor shares one task state and the value cannot rank them. That is not a defect to tune away — it is
    position-freeness, which is exactly what the task region was built for — but it does mean a primitive-action rollout is
    the wrong consumer, and that subgoal selection (H3) is the right one."""
    env = Environment(CollectAll())
    fd = env.reset()
    a = _agent()
    scored, discriminating = 0, 0
    for _ in range(40):
        action, coords = a.step(fd)
        fd = env.step(action, coords)
        if a._task is not None and a.task_reward.w:
            world, model = a.world_state(), a.world_model()
            values = [a._task_value_of(model.step(world, x)) for x in ACTIONS]
            scored += 1
            discriminating += 1 if max(values) - min(values) > 1e-9 else 0
        if fd.is_win() or fd.is_terminal():
            break
    assert scored > 5, "the run must reach a state where the task value is defined"
    assert discriminating == 0, (
        f"walking must not change the configuration, yet {discriminating}/{scored} steps ranked the successors")
