"""test_featurize.py — the world-state → SDR featuriser + the critic-scored rollout leaf (hippocampus/featurize.py).

This closes replay.py's honest seam: the rollout leaf now uses the real `ValueCritic` over the featurised world-state instead
of a stand-in. Three checks: (1) the featuriser is OVERLAP-BEARING (nearby world-states share bits, so value generalises);
(2) the critic LEARNS world-state value by TD (`Agent.learn_value`); (3) the rollout USES it at the leaf — a trained critic
pulls the plan toward a goal the horizon cannot reach, where an untrained one does not.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                              # noqa: E402
from tbt.hippocampus.featurize import WorldFeaturizer    # noqa: E402
from tbt.hippocampus.map import WorldMap                 # noqa: E402
from tbt.operator import eye                             # noqa: E402

ACTIONS = ["N", "S", "E", "W"]
_STEP = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _world(pos, objects=None) -> WorldMap:
    return WorldMap((tuple(float(c) for c in pos), eye(2)), objects or {}, [(0, 63)] * 2)


def _nav_agent() -> Agent:
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a.set_pose((10.0, 10.0), eye(2))
    for act, d in _STEP.items():
        a.learn_pose_move(act, ((0.0, 0.0), eye(2)), (d, eye(2)))
    return a


def test_featuriser_is_overlap_bearing():
    """Nearby world-states share more bits than distant ones (so the linear critic generalises), and the same place with a
    different object featurises differently (identity is bound to place)."""
    f = WorldFeaturizer(dims=2)
    near, close, far = f.encode(_world((10, 10))), f.encode(_world((11, 10))), f.encode(_world((45, 45)))
    assert len(near & close) > len(near & far), "adjacent world-states must share more bits than distant ones"
    assert len(near & far) < len(near), "distant states must not be identical"
    with_a = f.encode(_world((10, 10), {"A": ((5.0, 5.0), eye(2))}))
    with_b = f.encode(_world((10, 10), {"B": ((5.0, 5.0), eye(2))}))
    assert with_a != with_b, "the same place with a DIFFERENT object must featurise differently (id bound to place)"


def test_critic_learns_world_state_value():
    """TD over featurised worlds: after training a corridor toward a rewarded state, value increases toward the goal — the
    critic scores world-STATES, which it could not before (it only saw decision-context SDRs)."""
    a = _nav_agent()
    far, mid, goal = _world((5, 5)), _world((25, 25)), _world((45, 45))
    for _ in range(60):
        a.learn_value(goal, reward=1.0, done=True)
        a.learn_value(mid, reward=0.0, after=goal, done=False)
        a.learn_value(far, reward=0.0, after=mid, done=False)
    assert a.value_of(goal) > a.value_of(mid) > a.value_of(far), (
        f"value must increase toward the rewarded state, got far={a.value_of(far):.3f} "
        f"mid={a.value_of(mid):.3f} goal={a.value_of(goal):.3f}")


def test_rollout_uses_the_trained_critic_at_the_leaf():
    """The seam closed: with a goal beyond the horizon, an UNTRAINED critic gives the rollout no gradient (it stays put), but a
    critic TRAINED that eastward world-states pay off pulls the plan east — the learned leaf heuristic doing real work."""
    a = _nav_agent()
    goal = (16.0, 10.0)                                            # six east — far beyond horizon 2
    reward = lambda w: 1.0 if _dist(w.agent[0], goal) < 0.5 else 0.0

    assert a.plan(reward, ACTIONS, horizon=2) == [], "an untrained critic gives no leaf gradient to a far goal → no plan"

    corridor = [(11, 10), (12, 10), (13, 10), (14, 10), (15, 10), (16, 10)]
    for _ in range(80):
        for p, nxt in zip(corridor, corridor[1:]):
            a.learn_value(_world(nxt), reward=(1.0 if nxt == (16, 10) else 0.0),
                          after=None if nxt == (16, 10) else _world(nxt), done=(nxt == (16, 10)))
            a.learn_value(_world(p), reward=0.0, after=_world(nxt), done=False)

    plan = a.plan(reward, ACTIONS, horizon=2)                     # value defaults to the trained critic (value_of)
    assert plan, "a trained critic must give the rollout a gradient toward the goal"
    world = a.world_state()
    for act in plan:
        world = a.world_model().step(world, act)
    assert _dist(world.agent[0], goal) < _dist((10.0, 10.0), goal), (
        f"the trained critic must pull the plan toward the goal; ended {world.agent[0]}")


if __name__ == "__main__":
    f = WorldFeaturizer(dims=2)
    print(f"|featurised world| = {len(f.encode(_world((10, 10), {'A': ((5.0, 5.0), eye(2))})))} bits")
    print(f"overlap (10,10)~(11,10) = {len(f.encode(_world((10, 10))) & f.encode(_world((11, 10))))}")
    print(f"overlap (10,10)~(45,45) = {len(f.encode(_world((10, 10))) & f.encode(_world((45, 45))))}")
