"""test_hippocampus.py — the Hippocampus ORCHESTRATOR: the four subfields behind one handle (hippocampus/__init__.py; DESIGN §3, slice 6).

The last slice composes map ⊕ replay ⊕ CA3 ⊕ DG ⊕ CA1 into ONE region. Two checks: (1) the `Hippocampus` object exposes the
three roles — EPISODIC store/recall, REMAPPING, and PLANNING — standalone; (2) the agent routes all three through its single
`self.hippocampus` handle, end to end, sharing one memory.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                          # noqa: E402
from tbt.hippocampus import Hippocampus              # noqa: E402
from tbt.operator import eye                         # noqa: E402

ACTIONS = ["N", "S", "E", "W"]
_STEP = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _nav_agent() -> Agent:
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a.set_pose((10.0, 10.0), eye(2))
    for act, d in _STEP.items():
        a.learn_pose_move(act, ((0.0, 0.0), eye(2)), (d, eye(2)))
    return a


def test_hippocampus_composes_the_three_roles():
    """Standalone: one Hippocampus object does EPISODIC recall, REMAPPING, and DG separation — the subfields wired together."""
    h = Hippocampus(n_inputs=512, seed=0)

    scene = {("A", (5, 5)), ("B", (10, 20)), ("C", (30, 2))}
    h.remember(scene)
    assert h.recall({("A", (5, 5))}) == scene, "episodic: a glimpse recalls the whole remembered scene"

    ida, _ = h.visit({"a1", "a2", "a3"})
    idb, _ = h.visit({"b1", "b2", "b3"})
    assert idb != ida, "remapping: distinct environments get distinct charts"
    ida2, r = h.visit({"a1", "a2"})
    assert ida2 == ida and r.matched, "remapping: a partial view recalls the known chart"

    kA, kB = h.chart_key(set(range(0, 30))), h.chart_key(set(range(200, 230)))
    assert len(kA & kB) < 0.25 * len(kA), "separation: distinct signatures get well-separated chart keys"


def test_agent_routes_all_three_through_one_handle():
    """End to end: PLANNING, EPISODIC memory, and REMAPPING all flow through the agent's single `self.hippocampus`."""
    a = _nav_agent()
    assert isinstance(a.hippocampus, Hippocampus), "the agent holds ONE hippocampus handle"

    # PLANNING (replay) routes through the handle
    goal = (12.0, 11.0)
    plan = a.plan(lambda w: 1.0 if _dist(w.agent[0], goal) < 0.5 else 0.0, ACTIONS, horizon=6)
    world = a.world_state()
    for act in plan:
        world = a.world_model().step(world, act)
    assert _dist(world.agent[0], goal) < 0.5, f"the rollout plan (via the hippocampus) must reach the goal, ended {world.agent[0]}"

    # EPISODIC memory routes through the handle
    a.place_object("A", ((5.0, 5.0), eye(2)))
    a.place_object("B", ((10.0, 20.0), eye(2)))
    full = a.scene_tokens()
    a.remember_scene()
    assert a.recall_scene({("A", (5, 5))}) == set(full), "a glimpse recalls the whole scene through the handle"

    # REMAPPING routes through the handle
    ida, _ = a.visit_environment({"r1", "r2", "r3"})
    idb, _ = a.visit_environment({"s1", "s2", "s3"})
    ida2, res = a.visit_environment({"r1", "r2"})
    assert idb != ida and ida2 == ida and res.matched, "distinct environments separate; a partial view recalls its chart"

    # all three shared the ONE hippocampus (the episodic write is on the same handle)
    assert a.hippocampus.episodic.n_stored >= 1


if __name__ == "__main__":
    h = Hippocampus(n_inputs=512, seed=0)
    h.remember({("A", (5, 5)), ("B", (10, 20))})
    print(f"episodic: glimpse {{A}} recalls {sorted(h.recall({('A', (5, 5))}))}")
    print(f"remap: A->{h.visit({'a', 'b'})[0]}  B->{h.visit({'c', 'd'})[0]}  A-again->{h.visit({'a', 'b'})[0]}")
    print(f"one Hippocampus, three roles: episodic={type(h.episodic).__name__} dg={type(h.dg).__name__} remap={type(h.remapper).__name__}")
