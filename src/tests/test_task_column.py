"""test_task_column.py — H2: the TASK region, a column over the scene's configuration with NO position in it.

H0 measured that one frame over the joint `(position, configuration)` state does not factorise, and H1 built peer-to-peer
communication. H2 is the second column the plan allocates — and the legacy `HETERARCHY_PLAN`'s non-negotiable governs it:
*"A 'PFC/task column' is NOT a different kind of unit — it is the SAME `CorticalColumn` fed a different INPUT."* So there is
no task module here. There is `Column(graph=True)`, whose only difference from the sensory one is that its L6a is a LEARNED
graph frame (`successor.py`) instead of a given metric one, because task space has no coordinates to be handed.

ANATOMY DECIDES THE SPLIT. Nothing works out which variables are "task" variables. The task column simply never receives
position — its input is the SCENE region's relational output, and the body's pose goes to `nav` — so the split is in the
wiring, as it is in cortex, where what a region represents is settled by which axons arrive. The bold assumption, stated
(`feedback_prefer_generalize_then_correct`): a fixed wiring split serves any game. The falsifier is concrete and
measurable: a game whose task state genuinely depends on WHERE it holds would make this column's transitions
non-deterministic — one state, one action, two successors — and it would stop predicting.

TWO MEASURED CORRECTIONS GOT IT HERE, both found by looking at what it actually learned rather than at whether it ran:
  1. The scene held only MOVERS, so the pad was never in it, "the block rests on the pad" had no relatum, and a whole
     Sokoban play produced 1 task state and 4 self-loops. A compositional region represents WHAT OBJECTS ARE WHERE; being
     dynamic is the rollout's filter and now applies where the rollout consumes the scene.
  2. Routing every object then put the AGENT in the scene, and its relations moved as it walked — 106 task states against
     89 joint ones, i.e. worse than not factoring at all. The self is not a scene object; it is the body. Excluding it is
     the same wiring statement as "the task column never receives position", and which colour is the self is DISCOVERED
     (`self_color`), not declared.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.operator import eye                       # noqa: E402
from tasks.games.lockpath import LockPath          # noqa: E402
from tasks.harness import Environment              # noqa: E402


def _agent() -> Agent:
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)


def _scene(a: Agent, objects: dict) -> frozenset:
    """Put a configuration of objects into the scene region and read the task state off it."""
    a.clear_scene()
    for oid, cell in objects.items():
        a.place_object(oid, (tuple(float(c) for c in cell), eye(2)))
    return a.task_state()


_PLAYED = []


def _play(steps: int = 300):
    """Play LockPath, recording the task state and the agent's cell at each step. Memoised: four tests read different facts
    off ONE play, and replaying the game per assertion is pure suite time for an identical trajectory (seeded)."""
    if _PLAYED:
        return _PLAYED[0]
    env = Environment(LockPath())
    fd = env.reset()
    a = _agent()
    cells_per_task, joint = {}, set()
    for _ in range(steps):
        action, coords = a.step(fd)
        fd = env.step(action, coords)
        objs = a.transduce(fd.grid)
        cur, pos = a._self_pos(objs), a._positions(objs)
        if cur is not None:
            cells_per_task.setdefault(a.task_state(), set()).add(cur)
            joint.add((cur, frozenset((c, pos[c]) for c in a._movers if c in pos)))
        best = max((len(c) for c in cells_per_task.values()), default=0)
        if fd.is_terminal() or fd.is_win() or (best > 12 and len(joint) > 4 * len(cells_per_task)):
            break                                    # stop once the factoring is demonstrated: playing on restates it
    _PLAYED.append((a, cells_per_task, joint))
    return _PLAYED[0]


def test_the_task_state_is_translation_invariant():
    """A relation is a DIFFERENCE of poses, so absolute position cancels in the arithmetic — the task state is position-free
    by construction rather than by filtering something out afterwards. Shift the entire scene and it is the same state; move
    one object relative to another and it is a different one. That is the whole content of "a task space"."""
    a = _agent()
    base = _scene(a, {6: (3, 4), 7: (5, 4), 1: (0, 0)})
    shifted = _scene(a, {6: (3 + 9, 4 + 7), 7: (5 + 9, 4 + 7), 1: (0 + 9, 0 + 7)})
    on_the_pad = _scene(a, {6: (5, 4), 7: (5, 4), 1: (0, 0)})
    assert base == shifted, "translating the whole scene must not change its configuration"
    assert base != on_the_pad, "moving the block onto the pad must change it"


def test_a_lone_object_has_no_configuration_to_speak_of():
    """The degenerate case that motivated routing the whole scene, kept as the boundary of the claim: relations need
    relata, so a world with ONE object has no task structure — and the honest report is an empty relation set rather than
    an invented state. This is exactly what the region looked like when only movers were routed."""
    a = _agent()
    alone = _scene(a, {6: (3, 4)})
    assert all(not rel for _oid, rel in alone), f"one object can stand in no relation, got {alone}"


def test_the_agent_is_not_in_the_task_state():
    """THE SPLIT, on the live loop. The agent walks around a board it is not changing, and the task state holds constant
    through most of it — where a joint frame minted a fresh state at every step. The self is excluded because it is the
    body, not a scene object, and which colour that is was DISCOVERED from motion, never declared."""
    _a, cells_per_task, _joint = _play()
    unchanged = max(len(cells) for cells in cells_per_task.values())
    assert unchanged > 10, (
        f"one task state must abstract over many agent positions; the best abstracts over only {unchanged}")


def test_the_task_frame_is_far_smaller_than_the_joint_one():
    """H0'S BLOW-UP, undone — the point of building this at all. Over the SAME play, the joint `(cell, configuration)` state
    space is several times the task one, because it re-mints a state every time the agent takes a step. The ratio is the
    factoring, measured on a real game rather than a corridor."""
    _a, cells_per_task, joint = _play()
    assert len(joint) > 3 * len(cells_per_task), (
        f"the joint space must be several times larger; got {len(joint)} joint vs {len(cells_per_task)} task")


def test_the_task_region_learns_a_graph_and_is_the_same_column_class():
    """Mountcastle, and the plan's non-negotiable: this is `Column` — the one every other region is — fed a different input.
    What differs is its L6a, which is a LEARNED graph rather than a given grid, because "the block rests on the pad" is not
    a point in R^n. A column has ONE L6a: this one has no metric frame at all, and asking it for a metric location fails
    rather than quietly answering in the wrong space."""
    a, _cells, _joint = _play()
    task = a._task
    assert type(task) is type(a.sensory), "the task region must be the SAME column class, not a bespoke module"
    assert task.graph is not None and task.location is None, "its L6a is the learned graph, and it has no grid"
    assert len(task.graph.states()) > 1 and task.where_state() is not None, "and it learned a real transition graph"
    try:
        task.where()
        assert False, "a graph column must not answer a metric location query"
    except ValueError:
        pass


def test_the_declared_heterarchy_now_has_a_task_region_fed_by_cortex():
    """`region.py`'s thesis is that a region IS its wiring, so the wiring is what gets asserted. The task region's proximal
    input is the SCENE region — not a transducer — which is the structural difference between an association area and a
    sensory one, and the reason no "task modality" was ever needed."""
    a, _cells, _joint = _play()
    task = a.hierarchy.get("task")
    assert task is not None and task.proximal == "scene", "the task region must be driven by cortex, not a transducer"
    assert not task.is_peripheral({"vision", "touch"}), "an association area has no transducer"
    assert ("scene", "task") in a.hierarchy.edges(), "and the edge must show up in the declared heterarchy"
