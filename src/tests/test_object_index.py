"""test_object_index.py — OBJECT INDEXES: what makes two identical things two things, and what makes one thing persist.

The scene used to be keyed on APPEARANCE (the object's colour), so two identical crates collided in it and both were lost —
`_movers` stayed empty through a whole game while they were being shoved around. The fix is not a bigger data structure but
a different mechanism, and it is the one the brain uses.

THE BRAIN DOES NOT TELL OBJECTS APART BY THEIR PROPERTIES. It allocates a small set of pointers — Pylyshyn's FINSTs,
Kahneman & Treisman's object files — that stick to a thing and follow it, and the pointer is PRIOR TO and separable from
what the thing looks like. Multiple Object Tracking (Pylyshyn & Storm 1988) settles it: people track several targets among
IDENTICAL distractors, so nothing featural distinguishes any of them at any moment. Correspondence is solved by PREDICTION —
the pointer goes to whatever turns up nearest where the object was expected — which is why tracking survives a total absence
of features and fails on discontinuous jumps. Spatiotemporal individuation precedes featural individuation (Xu & Carey 1996),
the same finding `Column.commit` already leans on for segmentation, applied here to keeping a pointer ON a thing.

Object permanence is not a separate faculty: it is this mechanism run without input. An index survives when its object is
merely unseen and is dropped only when its place is VISIBLE and demonstrably empty — the mirror of
`reference_recognition_under_occlusion`'s "mint on refutation, never on incompleteness", one level down.
"""

from __future__ import annotations

import os
import sys

_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tbt.agent import Agent                        # noqa: E402
from tbt.operator import eye                       # noqa: E402

CRATE, PAD = 6, 7


def _column():
    return Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)._scene_col()


def _at(x, y):
    return ((float(x), float(y)), eye(2))


def _cells(col):
    return {i: tuple(round(c) for c in p[0]) for i, p in col.scene_snapshot().items()}


def test_two_identical_objects_are_two_things():
    """Nothing about either crate distinguishes it from the other — same colour, same shape, same everything. They still get
    separate indexes, because an index is ALLOCATED rather than derived from appearance. A scene keyed on the feature had
    room for one of them and therefore held neither."""
    col = _column()
    col.track([(CRATE, _at(3, 1)), (CRATE, _at(3, 3))])
    crates = [i for i in col.scene_snapshot() if col.feature_of(i) == CRATE]
    assert len(crates) == 2, f"two crates must be two tracked things, got {len(crates)}"
    assert crates[0] != crates[1], "with distinct indexes"


def test_a_pointer_follows_ITS_object_through_motion():
    """The correspondence, and the only thing that can do it here: which crate moved is answerable ONLY from where each one
    was, since the two are indistinguishable in every other respect."""
    col = _column()
    col.track([(CRATE, _at(3, 1)), (CRATE, _at(3, 3))])
    first = min(i for i in col.scene_snapshot() if col.feature_of(i) == CRATE)
    moved = _cells(col)[first]
    col.track([(CRATE, _at(moved[0] + 1, moved[1])), (CRATE, _at(3, 3) if moved != (3, 3) else _at(3, 1))])
    assert _cells(col)[first] == (moved[0] + 1, moved[1]), "the pointer must follow the crate it was on"
    assert len([i for i in col.scene_snapshot() if col.feature_of(i) == CRATE]) == 2, "and neither crate may be lost"


def test_an_occluded_object_PERSISTS_which_is_object_permanence():
    """The crate is pushed ONTO the pad, so the pad is no longer drawn and vanishes from the frame. Its index survives at the
    pose it had, because absence of evidence is not evidence of absence — the default is persistence and deletion needs a
    positive contradiction."""
    col = _column()
    col.track([(CRATE, _at(4, 1)), (PAD, _at(5, 1))])
    pad = next(i for i in col.scene_snapshot() if col.feature_of(i) == PAD)
    col.track([(CRATE, _at(5, 1))])                     # the crate now covers the pad; only the crate is visible
    assert pad in col.scene_snapshot(), "an occluded object must not be forgotten"
    assert _cells(col)[pad] == (5, 1), "and it is still where it was"


def test_an_object_whose_place_is_VISIBLY_EMPTY_is_dropped():
    """The other half of the same rule. Nothing stands where the crate was and the cell is plainly in view, so the object is
    demonstrably gone — consumed, or removed — and the index is released. Permanence is a default, not a refusal to update."""
    col = _column()
    col.track([(CRATE, _at(4, 1)), (PAD, _at(5, 1))])
    crate = next(i for i in col.scene_snapshot() if col.feature_of(i) == CRATE)
    col.track([(PAD, _at(5, 1))])
    assert crate not in col.scene_snapshot(), "a visibly absent object must be released"


def test_a_pointer_does_not_JUMP_to_a_distant_look_alike():
    """The bug this cost the most to find, kept as a test. With one pad occluded, the surviving pad's observation is the only
    feature-7 thing in the frame — and an unbounded nearest-match hands it to the occluded pad's pointer two cells away. The
    phantom displacement then teaches the agent that PADS MOVE, which is how a stationary landmark ended up in `_movers`.

    The bound is not a tuned radius: a pointer may follow its object as far as anything has actually been SEEN to move in one
    frame, floored at a cell. That is learned from motion, and it is why MOT survives smooth movement but not teleportation."""
    col = _column()
    col.track([(PAD, _at(5, 1)), (PAD, _at(5, 3)), (CRATE, _at(2, 3))])
    far = next(i for i in col.scene_snapshot() if col.feature_of(i) == PAD and _cells(col)[i] == (5, 3))
    col.track([(PAD, _at(5, 1)), (CRATE, _at(5, 3))])   # the pad at (5,3) is now covered by a crate
    assert _cells(col)[far] == (5, 3), "the occluded pad's pointer must stay put, not jump to the other pad"


def test_the_INDEX_says_which_one_and_the_FEATURE_says_what_kind():
    """The distinction the whole thing turns on. Learning belongs to the KIND — pressing a crate teaches you about crates,
    not about that particular crate — while the index picks out which body actually moves. They were the same thing only
    while every object had a unique colour, and the world map now carries both."""
    a = Agent(feat_n=8, n_content=2, n_state=2, n_cols=64, seed=0)
    a._movers.add(CRATE)
    col = a._scene_col()
    col.track([(CRATE, _at(3, 1)), (CRATE, _at(3, 3)), (PAD, _at(5, 1))])
    world = a.world_state()
    assert len(world.objects) == 2, f"both crates are BODIES the rollout simulates, got {len(world.objects)}"
    assert {world.kind_of(oid) for oid in world.objects} == {CRATE}, "each keyed by its own index, each of kind CRATE"
    assert world.kind_of("wall") == "wall", "a static feature is already a kind and passes through"
