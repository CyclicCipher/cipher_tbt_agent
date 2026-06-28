"""Real-ARC perception: segment a 64x64x16-colour frame into multi-cell OBJECTS and recognise each one
pose-invariantly (perception.perceive.ObjectRecognizer over tbt.recognize). The capability `Obj.shape` lacks: a
ROTATED object is tracked as the SAME object across frames — object permanence under rotation, the real-ARC
condition. Reuses the existing connected-component `segment` (no new segmentation)."""

from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from perception.perceive import ObjectRecognizer  # noqa: E402


def _frame(n=64):
    return [[0] * n for _ in range(n)]


def _place(grid, cells, color):
    for x, y in cells:
        grid[y][x] = color
    return grid


def test_rotated_object_is_recognised_as_the_same_id():
    """A multi-cell object, then the SAME object rotated 90 deg and moved elsewhere — recognised as one object id,
    where the translation-invariant shape key would call it new."""
    pr = ObjectRecognizer()
    L = [(0, 0), (0, 1), (0, 2), (1, 2)]                       # an asymmetric (chiral) 4-cell object
    o1, name1, pose1 = pr.perceive(_place(_frame(), [(10 + x, 10 + y) for x, y in L], 5))[0]
    assert name1 is not None and pose1 is not None

    Lrot = [(0, 1), (1, 1), (2, 1), (2, 0)]                    # L rotated 90 deg CW (normalised), placed far away
    o2, name2, pose2 = pr.perceive(_place(_frame(), [(30 + x, 30 + y) for x, y in Lrot], 5))[0]
    assert name2 == name1, f"rotated object read as a different id: {name1} vs {name2}"
    assert o1.shape != o2.shape, "Obj.shape differs under rotation — the gap pose-invariant recognition fills"


def test_different_shapes_get_distinct_ids():
    """Two genuinely different shapes in one frame → two distinct object ids (recognition discriminates, not just
    'an object is here')."""
    pr = ObjectRecognizer()
    L = [(0, 0), (0, 1), (0, 2), (1, 2)]
    T = [(0, 0), (1, 0), (2, 0), (1, 1)]
    grid = _place(_place(_frame(), [(10 + x, 10 + y) for x, y in L], 5), [(40 + x, 40 + y) for x, y in T], 7)
    scene = pr.perceive(grid)
    names = {name for _, name, _ in scene}
    assert len(scene) == 2 and len(names) == 2, f"shapes not distinguished: {[(n) for _, n, _ in scene]}"


def test_single_cell_objects_pass_through_without_identity():
    """A 1-cell object has no orientation — it passes through with no id (recognition is for multi-cell objects)."""
    pr = ObjectRecognizer()
    o, name, pose = pr.perceive(_place(_frame(), [(5, 5)], 3))[0]
    assert name is None and pose is None and o.size == 1
