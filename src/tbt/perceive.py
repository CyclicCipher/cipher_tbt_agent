"""Perception bridge -- raw frames to a scene of objects (the self + the others), for the playing loop.

The planner reasons over `(self_pose, others)`; this turns `(prev, action, cur)` frames into that, with no role decode:

  * SEGMENT the frame into objects -- connected components of non-background cells (background = the modal cell value).
    Connected-component objectness is a Core-Knowledge prior used ONLY as a sensor primitive (it proposes candidate
    objects), never a role decoder ([[feedback_bitter_lesson]]).
  * Identify the SELF as the object that MOVES under the agent's actions -- the controllable mover (its identity is
    remembered once found, so a momentarily-still self is still located). Everything else is `others`: the static
    landmarks the self navigates RELATIVE to (they are not in the dynamic residual, so the full frame is segmented).
  * Report TRUE positions (object centroids in the current frame), so the forward model's operator and the goal's
    relative encoding are exact -- unlike the change-blob centroid `objects.py` uses to recover the operator.

Identity is the object's dominant colour, so a goal generalises by what an object IS, not which instance it is. This
is the NEW perception, living in `tbt/` -- the dissolution direction (perception/ shrinks to a thin array sensor).
Several movers / many static objects (a busy real frame) are the deferred grouping step; here one self + a few
landmarks. Pure stdlib; reuses `objects.components`.
"""

from __future__ import annotations

from collections import Counter

from .objects import components


def background(frame):
    """The background colour = the most common cell value (a sensor primitive, not a colour assumption)."""
    return Counter(v for row in frame for v in row).most_common(1)[0][0]


def segment(frame, bg=None):
    """The frame's objects: each non-background 4-connected component as `(colour, cells, centroid)`, colour = the
    component's dominant value."""
    if bg is None:
        bg = background(frame)
    cells = {(x, y) for y, row in enumerate(frame) for x, v in enumerate(row) if v != bg}
    out = []
    for comp, centroid in components(cells):
        colour = Counter(frame[y][x] for (x, y) in comp).most_common(1)[0][0]
        out.append((colour, comp, centroid))
    return out


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_same_colour(colour, centroid, objs):
    """The centroid of the nearest same-colour object in `objs`, or None."""
    cands = [cen for (col, _cells, cen) in objs if col == colour]
    return min(cands, key=lambda c: _dist(centroid, c)) if cands else None


class ScenePerceiver:
    """Frames -> `(self_pose, others)`. Stateful only in remembering WHICH object is the self (by colour), learned
    from motion: the object that moves most under an action is the controllable self. Until that is known, the self
    is unlocated (the caller babbles to make something move). `others = [(colour, centroid), ...]`."""

    def __init__(self, move_threshold: float = 0.5):
        self.move_threshold = move_threshold                  # min centroid move to count an object as "the mover"
        self.self_colour = None

    def reset(self):
        self.self_colour = None

    def _identify_self(self, prev, cur_objs):
        """Set `self_colour` to the object that moved most between `prev` and the current objects (the mover = the
        controllable self). Called until the self is identified."""
        prev_objs = segment(prev)
        best, best_move = None, self.move_threshold
        for colour, _cells, cen in cur_objs:
            pc = _nearest_same_colour(colour, cen, prev_objs)
            move = _dist(cen, pc) if pc is not None else 0.0
            if move >= best_move:
                best, best_move = colour, move
        if best is not None:
            self.self_colour = best

    def perceive(self, prev, action, cur):
        """`(prev, action, cur)` -> `(self_pose, others)`. `self_pose` is None until the self is identified (so the
        caller babbles); `others` is always the non-self objects (static landmarks)."""
        objs = segment(cur)
        if not objs:
            return None, []
        if self.self_colour is None and prev is not None:
            self._identify_self(prev, objs)
        if self.self_colour is None:
            return None, [(col, cen) for col, _cells, cen in objs]
        mine = [(col, cells, cen) for col, cells, cen in objs if col == self.self_colour]
        if not mine:                                          # self temporarily not visible -> locate nothing this step
            return None, [(col, cen) for col, _cells, cen in objs]
        _col, _cells, self_pose = max(mine, key=lambda o: len(o[1]))   # the largest same-colour blob is the self body
        others = [(col, cen) for col, _cells, cen in objs if cen != self_pose]
        return self_pose, others
