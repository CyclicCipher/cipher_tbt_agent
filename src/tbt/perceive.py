"""Perception bridge -- raw frames to TRACKED OBJECTS, with NO privileged 'self'.

The self is not a perception concept. There are only OBJECTS; which one is controllable is decided NOWHERE here -- it
EMERGES downstream as the object whose learned operator is action-sensitive ("the self is the factor your operators
move"). Colour is never used (an arbitrary label that anything can share); an object is WHERE it is and its SIZE/shape.

Mechanism: segment each frame into objects (connected components of non-background cells, background = the modal
value) and link each to the nearest compatible object last frame (object permanence), keeping a stable id so a
per-object operator can be learned over a moving thing. On a movement game the controllable object is a distinct
shape that segments cleanly and tracks with one stable id (validated on cn04: ONE id with ACTION1→(0,-3) … ACTION4→
(3,0), conf 1.0; the static objects stay identity). On a state-change game (ls20) no component translates, so nothing
becomes action-sensitive -- correctly, no controllable object.

(An earlier salient-residual variant -- read movers from the motion residual -- was tried to isolate a mover embedded
in the static structure, but it FRAGMENTS a clean multi-part object into many ids and mis-reads an overlapping move as
its leading edge; full-frame tracking is cleaner for the distinct objects the real games actually have. Grouping a
genuinely MULTI-COMPONENT object by common motion is a future need, not a current one.) Pure stdlib; reuses
`objects.components`.
"""

from __future__ import annotations

from collections import Counter

from .objects import components


def background(frame):
    """The background colour = the most common cell value (a sensor primitive, not a colour assumption)."""
    return Counter(v for row in frame for v in row).most_common(1)[0][0]


def segment(frame, bg=None):
    """The frame's objects as `(cells, centroid)` -- the non-background 4-connected components. No colour."""
    if bg is None:
        bg = background(frame)
    cells = {(x, y) for y, row in enumerate(frame) for x, v in enumerate(row) if v != bg}
    return components(cells)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ObjectField:
    """The field of tracked objects -- no self. `perceive(frame)` segments the frame and links each object to the
    nearest compatible object from the previous frame (object permanence), returning `{object_id: (pose, size)}` with
    ids stable across frames. `max_jump` bounds how far an object may move and still be the same object; size must be
    roughly preserved to link (so a small object is not matched to a big one)."""

    def __init__(self, max_jump: float = 12.0):
        self.max_jump = max_jump
        self._last: dict = {}                                  # object_id -> (pose, size)
        self._next = 0

    def reset(self):
        self._last = {}
        self._next = 0

    def perceive(self, frame) -> dict:
        objs = [(centroid, len(cells)) for cells, centroid in segment(frame)]
        prev, result, used = dict(self._last), {}, set()
        for pose, size in objs:
            best, best_d = None, self.max_jump
            for oid, (ppose, psize) in prev.items():
                if oid in used or abs(psize - size) > max(3, 0.5 * psize):    # size must be roughly preserved
                    continue
                d = _dist(pose, ppose)
                if d <= best_d:
                    best, best_d = oid, d
            if best is None:                                  # a newly-appeared object
                best = self._next
                self._next += 1
            used.add(best)
            result[best] = (pose, size)
        self._last = result
        return result
