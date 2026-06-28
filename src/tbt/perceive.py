"""Perception bridge -- raw frames to TRACKED OBJECTS, with NO privileged 'self'.

The self is not a perception concept. There are only OBJECTS; which one is controllable is decided NOWHERE here -- it
EMERGES downstream as the object whose learned operator is action-sensitive ("the self is the factor your operators
move"). And colour is never used: it is an arbitrary label (anything sharing it breaks), so an object is identified by
WHERE it is and its SIZE/shape.

The hard lesson from the live games: a moving object is often EMBEDDED in the static structure (it shares cells/colour
with it), so segmenting the whole frame by connected components cannot isolate it -- on ls20 every non-background cell
falls into a few big blobs and the mover vanishes inside one. So movers are read from the DYNAMIC RESIDUAL instead:

  * MOVERS = connected components of the cells that CHANGED (`salient_cells`) and are occupied now -- the moving
    objects, isolated from the static structure by their MOTION. Tracked across frames by pose-continuity (permanence)
    so a per-object operator can be learned.
  * STATIC anchor = everything else occupied (the structure that did not change), lumped into one big reference
    object. Being the largest, it is the configuration anchor (a stable frame); it never decides "self".

Returns `{object_id: (pose, size)}` (the static anchor under id `STATIC`). Pure stdlib; reuses `objects.components`
and `retina.salient_cells`.
"""

from __future__ import annotations

from collections import Counter

from .objects import components
from .retina import salient_cells


def background(frame):
    """The background colour = the most common cell value (a sensor primitive, not a colour assumption)."""
    return Counter(v for row in frame for v in row).most_common(1)[0][0]


def _centroid(cells):
    n = len(cells)
    return (sum(x for x, _ in cells) / n, sum(y for _, y in cells) / n)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ObjectField:
    """The field of tracked objects -- no self. `perceive(prev, frame)` reads the moving objects from the dynamic
    residual (isolated from the static structure by motion), tracks them by pose-continuity, and lumps the rest into a
    single static anchor. Returns `{object_id: (pose, size)}`. `max_jump` bounds how far a mover may travel and still be
    the same object."""

    STATIC = -1                                                # the fixed id of the static-structure anchor

    def __init__(self, max_jump: float = 8.0):
        self.max_jump = max_jump
        self._last: dict = {}                                  # mover_id -> (pose, size)
        self._next = 0

    def reset(self):
        self._last = {}
        self._next = 0

    def perceive(self, prev, frame) -> dict:
        bg = background(frame)
        non_bg = {(x, y) for y, row in enumerate(frame) for x, v in enumerate(row) if v != bg}
        salient = salient_cells(prev, frame) if prev is not None else set()
        mover_cells = salient & non_bg                        # the movers' CURRENT footprints (changed + occupied now)
        static_cells = non_bg - salient                       # the structure that did not change

        result = {}
        if static_cells:
            result[self.STATIC] = (_centroid(static_cells), len(static_cells))

        movers = [(_centroid(cells), len(cells)) for cells, _ in components(mover_cells)]
        prev_tracks, used, tracked = dict(self._last), set(), {}
        for pose, size in movers:
            best, best_d = None, self.max_jump
            for oid, (ppose, _psize) in prev_tracks.items():
                if oid in used:
                    continue
                d = _dist(pose, ppose)
                if d <= best_d:
                    best, best_d = oid, d
            if best is None:                                  # a newly-seen moving object
                best = self._next
                self._next += 1
            used.add(best)
            tracked[best] = (pose, size)
        self._last = tracked
        result.update(tracked)
        return result
