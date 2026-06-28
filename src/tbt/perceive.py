"""Perception bridge -- raw frames to TRACKED OBJECTS, with NO privileged 'self', and OBJECT PERMANENCE through contact.

The self is not a perception concept. There are only OBJECTS; which one is controllable is decided NOWHERE here -- it
EMERGES downstream as the object whose learned operator is action-sensitive ("the self is the factor your operators
move"). Colour is never used (an arbitrary label that anything can share); an object is WHERE it is and its SIZE/shape.

An object's identity is its HISTORY, not its current appearance: appearance only PROPOSES a grouping; the dynamics
dispose. (This is why a dynamics model individuates objects where a pixel model cannot.) So we never re-segment from
scratch: segment each frame into connected components of non-background cells (background = the modal value), then
RE-FIND each previously-tracked object in the new frame, carrying its id forward. The key is PERMANENCE: a component is
matched to the previously-tracked objects whose last pose lies in it, and a component that contains SEVERAL of them is
those objects in CONTACT, not a new fused object. Such a blob is split back among them by (1) PERMANENCE -- each object
keeps the cells it already owned -- and (2) for genuinely new cells, MOTION: the cell goes to the object whose movement
best explains it (nearest of {it stayed, it moved as its operator predicts}), never a geometric size guess.

So a self touching a wall stays the self at its own pose (the model keeps tracking it; it never fuses into a self+wall
blob), and a block pushed out of a wall and shoved back to look contiguous is still remembered as a block where it now
is -- matter does not re-fuse just because it looks contiguous again. This is what lets the forward model record the
real blocked transition (the move did nothing) and learn the wall as a context-gated effect; without permanence the
contact erased the self and the obstacle could never be learned. On cn04 the lone mover is one component with one
claimant -- tracked exactly as before. Pure stdlib; reuses `objects.components`.
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


def _content_sig(frame, cells):
    """A translation-invariant signature of an object's CONTENT (its 'what'): each cell's value at its position
    RELATIVE to the object's corner. Stable as the object MOVES (movement lives in the pose, the 'where'); it changes
    only when the appearance changes in place -- a colour toggle, a deformation -- which is the behavior the content
    operator learns. Hashable, so it can key states and goals."""
    if not cells:
        return ()
    minx = min(x for x, _y in cells)
    miny = min(y for _x, y in cells)
    return tuple(sorted((x - minx, y - miny, frame[y][x]) for (x, y) in cells))


class ObjectField:
    """The field of tracked objects -- no self. `perceive(frame, predict)` segments the frame and RE-FINDS each
    previously-tracked object (object permanence), returning `{object_id: (pose, size)}` with ids stable across frames
    and `self.cells[id]` the object's cells this frame. Each object is mapped to the component its new position lies in
    (stayed, or moved as `predict` says); a component shared by several objects is them in contact, split back by
    permanence + motion. `max_jump` bounds how far an object may move and still be the same object."""

    def __init__(self, max_jump: float = 12.0):
        self.max_jump = max_jump
        self._last: dict = {}                                  # object_id -> (pose, size)
        self.cells: dict = {}                                  # object_id -> the object's cell set this frame
        self.contents: dict = {}                              # object_id -> its content signature this frame (the 'what')
        self._next = 0

    def reset(self):
        self._last = {}
        self.cells = {}
        self.contents = {}
        self._next = 0

    def perceive(self, frame, predict=None) -> dict:
        """Segment and re-find the tracked objects, carrying ids forward (permanence). `predict(oid, pose) -> pose` is
        where each object was expected to go under the action just taken (path integration via its learned operator);
        it disambiguates objects in contact. Default = identity (no dynamics), fine when objects never touch."""
        if predict is None:
            predict = lambda _oid, pose: pose
        comps = segment(frame)
        prev, prev_cells = dict(self._last), self.cells
        # Map each previously-tracked object to the ONE component its NEW position falls in (it stayed, or it moved as
        # its operator predicts -- nearest cell of either). So an object only claims the component it is actually in, and
        # a far object is never pulled into a neighbour's blob. A component then carries the object(s) really there:
        # 0 -> a new object, 1 -> that object, >=2 -> objects in CONTACT (split below). max_jump bounds real motion.
        claim: dict = {}
        for oid, (ppose, _ps) in prev.items():
            cand = (ppose, predict(oid, ppose))
            best_i, best_d = None, self.max_jump
            for i, (comp, _c) in enumerate(comps):
                d = min(min(_dist(p, c) for p in cand) for c in comp)
                if d <= best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                claim.setdefault(best_i, []).append(oid)
        result, cells = {}, {}
        for i, (comp, centroid) in enumerate(comps):
            claimants = claim.get(i, [])
            if len(claimants) <= 1:                            # a lone object (or a brand-new one) -> the whole component
                oid = claimants[0] if claimants else self._next
                if not claimants:
                    self._next += 1
                result[oid], cells[oid] = (centroid, len(comp)), set(comp)
            else:                                              # CONTACT: split the blob back into the objects it contains
                cand = {oid: (prev[oid][0], predict(oid, prev[oid][0])) for oid in claimants}   # {stayed, moved}
                groups: dict = {oid: set() for oid in claimants}
                for c in comp:
                    owner = next((o for o in claimants if c in prev_cells.get(o, ())), None)    # permanence: keep your cells
                    if owner is None:                          # a NEW cell -> whose MOTION best explains it (not size/geometry)
                        owner = min(claimants, key=lambda o: min(_dist(p, c) for p in cand[o]))
                    groups[owner].add(c)
                for oid, gcells in groups.items():
                    if gcells:
                        cx = sum(x for x, _ in gcells) / len(gcells)
                        cy = sum(y for _, y in gcells) / len(gcells)
                        result[oid], cells[oid] = ((cx, cy), len(gcells)), gcells
        self._last, self.cells = result, cells
        self.contents = {oid: _content_sig(frame, cs) for oid, cs in cells.items()}   # the 'what' of each tracked object
        return result
