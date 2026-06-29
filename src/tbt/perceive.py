"""Perception -- raw frames to TRACKED OBJECTS, with NO privileged 'self', and OBJECT PERMANENCE through contact.

The self is not a perception concept. There are only OBJECTS; which one is controllable is decided NOWHERE here -- it
EMERGES downstream as the object whose learned operator (L5) is action-sensitive ("the self is the factor your
operators move"). Colour is never used to RECOGNISE an object (an arbitrary label anything can share); an object is
WHERE it is, its SIZE/shape, and -- as a feature -- its content.

An object's identity is its HISTORY, not its current appearance: appearance only PROPOSES a grouping; the dynamics
dispose. So we never re-segment from scratch: segment each frame into connected components of non-background cells
(background = the modal value), then RE-FIND each previously-tracked object in the new frame, carrying its id forward.
PERMANENCE: a component is matched to the previously-tracked objects whose last pose lies in it; a component holding
SEVERAL of them is those objects in CONTACT, split back by (1) permanence (each keeps its own cells) and (2) for new
cells, MOTION (the cell goes to whichever object's stayed-or-predicted pose is nearest), never a geometric size guess.
So a self touching a wall stays the self at its own pose (it never fuses into a self+wall blob). Pure stdlib.
"""

from __future__ import annotations

import math
from collections import Counter, deque


def _r(v) -> int:
    """Round half UP (monotonic) -- not banker's rounding, whose half-to-even aliases adjacent half-integer poses."""
    return int(math.floor(v + 0.5))


def canonicalize(elements):
    """The TRANSLATION-INVARIANT config of `elements` = [(pose, feature), ...] where `feature` is a tuple
    `(size,)` or `(size, content, ...)`: each feature paired with its integer pose RELATIVE to the LARGEST element
    (max by (size, pose) -- a stable emergent anchor, not a privileged self), sorted. The same RELATIVE arrangement
    anywhere on the board yields the SAME tuple, so the loop's states RECUR and the SR / operator can learn them.

    ONE encoding, shared by the sensor (frames -> state) and L5 (operator -> predicted next state): each element comes
    out `(size, rel_pose, *content)` -- size first, the pose in the middle, any extra features last. A removed object
    is simply absent (so a 'required-absent' goal needs no special case)."""
    elements = list(elements)
    if not elements:
        return ()
    ax, ay = max(elements, key=lambda e: (e[1][0], e[0]))[0]            # the largest object (ties by pose) = the anchor
    def enc(e):
        pose, feat = e
        rp = (_r(pose[0] - ax), _r(pose[1] - ay))
        return (feat[0], rp) + tuple(feat[1:])
    return tuple(sorted(enc(e) for e in elements))


def components(cells):
    """All 4-connected components of `cells`, each as `(cellset, centroid)`."""
    cells = set(cells)
    seen, out = set(), []
    for s in cells:
        if s in seen:
            continue
        comp, q = set(), deque([s])
        seen.add(s)
        while q:
            x, y = q.popleft()
            comp.add((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (x + dx, y + dy)
                if p in cells and p not in seen:
                    seen.add(p)
                    q.append(p)
        cx = sum(x for x, _ in comp) / len(comp)
        cy = sum(y for _, y in comp) / len(comp)
        out.append((comp, (cx, cy)))
    return out


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


def content_sig(frame, cells):
    """A translation-invariant signature of an object's CONTENT (its 'what'): each cell's value at its position
    RELATIVE to the object's corner. Stable as the object MOVES (movement lives in the pose); it changes only when the
    appearance changes in place -- a colour toggle, a deformation -- the behaviour the content/dynamics column learns."""
    if not cells:
        return ()
    minx = min(x for x, _y in cells)
    miny = min(y for _x, y in cells)
    return tuple(sorted((x - minx, y - miny, frame[y][x]) for (x, y) in cells))


class ObjectField:
    """The field of tracked objects -- no self. `perceive(frame, predict)` segments the frame and RE-FINDS each
    previously-tracked object (object permanence), returning `{object_id: (pose, size)}` with ids stable across frames
    and `self.cells[id]` / `self.contents[id]` the object's cells + content this frame. `max_jump` bounds real motion."""

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
        """Segment and re-find the tracked objects, carrying ids forward. `predict(oid, pose) -> pose` is where each
        object was expected to go under the last action (path integration via L5); it disambiguates objects in contact.
        Default = identity (no dynamics), fine when objects do not touch."""
        if predict is None:
            predict = lambda _oid, pose: pose
        comps = segment(frame)
        prev, prev_cells = dict(self._last), self.cells
        # Map each previously-tracked object to the ONE component its NEW position falls in (it stayed, or it moved as
        # its operator predicts). A far object never gets pulled into a neighbour's blob; a component then carries the
        # object(s) really there: 0 -> new object, 1 -> that object, >=2 -> objects in CONTACT (split below).
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
                    if owner is None:                          # a NEW cell -> whose MOTION best explains it (not size)
                        owner = min(claimants, key=lambda o: min(_dist(p, c) for p in cand[o]))
                    groups[owner].add(c)
                for oid, gcells in groups.items():
                    if gcells:
                        cx = sum(x for x, _ in gcells) / len(gcells)
                        cy = sum(y for _, y in gcells) / len(gcells)
                        result[oid], cells[oid] = ((cx, cy), len(gcells)), gcells
        self._last, self.cells = result, cells
        self.contents = {oid: content_sig(frame, cs) for oid, cs in cells.items()}
        return result
