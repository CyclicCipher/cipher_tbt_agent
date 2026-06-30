"""Perception -- raw frames to STATELESS PROTO-OBJECTS (a pure sensor organ; Phase 4 of the refactor).

The self is not a perception concept. There are only proto-objects; which one is controllable is decided NOWHERE here
-- it EMERGES downstream as the object whose learned operator (L5) is action-sensitive ("the self is the factor your
operators move"). Colour is not used for IDENTITY (recognition is shape-based; colour is a weak feature), but a colour
BOUNDARY is a candidate object border (border-ownership) -- so `segment` is colour-aware.

These are Rensink's VOLATILE PROTO-OBJECTS: a fast, parallel, pre-attentive grouping PROPOSAL, recomputed every frame
with NO MEMORY -- not a tracker. PERMANENCE and IDENTITY (binding proto-objects across frames into stable objects) are
the COLUMN's job, via L2/3 recognition + path-integration, NOT a hand-coded position tracker. The proposal is
REVISABLE: the column may split a multi-colour over-segment or merge a same-colour touch (recognition mismatch). This
is the principled demotion (reference_tbt_segmentation_and_grouping): the sensor proposes, the column disposes.
Pure stdlib.
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
    """The frame's proto-objects as `(cells, centroid)` -- SAME-COLOUR 4-connected components of non-background cells.
    Colour-AWARE (a colour boundary is a candidate object border; border-ownership). Stateless, recomputed per frame
    (Rensink's volatile proto-objects). NB a multi-colour object is over-segmented into colour-parts -- the column's
    recognition/merge recombines it (deferred); colour-blind under-segmentation (different-colour touches MERGING, e.g.
    a mover bumping a wall) is the worse failure here and is what this avoids -- which is why it replaces contact-split."""
    if bg is None:
        bg = background(frame)
    H, W = len(frame), len(frame[0])
    seen, out = set(), []
    for y in range(H):
        for x in range(W):
            if frame[y][x] == bg or (x, y) in seen:
                continue
            val = frame[y][x]                                  # group only cells of the SAME colour (border = contrast)
            comp, q = set(), deque([(x, y)])
            seen.add((x, y))
            while q:
                cx, cy = q.popleft()
                comp.add((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen and frame[ny][nx] == val:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            cx_ = sum(p[0] for p in comp) / len(comp)
            cy_ = sum(p[1] for p in comp) / len(comp)
            out.append((comp, (cx_, cy_)))
    return out


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
    """A STATELESS proto-object proposer (Phase 4: ObjectField demoted to a pure sensor organ). Each frame it segments
    into colour-aware proto-objects and exposes `{index: (pose, size)}` with `self.cells[i]` / `self.contents[i]` the
    proto-object's cells + content -- NO cross-frame ids, NO permanence, NO tracking (Rensink's volatile proto-objects).
    Permanence + identity are the COLUMN's job (L2/3 recognition + path-integration), not a hand-coded position tracker
    (reference_tbt_segmentation_and_grouping). The per-frame `index` is provisional -- consumers key on recognition
    (barriers) or sort by size/pose (click-slots) or snapshot the set (config_state), none needing id stability."""

    def __init__(self):
        self._last: dict = {}                                  # index -> (pose, size) THIS frame
        self.cells: dict = {}                                  # index -> the proto-object's cell set this frame
        self.contents: dict = {}                              # index -> its content signature this frame (the 'what')

    def reset(self):
        self._last, self.cells, self.contents = {}, {}, {}

    def perceive(self, frame) -> dict:
        """Segment the frame into stateless colour-aware proto-objects: `{index: (centroid, size)}`, with cells +
        content per index. Recomputed every frame (no memory) -- the sensor PROPOSES; the column binds + disposes."""
        comps = segment(frame)
        self.cells = {i: set(comp) for i, (comp, _c) in enumerate(comps)}
        self._last = {i: (c, len(comp)) for i, (comp, c) in enumerate(comps)}
        self.contents = {i: content_sig(frame, cs) for i, cs in self.cells.items()}
        return self._last
