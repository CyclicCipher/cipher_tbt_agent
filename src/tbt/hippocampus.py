"""The hippocampus -- the allocentric cognitive map (place cells + content-at-place). H1 of HIPPOCAMPUS.md.

The world-anchored frame the egocentric sensor lacked. It binds the per-frame proto-objects (the entorhinal gateway:
read straight from the columns' perception, no thalamus on the way IN -- see HIPPOCAMPUS.md 3b) to allocentric world
SLOTS (place cells), producing a FACTORED scene state:

    scene(frame) -> ( (slot, colour, size), ... )   sorted, hashable

  * WHERE = the slot, an allocentric position binned at `pos_bin`. Anchored to the object's bounding-box MIN CORNER,
    not its centroid -- a growing/reshaping structure keeps a STABLE slot as it extends on one side (the centroid
    drifts; the corner does not). This is the place-cell code.
  * WHAT = (colour, size) -- a COARSE content feature BY DESIGN. The per-pixel content signature CHURNS (the 7b
    failure that drove the egocentric retreat) and the whole-board conjunction NEVER recurs (cn04: 70/76 distinct);
    (colour, size) compresses cn04's growing tree to its ~33 growth stages -- a small graph the L5 operator
    generalises over (predict size -> size+delta). Colour is the object's single segment colour (segment is
    colour-aware); size is the cell count, optionally binned.

Absolute slots = allocentric (the ARC board is a FIXED top-down world). `translation_invariant=True` drops the
absolute origin (slots relative to the largest object) for a PURE-transform game where position is irrelevant -- on
cn04 the absolute form already wins (corner-anchored slots barely move), so absolute is the default.

STATELESS per frame (Rensink's volatile proto-objects; the column binds permanence). Path-integration + landmark
loop-closure (H2/H3) are deferred -- this is the place+content core. The allocentric frame is broadcast back to the
columns as a top-down prior through the THALAMUS (`read_location`); that OUT path earns its keep at multi-column /
movement games and is not wired here. Pure stdlib.
"""

from __future__ import annotations

from .perceive import background, segment


def _anchor(cells):
    """A growth-stable anchor: the bounding-box MIN corner. Unlike the centroid it does not drift as the object
    grows/reshapes on one side, so a transforming structure keeps a stable allocentric slot."""
    return (min(x for x, _ in cells), min(y for _, y in cells))


class Hippocampus:
    """Frame -> allocentric factored scene state. `scene(frame)` returns `(state, objects)` where `state` is the
    hashable sorted tuple of `(slot, colour, size)` elements and `objects` is `[(slot, colour, size, cells, anchor)]`
    (the per-object detail downstream layers -- L2/3 recognition, the L5 operator -- consume)."""

    def __init__(self, pos_bin: int = 4, size_bin: int = 1, translation_invariant: bool = False):
        self.pos_bin = pos_bin
        self.size_bin = max(1, size_bin)
        self.translation_invariant = translation_invariant

    def scene(self, frame):
        bg = background(frame)
        objs = []
        for cells, _centroid in segment(frame, bg=bg):
            x0, y0 = next(iter(cells))
            colour = frame[y0][x0]
            ax, ay = _anchor(cells)
            slot = (ax // self.pos_bin, ay // self.pos_bin)
            size = len(cells) // self.size_bin
            objs.append((slot, colour, size, frozenset(cells), (ax, ay)))
        if self.translation_invariant and objs:
            # re-origin to the LARGEST object's slot (the stable emergent anchor canonicalize uses) -- position-free
            ox, oy = max(objs, key=lambda o: (o[2], o[0]))[0]
            objs = [((s[0] - ox, s[1] - oy), c, z, cs, a) for (s, c, z, cs, a) in objs]
        state = tuple(sorted((slot, colour, size) for (slot, colour, size, _cs, _a) in objs))
        return state, objs
