"""Object perception (E) — segment a raw colour grid into multi-cell OBJECTS, with no semantic priors.

Real ARC-AGI-3 gives a 64×64 grid where each cell is one of 16 colours, and the games are little worlds whose
pieces are multi-cell regions (a wall, a player, an item). The agent must turn that raw frame into OBJECTS
before it can reason — this is the perception gate E. An object here is just a connected region of non-
background cells (the Core-Knowledge "objectness" prior); its ROLE (key/wall/goal/…) is learned later from
dynamics, never read off its colour. Generic over grid size + palette, so the SAME parser handles the replica's
small frames and real 64×64 ones.

  segment(grid)       → [Obj]   connected-component objects (same colour by default = the ARC objectness prior).
  object_motion(a, b) → [(obj, (dx,dy))]   objects that TRANSLATED between two frames — the substrate for the
                        efference copy (the body is whichever object moved by the issued action's delta) and
                        for spotting pushable pieces (other objects that moved).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .perceive import modal_background                        # the cached canonical impl (one concept, shared)

_4 = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_8 = _4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass(frozen=True)
class Obj:
    color: int
    cells: frozenset            # frozenset of (x, y)

    @property
    def size(self):
        return len(self.cells)

    @property
    def bbox(self):             # (min_x, min_y, max_x, max_y)
        xs = [x for x, _ in self.cells]; ys = [y for _, y in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self):
        n = len(self.cells)
        return (sum(x for x, _ in self.cells) / n, sum(y for _, y in self.cells) / n)

    @property
    def shape(self):            # cells relative to the bbox corner — translation-invariant identity
        bx, by, _, _ = self.bbox
        return frozenset((x - bx, y - by) for x, y in self.cells)


def segment(grid, bg=None, conn=4, multicolor=False):
    """Connected-component objects: each is a region of connected non-background cells. Same colour by default
    (the ARC objectness prior); `multicolor=True` groups any adjacent non-bg cells into one blob. `conn` is 4
    (orthogonal) or 8 (incl. diagonals). O(H·W) — fine at 64×64."""
    H, W = len(grid), len(grid[0])
    if bg is None:
        bg = modal_background(grid)
    nbrs = _8 if conn == 8 else _4
    nonbg = {(x, y): grid[y][x] for y in range(H) for x in range(W) if grid[y][x] != bg}   # one scan
    seen, objs = set(), []
    for start, col in nonbg.items():                          # walk the (few) content cells, not the whole frame
        if start in seen:
            continue
        comp, q = set(), deque([start])
        seen.add(start)
        while q:
            x, y = q.popleft(); comp.add((x, y))
            for dx, dy in nbrs:
                p = (x + dx, y + dy)
                if p in nonbg and p not in seen and (multicolor or nonbg[p] == col):
                    seen.add(p); q.append(p)
        objs.append(Obj(col, frozenset(comp)))
    return objs


def object_motion(prev, cur):
    """Translations between two frames as [(prev_obj, (dx,dy))], for objects whose (colour, shape) recurs at a
    shifted position. Objects matched in place first (they stayed); the rest matched to the nearest same-shape
    object (they moved). The agent reads the body off this: the object that moved by the action's delta."""
    cur_by = defaultdict(list)
    for o in cur:
        cur_by[(o.color, o.shape)].append(o)
    used = defaultdict(set)
    moved = []
    for po in prev:
        key = (po.color, po.shape)
        cand = cur_by.get(key, [])
        px, py = po.bbox[0], po.bbox[1]
        in_place = [i for i, co in enumerate(cand) if i not in used[key] and (co.bbox[0], co.bbox[1]) == (px, py)]
        if in_place:                                  # the object stayed put
            used[key].add(in_place[0]); continue
        opts = [(abs(co.bbox[0] - px) + abs(co.bbox[1] - py), i, co)
                for i, co in enumerate(cand) if i not in used[key]]
        if opts:                                      # nearest same-shape object → it translated
            _, i, co = min(opts); used[key].add(i)
            moved.append((po, (co.bbox[0] - px, co.bbox[1] - py)))
    return moved


# ── validation ──────────────────────────────────────────────────────────────────────────────────────────
def _blank(n, bg=0):
    return [[bg] * n for _ in range(n)]


if __name__ == "__main__":
    import os, sys
    print("object perception (E): segment a raw colour grid into multi-cell objects\n")

    # (1) synthetic 64x64 with three multi-cell objects of different shapes
    g = _blank(64)
    for y in range(10, 13):
        for x in range(10, 13):
            g[y][x] = 2                                # a 3x3 square (size 9)
    g[30][30] = g[31][30] = g[31][31] = 3              # an L-tromino (size 3)
    for x in range(20, 26):
        g[50][x] = 4                                   # a horizontal line (size 6)
    objs = sorted(segment(g), key=lambda o: -o.size)
    print(f"  (1) synthetic 64x64: {len(objs)} objects "
          f"-> sizes {[o.size for o in objs]}  colours {[o.color for o in objs]}")
    print(f"      square bbox {objs[0].bbox}  line bbox {objs[1].bbox}  L bbox {objs[2].bbox}")

    # (2) a real replica frame -- the wall border becomes ONE multi-cell object; pieces are their own objects
    from tasks.games import LockPath                                      # noqa: E402
    game = LockPath(); game.load_level(3); frame = game.render()[0]
    fobjs = segment(frame)
    big = max(fobjs, key=lambda o: o.size)
    print(f"\n  (2) LockPath L3 frame ({len(frame)}x{len(frame[0])}): {len(fobjs)} objects; "
          f"largest = colour {big.color} size {big.size} (the wall border, grouped as one object)")
    print(f"      object colours+sizes: {sorted((o.color, o.size) for o in fobjs)}")

    # (3) object-level motion: shift the square by (+1, 0), leave the L; the line vanishes
    g2 = _blank(64)
    for y in range(10, 13):
        for x in range(11, 14):
            g2[y][x] = 2
    g2[30][30] = g2[31][30] = g2[31][31] = 3
    moved = object_motion(segment(g), segment(g2))
    print(f"\n  (3) motion: {[(o.color, d) for o, d in moved]}  "
          f"(square colour 2 moved (1,0); L stayed; line gone) -- the efference copy reads the body off this")
