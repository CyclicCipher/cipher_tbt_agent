"""The sensorimotor retina — turn a raw ARC frame into RECURRING (feature, pose) observations.

Global ARC frames never recur (measured: 0 revisits on the live games), so a column cannot treat the whole frame
as one state — there is nothing to learn. But LOCAL receptive fields recur ~99% (a 5x5 patch of a given local
pattern repeats across positions, frames, and games), so the column must sense the frame through narrow RFs. That
is also why TBT is sensorimotor in the first place. This module is the SENSOR side of the thousand-brains retina:
it sweeps narrow receptive fields over the frame and emits a (feature, pose) observation per RF, where

  feature = a label-free, online code for the local patch (the same patch -> the same id, so observations recur),
  pose    = the RF location (the 'where'; relative poses between RFs are how relations are read -- Monty's CMP vote).

No assumption about colours, objects, actions, body, or goal -- only "what local pattern is where". One
CorticalColumn will later sit on each RF (Monty's 1:1 sensor<->learning module); the column learns the layout and
the per-action operators over these recurring observations. Exogenous attention (the saccade-policy bootstrap) is
drawn to the dynamic RESIDUAL -- the cells that change between frames -- which the live games show is a small,
coherent, controllable object on a mostly-static layout. Pure stdlib (patch hashing): torch-free, runs in the
test venv. RF size / stride are ARC-calibrated hyperparameters (5x5-8x8 measured best), not first principles.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional


def background(frame):
    """The background colour = the most common cell value — retinal ADAPTATION to the dominant level (a sensor primitive,
    not a colour assumption). The one legitimate frame-wide reduction a retina does before delivering feature-at-locations."""
    return Counter(v for row in frame for v in row).most_common(1)[0][0]


def connected_figure(cells, seeds):
    """The 4-connected figure of `cells` that CONTAINS any of `seeds` — retinal figure-ground COMPLETION of a motion-defined
    seed into the whole cohesive body (Spelke cohesion). Motion-gated (the seed is the cells that MOVED), so it is NOT a
    global colour segmenter — it completes ONE figure the motion already picked out. Returns the cell set (or empty)."""
    cells, seeds = set(cells), set(seeds)
    figure, q = set(), deque(s for s in seeds if s in cells)
    figure.update(q)
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if p in cells and p not in figure:
                figure.add(p)
                q.append(p)
    return figure


def salient_targets(frame, exclude=(), k: int = 1, bg=None, suppress: float = 2.5):
    """The retina/SC SALIENCE peaks — the most feature-CONTRASTING locations, excluding the tracked self (`exclude` cells).
    Center-surround: a cell's pop-out = the fraction of its 4-neighbours whose colour DIFFERS (an isolated distinct cell —
    a marker — pops out; a uniform block's interior does not). Returns up to `k` winner-take-all peaks `[(x, y), ...]`
    with non-max suppression (radius `suppress`), so distinct objects get distinct slots. NO segmentation — just local
    maxima of a per-cell contrast map (colour as a peripheral feature, not identity)."""
    if bg is None:
        bg = background(frame)
    H, W = len(frame), len(frame[0])
    exclude = set((int(round(x)), int(round(y))) for x, y in exclude)
    scored = []
    for y in range(H):
        for x in range(W):
            v = frame[y][x]
            if v == bg or (x, y) in exclude:
                continue
            diff = tot = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    tot += 1
                    diff += (frame[ny][nx] != v)
            pop = diff / tot if tot else 0.0
            if pop > 0:
                scored.append((pop, x, y))
    scored.sort(reverse=True)
    peaks: list = []
    for _pop, x, y in scored:
        if all((x - px) ** 2 + (y - py) ** 2 > suppress ** 2 for px, py in peaks):   # non-max suppression
            peaks.append((x, y))
            if len(peaks) >= k:
                break
    return peaks


class Retina:
    """A tiling of narrow receptive fields with one shared, online-grown patch vocabulary. `rf` = the RF edge,
    `stride` = the step between RFs (stride < rf overlaps; default tiles without overlap). The codebook is
    label-free: a never-seen patch gets the next id, so the local alphabet is discovered by watching, never
    injected ([[feedback_bitter_lesson]])."""

    def __init__(self, rf: int = 5, stride: Optional[int] = None):
        self.rf = rf
        self.stride = stride if stride is not None else rf
        self.codebook: dict = {}                              # canonical patch -> feature id (grows online)

    def _patch(self, frame, x, y):
        return tuple(tuple(frame[y + i][x:x + self.rf]) for i in range(self.rf))

    def feature(self, patch) -> int:
        """The patch's feature id, adding it to the vocabulary if novel (label-free online discovery)."""
        fid = self.codebook.get(patch)
        if fid is None:
            fid = self.codebook[patch] = len(self.codebook)
        return fid

    def sense(self, frame, x, y):
        """One RF observation at top-left (x, y): (feature_id, pose=(x, y))."""
        return self.feature(self._patch(frame, x, y)), (x, y)

    def perceive(self, frame):
        """Sweep the retina over the frame -> [(feature_id, pose), ...] for the RF tiling."""
        H, W = len(frame), len(frame[0])
        return [self.sense(frame, x, y)
                for y in range(0, H - self.rf + 1, self.stride)
                for x in range(0, W - self.rf + 1, self.stride)]


def salient_cells(prev, cur):
    """Exogenous attention: the cells that CHANGED between two frames -- the dynamic residual the saccade motor is
    drawn to (the live games show this is a small, coherent, controllable object on a mostly-static layout). The
    bottom-up 'what moved' channel that bootstraps the learned (top-down) saccade policy."""
    return {(x, y) for y in range(len(cur)) for x in range(len(cur[0])) if prev[y][x] != cur[y][x]}


def view_signature(cloud, ndigits: int = 3):
    """The retina's CONTENT ENCODER (P1 slice 3): the rotation+translation-invariant, colour-aware descriptor of a local
    VIEW -- raw coloured cells -> an OPAQUE key the column encodes to an L4 feature id (an SDR). It pairs the sorted
    COLOUR multiset (composition; also covers a single cell) with the sorted multiset of `(colour-pair, pairwise distance)`
    over all cell pairs (the invariant GEOMETRY): pairwise distances are preserved by any rotation/translation, so the
    SAME shape+colours at any pose/position -> the SAME descriptor, while a different colouring or shape differs. This is
    MODALITY-SPECIFIC extraction and lives in the PERIPHERAL (reference_tbt_feature_definition); the column stays
    content-opaque. NB an EXACT-MATCH encoder: a truly general one would map SIMILAR views to OVERLAPPING SDRs (deferred).
    `cloud` = `[(x, y, colour), ...]`."""
    cells = [(float(x), float(y), c) for (x, y, c) in cloud]
    colours = tuple(sorted(c for _x, _y, c in cells))
    pairs = []
    for i in range(len(cells)):
        xi, yi, ci = cells[i]
        for j in range(i + 1, len(cells)):
            xj, yj, cj = cells[j]
            d = round(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5, ndigits)
            pairs.append((min(ci, cj), max(ci, cj), d))
    return colours, tuple(sorted(pairs))


def dominant_region(cells):
    """The largest 4-connected component of `cells` and its centroid -- where exogenous attention foveates (the
    primary moving object). Returns (component, (cx, cy)), or (set(), None) if there was no change."""
    cells = set(cells)
    seen, best = set(), set()
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
        if len(comp) > len(best):
            best = comp
    if not best:
        return set(), None
    cx = sum(x for x, _ in best) / len(best)
    cy = sum(y for _, y in best) / len(best)
    return best, (cx, cy)
