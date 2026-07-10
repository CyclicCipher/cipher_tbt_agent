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

from .encoders import SDR, CategoryEncoder, ScalarEncoder   # the SDR encoder library — the content SDR's substrate (M3)

# The content SDR's fixed sub-encoders (SDR_MIGRATION.md M3). Pre-populated + never grown, so `view_sdr` is a pure,
# deterministic function (no shared mutable state): ARC's 16-colour palette is UNORDERED (a CategoryEncoder — disjoint
# blocks, no false colour-nearness, the bitter lesson) and pairwise DISTANCE is a metric quantity (a ScalarEncoder —
# nearby distances OVERLAP). The two fields are concatenated into one content SDR.
_COLOUR_ENC = CategoryEncoder(categories=range(16), w=5, capacity=16)   # colour presence (composition)
_DIST_ENC = ScalarEncoder(0.0, 24.0, n=60, w=7, clip=True)             # pairwise distance -> an overlapping bump


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


def salient_cells(prev, cur):
    """Exogenous attention: the cells that CHANGED between two frames -- the dynamic residual the saccade motor is
    drawn to (the live games show this is a small, coherent, controllable object on a mostly-static layout). The
    bottom-up 'what moved' channel that bootstraps the learned (top-down) saccade policy."""
    return {(x, y) for y in range(len(cur)) for x in range(len(cur[0])) if prev[y][x] != cur[y][x]}


def view_sdr(cloud) -> SDR:
    """The retina's CONTENT ENCODER (SDR_MIGRATION.md M3): a local VIEW's raw coloured cells -> an OVERLAP-BEARING
    content SDR (retiring the exact-match `view_signature` key). Its point is that SIMILAR views share bits — the three
    SDR rules (Ahmad & Hawkins): overlap = similarity, determinism, fixed length + sparsity. It is the UNION of two
    fields, each an SDR that already has the right overlap structure:
      * COLOUR composition — each present colour's `CategoryEncoder` block (disjoint per colour: no false colour-nearness,
        the bitter lesson; ARC's 16-colour palette is UNORDERED). Views sharing a colour share those bits.
      * GEOMETRY — each pairwise DISTANCE's `ScalarEncoder` bump (nearby distances OVERLAP). Distances are preserved by
        any rotation/translation, so the SAME shape at ANY pose -> the SAME geometry bits (pose-invariant), and a SIMILAR
        shape (a cell moved / added) -> a HIGH-overlap code, where the exact-match key was orthogonal.
    So a different colouring differs only in the colour bits (geometry still overlaps), a different shape only in the
    geometry bits (colour still overlaps), and the same shape+colours at any pose is IDENTICAL. MODALITY-SPECIFIC
    extraction, in the PERIPHERAL (reference_tbt_feature_definition); the column stays content-opaque. `cloud` =
    `[(x, y, colour), ...]`; deterministic (the sub-encoders are fixed)."""
    cells = [(float(x), float(y), int(c)) for (x, y, c) in cloud]
    active = set()
    for _x, _y, c in cells:                                        # COLOUR composition: union of present-colour blocks
        active |= _COLOUR_ENC.encode(c).active
    off = _COLOUR_ENC.n
    for i in range(len(cells)):                                   # GEOMETRY: union of pairwise-distance bumps (invariant)
        xi, yi, _ci = cells[i]
        for j in range(i + 1, len(cells)):
            xj, yj, _cj = cells[j]
            d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            active |= {off + b for b in _DIST_ENC.encode(d).active}
    return SDR(_COLOUR_ENC.n + _DIST_ENC.n, active)
