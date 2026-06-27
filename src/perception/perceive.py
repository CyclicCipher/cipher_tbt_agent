"""Perception primitives — turn a raw colour grid into cells, objects, and motion (no semantic priors).

The cheap statistical background (modal_background), the non-bg cells (active_cells), one-cell motion
(detect_motion), and connected-component OBJECTS (segment / object_motion — the Core-Knowledge "objectness"
prior; an object's ROLE is learned later from dynamics, never read off its colour). There is NO agency code:
agency is the efference copy (the agent knows the action it issued), decided in the agent, not inferred here.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from tasks import GameState                                   # DynamicsPerceiver's terminal/score check

_UNITS = [(0, -1), (0, 1), (-1, 0), (1, 0)]                   # 4-neighbourhood
_8 = _UNITS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]            # 8-neighbourhood (incl. diagonals)

_BG_CACHE: dict = {}                                          # id(grid) -> (grid, bg), identity-checked + capped


def modal_background(grid):
    """The most common cell value — the (statistical, not semantic) background. Cached per grid OBJECT: a frame's
    grid is read several times per step (both perception passes + segmentation) and reused as the previous frame
    next step, so the Counter over the full 64x64 frame would otherwise run ~4x per frame redundantly."""
    key = id(grid)
    hit = _BG_CACHE.get(key)
    if hit is not None and hit[0] is grid:                    # identity check: a recycled id is not a false hit
        return hit[1]
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    if len(_BG_CACHE) > 16:                                   # keep only a small working set (the live frames)
        _BG_CACHE.clear()
    _BG_CACHE[key] = (grid, bg)                               # storing grid keeps it alive -> its id can't be reused
    return bg


def active_cells(grid, bg):
    """Non-background cells as {(x, y): colour} — the level inside the mostly-blank frame."""
    return {(x, y): v for y, row in enumerate(grid) for x, v in enumerate(row) if v != bg}


def bounding_box(cells):
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return min(xs), min(ys), max(xs), max(ys)


def detect_motion(prev_cells, cells):
    """Raw exafference+reafference: {colour: (dx, dy)} for every token that TRANSLATED by one cell. The
    agent later keeps the one its efference copy predicts (self) and treats the rest as environment."""
    moved = {}
    for (x, y), c in prev_cells.items():
        if cells.get((x, y)) == c:
            continue                                              # didn't leave — not a translation
        for dx, dy in _UNITS:
            p = (x + dx, y + dy)
            if cells.get(p) == c and prev_cells.get(p) != c:     # c newly appears one cell over
                moved[c] = (dx, dy)
                break
    return moved


# ── objects (E): segment a colour grid into multi-cell connected components ──────────────────────────────
@dataclass(frozen=True)
class Obj:
    color: int
    cells: frozenset                                          # frozenset of (x, y)

    @property
    def size(self):
        return len(self.cells)

    @property
    def bbox(self):                                           # (min_x, min_y, max_x, max_y)
        xs = [x for x, _ in self.cells]; ys = [y for _, y in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def centroid(self):
        n = len(self.cells)
        return (sum(x for x, _ in self.cells) / n, sum(y for _, y in self.cells) / n)

    @property
    def shape(self):                                          # cells relative to the bbox corner — translation-invariant
        bx, by, _, _ = self.bbox
        return frozenset((x - bx, y - by) for x, y in self.cells)


def segment(grid, bg=None, conn=4, multicolor=False):
    """Connected-component objects: each a region of connected non-background cells. Same colour by default (the
    ARC objectness prior); `multicolor=True` groups any adjacent non-bg cells. `conn` 4 (orthogonal) or 8 (incl.
    diagonals). Walks only the (few) non-bg cells, not the whole frame."""
    H, W = len(grid), len(grid[0])
    if bg is None:
        bg = modal_background(grid)
    nbrs = _8 if conn == 8 else _UNITS
    nonbg = {(x, y): grid[y][x] for y in range(H) for x in range(W) if grid[y][x] != bg}
    seen, objs = set(), []
    for start, col in nonbg.items():
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
    shifted position. Matched in place first (they stayed); the rest to the nearest same-shape object (they
    moved) — the substrate for the efference copy + spotting pushable pieces."""
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
        if in_place:
            used[key].add(in_place[0]); continue
        opts = [(abs(co.bbox[0] - px) + abs(co.bbox[1] - py), i, co)
                for i, co in enumerate(cand) if i not in used[key]]
        if opts:
            _, i, co = min(opts); used[key].add(i)
            moved.append((po, (co.bbox[0] - px, co.bbox[1] - py)))
    return moved


# ── E: learn body / pushable / blocking / consumable from motion (no colour priors) ─────────────────────
class ObjectPerceiver:
    def __init__(self):
        self.body_evidence = defaultdict(int)
        self.push_evidence = defaultdict(int)
        self.entered = set()                           # colours the body has walked ONTO (walkable)
        self.failed = set()                            # colours the body could NOT enter
        self.consume_evidence = defaultdict(int)       # colour vanished after the body stepped on it (consumed)
        self.occlude_evidence = defaultdict(int)       # colour reappeared after the body left (merely occluded)
        self.body_color = None
        self._pending = None                           # (cell, colour) the body just stepped onto, awaiting check

    def new_level(self):
        self._pending = None                           # the body teleports between levels; drop the pending check

    def observe(self, prev_grid, delta, cur_grid):
        """One transition: segment both frames, read the motion, accumulate body + pushable evidence.
        Returns (prev_objs, cur_objs, moved) so a caller can also reason over the perceived scene."""
        prev_objs, cur_objs = segment(prev_grid), segment(cur_grid)
        moved = object_motion(prev_objs, cur_objs)

        for obj, d in moved:                           # efference copy: moved by the issued delta
            if d == delta:
                self.body_evidence[obj.color] += 1
        if self.body_evidence:
            self.body_color = max(self.body_evidence, key=self.body_evidence.get)

        for obj, d in moved:                           # pushable: a NON-body object shoved along the move
            if obj.color != self.body_color and d == delta:
                self.push_evidence[obj.color] += 1

        # walkable vs obstacle: did the body enter the cell it stepped toward, or fail to?
        if self.body_color is not None:
            bobj = next((o for o in prev_objs if o.color == self.body_color), None)
            if bobj is not None:
                bx, by = next(iter(bobj.cells))
                tx, ty = bx + delta[0], by + delta[1]
                if 0 <= ty < len(prev_grid) and 0 <= tx < len(prev_grid[0]):
                    tcolor = prev_grid[ty][tx]
                    if any(o.color == self.body_color and d == delta for o, d in moved):
                        self.entered.add(tcolor)       # walked onto it
                    else:
                        self.failed.add(tcolor)        # could not enter it

        # consume vs occlude (a 2-step check): when the body LEAVES a cell it stepped onto, did the underlying
        # colour reappear (occluded, like a pad/goal) or vanish (consumed on contact, like a collect-all item)?
        if self.body_color is not None:
            bcur = next((o for o in cur_objs if o.color == self.body_color), None)
            here = next(iter(bcur.cells)) if bcur else None
            if self._pending is not None and here is not None and here != self._pending[0]:
                (cx, cy), under = self._pending
                back = cur_grid[cy][cx] if (0 <= cy < len(cur_grid) and 0 <= cx < len(cur_grid[0])) else under
                (self.occlude_evidence if back == under else self.consume_evidence)[under] += 1
                self._pending = None
            bprev = next((o for o in prev_objs if o.color == self.body_color), None)
            if here is not None and bprev is not None and here != next(iter(bprev.cells)):
                under = prev_grid[here[1]][here[0]]    # the colour at the destination before the body arrived
                if under != modal_background(prev_grid) and under != self.body_color:
                    self._pending = (here, under)
        return prev_objs, cur_objs, moved

    @property
    def pushable(self):
        return {c for c in self.push_evidence if c != self.body_color}

    @property
    def walkable(self):
        return set(self.entered)                       # colours the body can move onto

    @property
    def blocking(self):
        return self.failed - self.entered - self.pushable

    @property
    def consumable(self):
        """Colours the body removes by stepping on them (vanish and don't reappear) — collect-all items. A
        pad/goal is merely occluded and comes back; a PUSHED block's cell also goes empty, so exclude pushables.
        Over-broad colours (a used-up key) are filtered downstream (only a GOAL colour with multiple cells counts)."""
        return {c for c, n in self.consume_evidence.items()
                if n > self.occlude_evidence.get(c, 0)} - self.pushable


# ── F: learn the win-condition (goal colour + its conjunctive context) from the sparse score ────────────
class GoalModel:
    """`goal_colors` = reached when the score rises; `required_absent()` = colours present in some failed
    goal-reach yet absent in every win — the context that gates the goal (the conjunctive win's other half)."""

    def __init__(self):
        self.goal_colors = set()
        self.win_contexts = []                         # frozenset of colours present at each win (positives)
        self.reach_no_win = []                         # frozenset present when the goal was reached but no win

    def observe_win(self, present, goal_color):
        self.goal_colors.add(goal_color)
        self.win_contexts.append(frozenset(present))

    def observe_reach_no_win(self, present):
        self.reach_no_win.append(frozenset(present))

    def required_absent(self):
        if not self.win_contexts or not self.reach_no_win:
            return set()
        blocking = set().union(*self.reach_no_win)
        for won in self.win_contexts:
            blocking -= won
        return blocking

    def wins_now(self, present):
        return bool(self.goal_colors) and not (self.required_absent() & set(present))


# ── the per-step feature extractor: efference-copy body + presence-context + the symmetric world-diff ────
_PALETTE = 16                                                 # ARC's colour count: presence-context = present(0..15)


class DynamicsPerceiver:
    """Turn (prev_frame, action, frame) into (features, effect, present) for the DynamicsModel: body identity is
    the efference copy; features = stepped-on colour + the per-colour presence-context (makes a CONDITIONAL
    effect expressible); effect = death / score_up / a SYMMETRIC colour vanish⇄appear (occlusion-safe)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.body_color = None
        self.body_evidence = {}
        self.new_level()

    def new_level(self):
        self.body_pos = None                                  # the layout changed; keep the learned body colour

    def _find(self, grid, color):
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == color:
                    return (x, y)
        return None

    def _mover_cells(self, prev_cells, cur_cells):
        cells = set()
        for c in detect_motion(prev_cells, cur_cells):       # every translated colour (body + pushed block)
            cells |= {p for p, cc in prev_cells.items() if cc == c}
            cells |= {p for p, cc in cur_cells.items() if cc == c}
        return cells

    def observe(self, prev_frame, action, frame):
        prev, cur = prev_frame.grid, frame.grid
        prev_bg, cur_bg = modal_background(prev), modal_background(cur)
        prev_cells = active_cells(prev, prev_bg)
        cur_cells = active_cells(cur, cur_bg)

        if action.is_movement and frame.state == GameState.NOT_FINISHED:   # efference copy: body = moved by delta
            for c, d in detect_motion(prev_cells, cur_cells).items():
                if d == action.delta:
                    self.body_evidence[c] = self.body_evidence.get(c, 0) + 1
            if self.body_evidence:
                self.body_color = max(self.body_evidence, key=self.body_evidence.get)

        body = self.body_pos if self.body_pos is not None else self._find(prev, self.body_color)
        self.body_pos = self._find(cur, self.body_color)
        if body is None or self.body_color is None:
            return None, None, None

        dx, dy = action.delta                                 # features from the PRE-state (survive a level change)
        dest = (body[0] + dx, body[1] + dy)
        stepped_on = prev[dest[1]][dest[0]] if 0 <= dest[1] < len(prev) and 0 <= dest[0] < len(prev[0]) else -1
        present = {c for c in prev_cells.values()} - {self.body_color}
        features = (stepped_on,) + tuple(1 if c in present else 0 for c in range(_PALETTE))

        if frame.state == GameState.GAME_OVER:
            effect = "death"
        elif frame.score > prev_frame.score:
            effect = "score_up"
        else:                                                 # a symmetric world-diff (occlusion-safe): vanish⇄appear
            movers = self._mover_cells(prev_cells, cur_cells)
            gone = {}
            for (x, y), c in prev_cells.items():
                if (x, y) not in movers and cur[y][x] == cur_bg and c != cur_bg:
                    gone[c] = gone.get(c, 0) + 1              # a colour vanished where it was (door opened)
            appeared = {}
            for (x, y), c in cur_cells.items():
                if (x, y) not in movers and prev[y][x] == prev_bg and c != self.body_color:
                    appeared[c] = appeared.get(c, 0) + 1      # a colour materialised on bare bg (door closed)
            effect = (f"color_{max(gone, key=gone.get)}_gone" if gone else
                      f"color_{max(appeared, key=appeared.get)}_appeared" if appeared else None)
        return features, effect, present
