"""Reference frames for assoc columns (ARCHITECTURE.md sections 2-3): each is a coordinate system with a group of
operators, and lays its column's content out on the screen.

- translation: a rigid set of cells (part 12: translations in the world frame, quarter turns about its own pivot in
  its body frame). State: (tx, ty, r). Operator: (dx, dy, k); (0, 0, 0) is the identity.
- recolour: fixed cells whose colours change together by one colour map. State: the colours, in cell order.
  Operator: a colour map, as a sorted tuple of (old, new) pairs; () is the identity.
- bar: an ordered line of cells filled one at a time (Z_n, a count). State: how many cells are filled. Operator: a
  step (+1, -1, 0). The counters and timers the games draw are bars or translations driven by the clock.

Structure (the offsets, the order) is kept apart from content (the colours), so content can be relabelled without
touching structure.
"""
import numpy as np


H = W = 64


def rot(p, r):
    """Part 12-13's 90-degree rotations of the plane: points p (n x 2) turned r quarter turns."""
    p = np.array(p).reshape(-1, 2)
    for _ in range(r % 4):
        p = np.stack([-p[:, 1], p[:, 0]], -1)
    return p


class Translation:
    """A rigid set of cells (part 12): translations act in the world frame, 90-degree rotations about the piece's own
    pivot act in its body frame. State (tx, ty, r): the cells are rot(cells0 - pivot, r) + pivot + (tx, ty).
    Operator (dx, dy, k): move the pivot by (dx, dy) and turn the piece by k quarter turns about it."""
    kind = "translation"
    identity = (0, 0, 0)
    K = 33 * 33 * 4                                     # displacements within 16 cells, times 4 turns

    def __init__(self, cells, colours):
        self.cells0 = np.array(cells, int)              # (n, 2) as (x, y), where the column was found
        self.colours = np.array(colours, np.int8)
        self.pivot = np.round(self.cells0.mean(0)).astype(int)
        self.rel = {r: rot(self.cells0 - self.pivot, r).astype(int) for r in range(4)}

    def place(self, state):
        return self.rel[state[2] % 4] + self.pivot + np.array(state[:2], int)

    def apply(self, state, op):
        return (state[0] + op[0], state[1] + op[1], (state[2] + op[2]) % 4)

    def inside(self, state):
        p = self.place(state)
        return bool(np.all((p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)))

    def draw(self, grid, state):
        p = self.place(state)
        ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        grid[p[ok, 1], p[ok, 0]] = self.colours[ok]

    def matches(self, grid, state):
        if not self.inside(state):
            return False
        p = self.place(state)
        return bool(np.all(grid[p[:, 1], p[:, 0]] == self.colours))

    def score(self, grid, state):
        """The fraction of the column's cells whose colour is seen at this state (0 if any cell is off the screen)."""
        if not self.inside(state):
            return 0.0
        p = self.place(state)
        return float(np.mean(grid[p[:, 1], p[:, 0]] == self.colours))

    def _rings(self, near, radius, turns):
        for r in range(radius + 1):                     # nearest first
            for dx in range(-r, r + 1):
                for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                    for k in turns:
                        yield (near[0] + dx, near[1] + dy, (near[2] + k) % 4)

    def find(self, grid, near, radius=16):
        """The state nearest `near` at which the content is seen exactly: same turn first, then the other turns."""
        for s in self._rings(near, radius, (0,)):
            if self.matches(grid, s):
                return s
        for s in self._rings(near, 4, (1, 2, 3)):
            if self.matches(grid, s):
                return s
        return None

    def find_near(self, grid, near, radius, frac):
        """The best-matching state within radius (at least frac of the cells seen), nearest on ties, or None."""
        best = None
        for i, s in enumerate(self._rings(near, radius, (0, 1, 2, 3))):
            sc = self.score(grid, s)
            if sc >= frac and (best is None or sc > best[0]):
                best = (sc, s)
        return None if best is None else best[1]

    def find_all(self, grid):
        """Every state at which the content is seen exactly (for a new level)."""
        out = []
        for r in range(4):
            x0, y0 = self.rel[r][0] + self.pivot
            ys, xs = np.nonzero(grid == self.colours[0])
            for x, y in zip(xs, ys):
                s = (int(x - x0), int(y - y0), r)
                if self.matches(grid, s):
                    out.append(s)
        return out

    def op_between(self, s0, s1):
        return (s1[0] - s0[0], s1[1] - s0[1], (s1[2] - s0[2]) % 4)


class Choice:
    """A categorical frame (part 24, the PFC column's): which of a set of options holds. State: an option, or None
    before any. Operator: ("to", option), or ("stay",), the identity."""
    kind = "choice"
    identity = ("stay",)

    def __init__(self, options):
        self.options = tuple(options)
        self.K = len(self.options) + 1
        self.cells0 = np.zeros((1, 2), int)

    def apply(self, state, op):
        return state if op == ("stay",) else op[1]

    def op_between(self, s0, s1):
        return ("stay",) if s0 == s1 else ("to", s1)


class Line:
    """An integer that events add to or set (part 28: counters, timers, a clock). State: an int. Operator: an int to add
    (0, the identity) or ("to", v); a cap, if given, bounds the state."""
    kind = "line"
    identity = 0

    def __init__(self, cap=None, K=12):
        self.cap, self.K = cap, K
        self.cells0 = np.zeros((1, 2), int)

    def apply(self, state, op):
        s = op[1] if isinstance(op, tuple) else state + op
        return s if self.cap is None else min(s, self.cap)

    def op_between(self, s0, s1):
        return ("to", s1) if s1 < s0 else s1 - s0


class Recolour:
    kind = "recolour"
    identity = ()
    K = 256

    def __init__(self, cells, colours):
        self.cells0 = np.array(cells, int)
        self.colours = np.array(colours, np.int8)       # the colours when found (the first state)

    def place(self, state=None):
        return self.cells0

    def apply(self, state, op):
        m = dict(op)
        return tuple(m.get(c, c) for c in state)

    def inside(self, state):
        return True

    def draw(self, grid, state):
        grid[self.cells0[:, 1], self.cells0[:, 0]] = np.array(state, np.int8)

    def read(self, grid):
        return tuple(int(v) for v in grid[self.cells0[:, 1], self.cells0[:, 0]])

    def op_between(self, s0, s1):
        m = {}
        for a, b in zip(s0, s1):
            if a != b:
                if m.get(a, b) != b:
                    return None                         # not one colour map: the frame fails here
                m[a] = b
        return tuple(sorted(m.items()))


class Bar:
    kind = "bar"
    identity = 0
    K = 3

    def __init__(self, cells, empty, full):
        self.cells0 = np.array(cells, int)              # in fill order
        self.empty, self.full = int(empty), int(full)

    def place(self, state=None):
        return self.cells0

    def apply(self, state, op):
        return max(0, min(len(self.cells0), state + op))

    def inside(self, state):
        return True

    def draw(self, grid, state):
        c = self.cells0
        grid[c[:state, 1], c[:state, 0]] = self.full
        grid[c[state:, 1], c[state:, 0]] = self.empty

    def read(self, grid):
        v = grid[self.cells0[:, 1], self.cells0[:, 0]]
        return int(np.sum(v == self.full))

    def op_between(self, s0, s1):
        return int(np.sign(s1 - s0))

    def extend(self, cell):
        self.cells0 = np.vstack([self.cells0, np.array(cell, int)[None]])
