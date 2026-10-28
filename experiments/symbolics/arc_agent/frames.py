"""Reference frames for assoc columns (ARCHITECTURE.md sections 2-3): each is a coordinate system with a group of
operators, and lays its column's content out on the screen.

- translation: a set of cells that one displacement moves together (Z^2). State: the offset of the cells from where
  the column was found. Operator: a displacement (dx, dy); (0, 0) is the identity.
- recolour: fixed cells whose colours change together by one colour map. State: the colours, in cell order.
  Operator: a colour map, as a sorted tuple of (old, new) pairs; () is the identity.
- bar: an ordered line of cells filled one at a time (Z_n, a count). State: how many cells are filled. Operator: a
  step (+1, -1, 0). The counters and timers the games draw are bars or translations driven by the clock.

Structure (the offsets, the order) is kept apart from content (the colours), so content can be relabelled without
touching structure.
"""
import numpy as np

H = W = 64


class Translation:
    kind = "translation"
    identity = (0, 0)
    K = 33 * 33                                         # displacements within 16 cells: the operator universe's size

    def __init__(self, cells, colours):
        self.cells0 = np.array(cells, int)              # (n, 2) as (x, y), where the column was found
        self.colours = np.array(colours, np.int8)

    def place(self, state):
        return self.cells0 + np.array(state, int)

    def apply(self, state, op):
        return (state[0] + op[0], state[1] + op[1])

    def inside(self, state):
        p = self.place(state)
        return bool(np.all((p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)))

    def draw(self, grid, state):
        p = self.place(state)
        ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        grid[p[ok, 1], p[ok, 0]] = self.colours[ok]

    def matches(self, grid, state):
        return self.inside(state) and bool(np.all(grid[self.place(state)[:, 1], self.place(state)[:, 0]] == self.colours))

    def find(self, grid, near, radius=16):
        """The state nearest `near` at which the column's content is seen exactly, or None."""
        best = None
        for r in range(radius + 1):                     # rings of growing radius, so the nearest is found first
            for dx in range(-r, r + 1):
                for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                    s = (near[0] + dx, near[1] + dy)
                    if self.matches(grid, s):
                        return s
            if best is not None:
                return best
        return None

    def find_all(self, grid):
        """Every state at which the content is seen exactly (for a new level)."""
        x0, y0 = self.cells0[0]
        ys, xs = np.nonzero(grid == self.colours[0])
        return [(int(x - x0), int(y - y0)) for x, y in zip(xs, ys) if self.matches(grid, (int(x - x0), int(y - y0)))]

    def op_between(self, s0, s1):
        return (s1[0] - s0[0], s1[1] - s0[1])


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
