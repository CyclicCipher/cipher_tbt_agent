class Permute:
    """Contents moved among fixed cells (part 30): a frame learned from data. State: the colours of the cells, in order.
    Operator: a learned permutation of the cells, named by a token (1, 2, ...; 0 is staying). What a token means is
    learned: each observation it explains narrows, for every cell, the cells its colour may have come from (the
    intersection over those observations, `allowed`), and its permutation is the nearest assignment within them
    (unchanged cells stay put while nothing says otherwise). So one observation in which a cell kept its colour by
    coincidence (equal neighbours on a ring) does not fix a wrong permutation: a later one refutes the coincidence. An
    observation no token can explain starts a new one. Its cycles are its coordinates: applied k times, a permutation
    shifts each cycle by k (the cyclic group's characters, COORDINATES parts 6 and 9). The support grows when an
    overlapping change shows more cells taking part (extend)."""
    kind = "permute"
    identity = 0
    K = 4096
    BIG = 1e9

    def __init__(self, cells, colours):
        self.cells0 = np.array(cells, int)
        self.colours = np.array(colours, np.int8)
        self.allowed, self.perm, self.seen = {}, {}, {}  # token -> [dst, src] bool; token -> permutation; -> count
        self._dist()

    def _dist(self):
        c = self.cells0
        self.d = np.abs(c[:, None, :] - c[None, :, :]).sum(-1).astype(float)

    def place(self, state=None):
        return self.cells0

    def apply(self, state, op):
        p = self.perm.get(op)
        if p is None or len(p) != len(state):
            return state
        return tuple(state[j] for j in p)

    def inside(self, state):
        return True

    def draw(self, grid, state):
        grid[self.cells0[:, 1], self.cells0[:, 0]] = np.array(state, np.int8)

    def read(self, grid):
        return tuple(int(v) for v in grid[self.cells0[:, 1], self.cells0[:, 0]])

    def _assign(self, allowed):
        """The nearest permutation within allowed (cell i takes the colour of cell p[i]), or None if there is none."""
        cost = np.where(allowed, self.d, self.BIG)
        rows, cols = linear_sum_assignment(cost)
        if cost[rows, cols].max() >= self.BIG:
            return None
        return tuple(int(j) for j in cols[np.argsort(rows)])

    def op_between(self, s0, s1):
        """The token that explains s0 -> s1: the most used one whose permutation already does; else the most used one
        that can, narrowed to it; else a new one. None if the colours are not the same multiset."""
        s0, s1 = tuple(s0), tuple(s1)
        if len(s0) != len(self.cells0) or len(s1) != len(s0) or sorted(s0) != sorted(s1):
            return None
        if s0 == s1:
            return self.identity
        m = np.array(s1)[:, None] == np.array(s0)[None, :]
        order = sorted(self.allowed, key=lambda t: -self.seen[t])
        for t in order:
            if self.apply(s0, t) == s1:
                self.allowed[t] &= m
                self.seen[t] += 1
                return t
        for t in order:
            a = self.allowed[t] & m
            p = self._assign(a)
            if p is not None:
                self.allowed[t], self.perm[t] = a, p
                self.seen[t] += 1
                return t
        t = len(self.allowed) + 1
        self.allowed[t], self.perm[t], self.seen[t] = m, self._assign(m), 1
        return t

    def extend(self, cells, colours):
        """Grow the support by these cells. What they did in the observations before is unknown, so every token allows
        them any source, and them as anyone's source; each permutation keeps them where they are until told."""
        n, k = len(self.cells0), len(cells)
        self.cells0 = np.vstack([self.cells0, np.array(cells, int).reshape(-1, 2)])
        self.colours = np.concatenate([self.colours, np.array(colours, np.int8)])
        for t in self.allowed:
            a = np.ones((n + k, n + k), bool)
            a[:n, :n] = self.allowed[t]
            self.allowed[t] = a
            self.perm[t] = tuple(self.perm[t]) + tuple(range(n, n + k))
        self._dist()


