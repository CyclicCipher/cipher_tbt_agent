"""Spotlight memory (ARCHITECTURE.md section 4): a lattice of cells at addresses, kept apart from the models that use it.

Taken from Percepta's Spotlight: memory is a lattice, it grows without changing any model, and a cell is read and
written at its address, or through a neighbourhood when the address is uncertain. Changed for this agent (COORDINATES
part 5): addresses are computed and path-integrated by a column's reference frame, not learned by projecting a key,
so exact reads are the normal case.

A cell is a small associative memory updated by the delta rule with one-hot keys (DeltaNet's update in its simplest
form): for each key, w <- w + beta * (onehot(value) - w). With beta = 1 it holds the last value written; with beta < 1
it holds a distribution over values.
"""


class Spotlight:
    def __init__(self, beta=1.0):
        self.beta = beta
        self.cells = {}                                 # address -> {key: {value: weight}}

    def write(self, addr, key, value, beta=None):
        b = self.beta if beta is None else beta
        w = self.cells.setdefault(addr, {}).setdefault(key, {})
        for v in list(w):
            w[v] *= 1.0 - b
            if w[v] < 1e-6:
                del w[v]
        w[value] = w.get(value, 0.0) + b

    def read(self, addr, key):
        """{value: weight} at an exact address, or None if nothing was written there."""
        cell = self.cells.get(addr)
        return None if cell is None else cell.get(key)

    def value(self, addr, key, default=None):
        w = self.read(addr, key)
        return max(w, key=w.get) if w else default

    def read_soft(self, belief, key):
        """The belief-weighted mixture of the cells at the believed addresses ({address: probability})."""
        out = {}
        for addr, p in belief.items():
            for v, w in (self.read(addr, key) or {}).items():
                out[v] = out.get(v, 0.0) + p * w
        return out

    def forget(self, addrs=None):
        if addrs is None:
            self.cells.clear()
        else:
            for a in addrs:
                self.cells.pop(a, None)

    def __len__(self):
        return len(self.cells)
