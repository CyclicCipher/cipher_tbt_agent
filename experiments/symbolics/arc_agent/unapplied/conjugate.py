def turn(x, y, k):
    """(x, y) turned k quarter turns (frames.rot's sense)."""
    for _ in range(k % 4):
        x, y = -y, x
    return x, y


class Turned:
    """A context seen turned (part 31b): the content at offset o is read at R o about the column's centre, R = k
    quarter turns. Features other than positions pass through."""

    def __init__(self, ctx, k, centre):
        self.ctx, self.k, self.c = ctx, k, centre

    def _at(self, f):
        x, y = f[1] - self.c[0], f[2] - self.c[1]
        tx, ty = turn(x, y, self.k)
        return ("at", int(round(tx + self.c[0])), int(round(ty + self.c[1])))

    def __contains__(self, f):
        return (self._at(f) in self.ctx) if f[0] == "at" else (f in self.ctx)

    def get(self, f, default=None):
        return self.ctx.get(self._at(f), default) if f[0] == "at" else self.ctx.get(f, default)

    def __iter__(self):
        return iter(self.ctx)


def turn_op(op, k):
    if isinstance(op, tuple) and len(op) == 3 and all(isinstance(v, int) for v in op):
        x, y = turn(op[0], op[1], k)
        return (int(x), int(y), op[2])
    return op
