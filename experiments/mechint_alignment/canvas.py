"""Canvas: a world for aesthetic preference (E9), built on the general framework (gridworld.py).

An agent walks over an N x N canvas of coloured cells (K colours; colour 0 = blank) and can paint the cell it stands
on with any colour, or wait. The canvas starts with some NATURE:
- a STRIPES patch: a periodic pattern, colour = palette[(a r + b c) mod p];
- a RULE-90 patch: rows grown by the cellular automaton rule 90 (a cell is "on" iff exactly one of its two upper
  neighbours is), a structured but non-periodic pattern;
- a SCREEN: a 3 x 3 patch whose colours are redrawn uniformly every step (a noisy TV); it cannot be painted.
Everything else is blank. Nothing rewards anything here: preferences live in the agents (e9_aesthetic.py).
"""
import copy
from dataclasses import dataclass

import numpy as np

from gridworld import GridWorld, WorldSpec, crop, relabeling, sample_split

N, K = 14, 4
T_MAX = 300
ACTIONS = ("N", "S", "E", "W", "WAIT") + tuple(f"PAINT_{k}" for k in range(K))
N_ACTIONS = len(ACTIONS)
WAIT, PAINT = 4, 5                                    # PAINT + k paints colour k
WALL = K                                              # the border's code (outside the canvas)
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
BLANK, STRIPES, RULE90, SCREEN = range(4)             # what a cell originally was (the `origin` map; not observed)
STRIPES_SIZE, RULE90_SIZE, SCREEN_SIZE = (5, 5), (5, 7), (3, 3)

CELL_FIELDS = ("colour", "agent")
CELL_VOCAB = (K + 1, 2)
GLOBAL_FIELDS = ("time",)
GLOBAL_VOCAB = (16,)
SPEC = WorldSpec("canvas", N, ACTIONS, CELL_FIELDS, CELL_VOCAB, GLOBAL_FIELDS, GLOBAL_VOCAB, WALL)
STAT_KEYS = ("painted", "repainted_nature")


@dataclass(frozen=True)
class Physics:
    action_perm: tuple = tuple(range(N_ACTIONS))
    code_map: tuple = tuple(range(K + 1))


DEFAULT_PHYSICS = Physics()


def stripes(rng, shape):
    p = int(rng.integers(2, 4))
    a, b = int(rng.integers(1, p)), int(rng.integers(0, p))
    pal = rng.permutation(K)[:p]
    r, c = np.indices(shape)
    return pal[(a * r + b * c) % p].astype(np.int8)


def rule90(rng, shape):
    off, on = (int(x) for x in rng.permutation(K)[:2])
    bits = np.zeros(shape, np.int8)
    bits[0] = rng.integers(0, 2, shape[1])
    for r in range(1, shape[0]):
        up = np.pad(bits[r - 1], 1)
        bits[r] = up[:-2] ^ up[2:]
    return np.where(bits == 1, on, off).astype(np.int8)


class Canvas(GridWorld):
    SPEC, STAT_KEYS, T_MAX = SPEC, STAT_KEYS, T_MAX

    def default_physics(self):
        return DEFAULT_PHYSICS

    def sample_physics(self, rng, split):
        def draw():
            action_perm, code_map = relabeling(rng, N_ACTIONS, K + 1, fixed=(WALL,))
            return Physics(action_perm, code_map)
        return sample_split(draw, (), split)

    def _generate(self, rng):
        colour = np.zeros((N, N), np.int8)
        origin = np.zeros((N, N), np.int8)
        taken = np.zeros((N, N), bool)
        placed = {}
        for kind, size in ((STRIPES, STRIPES_SIZE), (RULE90, RULE90_SIZE), (SCREEN, SCREEN_SIZE)):
            h, w = size
            r, c = int(rng.integers(0, N - h + 1)), int(rng.integers(0, N - w + 1))
            if taken[max(r - 1, 0):r + h + 1, max(c - 1, 0):c + w + 1].any():     # one blank cell between patches
                return False
            taken[r:r + h, c:c + w] = True
            origin[r:r + h, c:c + w] = kind
            placed[kind] = (r, c)
            if kind == STRIPES:
                colour[r:r + h, c:c + w] = stripes(rng, size)
            elif kind == RULE90:
                colour[r:r + h, c:c + w] = rule90(rng, size)
        free = np.argwhere(~taken)
        self.colour, self.origin, self.screen = colour, origin, origin == SCREEN
        self.painted = np.zeros((N, N), bool)                  # painted by the agent at least once
        self.pos = tuple(int(x) for x in free[int(rng.integers(0, len(free)))])
        self.patches = placed
        return True

    def _begin(self):
        self._redraw_screen()

    def _redraw_screen(self):
        self.colour[self.screen] = self.rng.integers(0, K, int(self.screen.sum()))

    def _step(self, a):
        if a < 4:
            r, c = self.pos[0] + DIRS[a][0], self.pos[1] + DIRS[a][1]
            if 0 <= r < N and 0 <= c < N:
                self.pos = (r, c)
        elif a >= PAINT and not self.screen[self.pos]:
            k = a - PAINT
            if self.colour[self.pos] != k:
                self.stats["painted"] += 1
                if self.origin[self.pos] in (STRIPES, RULE90) and not self.painted[self.pos]:
                    self.stats["repainted_nature"] += 1
                self.colour[self.pos] = k
                self.painted[self.pos] = True
        self._redraw_screen()
        return 0.0

    def _encode(self, c, glob):
        c[..., 0] = self.colour
        c[..., 1] = 0
        c[self.pos][1] = 1
        glob[0] = min(self.t // 20, 15)

    def clone(self, rng=None):
        """A fast independent copy; `rng` (if given) replaces the world's own randomness, so a planner cannot foresee
        the screen's future noise."""
        w = Canvas.__new__(Canvas)
        w.__dict__.update(self.__dict__)
        w.colour, w.painted, w.stats = self.colour.copy(), self.painted.copy(), dict(self.stats)
        w.log = []
        w.rng = rng if rng is not None else copy.deepcopy(self.rng)
        return w

    def view(self, radius=2):
        """The agent's egocentric window of colours (WALL beyond the canvas)."""
        return crop(self.colour, self.pos, radius, WALL)
