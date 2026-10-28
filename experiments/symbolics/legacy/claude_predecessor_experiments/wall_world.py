"""W4 (docs/AGENT.md, part 16): push_block with a wall and two distractors, a random layout every level.

The wall is a straight segment of 3-4 cells (colour 5) that blocks the player and the box. The two distractors are
single passable cells, in fresh colours drawn from 6-13 each level, so curiosity is re-triggered by things irrelevant
to winning. Every layout is checked solvable.
"""
from collections import deque

import numpy as np

from arc_agent.claude.push_block_rebuild import DIRS, H, W, PushWorld


def solve(player, box, target, walls):
    """Fewest steps to put the box on the target (breadth-first), or None."""
    start = (player, box)
    seen, q = {start: 0}, deque([start])
    while q:
        p, b = q.popleft()
        if b == target:
            return seen[(p, b)]
        for dx, dy in DIRS.values():
            np_, nb = (p[0] + dx, p[1] + dy), b
            if not (0 <= np_[0] < W and 0 <= np_[1] < H) or np_ in walls:
                continue
            if np_ == b:
                nb = (b[0] + dx, b[1] + dy)
                if not (0 <= nb[0] < W and 0 <= nb[1] < H) or nb in walls:
                    continue
            if (np_, nb) not in seen:
                seen[(np_, nb)] = seen[(p, b)] + 1
                q.append((np_, nb))
    return None


class WallWorld(PushWorld):
    def __init__(self, rng):
        self.walls, self.deco = set(), {}
        super().__init__(rng, random_layout=True)

    def reset(self):
        r = self.rng
        while True:
            n, horiz = int(r.integers(3, 5)), bool(r.integers(2))
            x0, y0 = int(r.integers(0, W - (n if horiz else 0))), int(r.integers(1, H - (0 if horiz else n)))
            walls = {(x0 + k, y0) if horiz else (x0, y0 + k) for k in range(n)}
            free = [(x, y) for x in range(W) for y in range(1, H) if (x, y) not in walls]
            picks = r.choice(len(free), 5, replace=False)
            player, box, target, d1, d2 = (free[i] for i in picks)
            if not (1 <= box[0] <= W - 2 and 1 <= box[1] <= H - 2):
                continue
            if solve(player, box, target, walls) is None:
                continue
            c1, c2 = (int(c) for c in r.choice(np.arange(6, 14), 2, replace=False))
            self.walls, self.deco = walls, {d1: c1, d2: c2}
            self.player, self.box, self.target = player, box, target
            break
        self.t, self.pat_x = 0, 1
        return self.frame()

    def frame(self):
        g = np.zeros((H, W), int)
        for x, y in self.walls:
            g[y, x] = 5
        for (x, y), c in self.deco.items():
            g[y, x] = c
        g[self.target[1], self.target[0]] = 2
        g[self.box[1], self.box[0]] = 4
        g[self.player[1], self.player[0]] = 1
        return g

    def step(self, a):
        self.t += 1
        dx, dy = DIRS[a]
        nx, ny = self.player[0] + dx, self.player[1] + dy
        if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in self.walls:
            if (nx, ny) == self.box:
                bx, by = self.box[0] + dx, self.box[1] + dy
                if 0 <= bx < W and 0 <= by < H and (bx, by) not in self.walls:
                    self.box, self.player = (bx, by), (nx, ny)
            else:
                self.player = (nx, ny)
        return self.frame(), self.box == self.target
