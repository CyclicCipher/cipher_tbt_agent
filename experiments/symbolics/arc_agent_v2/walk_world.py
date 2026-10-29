"""Walk worlds (docs/LAB.md part 4): a graph walked by actions, with a label at each node, shown to the brain as SDRs.

**What the brain is shown:**
- **The "motor" sensor:** each action, as its efference copy. An action's SDR is fixed for the world's life.
- **The "retina" sensor:** each node's label, a random SDR, unique per node. `relabel` draws new labels.

The brain is told nothing else (ARCHITECTURE.md section 0). Node numbers, action names, the graph and distances live
only here, to drive walks and to score.

**The worlds:**
- **E1 torus:** 6 × 10, actions N, S, E, W.
- **E2 cycle:** 60 nodes, actions +1, −1, +7, −7. This is COORDINATES part 8's E2, where a family of rectangular tori
  settled on a 15 × 15 partial model.
- **O1 A5:** the Cayley graph of A₅ (the even permutations of 5 objects), actions a = (0 1)(2 3), b = (0 2 4) and
  b⁻¹, acting on the right. A group whose moves don't commute.
- **O2 box:** 6 × 10, N, S, E, W, with a move into a wall staying put. Not a group.

**The plane length** (E1, E2): the length of undoing a walk's net displacement without wrapping around (each action
pair counted as one axis of Z²). A trial is a shortcut trial when the true shortest way home is shorter.
"""
from collections import deque
from itertools import permutations

import numpy as np

from arc_agent_v2.HDClibrary import SDR, CategoryEncoder

KINDS = ("E1 torus", "E2 cycle", "O1 A5", "O2 box")
EXCURSION = {"E1 torus": (5, 9), "E2 cycle": (20, 8)}  # k in 1..K steps of action 0, m in 1..M of action 2


def graph(kind):
    """(next-node table, action names)."""
    if kind == "E1 torus" or kind == "O2 box":
        H, Wd = 6, 10
        T = np.zeros((H * Wd, 4), np.int64)
        for r in range(H):
            for c in range(Wd):
                for a, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    if kind == "E1 torus":
                        T[r * Wd + c, a] = ((r + dr) % H) * Wd + (c + dc) % Wd
                    else:
                        rr, cc = r + dr, c + dc
                        T[r * Wd + c, a] = rr * Wd + cc if 0 <= rr < H and 0 <= cc < Wd else r * Wd + c
        return T, ("N", "S", "E", "W")
    if kind == "E2 cycle":
        T = np.array([[(x + s) % 60 for s in (1, -1, 7, -7)] for x in range(60)], np.int64)
        return T, ("+1", "-1", "+7", "-7")
    if kind == "O1 A5":
        even = [p for p in permutations(range(5))
                if sum(1 for i in range(5) for j in range(i + 1, 5) if p[i] > p[j]) % 2 == 0]
        index = {p: i for i, p in enumerate(even)}
        a, b = (1, 0, 3, 2, 4), (2, 1, 4, 3, 0)
        binv = tuple(b.index(i) for i in range(5))
        T = np.array([[index[tuple(g[s[i]] for i in range(5))] for s in (a, b, binv)] for g in even], np.int64)
        return T, ("a", "b", "b^-1")
    raise ValueError(kind)


class WalkWorld:
    def __init__(self, kind, d=2048, w=40, seed=0):
        self.kind = kind
        self.T, self.names = graph(kind)
        self.n, self.A = self.T.shape
        enc = CategoryEncoder(d, w, seed=1000 + seed)
        self.action = [enc.encode(("action", nm)) for nm in self.names]
        self.index = {s: i for i, s in enumerate(self.action)}
        self.d, self.w = d, w
        self.relabel(np.random.default_rng((seed, 1)))

    def relabel(self, rng):
        self.labels = [SDR.random(self.d, self.w, rng) for _ in range(self.n)]
        self.node_of = {s: i for i, s in enumerate(self.labels)}

    def nodes(self, start, actions):
        out = [start]
        for a in actions:
            out.append(int(self.T[out[-1], a]))
        return out

    def stream(self, start, actions):
        """What the brain is shown: [(sensor, SDR)], the start's label, then each action and the label it leads to."""
        nodes = self.nodes(start, actions)
        out = [("retina", self.labels[start])]
        for a, v in zip(actions, nodes[1:]):
            out += [("motor", self.action[a]), ("retina", self.labels[v])]
        return out

    def distance(self, a, b):
        """The true shortest number of actions from node a to node b."""
        seen, q = {a: 0}, deque([a])
        while q:
            u = q.popleft()
            if u == b:
                return seen[u]
            for v in self.T[u]:
                if int(v) not in seen:
                    seen[int(v)] = seen[u] + 1
                    q.append(int(v))
        return None

    def plane(self, actions):
        """The length of undoing the walk's net displacement in the plane (E1, E2), or None."""
        if self.kind not in EXCURSION:
            return None
        net = np.zeros(2, np.int64)
        for a in actions:
            net[a // 2] += 1 if a % 2 == 0 else -1
        return int(np.abs(net).sum())
