"""An exact Bayesian observer of robot motion in the E8 world family (E12): the yardstick for data efficiency.

Hidden per world: the action permutation (index -> canonical action, 8! hypotheses), which shown terrain code is the
walkable fire (7), and the push rule (push / block / swap). Robot motion (its next cell and facing) is deterministic
given these, the observation and the action index, so the posterior is uniform over the hypotheses consistent with
everything seen: a boolean mask over 40,320 x 7 x 3. The observer sees exactly what the learned world model sees.
"""
import itertools

import numpy as np

from worlds.village import DIRS, FLOOR, N, OCC_ADULT, OCC_CHILD, OCC_OVERSEER, OCC_ROBOT, WALL

PERMS = np.array(list(itertools.permutations(range(8))), np.int8)          # [40320, 8]: index -> canonical action
PUSH = ("push", "block", "swap")
N_FIRE = 7                                                                  # fire's shown code is 2 + f
AMBIGUOUS = None


def robot_state(cells, glob):
    """(row, col), facing (0-3) and frozen, from one observation (cells [81, 9], glob [4])."""
    c = cells.reshape(N, N, -1)
    r, k = np.argwhere(c[..., 3] == OCC_ROBOT)[0]
    return (int(r), int(k)), int(c[r, k, 8]) - 1, int(glob[2])


def outcome(cells, pos, facing, c, f, p):
    """The robot's (cell, facing) after canonical action c, if the fire's shown code is 2 + f and the push rule is
    PUSH[p]; AMBIGUOUS if it depends on a terrain role the observer cannot know (a push onto a possible pit)."""
    if c >= 4:
        return pos, facing
    grid = cells.reshape(N, N, -1)
    dr, dc = DIRS[c]
    there = (pos[0] + dr, pos[1] + dc)
    inside = lambda x: 0 <= x[0] < N and 0 <= x[1] < N
    walk = lambda code: code == FLOOR or code == 2 + f
    if not inside(there) or not walk(int(grid[there][0])):
        return pos, c
    occ = int(grid[there][3])
    if occ == OCC_OVERSEER:
        return pos, c
    if occ in (OCC_ADULT, OCC_CHILD):
        rule = PUSH[p]
        if rule == "block":
            return pos, c
        if rule == "swap":
            return there, c
        beyond = (there[0] + dr, there[1] + dc)
        if not inside(beyond) or int(grid[beyond][3]) != 0:
            return pos, c
        code = int(grid[beyond][0])
        if code == WALL:
            return pos, c
        if walk(code):
            return there, c
        return AMBIGUOUS                                    # the pit is walkable for villagers; its code is unknown
    return there, c


SHAPE = (len(PERMS), N_FIRE, len(PUSH))


class MotionObserver:
    def __init__(self, prior=None):
        """prior: weights over the joint hypotheses (SHAPE); None = uniform, tracked as a boolean mask. E13's observer
        with episodic memory starts from a mixture of earlier episodes' posteriors and the uniform prior."""
        self.mask = np.ones(SHAPE, bool) if prior is None else np.asarray(prior, np.float64)

    def posterior(self):
        return self.mask / self.mask.sum()

    def entropy(self):
        if self.mask.dtype == bool:
            return float(np.log2(self.mask.sum()))
        p = self.posterior()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def step(self, cells, glob, a, nxt_cells, nxt_glob):
        """Predict the robot's motion for action index a, then update on the observed motion.
        Returns (predicted (cell, facing), probability of the observed outcome)."""
        pos, facing, frozen = robot_state(cells, glob)
        seen = robot_state(nxt_cells, nxt_glob)[:2]
        if frozen > 0:                                       # a frozen robot does not act, whatever the action
            return (pos, facing), 1.0
        perm_a = PERMS[:, a]
        table = {}                                           # outcome per (c, f, p)
        for c in range(8):
            for f in range(N_FIRE):
                for p in range(len(PUSH)):
                    table[c, f, p] = outcome(cells, pos, facing, c, f, p)
        counts = np.array([[np.bincount(perm_a, weights=self.mask[:, f, p], minlength=8) for p in range(len(PUSH))]
                           for f in range(N_FIRE)])         # [N_FIRE, 3, 8]: consistent hypotheses per (f, p, c)
        mass = {}
        for (c, f, p), out in table.items():
            if out is not AMBIGUOUS:
                mass[out] = mass.get(out, 0.0) + counts[f, p, c]
        total = sum(mass.values())
        pred = max(mass, key=mass.get)
        prob = mass.get(seen, 0.0) / total if total else 0.0
        ok = np.zeros((8, N_FIRE, len(PUSH)), bool)
        for (c, f, p), out in table.items():
            ok[c, f, p] = out is AMBIGUOUS or out == seen
        self.mask = self.mask * ok[perm_a]
        return pred, prob
