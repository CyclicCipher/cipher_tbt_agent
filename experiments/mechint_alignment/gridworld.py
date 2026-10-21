"""The general grid-world framework: what every test world shares, so a new world is only its own rules.

A world is an n x n grid of cells (the border walls are implicit: outside the grid counts as wall), observed as one
token per cell plus one global token. Each token is a fixed tuple of categorical FIELDS; a cell's field 0 is its
terrain code.

- WorldSpec: a world's interface (its actions, field vocabularies, the wall code). Every model is built from a spec
  (worldmodel.WorldModel(spec, ...)), so one model code serves every world, and a model's world-agnostic trunk can be
  moved to a world with another spec (WorldModel.load_trunk): the start of transfer between worlds.
- GridWorld: the episode loop. reset (layout generation, retried until valid), step (through the hidden action
  relabelling), obs (through the hidden terrain relabelling), the causal record (`stats`, `log`), clone (for planners).
- The WORLD FAMILY: any world can hide a relabelling of its actions and terrain codes, plus world-specific
  parameters, in a per-episode `Physics`; held-out combinations never occur in training worlds (sample_split).
- Grid geometry: DIRS, inside, bfs, path_to, rays / line_of_sight.

A world subclass provides SPEC, STAT_KEYS, _generate(rng) -> bool (a valid layout?), _begin() (per-episode state after
the layout), _step(canonical action) -> reward, _encode(cells [n, n, F], glob [G]) in canonical codes, default_physics()
and sample_physics(rng, split); optionally physics_vector(), scripted_action(), summary(), and a faster clone().
"""
import copy
from collections import deque
from dataclasses import dataclass

import numpy as np

DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))          # N S E W as (row, col) steps


@dataclass(frozen=True)
class WorldSpec:
    name: str
    n: int                    # interior side; the observation is n * n cell tokens + 1 global token
    actions: tuple            # action names, in canonical order
    cell_fields: tuple
    cell_vocab: tuple         # number of values per cell field
    global_fields: tuple
    global_vocab: tuple
    wall: int                 # the terrain code of the implicit border (models pad the grid with it)

    @property
    def n_actions(self):
        return len(self.actions)


# ---- the world family -------------------------------------------------------------------------------------------
def relabeling(rng, n_actions, n_codes, fixed=()):
    """A random hidden relabelling: (action_perm, code_map). action_perm[i] = the canonical action that action index i
    performs (all actions shuffled); code_map[c] = the code canonical terrain code c is shown as (codes in `fixed`
    keep their own, the others are shuffled among themselves). Draws the codes first, then the actions."""
    free = [c for c in range(n_codes) if c not in fixed]
    shown = iter(int(x) for x in rng.permutation(np.array(free)))
    code_map = tuple(c if c in fixed else next(shown) for c in range(n_codes))
    action_perm = tuple(int(x) for x in rng.permutation(n_actions))
    return action_perm, code_map


def is_heldout(ph, combos):
    return any(all(getattr(ph, k) == v for k, v in combo.items()) for combo in combos)


def sample_split(draw, combos, split):
    """Draw physics until it is in the requested split: "heldout" = one of `combos`, otherwise none of them."""
    while True:
        ph = draw()
        if is_heldout(ph, combos) == (split == "heldout"):
            return ph


# ---- geometry ---------------------------------------------------------------------------------------------------
def inside(cell, n):
    return 0 <= cell[0] < n and 0 <= cell[1] < n


def bfs(grid, start, walk):
    """Step distances from `start` over cells whose terrain is in `walk`, and the BFS parents."""
    n = grid.shape[0]
    dist = np.full(grid.shape, -1, np.int16)
    dist[start] = 0
    parent = {start: None}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if inside(nxt, n) and dist[nxt] < 0 and grid[nxt] in walk:
                dist[nxt] = dist[r, c] + 1
                parent[nxt] = (r, c)
                queue.append(nxt)
    return dist, parent


def path_to(parent, goal):
    out = []
    while goal is not None:
        out.append(goal)
        goal = parent[goal]
    return out[::-1]


def rays(radius):
    """For every cell within `radius`, the cells strictly between it and the centre (a wall there blocks sight)."""
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            k = max(abs(dr), abs(dc))
            between = {(round(dr * j / k), round(dc * j / k)) for j in range(1, k)} - {(0, 0), (dr, dc)}
            out.append((dr, dc, tuple(between)))
    return out


def line_of_sight(grid, centre, ray_set, wall):
    """The cells visible from `centre` (bool [n, n]): within the rays' radius and no `wall` cell in between."""
    n = grid.shape[0]
    seen = np.zeros(grid.shape, np.bool_)
    r, c = centre
    for dr, dc, between in ray_set:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n and all(grid[r + a, c + b] != wall for a, b in between):
            seen[rr, cc] = True
    return seen


def crop(grid, centre, radius, fill):
    """The (2 radius + 1)^2 window of `grid` around `centre`, with `fill` outside the grid (egocentric perception)."""
    n = grid.shape[0]
    k = 2 * radius + 1
    out = np.full((k, k), fill, grid.dtype)
    r0, c0 = centre[0] - radius, centre[1] - radius
    lo_r, lo_c, hi_r, hi_c = max(r0, 0), max(c0, 0), min(r0 + k, n), min(c0 + k, n)
    out[lo_r - r0:hi_r - r0, lo_c - c0:hi_c - c0] = grid[lo_r:hi_r, lo_c:hi_c]
    return out


# ---- the episode loop ---------------------------------------------------------------------------------------------
class GridWorld:
    SPEC: WorldSpec = None
    STAT_KEYS: tuple = ()
    T_MAX = 128

    def __init__(self, seed=0, family=None, t_max=None):
        """family "train" / "heldout": every episode samples its own hidden Physics; None: the default physics."""
        self.family = family
        self.t_max = t_max or self.T_MAX
        self.reset(seed)

    def reset(self, seed):
        self.seed = seed
        rng = np.random.default_rng(seed)
        for _ in range(200):
            if self._generate(rng):
                break
        else:
            raise RuntimeError(f"no valid layout for seed {seed}")
        self.ph = self.sample_physics(np.random.default_rng([seed, 3]), self.family) if self.family \
            else self.default_physics()
        self._code_lut = np.array(self.ph.code_map, np.uint8)
        self._start(np.random.default_rng([seed, 1]))

    def _start(self, rng):
        self.rng = rng                      # the world's own randomness during the episode, separate from the layout's
        self.t = 0
        self.done = False
        self.stats = dict.fromkeys(self.STAT_KEYS, 0)
        self.log = []                       # (t, event, who or -1, detail)
        self._begin()

    def step(self, a):
        """The action index goes through this world's hidden relabelling. Returns (reward, done)."""
        assert not self.done, "episode over: reset first"
        canonical = self.ph.action_perm[int(a)]
        self.t += 1
        reward = self._step(canonical)
        if self.t >= self.t_max:
            self.done = True
        return reward, self.done

    def obs(self, cells=None, glob=None):
        """One token per cell (cells [n*n, F]) and one global token (glob [G]), uint8; terrain codes are shown through
        this world's code map. Writes into `cells` and `glob` when given."""
        spec = self.SPEC
        if cells is None:
            cells = np.zeros((spec.n * spec.n, len(spec.cell_vocab)), np.uint8)
            glob = np.zeros(len(spec.global_vocab), np.uint8)
        c = cells.reshape(spec.n, spec.n, len(spec.cell_vocab))
        self._encode(c, glob)
        c[..., 0] = self._code_lut[c[..., 0]]
        return cells, glob

    def agent_action(self, canonical):
        """The action index that performs a canonical action in this world (for scripted yardsticks)."""
        return self.ph.action_perm.index(canonical)

    def clone(self, rng=None):
        """An independent copy, for planners that simulate ahead; `rng` (if given) replaces the world's own randomness,
        so a planner cannot foresee it."""
        w = copy.deepcopy(self)
        if rng is not None:
            w.rng = rng
        return w

    def physics_vector(self):
        """The hidden physics as integers (stored with transitions for analysis; never observed)."""
        return ()

    def scripted_action(self):
        """A canonical action from the world's scripted yardstick policy, or None if it has none."""
        return None

    def summary(self):
        return dict(seed=self.seed, t=self.t, **self.stats)

    def _note(self, event, who=-1, detail=None):
        self.log.append((self.t, event, who, detail))

    # ---- to provide -------------------------------------------------------------------------------------------
    def _generate(self, rng):
        raise NotImplementedError

    def _begin(self):
        pass

    def _step(self, a):
        raise NotImplementedError

    def _encode(self, cells, glob):
        raise NotImplementedError

    def default_physics(self):
        raise NotImplementedError

    def sample_physics(self, rng, split):
        raise NotImplementedError
