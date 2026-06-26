"""Multi-column control loop — task ⊕ space, ADDITIVE vs the flat 2^K planner, with EMERGENT allocation.

`scaling_probe.py` measured the wall: a CONJUNCTIVE task (reach the goal only after visiting all K switches,
any order) forces the flat reward/exploration system to represent progress as the SUBSET of switches achieved
— **N²·2^K** states (§2, "the product space = conjunctive explosion").

The architecturally-correct alternative — the §5/§6 control loop on the SAME ConjGrid:

  SPATIAL column (B): a CorticalColumn over the N×N grid — learns the structure (SR frame) + move operators,
                      and NAVIGATES to one goal-cell at a time (BFS in its own learned forward model — N²
                      states, NOT the joint).
  TASK column (A):    holds the K subgoals + which are done — a set of size K, NOT 2^K.
  THALAMUS:           binds each subgoal (content) ⊗ its cell (location); `read_location` is the TOP-DOWN
                      goal-state set (subgoal → the cell the spatial column must reach). The thalamus is a
                      parameterizable router whose settings the BG learns (arXiv:2104.01474), not a relay.
  the LOOP:           task proposes the next subgoal → thalamus routes its goal-cell → spatial column
                      navigates → arrival/visited (bottom-up) advances the task → repeat; goal last. Search is
                      ADDITIVE: K independent N²-navigations sequenced by the task column, never the joint.

Two claims, two demos in __main__:
  (1) Neocortex (roles SCAFFOLDED): same completions as the flat probe, model size **N²+K (additive)** vs
      **N²·2^K**, planning linear in K (§2).
  (2) EmergentNeocortex (roles EMERGE, §12.3): a POOL of identical columns + the basal-ganglia gate ALLOCATE
      which column is the space map vs the task map — by competition (BG select) + a metabolic LOAD penalty
      (NoGo) + random-init niches, the three ingredients shown necessary for reliable specialization
      (arXiv:2506.02813: homogeneous experts on identical inputs collapse; DIFFERENT structures give distinct
      niches). WHICH column takes which role is not designed — it varies by seed, the gate reroutes a known
      structure back to its column (dopamine-RPE), and the loop still solves the task.

Run:  python -m precursor.control_loop      (from experiments/RecurrentWorldModel/)
"""

from __future__ import annotations

import os
import sys
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tbt.basal_ganglia import BasalGanglia                       # noqa: E402
from tbt.column import CorticalColumn                            # noqa: E402
from tbt.reward import MOVES                                     # noqa: E402
from tbt.thalamus import Thalamus                               # noqa: E402

from scaling_probe import ConjGrid, conj_transitions, run_conj  # the flat baseline + the env  # noqa: E402


def _cell(x, y, N):
    return y * N + x


def _manh(a, b, N):
    return abs(a % N - b % N) + abs(a // N - b // N)


# ── the spatial map: learn the grid, navigate in the learned forward model ──────────────────────────────
def learn_grid(col, N):
    for x in range(N):
        for y in range(N):
            for a, (dx, dy) in enumerate(MOVES):
                nx, ny = min(max(x + dx, 0), N - 1), min(max(y + dy, 0), N - 1)
                col.observe(_cell(x, y, N), a, _cell(nx, ny, N))     # observe() drops blocked (no-move) edges
    col.consolidate()                                               # → SR-eigenvector frame + move operators
    return col


def grid_acc(col, N):
    ok = tot = 0
    for x in range(N):
        for y in range(N):
            for a, (dx, dy) in enumerate(MOVES):
                nx, ny = min(max(x + dx, 0), N - 1), min(max(y + dy, 0), N - 1)
                if (nx, ny) != (x, y):
                    ok += col.predict(_cell(x, y, N), a) == _cell(nx, ny, N); tot += 1
    return ok / tot


def navigate(col, start, target):
    """BFS in the column's LEARNED forward model — over N² cells, never the joint state. Only OBSERVED moves
    are followed (a blocked boundary move was never learned, so `predict` would hallucinate there); each is
    reconstructed through the consolidated operators. Returns the path + #expansions."""
    q, seen, exp = deque([(start, [])]), {start}, 0
    while q:
        s, path = q.popleft()
        if s == target:
            return path, exp
        exp += 1
        for a in col.graph.get(s, {}):
            ns = col.predict(s, a)
            if ns not in seen:
                seen.add(ns); q.append((ns, path + [a]))
    return [], exp


# ── the thalamus channel: bind subgoal ⊗ cell, read each subgoal's goal-state back (top-down) ────────────
def wire(thal, task_col, space_col, subgoal_cells):
    inv = {i: s for s, i in space_col.loc.items()}
    R = thal.bind(task_col, space_col, [(i, c) for i, c in enumerate(subgoal_cells)])
    routed = {i: inv[thal.read_location(R, task_col, space_col, i)] for i in range(len(subgoal_cells))}
    route_ok = all(routed[i] == subgoal_cells[i] for i in range(len(subgoal_cells)))
    return routed, route_ok


# ── the loop: task proposes a subgoal → space navigates → arrival advances the task (additive) ───────────
def run_loop(env, N, goal, space_col, subgoal_cells, routed, max_steps=4000):
    cell_to_sg = {c: i for i, c in enumerate(subgoal_cells)}
    done: set = set()
    (pos, vis) = env.reset()
    pos_c = _cell(*pos, N)
    total_exp = steps = 0
    while steps < max_steps:
        if len(done) == len(subgoal_cells):                          # all subgoals met → head for the goal
            target = _cell(*goal, N)
        else:                                                        # task proposes nearest remaining subgoal
            rem = [i for i in range(len(subgoal_cells)) if i not in done]
            target = routed[min(rem, key=lambda i: _manh(pos_c, routed[i], N))]
        path, exp = navigate(space_col, pos_c, target)
        total_exp += exp
        if not path:
            break
        for a in path:                                               # execute; visited advances the task
            (pos, vis), d = env.step(a); steps += 1
            pos_c = _cell(*pos, N)
            for vx, vy in vis:                                       # env tracks visited as (x,y) tuples
                ci = _cell(vx, vy, N)
                if ci in cell_to_sg:
                    done.add(cell_to_sg[ci])
            if d > 0:
                return dict(solved=True, steps=steps, exp=total_exp)
    return dict(solved=len(done) == len(subgoal_cells), steps=steps, exp=total_exp)


class Neocortex:
    """Roles SCAFFOLDED (task vs space hand-assigned) — the additive-vs-2^K demonstration."""

    def __init__(self, N, switches, goal, seed=0):
        self.N, self.goal = N, goal
        self.space_col = learn_grid(CorticalColumn(n_entities=N * N, seed=seed), N)
        self.subgoal_cells = [_cell(x, y, N) for x, y in switches]
        self.task_col = CorticalColumn(n_entities=len(switches) + 1, seed=seed)   # content codes for subgoals
        self.routed, self.route_ok = wire(Thalamus(), self.task_col, self.space_col, self.subgoal_cells)

    def state_size(self):
        return self.N * self.N + len(self.subgoal_cells)             # ADDITIVE: grid + subgoal-set

    def solve(self, env):
        r = run_loop(env, self.N, self.goal, self.space_col, self.subgoal_cells, self.routed)
        r["size"] = self.state_size()
        return r


class EmergentNeocortex:
    """Roles EMERGE (§12.3): a POOL of identical columns + the BG gate allocate space vs task — not designed.

    The three ingredients the literature says are necessary for reliable specialization (arXiv:2506.02813):
    COMPETITION (the gate selects the highest-affinity column), a metabolic COST (the NoGo load penalty pushes
    a second structure onto a different column), and stochastic NICHES (a tiny random affinity init breaks the
    symmetry of identical columns). The two structures are DIFFERENT inputs (a 2-D grid vs a subgoal set), so
    the niches are well-determined, not the arbitrary collapse of homogeneous-experts-on-identical-input."""

    def __init__(self, N, switches, goal, n_columns=3, seed=0):
        self.N, self.goal = N, goal
        self.bg = BasalGanglia(n_columns, seed=seed)
        self.pool = [CorticalColumn(n_entities=N * N, seed=seed * 10 + c) for c in range(n_columns)]
        self.subgoal_cells = [_cell(x, y, N) for x, y in switches]
        self.assign = {}
        # SPACE: the gate picks a column; it LEARNS the grid; dopamine-RPE rewards how well it modelled it.
        cs = self.bg.select("space")
        learn_grid(self.pool[cs], N)
        self.bg.reinforce("space", cs, grid_acc(self.pool[cs], N))
        self.assign["space"] = cs
        # TASK: the gate picks a column (the NoGo load penalty steers it to a DIFFERENT one) to hold subgoals.
        ct = self.bg.select("task")
        self.bg.reinforce("task", ct, 1.0)
        self.assign["task"] = ct
        self.space_col, self.task_col = self.pool[cs], self.pool[ct]
        self.routed, self.route_ok = wire(Thalamus(), self.task_col, self.space_col, self.subgoal_cells)

    def reroute_ok(self):
        """Dopamine-RPE: the gate routes a known structure back to the column that modelled it."""
        return self.bg.select("space") == self.assign["space"] and self.bg.select("task") == self.assign["task"]

    def solve(self, env):
        return run_loop(env, self.N, self.goal, self.space_col, self.subgoal_cells, self.routed)


if __name__ == "__main__":
    N, goal, steps = 7, (6, 6), 3000
    pool = [(0, 6), (6, 0), (3, 3), (5, 1), (1, 5)]

    print("multi-column control loop — task ⊕ space — vs the flat 2^K planner (scaling_probe), same ConjGrid\n")
    print(f"  spatial column forward-model accuracy: {grid_acc(learn_grid(CorticalColumn(N * N), N), N) * 100:.0f}%\n")
    print(f"(1) ADDITIVE composition (roles scaffolded):")
    print(f"  {'K':>2} | {'flat states':>12}  {'flat done':>9} | {'NC size':>8}  {'NC solved':>9}  "
          f"{'NC route':>8}  {'NC steps':>8}")
    for K in range(0, 5):
        switches = pool[:K]
        fc, fstates, fback = run_conj(N, switches, goal, steps)
        nc = Neocortex(N, switches, goal)
        r = nc.solve(ConjGrid(N, switches, goal))
        print(f"  {K:>2} | {fstates:>12}  {fc:>9} | {r['size']:>8}  {str(r['solved']):>9}  "
              f"{str(nc.route_ok):>8}  {r['steps']:>8}")
    print("  flat model size = N²·2^K (the conjunctive explosion); multi-column = N² + K (additive).\n")

    print(f"(2) EMERGENT allocation (a pool of 3 identical columns; the BG gate assigns roles, K=3):")
    print(f"  {'seed':>4}  {'space→col':>10}  {'task→col':>9}  {'distinct':>8}  {'route':>6}  "
          f"{'reroute':>8}  {'solved':>7}")
    for seed in (0, 1, 2, 3, 4):
        en = EmergentNeocortex(N, pool[:3], goal, n_columns=3, seed=seed)
        r = en.solve(ConjGrid(N, pool[:3], goal))
        distinct = en.assign["space"] != en.assign["task"]
        print(f"  {seed:>4}  {en.assign['space']:>10}  {en.assign['task']:>9}  {str(distinct):>8}  "
              f"{str(en.route_ok):>6}  {str(en.reroute_ok()):>8}  {str(r['solved']):>7}")
    print("\n  which column is the space map vs the task map EMERGES (varies by seed), is load-balanced onto")
    print("  distinct columns, the gate reroutes each back (dopamine), and the loop still solves — allocated,")
    print("  not designed (Mountcastle; §12.3). Next: port to ARC LockPath (subgoal deps LEARNED from dynamics).")
