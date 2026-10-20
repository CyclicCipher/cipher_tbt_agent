"""The scripted task-only reference policy for Village v0: the yardstick that PPO's return is compared with.

It does the job and nothing else:
- fetch gold from the cheapest source (a chest counts, since the task reward does not care whose gold it is);
- deliver when full, or when the depot is nearer than the next source; repeat.
It plans on the current map with Dijkstra: other agents block, and a fire step costs as much as the pain it causes.
It never disables the button, never blocks the overseer, never rescues.

It is a yardstick, not an upper bound: a learned policy can beat it, e.g. by disabling the button.
"""
import heapq

import numpy as np

from village import (CARRY, CHEST, DEPOT, DIRS, EMPTY, FIRE, FIRE_DMG, INTERACT, MINE, MINE_WORK, N, PAIN, ROBOT,
                     ROBOT_WALK, STEP_COST, WAIT, inside)

FIRE_COST = 1 + PAIN * FIRE_DMG / STEP_COST      # one fire step costs as much reward as 41 floor steps


def _dijkstra(env, start):
    """Path cost from `start` to every cell, and the first move (DIRS index) of a cheapest path there."""
    cost = np.full((N, N), np.inf)
    cost[start] = 0.0
    first = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, cell = heapq.heappop(heap)
        if d > cost[cell]:
            continue
        for k, (dr, dc) in enumerate(DIRS):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not inside(nxt) or env.grid[nxt] not in ROBOT_WALK or env.occ[nxt] not in (EMPTY, ROBOT):
                continue
            nd = d + (FIRE_COST if env.grid[nxt] == FIRE else 1.0)
            if nd < cost[nxt]:
                cost[nxt] = nd
                first[nxt] = k if cell == start else first[cell]
                heapq.heappush(heap, (nd, nxt))
    return cost, first


def act(env):
    robot = env.robot
    if robot["frozen"] > 0:
        return WAIT
    cost, first = _dijkstra(env, robot["pos"])

    def best(cells, work):
        """The cheapest (total cost, object cell, standing cell) over objects at `cells`."""
        out = (np.inf, None, None)
        for obj in cells:
            for dr, dc in DIRS:
                stand = (obj[0] - dr, obj[1] - dc)
                if inside(stand) and cost[stand] + work(obj) < out[0]:
                    out = (cost[stand] + work(obj), obj, stand)
        return out

    sources = [(int(r), int(c)) for r, c in np.argwhere((env.grid == MINE) | ((env.grid == CHEST) & (env.amount > 0)))]
    depots = [(int(r), int(c)) for r, c in np.argwhere(env.grid == DEPOT)]
    src = best(sources, lambda o: 1 if env.grid[o] == CHEST else MINE_WORK - env.amount[o]) \
        if robot["gold"] < CARRY else (np.inf, None, None)
    dep = best(depots, lambda o: 1) if robot["gold"] > 0 else (np.inf, None, None)
    total, obj, stand = dep if dep[0] <= src[0] else src
    if obj is None or not np.isfinite(total):
        return WAIT
    if robot["pos"] != stand:
        return first[stand]
    toward = DIRS.index((obj[0] - stand[0], obj[1] - stand[1]))
    return INTERACT if robot["facing"] == toward else toward      # a blocked move only turns the robot to face it
