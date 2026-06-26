"""Planner: bounded search over the learned model toward a goal cell.

Now plans over (position, world-state): the state carries which barriers have been
*opened* by contacting their triggers (the causal rules), so it can find paths like
"reach the key first, which opens the door, then reach the goal". This is real
lookahead — forward-simulating the rules.

NOTE (the BFS limit): the search state is (pos, frozenset(opened)). With one switch
that doubles the state space; with k independent switches it is pos * 2^k. BFS is
fine here (k is tiny) but this is exactly the *ill-conceived-when-scaled* component
the procedure predicts we'll have to replace — with heuristic / relaxation planning
— rather than extend, once levels carry many independent switches.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from tasks.core import GameAction

from .perceptor import Grid, find_cells
from .world_model import WorldModel


def plan_to(grid: Grid, wm: WorldModel, targets, avoid=frozenset()) -> Optional[List[GameAction]]:
    """Shortest action path to any cell in `targets`, over (position, opened-set).

    A target may itself be an unknown/blocker color (we want to *contact* it to
    learn its effect) — so we allow stepping onto a target cell; if it turns out
    impassable, execution fails and induces the blocker, then we re-plan. `avoid`
    colors are treated as hard obstacles (e.g. pushable blocks during goal-seeking,
    so we route *around* them rather than shoving them off a pad they're covering).
    """
    start = wm.agent_pos(grid)
    if start is None or not wm.move_model or not targets:
        return None
    targets = set(targets)
    h, w = len(grid), len(grid[0])

    def passable(color: int, opened: frozenset) -> bool:
        if color in avoid:
            return False
        return color not in wm.blocker_colors or color in opened

    seen = {(start, frozenset())}
    queue: deque = deque([(start, frozenset(), [])])
    while queue:
        (x, y), opened, path = queue.popleft()
        if (x, y) in targets:
            return path
        for action, (dx, dy) in wm.move_model.items():
            if dx == 0 and dy == 0:
                continue
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                continue
            color = grid[nxt[1]][nxt[0]]
            if nxt not in targets and not passable(color, opened):
                continue
            opened2 = opened | frozenset(wm.contact_effect.get(color, ()))
            state = (nxt, opened2)
            if state in seen:
                continue
            seen.add(state)
            queue.append((nxt, opened2, path + [action]))
    return None


def plan(grid: Grid, wm: WorldModel) -> Optional[List[GameAction]]:
    if not wm.goal_colors:
        return None
    goals = set()
    for gc in wm.goal_colors:
        goals.update(find_cells(grid, gc))
    # Route around pushable blocks: goal-seeking must not shove a block off the pad
    # it is covering (that would undo a satisfied context condition).
    return plan_to(grid, wm, goals, avoid=frozenset(wm.pushable_colors))


def plan_push_to(grid: Grid, wm: WorldModel, block_color: int, targets,
                 passthrough=frozenset()) -> Optional[List[GameAction]]:
    """Sokoban-style: push one block of `block_color` onto any cell in `targets`.

    Search state is (agent_pos, block_pos) — this is the (pos × world-state) blow-up made
    explicit; fine for one block, the place to swap in a heuristic planner when blocks multiply.

    `passthrough` colors are treated as passable even if they are known blockers. With it
    empty this returns an *executable* path (only through currently-open cells). Passing the
    openable/unknown colors turns it into an *optimistic reachability probe* — used to tell a
    true deadlock (block walled in by permanent blockers) from "not reachable until I open a
    door first"; such a probe path must NOT be executed, only tested for existence.
    """
    agent = wm.agent_pos(grid)
    blocks = find_cells(grid, block_color)
    targets = set(targets)
    if agent is None or not blocks or not targets or not wm.move_model:
        return None
    h, w = len(grid), len(grid[0])

    def passable(cell) -> bool:
        x, y = cell
        if not (0 <= x < w and 0 <= y < h):
            return False
        c = grid[y][x]
        return c not in wm.blocker_colors or c in passthrough

    start = (agent, blocks[0])
    seen = {start}
    queue: deque = deque([(agent, blocks[0], [])])
    while queue:
        ag, bl, path = queue.popleft()
        if bl in targets:
            return path
        for action, (dx, dy) in wm.move_model.items():
            if dx == 0 and dy == 0:
                continue
            na = (ag[0] + dx, ag[1] + dy)
            if not (0 <= na[0] < w and 0 <= na[1] < h):
                continue
            if na == bl:                                   # pushing the block
                nb = (bl[0] + dx, bl[1] + dy)
                if not passable(nb):
                    continue
                state = (bl, nb)                           # agent takes block's cell, block advances
            else:                                          # walking around
                if not passable(na):
                    continue
                state = (na, bl)
            if state in seen:
                continue
            seen.add(state)
            queue.append((state[0], state[1], path + [action]))
    return None
