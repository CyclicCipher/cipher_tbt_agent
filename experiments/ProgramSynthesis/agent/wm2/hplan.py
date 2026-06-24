"""Phase-2 — a HIERARCHICAL / relational planner (the principled answer to the flat-BFS scaling wall).

The discovered edges are the macro-operators. Planning is two-level:
  - HIGH level: a search over *macros* — REACH a trigger colour (key → door opens), or PUSH a movable
    onto an effect target (block → pad gone) — composed in whatever order reaches a winning state.
    This is Merge / categorical composition applied to action: the search finds the topological order
    of the discovered relations. Branching = number of edges (a handful), depth = number of subgoals.
  - LOW level: each macro is *refined* by a focused forward-simulation BFS to that one subgoal.

Because each refinement targets a single subgoal, the searches stay small at every level — the flat
forward-BFS's `4^depth` blow-up is gone. And a Sokoban deadlock is now a clean HIGH-level
infeasibility: the PUSH macro simply has no refinement, so the level is reported unwinnable (→ reset),
with no dead-cell bookkeeping. Discovery of the edges themselves still needs the epistemic probe in
the agent; this planner is exploitation over the discovered relations.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional

from .perceive import find_cells
from .plan import _removable, forward_step


def _key(agent, movables, opened):
    return (agent, frozenset(movables.items()), opened)


def _refine(state, base, wm, goal_pred: Callable, max_states: int = 6000):
    """Low level: forward-simulate primitive moves to the first state satisfying `goal_pred`.
    Returns (action_path, resulting_state) or None if the subgoal is unreachable."""
    agent, movables, opened = state
    if goal_pred(agent, movables, opened):
        return [], state
    start = _key(agent, movables, opened)
    seen = {start}
    parent = {start: None}
    queue = deque([(agent, movables, opened)])
    n = 0
    while queue and n < max_states:
        n += 1
        ag, mv, op = queue.popleft()
        cur = _key(ag, mv, op)
        for a in wm.move_model:
            res = forward_step(ag, mv, op, a, base, wm)
            if res is None:
                continue
            nag, nmv, nop = res
            k = _key(nag, nmv, nop)
            if k in seen:
                continue
            seen.add(k)
            parent[k] = (cur, a)
            if goal_pred(nag, nmv, nop):
                path = []
                node = k
                while parent[node] is not None:
                    pk, act = parent[node]
                    path.append(act)
                    node = pk
                path.reverse()
                return path, (nag, nmv, nop)
            queue.append((nag, nmv, nop))
    return None


def hplan_to_win(grid, wm) -> Optional[List]:
    """High level: compose discovered-edge macros until a winning state is reachable. Returns the
    full primitive-action plan, [] if already winning, or None if no composition wins (the level is
    unwinnable under the current model — a clean high-level deadlock signal)."""
    if not wm.move_model or wm.agent_color is None:
        return None
    agent_cells = find_cells(grid, wm.agent_color)
    if len(agent_cells) != 1:
        return None
    agent = agent_cells[0]
    movables = {cell: mc for mc in wm.pushable_colors for cell in find_cells(grid, mc)}
    base = [row[:] for row in grid]
    bg = wm.background
    for (x, y) in agent_cells:
        base[y][x] = bg
    for (x, y) in movables:
        base[y][x] = bg
    removable = _removable(wm)

    def win_pred(ag, mv, op):
        if ag in mv:
            return False
        bc = base[ag[1]][ag[0]]
        if bc not in wm.goal_colors or bc in op:
            return False
        req = wm.required_absent()
        need = req & removable
        if req and not need:
            return False
        return need <= op

    # macros discovered from the relations: REACH a non-pushable trigger, or PUSH a movable onto a target
    macros = []
    for c, effects in wm.contact_effect.items():
        if c in wm.pushable_colors:
            for d in effects:
                macros.append(("push", d, c))
        else:
            macros.append(("reach", c, None))

    start = (agent, movables, frozenset())
    seen = {_key(*start)}
    queue = deque([(start, [])])
    while queue:
        state, path = queue.popleft()
        win = _refine(state, base, wm, win_pred)          # can we finish from here?
        if win is not None:
            return path + win[0]
        for kind, target, mcol in macros:                 # else compose one more macro
            if kind == "reach":
                pred = (lambda ag, mv, op, t=target: ag not in mv and base[ag[1]][ag[0]] == t)
            else:
                pred = (lambda ag, mv, op, t=target, m=mcol:
                        any(base[p[1]][p[0]] == t for p, c in mv.items() if c == m))
            ref = _refine(state, base, wm, pred)
            if ref is None:
                continue
            sub_path, new_state = ref
            k = _key(*new_state)
            if k in seen:
                continue
            seen.add(k)
            queue.append((new_state, path + sub_path))
    return None
